from common import *
from itertools import *
import qdr
from operator import itemgetter, attrgetter
from scipy.spatial import distance
from importlib import reload
from functools import partial
import multiprocessing
import threading
import time
import ujson
from multiprocessing.pool import ThreadPool
import features as ft
import fetching as fc

reload(ft)
reload(fc)

SEED = 0


class Data:
    def __init__(self, ixs):
        """
        load data to RAM for fast access
        :param ixs: document indexes
        """
        ixs = sorted(list(set(ixs)))
        self.ixs = ixs
        self.all_ids = fc.load_keys('../data/keys.json')
        self.ids = set(itemgetter(*ixs)(self.all_ids))
        corpus_files = glob('../data/documents/*')
        corpus_files.sort(key=natural_keys)
        self.corpus_files = corpus_files

        self.dictionary = Dictionary.load('../data/corpus.dict')
        self.token2id = self.dictionary.token2id
        logging.info("loading docs to RAM")
        self.docs_ram_dict = self.push_docs_to_ram()

        self.wv = Word2Vec.load('../data/w2v_200_5_w8').wv

        logging.info("loading qdr model")
        self.qdr = qdr.QueryDocumentRelevance.load_from_file('../data/qdr_model.gz')

        logging.info("loading mpk data")
        with open('../data/all_mpk.pkl', 'rb') as f:
            self.all_mpk = pickle.load(f)
        self.mpk = ft.MPK()
        logging.info('data is loaded')

    def _push_worker(self, filename):
        with GzipFile(filename) as f:
            data = ujson.load(f)
        common_keys = self.ids.intersection(data.keys())
        _docs = {_id: {k: [self.token2id[w] for s in v for w in s]
                       for k, v in data[_id].items()}
                 for _id in common_keys}
        return _docs

    def push_docs_to_ram(self):
        fname = '../data/docs_ram.pkl'
        if path.exists(fname):
            with open(fname, 'rb') as f:
                docs_ram = pickle.load(f)
            logging.info('docs loaded from %s' % fname)
            return docs_ram

        start = time.time()
        docs_ram = {}

        for filename in tqdm(self.corpus_files):
            docs_ram.update(self._push_worker(filename))

        end = time.time()
        logging.info('docs loaded in %f m.' % ((end - start) / 60.))
        with open(fname, 'wb') as f:
            pickle.dump(docs_ram, f)
        logging.info('docs saved to %s' % fname)
        return docs_ram

    @staticmethod
    def keywise(func, *args, prefix=''):
        keys = [set(d.keys()) for d in args]
        keys = set.intersection(*keys)
        _ft = {'%s_%s' % (k, prefix): func(*(d[k] for d in args)) for k in keys}
        _ft = {'%s_%s' % (k, ki): vi for k, v in _ft.items() for ki, vi in v.items()}
        return _ft

    @staticmethod
    def jaccard(v1, v2):
        s1 = set(v1)
        s2 = set(v2)
        lu = len(s1.union(s2))
        if lu:
            j = len(s1.intersection(s2)) / lu
        else:
            j = 0
        return {'j': j}

    @staticmethod
    def independent(words):
        return {"all": len(words), "unique": len(set(words))}

    @staticmethod
    def mean_vector(vectors):
        if vectors.ndim > 1:
            return vectors.mean(axis=0)
        else:
            return vectors

    def words2vec(self, words):
        words = [w for w in words if w in self.wv]
        if len(words):
            word_vecs = self.wv[words]
            vec = self.mean_vector(word_vecs)
            return vec

    def tag_vectors(self, doc):
        vecs = {k: self.words2vec(v) for k, v in doc.items()}
        vecs = {k: v for k, v in vecs.items() if v is not None}
        return vecs

    @staticmethod
    def cosine(v1, v2):
        return {'cos': distance.cosine(v1, v2)}

    def doc_info(self, d_ix):
        info = {}
        _id = self.all_ids[d_ix]

        info['id'] = _id
        info['words'] = {k: [self.dictionary[vi] for vi in v]
                         for k, v in self.docs_ram_dict[_id].items()}
        info['w2v'] = self.tag_vectors(info['words'])
        info['mpk'] = self.all_mpk[_id]['mpk']

        return info

    def score(self, q, d):
        ftrs = {'q': q['id'], 'd': d['id']}
        q_words = q['words']
        d_words = d['words']
        q_text = (w.encode() for w in chain.from_iterable(q_words.values()))
        d_text = (w.encode() for w in chain.from_iterable(d_words.values()))

        ftrs.update(self.qdr.score(q_text, d_text))
        ftrs.update(self.keywise(self.independent, q_words, prefix='q'))
        ftrs.update(self.keywise(self.independent, d_words, prefix='d'))
        ftrs.update(self.keywise(self.jaccard, q_words, d_words))
        ftrs.update(self.keywise(self.cosine, q['w2v'], d['w2v']))

        n = self.mpk.compare_mpk(q['mpk'], d['mpk'], self.mpk.way)
        ftrs['mpk'] = n

        return ftrs

    def process_one(self, q, d_ix, rank):
        d = self.doc_info(d_ix)
        _ft = self.score(q, d)
        _ft['rank'] = rank
        return _ft

    def process_triple(self, anc, pos, neg):
        q = self.doc_info(anc)
        ranks = [1] * len(pos) + [0] * len(neg)
        _s = [self.process_one(q, d_ix, rank)
              for rank, d_ix in zip(ranks, pos + neg)]
        return _s

    def scores_worker(self, samples):
        ftrs = []
        _l = sum([1+len(pos) +len(neg) for anc, pos, neg in samples])
        i = 0
        for anc, pos, neg in tqdm(samples):
            for _ft in self.process_triple(anc, pos, neg):
                ftrs.append(_ft)
            i += 1.+len(pos)+len(neg)
            print('%f%%' % (100.*i/_l))
        return ftrs

    def scores(self, samples, n_threads=cpu_count):
        with ThreadPool(processes=n_threads) as pool:
            res = pool.map(self.scores_worker, chunkify(samples, n_threads))
        ftrs = list(chain.from_iterable(res))
        return ftrs


with open('../data/sampled.json', 'r') as f:
    samples = json.load(f)

unique = sorted(list(set(chain.from_iterable([[anc] + pos + neg for anc, pos, neg in samples]))))

data = Data(unique)
ftrs = data.scores_worker(samples)
df = ft.to_dataframe(ftrs)
ft.save(df, '../data/ftrs_new.csv.gz', compression='gzip')
ftrs[:10000].to_csv('../data/ftrs_new_show.csv')


ftrs = pd.read_csv('../data/ftrs_new.csv.gz')


#   ################# test 184 ##############

all_ids = fc.load_keys('../data/keys.json')
with open('../data/gold_mongo.json', 'r') as f:
    gold = json.load(f)
ix_map = {vi: i for i, vi in enumerate(all_ids)}
_gold = {ix_map[k]: [ix_map[vi] for vi in v] for k,v in gold.items()}
keys = list(_gold.keys())

with open('../data/test_ixs.json', 'r') as f:
    test_ixs = json.load(f)

samples_test = []
for q_ix, vals in zip(keys, test_ixs):
    _g = _gold[q_ix]
    s = [q_ix, _g, [i for i,cos in vals if i not in set(_g)]]
    samples_test.append(s)

unique = sorted(list(set(chain.from_iterable([[anc] + pos + neg for anc, pos, neg in samples_test]))))

data = Data(unique)
ftrs = data.scores_worker(samples_test)
df = ft.to_dataframe(ftrs)
ft.save(df, '../data/ftrs_test_new.csv.gz', compression='gzip')

with open('../data/all_mpk.pkl', 'rb') as f:
    all_mpk = pickle.load(f)

#   ############################################

# all_ids = fc.load_keys('../data/keys.json')
# q_ix = all_ids.index('5984b7c2b6b1132856638528')
# d_ix = all_ids.index('5984b65cb6b1131291638512')
#
# data = Data([q_ix, d_ix])
#
# q = data.doc_info(q_ix)
# d = data.doc_info(d_ix)
#
# data.score(q, d)
