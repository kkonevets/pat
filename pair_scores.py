
from common import *
from itertools import *
import qdr
from operator import itemgetter
from scipy.spatial import distance
from importlib import reload
from functools import partial
import multiprocessing
import threading
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
        self.docs_ram_dict = self.push_docs_to_ram()

        self.wv = Word2Vec.load('../data/w2v_200_5_w8').wv

        logging.info("loading qdr model")
        self.qdr = qdr.QueryDocumentRelevance.load_from_file('../data/qdr_model.gz')

        logging.info("loading mpk data")
        with open('../data/all_mpk.pkl', 'rb') as f:
            self.all_mpk = pickle.load(f)
        self.mpk = ft.MPK()

    def _push_worker(self, files):
        docs_ram = {}

        for _id, doc in fc.iter_docs(files, encode=False, with_ids=True, as_is=True):
            if _id not in self.ids:
                continue
            _doc = {k: [self.token2id[w] for s in v for w in s] for k, v in doc.items()}
            docs_ram[_id] = _doc
        return docs_ram

    def push_docs_to_ram(self):
        logging.info("loading docs to RAM")
        docs_ram = {}

        func = partial(self._push_worker)
        with ThreadPool(processes=cpu_count) as pool:
            res = pool.map(func, np.array_split(self.corpus_files, cpu_count))

        for doc in res:
            docs_ram.update(doc)

        logging.info("loaded docs to RAM")
        return docs_ram

    @staticmethod
    def jaccard(d1, d2):
        keys = set(d1.keys()).intersection(d2.keys())
        jac = {}
        for k in keys:
            s1 = set(d1[k])
            s2 = set(d2[k])
            lu = len(s1.union(s2))
            if lu:
                j = len(s1.intersection(s2)) / lu
                jac['%s_j' % k] = j
        return jac

    @staticmethod
    def independent(doc, prefix='d'):
        _ft = {}
        _l = 0
        for k, words in doc.items():
            tag = '%s_%s' % (k, prefix)
            _ft[tag] = len(words)
            _ft['unique_%s' % tag] = len(set(words))
            _l += _ft[tag]
        _ft['total_len'] = _l
        return _ft

    @staticmethod
    def mean_vector(vectors):
        if vectors.ndim > 1:
            return vectors.mean(axis=0)
        else:
            return vectors

    def tag_vectors(self, doc):
        vecs = {}
        for k,v in doc.items():
            words = [vi for vi in v if vi in self.wv]
            if len(words):
                word_vecs = self.wv[words]
                vec = self.mean_vector(word_vecs)
                vecs[k] = vec
        return vecs

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

        ftrs.update(self.independent(q_words, prefix='q'))
        ftrs.update(self.independent(d_words, prefix='d'))

        ftrs.update(self.jaccard(q_words, d_words))

        q_vecs = q['w2v']
        d_vecs = d['w2v']
        for k in set(q_vecs.keys()).intersection(d_vecs.keys()):
            ftrs['%s_cos' % k] = distance.cosine(q_vecs[k], d_vecs[k])

        n = self.mpk.compare_mpk(q['mpk'], d['mpk'], self.mpk.way)
        ftrs['mpk'] = n

        return ftrs

    def scores(self, samples):
        ftrs = []
        for anc, pos, neg in tqdm(samples):
            q = self.doc_info(anc)
            ranks = [1] * len(pos) + [0] * len(neg)
            for rank, d_ix in zip(ranks, pos):
                d = data.doc_info(d_ix)
                _ft = data.score(q, d)
                _ft['rank'] = rank
                ftrs.append(_ft)
        return ftrs


with open('../data/sampled.json', 'r') as f:
    samples = json.load(f)

unique = sorted(list(chain.from_iterable([[anc] + pos + neg for anc, pos, neg in samples])))

data = Data(unique)

ftrs = []
for anc, pos, neg in tqdm(samples):
    q = data.doc_info(anc)
    ranks = [1]*len(pos) + [0]*len(neg)
    for rank, d_ix in zip(ranks, pos):
        d = data.doc_info(d_ix)
        _ft = data.score(q, d)
        _ft['rank'] = rank
        ftrs.append(_ft)


corpus = MmCorpus('../data/corpus.mm')
# build_tfidf_index(dictionary, corpus, anew=True)

index = Similarity.load('../data/sim_index/sim')
tfidf = TfidfModel.load('../data/tfidf.model')
all_ids = fc.load_keys('../data/keys.json')

tfidf_blob = ft.TfIdfBlob(corpus, tfidf, index, all_ids)
tfidf_scores = tfidf_blob.extract(samples, '../data/tfidf.csv', n_chunks=150)


# q_ix = all_ids.index('5984b7c2b6b1132856638528')
# d_ix = all_ids.index('5984b65cb6b1131291638512')
#
# data = Data([q_ix, d_ix], all_ids, corpus_files)
#
# q = data.doc_info(q_ix)
# d = data.doc_info(d_ix)
#
# data.scores(q, d)
