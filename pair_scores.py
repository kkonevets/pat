
from common import *
from sklearn.model_selection import train_test_split
from itertools import *
from collections import Counter
import cProfile, pstats
import qdr
from operator import itemgetter
from scipy.spatial import distance
from importlib import reload
from functools import partial
import multiprocessing
import features as ft
import fetching as fc

from scipy.sparse import csc_matrix
from gensim.matutils import corpus2csc


reload(ft)
reload(fc)

SEED = 0


class Data:
    def __init__(self, ixs, all_ids, corpus_files):
        """
        load data to RAM for fast access
        :param ixs: document indexes
        """
        ixs = sorted(list(set(ixs)))
        self.ixs = ixs
        self.ix_map = {ix:i for i, ix in enumerate(ixs)}
        self.all_ids = all_ids
        self.ids = set(itemgetter(*ixs)(all_ids))
        self.corpus_files = corpus_files

        self.dictionary = Dictionary.load('../data/corpus.dict')
        self.docs_ram_dict = self.push_docs_to_ram(corpus_files)
        self.corpus = MmCorpus('../data/corpus.mm')

        self.tfidf = TfidfModel.load('../data/tfidf.model')
        # documents as columns
        self.tfidf_vectors = corpus2csc(self.tfidf[self.corpus[ixs]])

        self.wv = Word2Vec.load('../data/w2v_200_5_w8').wv

        logging.info("loading qdr model")
        self.qdr = qdr.QueryDocumentRelevance.load_from_file('../data/qdr_model.gz')

        logging.info("loading mpk data")
        with open('../data/all_mpk.pkl', 'rb') as f:
            self.all_mpk = pickle.load(f)
        self.mpk = ft.MPK()
        logging.info('All data loaded')

    @staticmethod
    def dictionary_w2v_map(dictionary, wv):
        maped = {dictionary.token2id[k]: v.index for k, v in wv.vocab.items()}
        return maped

    def _push_worker(self, files):
        docs_ram = {}
        token2id = self.dictionary.token2id

        for _id, doc in fc.iter_docs(files, encode=False, with_ids=True, as_is=True):
            if _id not in self.ids:
                continue
            _doc = {}
            for k, v in doc.items():
                _ids = [token2id[w] for s in v for w in s]
                _doc[k] = _ids
            docs_ram[_id] = _doc
        return docs_ram

    def push_docs_to_ram(self, corpus_files):
        logging.info("loading docs to RAM")
        docs_ram = {}

        func = partial(self._push_worker)
        with multiprocessing.Pool(processes=cpu_count) as pool:
            res = pool.map(func, np.array_split(corpus_files, cpu_count))

        for d in res:
            docs_ram.update(d)

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
            words = self.wv[[vi for vi in v if vi in self.wv]]
            if len(words):
                vec = self.mean_vector(words)
                vecs[k] = vec
        return vecs

    def doc_info(self, d_ix):
        info = {}
        _id = self.all_ids[d_ix]

        info['id'] = _id
        info['tfidf'] = self.tfidf_vectors.getcol(self.ix_map[d_ix]).toarray()
        info['words'] = {k: [self.dictionary[vi] for vi in v]
                         for k, v in self.docs_ram_dict[_id].items()}
        info['w2v'] = self.tag_vectors(info['words'])
        info['mpk'] = self.all_mpk[_id]['mpk']

        return info

    def scores(self, q, d):
        ftrs = {'q': q['id'], 'd': d['id']}
        q_words = q['words']
        d_words = d['words']
        q_text = (w.encode() for w in chain.from_iterable(q_words.values()))
        d_text = (w.encode() for w in chain.from_iterable(d_words.values()))

        ftrs['tfidf_gs'] = distance.cosine(q['tfidf'], d['tfidf'])
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


all_ids = fc.load_keys('../data/keys.json')
corpus_files = glob('../data/documents/*')
corpus_files.sort(key=natural_keys)

q_ix = all_ids.index('5984b7c2b6b1132856638528')
d_ix = all_ids.index('5984b65cb6b1131291638512')

data = Data([q_ix, d_ix], all_ids, corpus_files)

q = data.doc_info(q_ix)
d = data.doc_info(d_ix)

data.scores(q, d)