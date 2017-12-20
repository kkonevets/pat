
from common import *
from sklearn.model_selection import train_test_split
from itertools import *
from collections import Counter
import cProfile, pstats
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
        self.ixs = list(set(ixs))
        self.all_ids = all_ids
        self.ids = set(itemgetter(*ixs)(all_ids))

        self.dictionary = Dictionary.load('../data/corpus.dict')
        self.corpus = MmCorpus('../data/corpus.mm')

        self.index = Similarity.load('../data/sim_index/sim')
        self.tfidf = TfidfModel.load('../data/tfidf.model')
        # documents as columns
        self.tfidf_vectors = corpus2csc(self.tfidf[self.corpus[ixs]])

        self.wv = Word2Vec.load('../data/w2v_200_5_w8').wv
        self.docs_ram_wv = self.push_docs_to_ram(self.wv.vocab, corpus_files, True)

        self.docs_ram_dict = self.push_docs_to_ram(self.ids, self.dictionary.token2id, corpus_files)

    def _push_worker(self, token2id, is_gensim, files):
        docs_ram = {}

        for _id, doc in fc.iter_docs(files, encode=False, with_ids=True, as_is=True):
            if _id not in self.ids:
                continue
            _doc = {}
            for k, v in doc.items():
                if is_gensim:
                    _ids = [token2id[w].index for s in v for w in s if w in token2id]
                else:
                    _ids = [token2id[w] for s in v for w in s]
                _doc[k] = _ids
            docs_ram[_id] = _doc
        return docs_ram

    def push_docs_to_ram(self, token2id, corpus_files, is_gensim=False):
        docs_ram = {}

        func = partial(self._push_worker, token2id, is_gensim)
        with multiprocessing.Pool(processes=cpu_count) as pool:
            res = pool.map(func, np.array_split(corpus_files, cpu_count))

        for d in res:
            docs_ram.update(d)
        print("docs in ram\n")
        return docs_ram

    @staticmethod
    def jaccard(self, d1, d2):
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
    def independent(self, doc, prefix='d'):
        _ft = {}
        _l = 0
        for k, words in doc.items():
            _ft['%s_%s' % (k, prefix)] = len(words)
            _ft['unique_%s_%s' % (k, prefix)] = len(set(words))
            _l += _ft[k]
        _ft['total_len'] = _l
        return _ft

    def scores(self, q, d):
        """
        get features for query and document
        """
        pass


# dictionary.token2id['стол']
#
# wv.vocab['стол'].index