
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
        self.ixs = ixs
        self.ix_map = {ix:i for i,ix in ixs}
        self.all_ids = all_ids
        self.ids = set(itemgetter(*ixs)(all_ids))
        self.corpus_files = corpus_files

        self.dictionary = Dictionary.load('../data/corpus.dict')
        self.corpus = MmCorpus('../data/corpus.mm')

        self.index = Similarity.load('../data/sim_index/sim')
        self.tfidf = TfidfModel.load('../data/tfidf.model')
        # documents as columns
        self.tfidf_vectors = corpus2csc(self.tfidf[self.corpus[ixs]])

        self.wv = Word2Vec.load('../data/w2v_200_5_w8').wv
        self.docs_ram_w2v = self.push_docs_to_ram(self.wv.vocab,
                                                  corpus_files, is_gensim=True)

        self.docs_ram_dict = self.push_docs_to_ram(self.dictionary.token2id,
                                                   corpus_files)
        self.qdr = qdr.QueryDocumentRelevance.load_from_file('../data/qdr_model.gz')
        with open('../data/all_mpk.pkl', 'rb') as f:
            self.all_mpk = pickle.load(f)

        self.mpk = ft.MPK()

    def _push_worker(self, token2id, is_gensim, files):
        docs_ram = {}

        for _id, doc in fc.iter_docs(files, encode=False, with_ids=True, as_is=True):
            if _id not in self.ids:
                continue
            _doc = {}
            for k, v in doc.items():
                if is_gensim:
                    words = [w for s in v for w in s if w in token2id]
                    if len(words):
                        vec = self.mean_vector(self.wv[words])
                        _doc[k] = vec
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
        print("docs in ram, is_gensim %s\n" % is_gensim)
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
            _ft['%s_%s' % (k, prefix)] = len(words)
            _ft['unique_%s_%s' % (k, prefix)] = len(set(words))
            _l += _ft[k]
        _ft['total_len'] = _l
        return _ft

    @staticmethod
    def mean_vector(vectors):
        if vectors.ndim > 1:
            return vectors.mean(axis=0)
        else:
            return vectors

    def doc_info(self, d_ix):
        info = {}
        info['ix'] = d_ix
        info['id'] = self.all_ids[d_ix]
        info['iix'] = self.ix_map[d_ix]
        info['tfidf_vec'] = self.tfidf_vectors[:, info['iix']]
        info['words'] = self.docs_ram_dict[info['id']]
        info['w2v'] = self.docs_ram_w2v[info['id']]
        info['mpk'] = self.all_ids[info['id']]

        return info

    def scores(self, q, d):
        ftrs = {'q': q['id'], 'd': d['id']}

        ftrs['tfidf_gs'] = distance.cosine(q['tfidf_vec'], d['tfidf_vec'])
        ftrs.update(self.qdr.score(q['words'], d['words']))

        q_text = chain.from_iterable(q['words'].values())
        d_text = chain.from_iterable(d['words'].values())
        ftrs.update(self.independent(q_text, prefix='q'))
        ftrs.update(self.independent(d_text, prefix='d'))

        ftrs.update(self.jaccard(q['words'], d['words']))

        q_vecs = q['w2v']
        d_vecs = d['w2v']
        for k in set(q_vecs.keys()).intersection(d_vecs.keys()):
            ftrs['%s_cos' % k] = distance.cosine(q[k], d[k])

        n = self.mpk.compare_mpk(q['mpk'], d['mpk'])
        ftrs['mpk'] = n

        return ftrs


# dictionary.token2id['стол']
#
# wv.vocab['стол'].index