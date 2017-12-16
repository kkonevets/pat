
from common import *
import random
from operator import itemgetter
import qdr
from sklearn.metrics import accuracy_score, f1_score, \
    roc_auc_score, confusion_matrix
from scipy.spatial import distance
from itertools import filterfalse, starmap, chain
import fetching as fc


def to_dataframe(ftr_list):
    ftrs = pd.DataFrame.from_dict(ftr_list, orient='columns')
    ftrs.fillna(0, inplace=True)
    return ftrs


def save(ftrs, fname, index_names=('q', 'd'), compression=None):
    ftrs.sort_values(index_names, inplace=True)
    ftrs.set_index(index_names, inplace=True)

    print('saving to %s ...\n' % fname)
    ftrs.to_csv(fname, compression=compression)


def build_tfidf_index(dictionary, corpus, anew=True):
    fmodel = '../data/tfidf.model'
    if anew or not path.exists(fmodel):
        tfidf = TfidfModel(corpus)
        tfidf.save(fmodel)
    else:
        tfidf = TfidfModel.load(fmodel)

    fname = '../data/sim_index/sim'
    index = Similarity(fname, tfidf[corpus],
                       num_features=len(dictionary), num_best=None,
                       chunksize=10 * 256, shardsize=10 * 32768)
    index.save(fname)


class TfIdfBlob:
    def __init__(self, corpus, tfidf, index):
        self.corpus = corpus
        self.tfidf = tfidf
        self.index = index

    def get_cosines(self, ixs):
        if len(ixs) == 0:
            return None
        batch = self.tfidf[itemgetter(*ixs)(self.corpus)]
        cosines = self.index[batch]
        return cosines

    def predict(self, ixs, limit=None):
        cosines = self.get_cosines(ixs)
        # reversed sort
        argsorted = np.argsort(-cosines)[:, :limit]
        return argsorted

    def extract(self, samples, all_ids, fname):
        def pick_scores(cosine_list, sample):
            q = all_ids[sample[0]]
            negs = chain(sample[1], sample[2])
            cosines = [{'q': q, 'd': all_ids[ix],
                        'tfidf_gs': cosine_list[ix]} for ix in negs]
            return cosines

        def worker(part):
            if len(part) == 0:
                return []
            ixs = [el[0] for el in part]
            cosines = self.get_cosines(ixs)
            args = zip(cosines, part)
            picked = list(starmap(pick_scores, args))
            return picked

        ftrs = []
        for samples_part in tqdm(chunkify(samples, 100)):
            res = Parallel(n_jobs=cpu_count, backend="threading") \
                (delayed(worker)(part) for
                 part in chunkify(samples_part, cpu_count))
            ftrs += chain.from_iterable(res)

        ftrs = to_dataframe(ftrs)
        save(ftrs, fname)
        return ftrs


def neg_sampler_worker(ns, ixs):
    argsorted = ns.tfidf_blob.predict(ixs)
    # first element is a query itself
    args = ((iix[1:], ix) for iix, ix in
            zip(argsorted, ixs))
    samps = list(starmap(ns.sample_negs, args))
    # found = list(chain.from_iterable((starmap(found_at, args2))))
    return samps


class NegativeSampler:
    def __init__(self, sims, tfidf_blob, neg_ixs_distr,
                 k=1, percentile=90, seed=0):
        self.sims = sims
        self.tfidf_blob = tfidf_blob
        self.neg_ixs_distr = neg_ixs_distr.copy()
        self.k = k
        self.percentile = percentile

        q = range(90, 100, 1)
        self.percentiles = pd.Series(np.percentile(neg_ixs_distr, q),
                                        index=q)
        self.worst_pos = int(self.percentiles[percentile])

        random.seed(seed)
        random.shuffle(self.neg_ixs_distr)
        self.samples = None

    def gen_samples(self, fname, n_chunks=500):
        # try:
        #     os.remove('../data/foundat.csv')
        # except OSError:
        #     pass
        samples = []

        keys = list(self.sims.keys())
        for ixs_part in tqdm(np.array_split(keys, n_chunks)):
            res = Parallel(n_jobs=cpu_count, backend="threading") \
                (delayed(neg_sampler_worker)(self, part) for
                 part in np.array_split(ixs_part, cpu_count))
            samples += list(chain.from_iterable(res))

        # with open(fname, 'w') as f:
        #     json.dump(samples, f)
        return samples

    def sample_negs(self, iix, key_ix):
        posvs = self.sims[key_ix]
        exclude = posvs + [key_ix]
        size = len(posvs) * self.k

        # slice starting from 3 - could be duplicates
        filtered = filterfalse(lambda x: x in exclude, map(int, iix[3:]))
        close_negs = list(islice(filtered, size))

        i = random.randint(0, len(self.neg_ixs_distr) - size)
        filtered = filterfalse(lambda x: x in exclude + close_negs,
                               (int(iix[j]) for j in self.neg_ixs_distr[i:]))
        far_negs = list(first_unique(filtered, size)) + [int(iix[self.worst_pos])]

        return key_ix, posvs, close_negs + far_negs

    def found_at(self, iix, key_ix):
        posvs = self.sims[key_ix]
        ixs = np.isin(iix, posvs)
        return np.where(ixs)[0]


class QDR:
    def __init__(self, fmodel):
        self.fmodel = fmodel
        self.model = None
        self._corpus = None
        self._map = None

    def train(self, corpus_files):
        corpus_iter = fc.iter_docs(corpus_files, encode=True)
        self.model = qdr.Trainer()
        self.model.train(corpus_iter)
        self.model.serialize_to_file(self.fmodel)

    def load(self):
        self.model = qdr.QueryDocumentRelevance.load_from_file(self.fmodel)
        return self.model

    def _get(self, ix):
        doc = self._corpus[self._map[ix]]
        doc = {str(k).encode(): v for k, v in doc}
        return doc

    def extract(self, doc_pairs, all_ids, corpus, fname):
        ixs = sorted(list(set(chain.from_iterable(doc_pairs))))
        self._map = {ix: i for i, ix in enumerate(ixs)}

        # load all corpus into RAM
        self._corpus = list(corpus[ixs])
        print('loaded')

        ftrs = []
        for qix, dix in tqdm(doc_pairs):
            _s = {'q': all_ids[qix], 'd': all_ids[dix]}
            pred = self.model.score(self._get(dix), self._get(qix))
            _s.update(pred)
            ftrs.append(_s)

        print("got scores")

        ftrs = to_dataframe(ftrs)
        save(ftrs, fname)
        return ftrs

    def bm25_sanity_check(self, corpus, all_ids):
        all_sampled_docs = list(corpus[list(range(len(all_ids)))])
        print('loaded')

        scores = []
        for ix in tqdm(range(len(all_ids))):
            if ix == 0:
                print(all_ids[ix])
                q = {str(k).encode(): v for k, v in all_sampled_docs[ix]}
                continue

            doc = {str(k).encode(): v for k, v in all_sampled_docs[ix]}
            if len(doc):
                _sc = self.model.score(doc, q)
                scores.append(_sc)

        with open('../data/qdr_scores_sanity.pkl', 'wb') as f:
            pickle.dump(scores, f)

        # scored = zip(all_ids[1:], (s['bm25'] for s in scores if s))
        # scored = list(sorted(scored, key=itemgetter(1), reverse=True))
        # scored[:10]

    @staticmethod
    def describe_bm25(qdr_scores):
        import matplotlib.pylab as plt

        sim_bms, close_bms, far_bms, sfar_bms = [], [], [], []
        wrong = 0
        for el in qdr_scores:
            _l = len(el[1])
            if len(el[2]) != 2 * _l + 1:
                wrong += 1
            for s in el[1]:
                sim_bms.append(s['bm25'])
            for s in el[2][:_l]:
                close_bms.append(s['bm25'])
            for s in el[2][_l:-1]:
                far_bms.append(s['bm25'])
            sfar_bms.append(el[2][-1]['bm25'])

        pprint(pd.Series(sim_bms).describe())
        pprint(pd.Series(close_bms).describe())
        pprint(pd.Series(far_bms).describe())
        pprint(pd.Series(sfar_bms).describe())

        pd.Series(sim_bms).plot.hist(xlim=(0, 10000), ylim=(0, 800000))
        plt.show()

        pd.Series(close_bms).plot.hist(xlim=(0, 10000), ylim=(0, 800000))
        plt.show()

        pd.Series(far_bms).plot.hist(xlim=(0, 10000), ylim=(0, 800000))
        plt.show()

        pd.Series(sfar_bms).plot.hist(xlim=(0, 10000), ylim=(0, 800000))
        plt.show()


class Independent:
    def __init__(self, corpus_files, ids):
        self.corpus_files = corpus_files
        self.ids = ids

    def extract(self, fname):
        ftrs = []
        for _id, doc in fc.iter_docs(self.corpus_files, encode=False, with_ids=True, as_is=True):
            if _id not in self.ids:
                continue
            _ft = {'q': _id}
            _l = 0
            for k, v in doc.items():
                words = [w for s in v for w in s]
                _ft[k] = len(words)
                _ft['unique_%s' % k] = len(set(words))
                _l += _ft[k]
            _ft['total_len'] = _l
            ftrs.append(_ft)

        ftrs = to_dataframe(ftrs)
        save(ftrs, fname, index_names='q')
        return ftrs


class Jaccard:
    def __init__(self, ids, token2id, corpus_files):
        self.docs_ram = push_docs_to_ram(ids, token2id, corpus_files)

    def extract(self, samples, all_ids, fname=None):
        ftrs = []
        for anc, pos, neg in tqdm(samples):
            key = all_ids[anc]
            q = self.docs_ram[key]
            q_sets = {}
            for k, v in q.items():
                q_sets[k] = set(v)
            for _ix in pos + neg:
                _id = all_ids[_ix]
                jac = {'q': key, 'd': _id}
                d = self.docs_ram[_id]
                for k, v in d.items():
                    if k in q_sets:
                        lu = len(q_sets[k].union(v))
                        if lu:
                            j = len(q_sets[k].intersection(v)) / lu
                            jac['%s_j' % k] = j
                ftrs.append(jac)

        ftrs = to_dataframe(ftrs)
        save(ftrs, fname, index_names='q')
        return ftrs


class Distribured:
    def __init__(self, w2v_model, corpus_files, all_ids):
        self.wv = w2v_model.wv
        self.index2word = self.wv.index2word
        self.all_ids = all_ids
        self.cosines = []
        self.docs_in_ram = push_docs_to_ram(self.wv.vocab, self.index2word, corpus_files, is_gensim=True)

    def extract(self, samples, fname=None, n_chunks=50):
        ftrs = []
        for keys_part in tqdm(chunkify(samples, n_chunks)):
            res = Parallel(n_jobs=cpu_count, backend="multiprocessing") \
                (delayed(self.cosines_worker)(part) for
                 part in chunkify(keys_part, cpu_count))
            ftrs += list(chain.from_iterable(res))

        ftrs = to_dataframe(ftrs)
        save(ftrs, fname, index_names='q')
        return ftrs

    def cosines_worker(self, samples_part):
        def mean_vector(vectors):
            if vectors.ndim > 1:
                return vectors.mean(axis=0)
            else:
                return vectors

        cosines = []
        for count, anc, pos, neg in enumerate(samples_part):
            key = self.all_ids[anc]
            q = self.docs_in_ram[key]
            q_vecs = {}
            for k, v in q.items():
                if len(v):
                    q_vecs[k] = mean_vector(self.wv[itemgetter(*v)(self.index2word)])
            for _ix in pos + neg:
                _id = self.all_ids[_ix]
                _cos = {'q': key, 'd': _id}
                d = self.docs_in_ram[_id]
                for k, v in d.items():
                    if len(v):
                        vec = mean_vector(self.wv[itemgetter(*v)(self.index2word)])
                        if k in q_vecs and len(vec) and len(q_vecs[k]):
                            _cos['%s_cos' % k] = distance.cosine(vec, q_vecs[k])
                cosines.append(_cos)
            # if count % 1000 == 0:
            #     print('%d%%' % (100.*count/len(samples_part)))
        return cosines


class MPK:
    def __init__(self):
        self.way = ['section', 'class_',
                    'subclass', 'main_group', 'subgroup']

    def extract(self, all_mpk, samples, all_ids, fname):
        ftrs = []
        for q_ix, pos, neg in tqdm(samples):
            q_id = all_ids[q_ix]
            q = all_mpk[q_id]
            for _ix in pos + neg:
                _id = all_ids[_ix]
                _ft = {'q': q_id, 'd': _id}
                d = all_mpk[_id]
                n = self.compare_mpk(q['mpk'], d['mpk'], self.way)
                _ft['mpk'] = n
                ftrs.append(_ft)

        ftrs = to_dataframe(ftrs)
        save(ftrs, fname, index_names='q')
        return ftrs

    @staticmethod
    def compare_mpk_level(mpk1, mpk2, tag):
        matched = {}
        for i, m1 in enumerate(mpk1):
            v1 = m1.get(tag)
            if v1 is None:
                continue
            for j, m2 in enumerate(mpk2):
                v2 = m2.get(tag)
                if v2 is None:
                    continue
                if v1 == v2:
                    ixs = matched.get(v1)
                    if ixs is None:
                        ixs = (set(), set())
                        matched[v1] = ixs
                    ixs[0].update([i])
                    ixs[1].update([j])
        return matched

    @staticmethod
    def compare_mpk(mpk1, mpk2, way):
        def make_tuple(val):
            if type(val) == tuple:
                return val
            else:
                return (val,)

        if len(way):
            matched = MPK.compare_mpk_level(mpk1, mpk2, way[0])
        else:
            return 0
        count = int(bool(len(matched)))
        sub_count = 0
        for ixs1, ixs2 in matched.values():
            _mpk1 = make_tuple(itemgetter(*ixs1)(mpk1))
            _mpk2 = make_tuple(itemgetter(*ixs2)(mpk2))
            sub_count = max(sub_count, MPK.compare_mpk(_mpk1, _mpk2, way[1:]))
        return count + sub_count


def evaluate_probs(probs, y_test, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1score = f1_score(y_test, y_pred) * 100
    if len(np.unique(y_test)) != 1:
        auc = roc_auc_score(y_test, probs) * 100
    else:
        auc = 0
    print("Accuracy: %.2f%%, f1 score: %.2f%%, auc score: %.2f%%" % (accuracy, f1score, auc))

    cm = confusion_matrix(y_test, y_pred)
    pprint(cm)
    return cm


def push_docs_to_ram(ids, token2id, corpus_files, is_gensim=False):
    docs_ram = {}

    for _id, doc in fc.iter_docs(corpus_files, encode=False, with_ids=True, as_is=True):
        if _id not in ids:
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


def save_letor(ftrs, fname):
    qid = 0
    q_prev = None
    with open(fname, 'w') as f:
        for _, row in tqdm(ftrs.iterrows(), total=len(ftrs)):
            q = row['q']
            if q != q_prev:
                qid += 1
                q_prev = q
            s = '%d qid:%d' % (row['rank'], qid)
            _sft = ' '.join(['%d:%s' % (i + 1, v) for i, v in enumerate(row[3:].values)])
            s = ' '.join([s, _sft, '\n'])
            f.write(s)
