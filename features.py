
from common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymystem3 import Mystem
import ujson
import random
from sklearn.model_selection import train_test_split
from itertools import *
from importlib import reload
from operator import attrgetter, itemgetter
from qdr import Trainer, QueryDocumentRelevance
from sklearn.metrics import accuracy_score, f1_score, \
    roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from collections import Counter
from scipy.spatial import distance
import copy
import collections
from itertools import combinations
import cProfile, pstats


SEED = 0


def evaluate(probs, y_test, threshold=0.5):
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


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def save_json(ids, prefix, raw=True):
    client = MongoClient()
    db = client.fips
    mstem = Mystem()
    prog = re.compile("[\\W\\d]", re.UNICODE)  # only letters and '_'

    raw_str = '_raw' if raw else ''
    tags = ['name', 'abstract', 'description', 'claims']
    tags = [tag + raw_str for tag in tags]

    docs = {}
    for _id in ids:
        doc_tokenized = {}
        doc = db.patents.find_one({'_id': _id},
                                  {tag: 1 for tag in tags})
        for tag in tags:
            if tag not in doc:
                continue
            tag_sents = []
            for sentence in nltk.sent_tokenize(doc[tag]):
                tokens = word_tokenize(sentence)
                tokens = (prog.sub('', w) for w in tokens)
                if raw:
                    joined = ' '.join(tokens)
                    tokens = mstem.lemmatize(joined)[:-1]
                tokens = [w for w in tokens if not w.isspace() and len(w) > 1 and w not in stop_list]
                if len(tokens):
                    tag_sents.append(tokens)
            if len(tag_sents):
                doc_tokenized[tag.rstrip('_raw')] = tag_sents

        docs[str(_id)] = doc_tokenized

    fname = join(DATA_FOLDER, 'documents', '%s' % prefix + '.json.gz')
    save_texts(docs, fname)

    gc.collect()


def save_texts(docs, fname):
    with GzipFile(fname, 'w') as fout:
        json_str = json.dumps(docs, ensure_ascii=False)
        json_bytes = json_str.encode('utf-8')
        fout.write(json_bytes)


def iter_docs(fnames, encode=False, with_ids=False, as_is=False):
    def do_encode(w):
        return w.encode() if encode else w

    for filename in tqdm(fnames):
        with GzipFile(filename) as f:
            data = ujson.load(f)
        for _id, doc in data.items():
            if as_is:
                text = doc
            else:
                text = [do_encode(w) for t in doc.values() for s in t for w in s]

            if with_ids:
                yield (_id, text)
            else:
                yield text


def keys_from_json(fnames):
    keys = []
    for filename in tqdm(fnames):
        with GzipFile(filename) as f:
            data = ujson.load(f)
            keys += list(data.keys())
    return keys


def load_keys(keys_path):
    if path.exists(keys_path):
        with open(keys_path) as f:
            keys = json.load(f)
    else:
        list_block = glob('../data/documents/*')
        list_block.sort(key=natural_keys)
        keys = keys_from_json(list_block)
        with open(keys_path, 'w') as f:
            json.dump(keys, f)
    return keys


class Corpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        for doc in iter_docs(self.input):
            yield doc


def save_corpus(list_block, dir_name, prefix='corpus'):
    dic_name = join(dir_name, '%s.dict' % prefix)
    corp_name = join(dir_name, '%s.mm' % prefix)

    dictionary = Dictionary(iter_docs(list_block), prune_at=None)
    dictionary.save(dic_name)

    corpus = Corpus(list_block, dictionary=dictionary)
    corpora.MmCorpus.serialize(corp_name, corpus)

    return dic_name, corp_name


def build_tfidf_index(dictionary, corpus, anew=True):
    fmodel = '../data/tfidf.model'
    if anew or not path.exists(fmodel):
        tfidf = models.TfidfModel(corpus)
        tfidf.save(fmodel)
    else:
        tfidf = models.TfidfModel.load(fmodel)

    fname = '../data/sim_index/sim'
    index = similarities.Similarity(fname, tfidf[corpus],
                                    num_features=len(dictionary), num_best=None,
                                    chunksize=10 * 256, shardsize=10 * 32768)
    index.save(fname)


def load_sims(fname):
    if path.exists(fname):
        with open(fname) as f:
            sims = json.load(f)
    else:
        client = MongoClient()
        db = client.fips

        sims = {}
        topn = db.patents.find({'similar': {'$exists': True}}, {'similar': 1})
        for doc in tqdm(topn):
            sims[str(doc['_id'])] = doc['similar']
        with open(fname, 'w') as f:
            json.dump(sims, f)

    return sims


def tfidf_batch_predict(all_ids, ixs, limit=None):
    preds = {}
    batch = tfidf[corpus[ixs]]
    for ix, similarities in zip(ixs, index[batch]):
        limit = limit + 1 if limit else limit
        sorted_ixs = similarities.argsort()[::-1][:limit]
        preds[all_ids[ix]] = [all_ids[i] for i in sorted_ixs if i != ix]
    return preds


def first_unique(iterable, n):
    """
    iterable = [1,2,1,7,2,5,6,1,3], n = 4 -> [1, 2, 7, 5]
    """
    unique = set()
    def condition(x):
        nonlocal unique, n
        unique.update([x])
        return len(unique) <= n
    filtered = filterfalse(lambda x: x in unique, iterable)
    return takewhile(condition, filtered)


def sample_negs(iix, key, k=1):
    key_ix = ix_map[key]
    posvs = [ix_map[k] for k in sims[key]]
    exclude = posvs + [key_ix]
    size = len(posvs) * k

    # slice starting from 3 - could be duplicates
    filtered = filterfalse(lambda x: x in exclude, map(int, iix[3:]))
    close_negs = list(islice(filtered, size))

    i = random.randint(0, len(neg_ixs) - size)
    worst = int(percentiles[90][0])
    filtered = filterfalse(lambda x: x in exclude + close_negs,
                           (int(iix[j]) for j in neg_ixs[i:]))
    far_negs = list(first_unique(filtered, size)) + [int(iix[worst])]

    return key_ix, posvs, close_negs + far_negs


def found_at(iix, key):
    posvs = [ix_map[k] for k in sims[key]]
    ixs = np.isin(iix, posvs)
    return np.where(ixs)[0]


def get_cosines(keys):
    if len(keys) == 0:
        return None
    ixs = [ix_map[k] for k in keys]
    batch = tfidf[itemgetter(*ixs)(corpus)]
    cosines = index[batch]
    return cosines


def argsort(keys):
    cosines = get_cosines(keys)
    # reversed sort
    argsorted = np.argsort(-cosines)
    return argsorted


def tfidf_worker(keys):
    argsorted = argsort(keys)
    # first element is a query itself
    args = ((iix[1:], key) for iix, key in
            zip(argsorted, keys))
    return args


def gen_train_samples(keys_tv):
    samples = []
    # try:
    #     os.remove('../data/foundat.csv')
    # except OSError:
    #     pass
    def worker(keys):
        args = tfidf_worker(keys)
        samples = list(starmap(sample_negs, args))
        # found = list(chain.from_iterable((starmap(found_at, args2))))
        return samples

    for keys_part in tqdm(np.array_split(keys_tv, 500)):
        res = Parallel(n_jobs=cpu_count, backend="threading") \
            (delayed(worker)(part) for
             part in np.array_split(keys_part, cpu_count))
        samples += list(chain.from_iterable(res))

    with open('../data/sampled.json', 'w') as f:
        json.dump(samples, f)

    return samples


def save_qdr_features(model, corpus, samples):
    l = []
    for el in samples:
        l.append(el[0])
        for ix in chain(*el[1:]):
            l.append(ix)
    sample_ids = sorted(list(set(l)))
    sample_ids_map = {ix: i for i, ix in enumerate(sample_ids)}

    # load all corpus in to RAM
    all_sampled_docs = list(corpus[sample_ids])
    print('loaded')

    scores = []
    for el in tqdm(samples):
        _scores = [el[0]]
        ixs = [el[0]] + el[1] + el[2]
        docs = [all_sampled_docs[sample_ids_map[ix]] for ix in ixs]
        q = {str(k).encode(): v for k, v in docs[0]}

        sim_scores = [model.score({str(k).encode(): v for k, v in doc}, q)
            for doc in docs[1:len(el[1])+1]]
        _scores.append(sim_scores)
        neg_scores = [model.score({str(k).encode(): v for k, v in doc}, q)
            for doc in docs[len(el[1])+1:]]
        _scores.append(neg_scores)
        scores.append(_scores)

    print("got scores")

    with open('../data/qdr_scores.pkl', 'wb') as f:
        pickle.dump(scores, f)


def pick_scores(cosine_list, sample):
    return [sample[0], cosine_list[sample[1]], cosine_list[sample[2]]]


def save_tfidf_features(corpus, samples):
    def worker(part):
        if len(part) == 0:
            return []
        keys_sample = [all_ids[el[0]] for el in part]
        cosines = get_cosines(keys_sample)
        args = zip(cosines, part)
        picked = list(starmap(pick_scores, args))
        return picked

    scores = []
    for samples_part in tqdm(chunkIt(samples, 100)):
        res = Parallel(n_jobs=cpu_count, backend="threading") \
            (delayed(worker)(part) for
             part in chunkIt(samples_part, cpu_count))
        scores += list(chain.from_iterable(res))

    with open('../data/tfidf_scores.pkl', 'wb') as f:
        pickle.dump(scores, f)


def bm25_sanity_check(model, corpus, samples):
    all_ids = load_keys('../data/keys.json')
    ix_map = {vi: i for i, vi in enumerate(all_ids)}

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
            _sc = model.score(doc, q)
            scores.append(_sc)

    with open('../data/qdr_scores_0.pkl', 'wb') as f:
        pickle.dump(scores, f)

    # scored = zip(all_ids[1:], (s['bm25'] for s in scores if s))
    # scored = list(sorted(scored, key=itemgetter(1), reverse=True))
    # scored[:10]


def save_ftrs_to_dataframe(scores_list, names, samples):
    def flatten_scores(scs_list, rank_level, ixs, qid):
        scs_list = list(zip(*scs_list))
        ret = []
        for ix, scs in zip(ixs, scs_list):
            _l = [qid, rank_level]
            for sc in scs:
                if type(sc) == dict:
                    _l += list(sc.values())
                else:
                    _l.append(sc)
            _l.append(all_ids[ix])
            ret.append(_l)
        return ret

    samples_dict = {el[0]: el[1:] for el in samples}
    ftrs = []
    for scores in zip(*scores_list):
        _fs = []
        ancors, pos, neg = list(zip(*scores))
        assert len(set(ancors)) == 1

        anc = ancors[0]
        samp_ixs = samples_dict[anc]
        pos_ss = flatten_scores(pos, 1, samp_ixs[0], all_ids[anc])
        neg_ss = flatten_scores(neg, 2, samp_ixs[1], all_ids[anc])
        ftrs += pos_ss
        ftrs += neg_ss

    ftrs_df = pd.DataFrame(ftrs, columns=names)
    ftrs_df.to_csv('../data/qdr_gens_ftrs.csv', index=False)


def push_docs_to_ram(token2id, is_gensim=False):
    unique = set(ftrs_df['q']).union(ftrs_df['d'])
    docs_ram = {}
    list_block = glob('../data/documents/*')
    list_block.sort(key=natural_keys)

    for _id, doc in iter_docs(list_block, encode=False, with_ids=True, as_is=True):
        if _id not in unique:
            continue
        _doc = {}
        for k,v in doc.items():
            if is_gensim:
                _ids = [token2id[w].index for s in v for w in s if w in token2id]
            else:
                _ids = [token2id[w] for s in v for w in s]
            _doc[k] = _ids
        docs_ram[_id] = _doc 
    return docs_ram


def mean_vector(vectors):
    if vectors.ndim > 1:
        return vectors.mean(axis=0)
    else:
        return vectors


def cosines_worker(samples_part):
    cosines = []
    for count, el in enumerate(samples_part):
        key = all_ids[el[0]]
        q = docs_ram[key]
        q_vecs = {}
        for k,v in q.items():
            if len(v):
                q_vecs[k] = mean_vector(wv[itemgetter(*v)(index2word)])
        for _ix in el[1] + el[2]:
            _id = all_ids[_ix]
            _cos = {'q':key, 'd':_id}
            d = docs_ram[_id]
            for k,v in d.items():
                if len(v):
                    vec = mean_vector(wv[itemgetter(*v)(index2word)])
                    if k in q_vecs and len(vec) and len(q_vecs[k]):
                        _cos['%s_cos' % k] = distance.cosine(vec, q_vecs[k])
            cosines.append(_cos)
        # if count % 1000 == 0:
        #     print('%d%%' % (100.*count/len(samples_part)))
    return cosines


class Sentences(object):
    def __init__(self, folder):
        self.folder = folder
 
    def __iter__(self):
        fnames = glob(join(self.folder, '*.json.gz'))
        for doc in iter_docs(fnames, as_is=True):
            for t in doc.values():
                for s in t:
                    yield s


def fix(name):
    fixes = {u'\u041d':u'H', u'\u0410':u'A', u'\u0412':u'B'}
    if name in fixes:
        return fixes[name]
    else:
        return name


def clean_mpk(mpk):
    mpk = copy.deepcopy(mpk)
    for k,v in mpk.items():
        mpk[k] = fix(v.strip())
        if k in (u'main_group', u'class_',  u'subgroup'):
            mpk[k] = int(mpk[k])
    return mpk


def rename(doc):
    mpk_list = []
    for mpk in doc['mpk']:
        if u'class' in mpk:
            mpk[u'class_'] = mpk.pop(u'class')
        if u'main-group' in mpk:
            mpk[u'main_group'] = mpk.pop(u'main-group')
        mpk_list.append(mpk)
    doc['mpk'] = mpk_list
    return doc


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


def make_tuple(val):
    if type(val) == tuple:
        return val 
    else:
        return (val,)


def compare_mpk(mpk1, mpk2, way):
    if len(way):
        matched = compare_mpk_level(mpk1, mpk2, way[0])
    else: 
        return 0
    count = int(bool(len(matched)))
    sub_count = 0
    for ixs1, ixs2 in matched.values():
        _mpk1 = make_tuple(itemgetter(*ixs1)(mpk1))
        _mpk2 = make_tuple(itemgetter(*ixs2)(mpk2))
        sub_count = max(sub_count, compare_mpk(_mpk1, _mpk2, way[1:]))
    return count + sub_count


def save_letor(ftrs, fname):
    qid = 0
    q_prev = None
    with open(fname, 'w') as f:
        for _, row in tqdm(ftrs.iterrows(), total=len(ftrs)):
            q = row['q']
            if q != q_prev:
                qid += 1
                q_prev = q
            strings = [row['rank'], ]
            s = '%d qid:%d' % (2-row['rank'], qid)
            _sft = ' '.join(['%d:%s' % (i+1,v) for i,v in enumerate(row[3:].values)])
            s = ' '.join([s, _sft, '\n'])
            f.write(s)


#   ###################################################################

client = MongoClient()
db = client.fips

#   ####################### save to json ###############################

docs_ids = [doc['_id'] for doc in db.patents.find({}, {'_id': 1})]

parallelizer = Parallel(n_jobs=cpu_count)
tasks_iterator = (delayed(save_json)(list_block, i, raw=True) for
                  i, list_block in enumerate(grouper(len(docs_ids) // 1000, docs_ids)))
result = parallelizer(tasks_iterator)

#   ####################### save corpus #################################

list_block = glob('../data/documents/*')
list_block.sort(key=natural_keys)
save_corpus(list_block, '../data', prefix='corpus')

#   ########################### tfidf ####################################

dictionary = corpora.Dictionary.load('../data/corpus.dict')
corpus = corpora.MmCorpus('../data/corpus.mm')
# build_tfidf_index(dictionary, corpus, anew=True)

index = similarities.Similarity.load('../data/sim_index/sim')
tfidf = models.TfidfModel.load('../data/tfidf.model')

#   ####################### fetch ids data ###############################

all_ids = load_keys('../data/keys.json')
ix_map = {vi: i for i, vi in enumerate(all_ids)}
sims = load_sims('../data/sims.json')
with open('../data/gold_mongo.json', 'r') as f:
    gold = json.load(f)

# ############################ small test  ##################################

ixs = [ix_map[k] for k in gold.keys()]
preds = tfidf_batch_predict(all_ids, ixs, limit=200)
res = evaluate(preds, gold)
"""
acc10 0.286738
acc20 0.347670
acc200 0.573477
"""

# ######################### train val test split ############################

# train, val, test = train_val_test_tups(ix_map, sims, n=1, seed=SEED)
keys_tv, keys_test = train_test_split(list(sims.keys()), test_size=0.2, random_state=SEED)
keys_train, keys_val = train_test_split(keys_tv, test_size=0.2, random_state=SEED)

ixs_train = [ix_map[k] for k in keys_train]
ixs_val = [ix_map[k] for k in keys_val]
ixs_test = [ix_map[k] for k in keys_test]

# ############################ get stat #####################################

df = pd.read_csv('../data/foundat.csv', header=None, names=['rank'])
# df.plot.hist(bins=100)
df.describe()
q = range(10, 100, 10)
percentiles = pd.DataFrame([np.percentile(df['rank'], q)], columns=q)
neg_ixs = df['rank'].values
random.seed(SEED)
random.shuffle(neg_ixs)

# ############################# smart neg sample ############################

samples = gen_train_samples(keys_tv)
with open('../data/sampled.json', 'r') as f:
    samples = json.load(f)

# i = 33
# samples[i]
# print(all_ids[samples[i][0]])
# print([all_ids[ix] for ix in samples[i][1]])
# print([all_ids[ix] for ix in samples[i][2]])

# ################################## BM25 #####################################

fname = '../data/qdr_model.gz'

model = Trainer()
model.train(corpus_iter)
model.serialize_to_file(fname)

model = QueryDocumentRelevance.load_from_file(fname)
corpus = corpora.MmCorpus('../data/corpus.mm')

save_tfidf_features(corpus, samples)
save_qdr_features(model, corpus, samples)

with open('../data/qdr_scores.pkl', 'rb') as f:
    qdr_scores = pickle.load(f)
with open('../data/tfidf_scores.pkl', 'rb') as f:
    tfidf_scores = pickle.load(f)

i = 0
pprint([s if type(s) == int else [si['tfidf'] for si in s] for s in qdr_scores[i]])

pprint(qdr_scores[i])
pprint(tfidf_scores[i])

sim_bms, close_bms, far_bms, sfar_bms = [], [], [], []
wrong = 0
for el in qdr_scores:
    _l = len(el[1])
    if len(el[2]) != 2*_l+1:
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


[el for el in samples if el[0] == 922222]

import matplotlib.pylab as plt

pd.Series(sim_bms).plot.hist(xlim=(0,10000), ylim=(0,800000))
plt.show()

pd.Series(close_bms).plot.hist(xlim=(0,10000), ylim=(0,800000))
plt.show()

pd.Series(far_bms).plot.hist(xlim=(0,10000), ylim=(0,800000))
plt.show()

pd.Series(sfar_bms).plot.hist(xlim=(0,10000), ylim=(0,800000))
plt.show()


bm25_sanity_check(model, corpus, samples)

# ############################## gen features ##################################

names = ['q', 'rank', 'tfidf_qdr', 'bm25', 
    'lm_jm', 'lm_dirichlet', 'lm_ad', 'tfidf_gs', 'd']
scores_list = (qdr_scores, tfidf_scores)

save_ftrs_to_dataframe(scores_list, names, samples)
ftrs_df = pd.read_csv('../data/qdr_gens_ftrs.csv')

# ############################################################################

data = ftrs_df.drop(columns=['Unnamed: 0', 'q', 'd', 'rank'])

from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import seaborn as sns

train_val = ftrs_df['q'].unique()
t_ixs, v_ixs = train_test_split(train_val, test_size=0.2, random_state=SEED)
x_train = data.loc[ftrs_df['q'].isin(t_ixs)]
y_train = ftrs_df.loc[ftrs_df['q'].isin(t_ixs),'rank']
x_val = data.loc[ftrs_df['q'].isin(v_ixs)]
y_val = ftrs_df.loc[ftrs_df['q'].isin(v_ixs),'rank']

x_train, y_train = shuffle(x_train, y_train, random_state=SEED)
x_val, y_val = shuffle(x_val, y_val, random_state=SEED)

scaler = StandardScaler()
print(scaler.fit(x_train))

x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)

plt.scatter(x_val['bm25'], x_val['tfidf_gs'], c=y_val, alpha=0.1)
plt.show()

from sklearn import linear_model

model = linear_model.LogisticRegression(C=1)
model.fit(x_train, y_train)
probs = model.predict_proba(x_val)[:,0]

ones = probs[np.where(y_val == 1)]
twoes = probs[np.where(y_val == 2)]
evaluate(probs, y_val == 1, threshold=0.7)

sns.distplot(ones, label='sim')
sns.distplot(twoes, label='dissim')
plt.xlabel('prob')
plt.ylabel('freq')
plt.legend(loc="best")
plt.show(block=False)


probs = model.predict_proba(x_train)[:,0]
ones = probs[np.where(y_train == 1)]
twoes = probs[np.where(y_train == 2)]
evaluate(probs, y_train == 1, threshold=0.3)


# WHY NOT OVERFIT ON TRAIN DATA WITH XGBOOST???

import xgboost as xgb

model = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=2000, 
    learning_rate=0.3)
model.fit(x_train, y_train)

probs = model.predict_proba(x_train)[:,0]
ones = probs[np.where(y_train == 1)]
twoes = probs[np.where(y_train == 2)]
evaluate(probs, y_train == 1, threshold=0.45)

probs = model.predict_proba(x_val)[:,0]
ones = probs[np.where(y_val == 1)]
twoes = probs[np.where(y_val == 2)]
evaluate(probs, y_val == 1, threshold=0.34)


print(shuffle([1,2,3,4], random_state=1))

roc_auc_score(y_train==1, probs)

y_pred = [1 if p >= 0.35 else 2 for p in probs]
accuracy_score(y_train, y_pred)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_val, y_pred, average='macro')

#   #########################################################################

list_block = glob('../data/documents/*')
list_block.sort(key=natural_keys)
# all_ids = load_keys('../data/keys.json')

unique = set(ftrs_df['q']).union(ftrs_df['d'])
ftrs = []
for _id, doc in iter_docs(list_block, encode=False, with_ids=True, as_is=True):
    if _id not in unique:
        continue
    _ft = {'q':_id}
    _l = 0
    for k,v in doc.items():
        words = [w for s in v for w in s]
        _ft[k] = len(words)
        _ft['unique_%s' % k] = len(set(words))
        _l += _ft[k]
    _ft['total_len'] = _l
    ftrs.append(_ft)

ftrs = pd.DataFrame.from_dict(ftrs, orient='columns')
ftrs.fillna(0, inplace=True)
ftrs.set_index('q', inplace=True)
ftrs = ftrs.astype(int)
ftrs.to_csv('../data/independ_ftrs.csv')

ftrs_independent = pd.read_csv('../data/independ_ftrs.csv')


docs_ram = push_docs_to_ram(dictionary.token2id)

jaccard = []
for el in tqdm(samples):
    key = all_ids[el[0]]
    q = docs_ram[key]
    q_sets = {}
    for k,v in q.items():
        q_sets[k] = set(v)
    for _ix in el[1] + el[2]:
        _id = all_ids[_ix]
        jac = {'q':key, 'd':_id}
        d = docs_ram[_id]
        for k,v in d.items():
            if k in q_sets:
                lu = len(q_sets[k].union(v))
                if lu:
                    j = len(q_sets[k].intersection(v))/lu
                    jac['%s_j'%k] = j
        jaccard.append(jac)

jaccard = pd.DataFrame.from_dict(jaccard)
jaccard.fillna(0, inplace=True)
jaccard.to_csv('../data/jaccard.csv', index=False)

jaccard = pd.read_csv('../data/jaccard.csv')

assert len(ftrs_df) == len(jaccard)

#   ######################## word2vec ####################################

dim = 200
model = Word2Vec(Sentences('../data/documents/'), size=dim, 
    min_count=5, window=8, workers=cpu_count)
model.save('../data/w2v_200_5_w8')

all_ids = load_keys('../data/keys.json')
model = Word2Vec.load('../data/w2v_200_5_w8')
ftrs_df = pd.read_csv('../data/qdr_gens_ftrs.csv')
with open('../data/sampled.json', 'r') as f:
    samples = json.load(f)

# for w ,s in model.most_similar('стол', topn=10):
#     print('%s %s' % (w,s))

wv = model.wv
index2word = wv.index2word
docs_ram = push_docs_to_ram(wv.vocab, is_gensim=True)

cosines = []
for keys_part in tqdm(chunkIt(samples, 50)):
    res = Parallel(n_jobs=cpu_count, backend="multiprocessing") \
        (delayed(cosines_worker)(part) for
         part in chunkIt(keys_part, cpu_count))
    cosines += list(chain.from_iterable(res))


cosines = pd.DataFrame.from_dict(cosines, orient='columns')
cosines.fillna(0, inplace=True)
cosines.sort_values(['q', 'd'], inplace=True)
cosines.set_index(['q', 'd'], inplace=True)
cosines.to_csv('../data/cosines.csv')
cosines = pd.read_csv('../data/cosines.csv')

#   ############################### MPK #######################################

client = MongoClient()
db = client.fips
all_ids = load_keys('../data/keys.json')
with open('../data/sampled.json', 'r') as f:
    samples = json.load(f)

way = ['section', 'class_', 'subclass', 'main_group', 'subgroup']
combs = [way[:i+1] for i in range(len(way))]

def fetch_mkp(_id):
    doc = db.patents.find_one({'_id': ObjectId(_id)}, {'mpk': 1, '_id':0})
    if 'mpk' in doc:
        doc = rename(doc)
        doc['mpk'] = [clean_mpk(mpk) for mpk in doc['mpk']]
    else:
        doc['mpk'] = []
    return doc


all_mpk = {}
for q_ix, pos, neg in tqdm(samples):
    for _ix in [q_ix] + pos + neg:
        _id = all_ids[_ix]
        if _id not in all_mpk:
            all_mpk[_id] = fetch_mkp(_id)


with open('../data/all_mpk.pkl', 'wb') as f:
    pickle.dump(all_mpk, f)
with open('../data/all_mpk.pkl', 'rb') as f:
    all_mpk = pickle.load(f)

mpk_ftrs = []
for q_ix, pos, neg in tqdm(samples):
    q_id = all_ids[q_ix]
    q = all_mpk[q_id]
    for _ix in pos + neg:
        _id = all_ids[_ix]
        _ft = {'q':q_id, 'd': _id}
        d = all_mpk[_id]
        n = compare_mpk(q['mpk'], d['mpk'], way)
        _ft['mpk'] = n
        mpk_ftrs.append(_ft)

mpk_ftrs = pd.DataFrame.from_dict(mpk_ftrs, orient='columns')
mpk_ftrs.fillna(0, inplace=True)
mpk_ftrs.sort_values(['q', 'd'], inplace=True)
mpk_ftrs.set_index(['q', 'd'], inplace=True)
mpk_ftrs.to_csv('../data/mpk_ftrs.csv')
mpk_ftrs = pd.read_csv('../data/mpk_ftrs.csv')

#   ################################# unite features #############################

joined = ftrs_df.merge(ftrs_independent, on='q')
cp = ftrs_independent.copy()
cp['d'] = cp['q']
del cp['q']
joined = joined.merge(cp, on='d', suffixes=('_q', '_d'))

assert len(ftrs_df) == len(joined)

joined = joined.merge(jaccard, on=['q', 'd'])
joined.drop_duplicates(['q', 'd'], inplace=True)
pprint('%s %s' % (len(ftrs_df), len(joined)))
pprint(joined.isnull().sum())

for _ix, g in tqdm(joined.groupby(['q'])):
    assert len(g['rank'].unique()) > 1
    break

joined = joined.merge(cosines, on=['q', 'd'])

joined = joined.merge(mpk_ftrs, on=['q', 'd'])

joined.sort_values(['q', 'rank'], inplace=True)
joined.set_index(['q', 'd'], inplace=True)

joined.to_csv('../data/ftrs.csv.gz', compression='gzip')
joined[:10000].to_csv('../data/ftrs_show.csv')

joined = pd.read_csv('../data/ftrs.csv.gz')

#   ######################### LETOR format #############################

sims = load_sims('../data/sims.json')
keys_tv, keys_test = train_test_split(list(sims.keys()), test_size=0.2, random_state=SEED)
keys_train, keys_val = train_test_split(keys_tv, test_size=0.2, random_state=SEED)

train = joined[joined['q'].isin(keys_train)]
val = joined[joined['q'].isin(keys_val)]

save_letor(train, '../data/train.txt')
save_letor(val, '../data/vali.txt')























