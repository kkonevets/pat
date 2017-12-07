
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


SEED = 0


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


def sample_tups(ix_map, sims, n=1, seed=0):
    """
    :param n: "number of negatives" = n * "number of positives" for each query
    :return: pairs of documents with relevance
    """
    tups = []
    ixs = list(range(len(ix_map)))
    random.seed(seed)
    random.shuffle(ixs)

    it = iter(ixs)
    for k, v in sims.items():
        k_ix = ix_map[k]
        v_ixs = [ix_map[vi] for vi in v]
        positives = set([k_ix] + v_ixs)
        for pos in v_ixs:
            tups.append((k_ix, pos, 1))
        for i in range(n * len(v)):
            neg = next(it)
            while neg in positives:
                neg = next(it)
            tups.append((k_ix, neg, 0))

    random.shuffle(tups)
    return tups


def train_val_test_tups(ix_map, sims, n=1, seed=0, test_size=0.2):
    keys, keys_test = train_test_split(list(sims.keys()), test_size=test_size, random_state=seed)
    keys_train, keys_val = train_test_split(keys, test_size=test_size, random_state=seed)

    sims_train = {k: sims[k] for k in keys_train}
    sims_val = {k: sims[k] for k in keys_val}
    sims_test = {k: sims[k] for k in keys_test}

    train = sample_tups(ix_map, sims_train, n=n, seed=seed)
    val = sample_tups(ix_map, sims_val, n=n, seed=seed)
    test = sample_tups(ix_map, sims_test, n=n, seed=seed)

    logging.info("train %s, val %s, test %s" % (len(train), len(val), len(test)))

    return train, val, test


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

    # slice starting from 1 - could be duplicate
    filtered = filterfalse(lambda x: x in exclude, map(int, iix[1:]))
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
    for samples_part in tqdm(chunkIt(samples, 500)):
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


    scored = zip(all_ids[1:], (s['bm25'] for s in scores if s))
    scored = list(sorted(scored, key=itemgetter(1), reverse=True))
    scored[:10]



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
with open('../data/sampled2.json', 'r') as f:
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

save_qdr_features(model, corpus, samples)

with open('../data/qdr_scores.pkl', 'rb') as f:
    scores = pickle.load(f)
pprint(scores[4])

sim_bms, close_bms, far_bms, sfar_bms = [], [], [], []
wrong = 0
for el in scores:
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

l = [1,2,1,7,2,5,6,1,3]
unique = set()
n = 3

list(first_unique(l, 3))

###################### bm25 sanity check #######################################

bm25_sanity_check(model, corpus, samples)

# ############################## gen features ##################################

doc_ix = 591814


def get_features(doc_ix):
    1

# ############################################################################









