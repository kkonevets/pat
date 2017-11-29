from common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymystem3 import Mystem
import ujson
import random
from sklearn.model_selection import train_test_split
from itertools import *

SEED = 0


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
    with GzipFile(fname, 'w') as fout:
        json_str = json.dumps(docs, ensure_ascii=False)
        json_bytes = json_str.encode('utf-8')
        fout.write(json_bytes)

    gc.collect()


def iter_docs(fnames):
    for filename in tqdm(fnames):
        with GzipFile(filename) as f:
            data = ujson.load(f)
        for doc in data.values():
            yield [w for t in doc.values() for s in t for w in s]


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
    far_negs = list(set(islice(filtered, size))) + [int(iix[worst])]

    return key_ix, posvs, close_negs + far_negs


def found_at(iix, key):
    posvs = [ix_map[k] for k in sims[key]]
    ixs = np.isin(iix, posvs)
    return np.where(ixs)[0]


def argsort(keys):
    ixs = [ix_map[k] for k in keys]
    batch = tfidf[corpus[ixs]]
    cosines = index[batch]
    # reversed sort
    argsorted = np.argsort(-cosines)
    return argsorted


def tfidf_worker(keys):
    argsorted = argsort(keys)
    # first element is a query itself
    args = ((iix[1:], key) for iix, key in
            zip(argsorted, keys))

    samples = list(starmap(sample_negs, args))

    # found = list(chain.from_iterable((starmap(found_at, args2))))

    return samples


def gen_train_samples(keys_tv):
    samples = []
    # try:
    #     os.remove('../data/foundat.csv')
    # except OSError:
    #     pass
    for keys_part in tqdm(np.array_split(keys_tv, 2000)):
        res = Parallel(n_jobs=cpu_count, backend="threading") \
            (delayed(tfidf_worker)(part) for
             part in np.array_split(keys_part, cpu_count))
        samples += list(chain.from_iterable(res))

    with open('../data/sampled.json', 'w') as f:
        json.dump(samples, f)

    return samples


if __name__ == '__main__':
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

    # i = 33
    # samples[i]
    # print(all_ids[samples[i][0]])
    # print([all_ids[ix] for ix in samples[i][1]])
    # print([all_ids[ix] for ix in samples[i][2]])

    # ################################## BM25 #####################################

    from gensim.summarization.bm25 import *

    dictionary = corpora.Dictionary.load('../data/corpus.dict')
    corpus = corpora.MmCorpus('../data/corpus.mm')

    bm25_model = BM25(corpus)
    with open('../data/BM25.model', 'wb') as f:
        pickle.dump(bm25_model, f)

    # ############################## gen features ##################################

    doc_ix = 591814

    def get_features(doc_ix):
        1

    # ############################################################################












