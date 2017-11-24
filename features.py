from common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymystem3 import Mystem
import ujson


def save_json(ids, prefix, raw=True):
    client = MongoClient()
    db = client.fips
    mstem = Mystem()
    prog = re.compile("[\\W\\d]", re.UNICODE) # only letters and '_'

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
        topn = db.patents.find({'similar': {'$exists': True}}, {'similar':1})
        for doc in tqdm(topn):
            sims[str(doc['_id'])] = doc['similar']
        with open(fname, 'w') as f:
            json.dump(sims, f)

    return sims


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
    build_tfidf_index(dictionary, corpus, anew=True)

    index = similarities.Similarity.load('../data/sim_index/sim')
    tfidf = models.TfidfModel.load('../data/tfidf.model')

#   ####################### fetch ids data ###############################

    ids = load_keys('../data/keys.json')
    sims = load_sims('../data/sims.json')
    with open('../data/gold_mongo.json', 'r') as f:
        gold = json.load(f)

#   ############################ test  ########################################

    preds = {}
    ixs = [ids.index(k) for k in gold.keys()]
    batch = tfidf[corpus[ixs]]
    for _id, similarities in zip(gold.keys(), index[batch]):
        sorted_ixs = similarities.argsort()[::-1][:201]
        preds[_id] = [ids[i] for i in sorted_ixs if ids[i] != _id]

    res = evaluate(preds, gold)

#   ##########################################################################

    for i, similarities in enumerate(index):
        print(i)


#   #############################################################################

