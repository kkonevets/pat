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

    fmodel = join(DATA_FOLDER, 'tfidf.model')
    if not path.exists(fmodel):
        tfidf = models.TfidfModel(corpus)
        tfidf.save(fmodel)
    else:
        tfidf = models.TfidfModel.load(fmodel)

    fname = join(DATA_FOLDER, '../data/sim_index/sim')
    index = similarities.Similarity(fname, tfidf[corpus],
                                    num_features=len(dictionary), num_best=None,
                                    chunksize=10 * 256, shardsize=10 * 32768)
    index.save(fname)

    # index = similarities.Similarity.load(fname)


#   ########################################################################

    with open('../data/gold_mongo.json', 'r') as f:
        gold = json.load(f)

    topn = db.patents.find({})
    for doc in tqdm(topn, total=len(docs_ids)):
        break
