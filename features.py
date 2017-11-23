from common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymystem3 import Mystem


def save_corpus(ids, prefix, raw=True):
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


# class Corpus(gensim.corpora.TextCorpus):
#     def get_texts(self):
#         for filename in tqdm(self.input):  # for each relevant file
#             yield tokenize(open(filename).read())
#
#
# def save_corpus(list_block, dir_name, prefix):
#     corpus = Corpus(list_block)
#
#     dic_name = join(dir_name, '%s.dict' % prefix)
#     corp_name = join(dir_name, '%s_corpus.mm' % prefix)
#
#     corpus.dictionary.save(dic_name)
#     corpora.MmCorpus.serialize(corp_name, corpus)
#
#     return dic_name, corp_name


client = MongoClient()
db = client.fips

with open('../data/gold_mongo.json', 'r') as f:
    gold = json.load(f)

docs_ids = [doc['_id'] for doc in db.patents.find({}, {'_id': 1})]


parallelizer = Parallel(n_jobs=cpu_count)
tasks_iterator = (delayed(save_corpus)(list_block, i, raw=True) for
                  i, list_block in enumerate(grouper(len(docs_ids) // 1000, docs_ids)))
result = parallelizer(tasks_iterator)


save_corpus(docs_ids[:10], 'test', True)


topn = db.patents.find({})
for doc in tqdm(topn, total=len(docs_ids)):
    break


mstem.lemmatize('папа пошел домой  девочки. сегодня')[:-1]