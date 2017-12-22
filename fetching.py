from common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymystem3 import Mystem
import ujson
import copy


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


def iter_docs(fnames, encode=False, with_ids=False, as_is=False, visual=True):
    def do_encode(w):
        return w.encode() if encode else w

    iterator = tqdm(fnames) if visual else fnames

    for filename in iterator:
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


class Sentences(object):
    def __init__(self, folder):
        self.folder = folder

    def __iter__(self):
        fnames = glob(join(self.folder, '*.json.gz'))
        for doc in iter_docs(fnames, as_is=True):
            for t in doc.values():
                for s in t:
                    yield s


def save_corpus(list_block, dir_name, prefix='corpus'):
    dic_name = join(dir_name, '%s.dict' % prefix)
    corp_name = join(dir_name, '%s.mm' % prefix)

    dictionary = Dictionary(iter_docs(list_block), prune_at=None)
    dictionary.save(dic_name)

    corpus = Corpus(list_block, dictionary=dictionary)
    MmCorpus.serialize(corp_name, corpus)

    return dic_name, corp_name


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


class MPKFetcher:
    def __init__(self):
        self.all_mpk = {}
        self.client = MongoClient()
        self.db = self.client.fips

    @staticmethod
    def clean_mpk(mpk):
        def fix(name):
            fixes = {u'\u041d': u'H', u'\u0410': u'A', u'\u0412': u'B'}
            if name in fixes:
                return fixes[name]
            else:
                return name

        mpk = copy.deepcopy(mpk)
        for k, v in mpk.items():
            mpk[k] = fix(v.strip())
            if k in (u'main_group', u'class_', u'subgroup'):
                mpk[k] = int(mpk[k])
        return mpk

    @staticmethod
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

    def fetch_one(self, _id):
        doc = self.db.patents.find_one({'_id': ObjectId(_id)}, {'mpk': 1, '_id': 0})
        if 'mpk' in doc:
            doc = self.rename(doc)
            doc['mpk'] = [self.clean_mpk(mpk) for mpk in doc['mpk']]
        else:
            doc['mpk'] = []
        return doc

    def fetch_all(self, samples, all_ids, fname=None):
        for q_ix, pos, neg in tqdm(samples):
            for _ix in [q_ix] + pos + neg:
                _id = all_ids[_ix]
                if _id not in self.all_mpk:
                    self.all_mpk[_id] = self.fetch_one(_id)

        if fname:
            with open(fname, 'wb') as f:
                pickle.dump(self.all_mpk, f)

        return self.all_mpk


if __name__ == "__main__":
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
