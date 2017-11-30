import ujson
from glob import glob
from gzip import GzipFile
from tqdm import tqdm
import re
from qdr import Trainer, QueryDocumentRelevance


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def iter_docs(fnames):
    for filename in tqdm(fnames):
        with GzipFile(filename) as f:
            data = ujson.load(f)
        for doc in data.values():
            yield [w for t in doc.values() for s in t for w in s]


if __name__ == '__main__':
    list_block = glob('../data/documents/*')
    list_block.sort(key=natural_keys)
    corpus = iter_docs(list_block)

    model = Trainer()
    model.train(corpus)
    model.serialize_to_file('../data/qdr_model.gz')

    model = scorer = QueryDocumentRelevance.load_from_file('../data/qdr_model.gz')
    computed_score = model.score(corpus[0], corpus[1])['bm25']
