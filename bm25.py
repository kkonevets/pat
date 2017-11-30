import gzip
from qdr.ranker import QueryDocumentRelevance
from contextlib import closing


def load_model(inputfile):
    '''
    Return total docs, counts dict
    '''
    with closing(gzip.GzipFile(inputfile, 'r')) as f:
        ndocs = int(f.readline().strip())
        counts = {}
        for line in f:
            word, count1, count2 = line.decode().strip().split('\t')
            counts[word.encode()] = [int(count1), int(count2)]
    return ndocs, counts


def write_model(ndocs, counts, outputfile):
    '''Write to output file'''
    with closing(gzip.GzipFile(outputfile, 'w')) as f:
        f.write(("%s\n" % ndocs).encode())
        for word, count in counts.items():
            f.write(("%s\t%s\t%s\n" % (word, count[0], count[1])).encode())


def load_from_file(inputfile):
    ndocs, counts = load_model(inputfile)
    ret = QueryDocumentRelevance(counts, ndocs)
    return ret


def serialize_to_file(model, outputfile):
    write_model(model._total_docs, model._counts, outputfile)
