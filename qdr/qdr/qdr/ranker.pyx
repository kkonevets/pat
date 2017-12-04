
# c imports
cimport cython
from ranker cimport *
from .trainer import load_model

ctypedef fused doc_type:
    doc_t
    word_counts_t

cdef class QueryDocumentRelevance:
    def __cinit__(self, counts, total_docs):
        '''
        Load the model and construct the C++ class

        counts: the token -> (word count, document count) map from the corpus
        total_docs: total documents in the corpus
        '''
        self._qdr_ptr = new QDR(counts, total_docs)

    def __dealloc__(self):
        del self._qdr_ptr

    def score(self, doc_t document, doc_t query):
        '''
        Compute the query-document relevance scores

        document and query are tokenized lists of words
        '''
        # cython will handle the conversion for us...
        return self._qdr_ptr.score(document, query)

    def score(self, word_counts_t document, word_counts_t query):
        # cython will handle the conversion for us...
        return self._qdr_ptr.score(document, query)


    def score_batch(self, document, queries):
        '''
        Compute the query-document relevance scores for a group of queries
            against a single document

        document is a list of tokenized words
        queries is a list of queries, each query is a list of tokenized words
        '''
        return self._qdr_ptr.score_batch(document, queries)

    def get_idf(self, word):
        return self._qdr_ptr.get_idf(word)

    @classmethod
    def load_from_file(cls, inputfile):
        ndocs, counts = load_model(inputfile)
        ret = cls(counts, ndocs)
        return ret

