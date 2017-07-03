import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import string
from itertools import islice
import re
from tqdm import tqdm
from os.path import join, exists
from glob import glob

stop_list = stopwords.words('russian')
stop_list.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на', 'ко'])
stop_list.extend(list(string.punctuation))

punkts = [s for s in string.punctuation if s not in '.!?']


class Corpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        for filename in tqdm(self.input): # for each relevant file
            yield tokenize(open(filename).read())

            
def save_corpus(list_block, dir_name, prefix):
    corpus = Corpus(list_block)
    
    dic_name = join(dir_name, '%s.dict' % prefix)
    corp_name = join(dir_name, '%s_corpus.mm' % prefix)
    
    corpus.dictionary.save(dic_name)
    corpora.MmCorpus.serialize(corp_name, corpus)
    
    return dic_name, corp_name
    
            
def tokenize(file_text):
    tokens = nltk.word_tokenize(file_text)

    tokens = [validate_word(word) for word in tokens if check_word(word)]
    tokens = filter(lambda s: s != '', tokens)
    
    return tokens

            
def validate_word(word):
    word = word.replace("«", "").replace("»", "").replace("`", '').replace(".", '')
    if len(word) == 1:
        return ''

    return word


def check_word(word):
    if word in stop_list:
        return False
    if any(char.isdigit() for char in word):
        return False
    
    return True


def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(islice(it, n))
       if not chunk:
           return
       yield chunk


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def get_all_docs(data_folder):
    fname = join(data_folder, 'all_docs.txt')

    if not exists(fname):
        all_docs = glob(join(data_folder, 'FIPS') + '/**/*.txt', recursive=True)
        with open(fname, mode='wt', encoding='utf-8') as f:
            f.write('\n'.join(all_docs))
    else:
        with open(fname, mode='r', encoding='utf-8') as f:
            all_docs = f.read().splitlines()
    
    return all_docs