import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string
from itertools import islice
import re
from tqdm import tqdm
from os.path import join, exists, dirname, realpath
from glob import glob

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

stop_list = stopwords.words('russian')
stop_list.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на', 'ко'])
stop_list.extend(list(string.punctuation))
cur_dir = dirname(realpath(__file__))
with open(join(cur_dir, 'SimilarStopWords.txt'), 'r') as f:
    extra_stop_words = f.read().splitlines()
stop_list.extend(extra_stop_words)
stop_list = set(stop_list)

punkts = [s for s in string.punctuation if s not in '.!?']

# only Russian letters and minimum 2 symbols in a word
word_tokenizer = RegexpTokenizer(r'[а-яА-Яa-zA-Z]{2,}')

                
def tokenize(file_text, stop_list=stop_list):
    tokens = word_tokenizer.tokenize(file_text)
    tokens = [word for word in tokens if word not in stop_list]
    
    return tokens


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


def softmax(w, t = 1.0):
    e = np.exp(np.array(w).astype(float) / t)
    dist = e / np.sum(e)
    return dist


def evaluate(preds, gold):
    result = []
    for key, val in preds.items():
        true_val = gold[key]
        gold_len = len(true_val)

        inter10 = set(val[0:10]).intersection(true_val)
        inter20 = set(val[0:20]).intersection(true_val)
        inter200 = set(val[0:200]).intersection(true_val)

        acc10 = len(inter10)/gold_len
        acc20 = len(inter20)/gold_len
        acc200 = len(inter200)/gold_len

        result.append([acc10, acc20, acc200])

    result = pd.DataFrame(result, columns=['acc10', 'acc20', 'acc200'])
    
    print('median')
    print(result.median(axis=0))
    
    print('mean')
    print(result.mean(axis=0))
    
    ax = result['acc10'].hist()
    ax.set_xlabel("acc10")
    plt.show()
    
    ax = result['acc20'].hist()
    ax.set_xlabel("acc20")
    plt.show()
    
    ax = result['acc200'].hist()
    ax.set_xlabel("acc200")
    plt.show()
    
    return result
