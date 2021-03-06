# -*- coding: utf-8 -*-

from __future__ import division

# import tensorflow as tf
import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.similarities import Similarity
from gensim.models import Word2Vec, Doc2Vec, TfidfModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from itertools import islice, filterfalse, starmap, chain, takewhile, zip_longest
import re, json
from tqdm import tqdm, tqdm_notebook
from os.path import join, exists, dirname, realpath, basename
import os
from glob import glob, iglob
from os import path
from gzip import GzipFile
from pprint import pprint
import pickle
import gc, io
import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import multiprocessing

cpu_count = multiprocessing.cpu_count()

DATA_FOLDER = '../data/'

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

fileHandler = logging.FileHandler("{0}/{1}.log".format(DATA_FOLDER, 'pat'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

##################################################################################################

simple_stop_list = stopwords.words('russian')
simple_stop_list.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на', 'ко'])

cur_dir = dirname(realpath(__file__))
with io.open(join(cur_dir, 'SimilarStopWords.txt'), 'r', encoding='utf8') as f:
    extra_stop_words = f.read().splitlines()

stop_list = set(simple_stop_list)
simple_stop_list = set(simple_stop_list)
stop_list.update(extra_stop_words)

# only letters and '_'
prog = re.compile("[\\W\\d]", re.UNICODE)


def tokenize(sentence, stop_words):
    tokens = word_tokenize(sentence)
    tokens = (prog.sub('', w) for w in tokens)
    tokens = (w for w in tokens if len(w) > 1 and w not in stop_words)

    return tokens


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def get_all_docs(data_folder, ext='txt'):
    fname = join(data_folder, 'all_docs.txt')

    if not exists(fname):
        all_docs = glob(data_folder + '/**/*.%s' % ext, recursive=True)
        all_docs = sorted(all_docs)
        with open(fname, mode='wt') as f:
            f.write('\n'.join(all_docs))
    else:
        with open(fname, mode='r') as f:
            all_docs = f.read().splitlines()

    return all_docs


def softmax(w, t=1.0):
    e = np.exp(np.array(w).astype(float) / t)
    dist = e / np.sum(e)
    return dist


def evaluate(preds, gold):
    result = []
    l = 0
    for key, val in preds.items():
        true_val = gold[key]

        inter10 = set(val[0:10]).intersection(true_val)
        inter20 = set(val[0:20]).intersection(true_val)
        inter200 = set(val[0:200]).intersection(true_val)

        acc10 = len(inter10)
        acc20 = len(inter20)
        acc200 = len(inter200)

        result.append([acc10, acc20, acc200])
        l += len(true_val)

    result = pd.DataFrame(result, columns=['acc10', 'acc20', 'acc200'])

    logging.info('\n%s' % (result.sum(axis=0) / float(l)))

    return result


def send_email(notebook_url,
               sender='guyos@bk.ru',
               receivers=['kkonevets@gmail.com'],
               subject='jupyter notification',
               body=''):
    import smtplib
    from os.path import expanduser

    home = expanduser("~")
    with open(home + '/hash') as f:
        password = f.read().strip()

    msg = "\r\n".join([
        "From: %s" % sender,
        "To: %s" % (','.join(receivers)),
        "Subject: %s" % subject,
        "%s" % body,
        "%s" % notebook_url
    ])

    server = smtplib.SMTP('smtp.mail.ru:587')
    server.ehlo()
    server.starttls()

    server.login('guyos@bk.ru', password)
    server.sendmail(sender, receivers, msg)
    server.quit()


def first_unique(iterable, n):
    """
    iterable = [1,2,1,7,2,5,6,1,3], n = 4 -> [1, 2, 7, 5]
    """
    unique = set()
    def condition(x):
        nonlocal unique, n
        unique.update([x])
        return len(unique) <= n
    filtered = filterfalse(lambda x: x in unique, iterable)
    return takewhile(condition, filtered)


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

