# -*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf

import gensim
from gensim import corpora, models, similarities
from gensim.models import Word2Vec, Doc2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
import string
from itertools import islice
from operator import itemgetter
import re, json
from tqdm import tqdm, tqdm_notebook
from os.path import join, exists, dirname, realpath, basename
from glob import glob, iglob
from os import path
from gzip import GzipFile
from pprint import pprint
import pickle
import gc, ujson, io

import numpy as np
import pandas as pd

import logging
from sklearn.metrics.pairwise import cosine_similarity

from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory

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
simple_stop_list.extend(list(string.punctuation))

cur_dir = dirname(realpath(__file__))
with io.open(join(cur_dir, 'SimilarStopWords.txt'), 'r', encoding='utf8') as f:
    extra_stop_words = f.read().splitlines()

stop_list = set(simple_stop_list)
simple_stop_list = set(simple_stop_list)
stop_list.update(extra_stop_words)

punkts = [s for s in string.punctuation if s not in '.!?']

# only Russian letters and minimum 2 symbols in a word
word_tokenizer = RegexpTokenizer(u'[а-яА-Яa-zA-Z]{2,}')

def tokenize(file_text, stop_list=stop_list):
    tokens = word_tokenize(file_text)
    if stop_list is not None:
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


def softmax(w, t = 1.0):
    e = np.exp(np.array(w).astype(float) / t)
    dist = e / np.sum(e)
    return dist


def evaluate(preds, gold):

    result = []
    for key, val in tqdm(preds.items()):
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

    logging.info('median')
    logging.info(result.median(axis=0))

    logging.info('mean')
    logging.info(result.mean(axis=0))

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
        "Subject: %s" %subject,
        "%s" % body,
        "%s" % notebook_url
      ])


    server = smtplib.SMTP('smtp.mail.ru:587')
    server.ehlo()
    server.starttls()

    server.login('guyos@bk.ru', password)
    server.sendmail(sender, receivers, msg)
    server.quit()
