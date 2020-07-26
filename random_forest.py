# -*- coding: utf-8 -*-
import sklearn
import gensim
import pandas as pd
import math
import json
import configparser
import os
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
import numpy as np
import argparse
import sys
from sklearn.externals import joblib
from sklearn.utils import shuffle

# Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)

EMBEDDING_DIM = 100
W2VEC_MODEL_FILE = None
NO_OF_CLASSES = 2
MAX_SEQUENCE_LENGTH = 25
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
INITIALIZE_WEIGHTS_WITH = 'word2vec'
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 30
SCALE_LOSS_FUN = True
MODEL_NAME = 'cnn_model_'
DICT_NAME = 'cnn_dict_'
POLITICS_FILE = 'politics.txt'
NON_POLITICS_FILE = 'non-politics.txt'
word2vec_model = None


def select_texts(texts):
    # selects the texts as in embedding method
    # Processing
    text_return = []
    for text in texts:
        _emb = 0
        for w in text:
            if w in word2vec_model:  # Check if embeeding there in embedding model
                _emb += 1
        if _emb:   # Not a blank text
            text_return.append(text)
    print('texts selected:', len(text_return))
    return text_return


def gen_vocab(model_vec):
    vocab = dict([(k, v.index) for k, v in model_vec.vocab.items()])
    vocab['UNK'] = len(vocab) + 1
    print(vocab['UNK'])
    return vocab


def gen_data(tweets, tw_class):
    y_map = dict()
    for i, v in enumerate(sorted(set(tw_class))):
        y_map[v] = i
    print(y_map)

    X, y = [], []
    for i, tweet in enumerate(tweets):
        emb = np.zeros(EMBEDDING_DIM)
        for word in tweet:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(tweet)
        X.append(emb)
        y.append(y_map[tw_class[i]])
    return X, y



if __name__ == "__main__":

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_data = path['dir_data']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_data + 'vector_representation/cbow_s100.txt',
                                                                     binary=False,
                                                                     unicode_errors="ignore")

    df = pd.read_pickle(dir_data + 'trainning/trainning.pck')
    dfe = df['2013-10-04':'2014-04-05']
    texts = list()
    tx_class = list()

    political = dfe[dfe.apply(lambda x: x['political']
                              == True, axis=1)]['text_processed'].tolist()
    tx_class = ['politics'] * len(political)

    npolitical = dfe[dfe.apply(
        lambda x: x['political'] == False, axis=1)]['text_processed'].tolist()
    tx_class += ['non-politics'] * len(npolitical)

    texts = political + npolitical

    texts = select_texts(texts)
    X, y = gen_data(texts, tx_class)
    model = RandomForestClassifier(n_estimators=500)
    X, Y = shuffle(X, y, random_state=1)
    model.fit(X,Y)
    joblib.dump(model, 'rf_model.skl')

# python cnn.py -f cbow_s300.txt  -d 300 --epochs 10 --batch-size 30 --initialize-weights word2vec --scale-loss-function
