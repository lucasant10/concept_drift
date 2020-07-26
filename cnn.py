# -*- coding: utf-8 -*-
import sklearn
import gensim
import pandas as pd
import math
import h5py
import json
import configparser
import os
from batch_gen import batch_gen
from collections import defaultdict
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
import numpy as np
from keras.layers import Activation, Dense, Dropout, Flatten, concatenate, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential, Model
from keras.layers import Embedding, Input
from keras.preprocessing.sequence import pad_sequences
import argparse
import sys
import warnings

warnings.simplefilter("ignore")


def warn(*args, **kwargs):
    pass


warnings.warn = warn


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    print("%d embedding missed" % n)
    print("%d embedding found" % len(embedding))
    return embedding


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


def gen_sequence(vocab, texts, tx_class):
    y_map = dict()
    for i, v in enumerate(sorted(set(tx_class))):
        y_map[v] = i
    print(y_map)
    X, y = [], []
    for i, text in enumerate(texts):
        seq = []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tx_class[i]])
    return X, y


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def cnn_model(sequence_length, embedding_dim):
    model_variation = 'CNN-rand'  # CNN-rand | CNN-non-static | CNN-static
    print('Model variation is %s' % model_variation)

    # Model Hyperparameters
    n_classes = NO_OF_CLASSES
    embedding_dim = EMBEDDING_DIM
    filter_sizes = (3, 4, 5)
    num_filters = 120
    dropout_prob = (0.25, 0.25)
    hidden_dims = 100

    # Training parameters
    # Word2Vec parameters, see train_word2vec
    # min_word_count = 1  # Minimum word count
    # context = 10        # Context window size

    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu')(graph_in)
        # ,subsample_length=1)(graph_in)
        pool = GlobalMaxPooling1D()(conv)
        #flatten = Flatten()(pool)
        convs.append(pool)

    if len(filter_sizes) > 1:
        out = concatenate(convs)
        #out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    # if not model_variation=='CNN-rand':
    model.add(Embedding(len(vocab)+1, embedding_dim,
                        input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    # , input_shape=(sequence_length, embedding_dim)))
    model.add(Dropout(dropout_prob[0]))
    model.add(graph)
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(len(set(tx_class)), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


def train_CNN(X, y, inp_dim, model, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print(cv_object)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "word2vec":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print("ERROR!")
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in range(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    for cw in range(len(set(tx_class))):
                        class_weights[cw] = np.where(y_temp == cw)[0].shape[
                            0]/float(len(y_temp))
                try:
                    y_temp = np_utils.to_categorical(
                        y_temp, num_classes=len(set(tx_class)))
                except Exception as e:
                    print(e)
                    print(y_temp)
                #print(x.shape, y.shape)
                loss, acc = model.train_on_batch(
                    x, y_temp, class_weight=class_weights)

        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        #print(classification_report(y_test, y_pred))
        #print(precision_recall_fscore_support(y_test, y_pred))
        # print(y_pred)
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    print("macro results are")
    print("average precision is %f" % (p / NO_OF_FOLDS))
    print("average recall is %f" % (r / NO_OF_FOLDS))
    print("average f1 is %f" % (f1 / NO_OF_FOLDS))

    print("micro results are")
    print("average precision is %f" % (p1 / NO_OF_FOLDS))
    print("average recall is %f" % (r1 / NO_OF_FOLDS))
    print("average f1 is %f" % (f11 / NO_OF_FOLDS))

    return ((p / NO_OF_FOLDS), (r / NO_OF_FOLDS), (f1 / NO_OF_FOLDS))

def train_CNN_total(X, y, model):
    class_weights = {}
    for cw in range(len(set(y))):
        class_weights[cw] = np.where(y == cw)[0].shape[
            0]/float(len(y))
    model.fit(X,to_categorical(y,num_classes=2),epochs=20,class_weight=class_weights)



if __name__ == "__main__":

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_data = path['dir_data']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_data + 'vector_representation/cbow_s100.txt',
                                                                     binary=False,
                                                                     unicode_errors="ignore")

    df = pd.read_pickle(dir_data + 'trainning/trainning.pck')
    dfe = df['2013-10-04':'2015-10-01']
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
    vocab = gen_vocab(word2vec_model)
    X, y = gen_sequence(vocab, texts, tx_class)

    data = pad_sequences(X, maxlen=28)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()
    model = cnn_model(data.shape[1], EMBEDDING_DIM)
    #p, r, f1 = train_CNN(data, y, EMBEDDING_DIM, model, W)
    train_CNN_total(data, y, model)

    model.save("model_cnn.h5")
    np.save('model_cnn.npy', vocab)
    

# python cnn.py -f cbow_s300.txt  -d 300 --epochs 10 --batch-size 30 --initialize-weights word2vec --scale-loss-function
