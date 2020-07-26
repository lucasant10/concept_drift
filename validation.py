import configparser
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from scipy.stats import norm
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_fscore_support, f1_score, accuracy_score, recall_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt
H5_FILE = 'model_cnn.h5'
NPY_FILE = 'model_cnn.npy'


def is_political(tweet):
    X = list()
    seq = list()
    for word in tweet:
        seq.append(vocab.get(word, vocab['UNK']))
    X.append(seq)
    data = pad_sequences(X, maxlen=28)
    y_pred = model.predict(data)
    y_pred = np.argmax(y_pred, axis=1)
    return True if y_pred == 1 else False


if __name__ == "__main__":

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_data = path['dir_data']
    #model = load_model('transfer_cnn_2.h5')
    model = load_model('drift_model_6.h5')
    #model = load_model('model_cnn.h5')
    vocab = np.load('model_cnn.npy', allow_pickle=True).item()
    df = pd.read_pickle(dir_data + 'trainning/trainning.pck')
    #df = pd.read_pickle(dir_data + 'validation/validation.pck')
    df = df.sort_index()

    for date, group in df.groupby(pd.Grouper(freq='6m')):
        print(date)
        y_pred = list()
        political = group[group.apply(
            lambda x: x['political'] == True, axis=1)]['text_processed'].tolist()
        y_true = [1] * len(political)

        npolitical = group[group.apply(
            lambda x: x['political'] == False, axis=1)]['text_processed'].tolist()
        y_true += [0] * len(npolitical)

        texts = political + npolitical

        for txt in texts:
            if is_political(txt):
                y_pred.append(1)
            else:
                y_pred.append(0)
        print(classification_report(y_true, y_pred))
        #p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
