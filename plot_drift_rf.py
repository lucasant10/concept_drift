import matplotlib
matplotlib.use('MacOSX')
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_fscore_support, f1_score, accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.utils import shuffle
import gensim
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import matplotlib.dates as mdates
import drift


def is_political(text):
    X = list()
    emb = np.zeros(100)
    for word in text:
        try:
            emb += word2vec_model[word]
        except:
            pass
    if text:    
        emb /= len(text)
    X.append(emb)
    y_pred = model.predict(X)
    return y_pred

def retrain(train):
        group = pd.concat(train)
        x = list()
        # generate vectors of words
        for text in group.text_processed.tolist():
            emb = np.zeros(100)
            for word in text:
                try:
                    emb += word2vec_model[word]
                except:
                    pass
            if text:    
                emb /= len(text)
            x.append(emb)
        # true label
        y = group.political.tolist()
        y = np.array(y)
        # compute balance among classes
        class_weights = {}
        for cw in range(len(set(y))):
            class_weights[cw] = np.where(y == cw)[0].shape[
                0]/float(len(y))
        # update model
        model = RandomForestClassifier(n_estimators=500, class_weight = class_weights)
        X, Y = shuffle(x, y, random_state=1)
        model.fit(X,Y)
        return model

def gen_data(tweet):
    emb = np.zeros(100)
    for word in tweet:
        try:
            emb += word2vec_model[word]
        except:
            pass
    if tweet:
        emb /= len(tweet)
    return [emb]



if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_data = path['dir_data']
    #model = load_model('transfer_cnn_2.h5')
    #model = load_model('drift_model_6.h5')
    #model = load_model('model_cnn.h5')
    vocab = np.load('model_cnn.npy', allow_pickle=True).item()
    #df = pd.read_pickle(dir_data + 'trainning/trainning.pck')
    #df = pd.read_pickle(dir_data + 'validation/validation.pck')
    #df = pd.read_pickle(dir_data + 'coleta/df_label.pck')
    df = pd.read_pickle(dir_data + 'validation/validation_drift.pck')
    df = df.sort_index()
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_data + 'vector_representation/cbow_s100.txt',
                                                                     binary=False,
                                                                     unicode_errors="ignore")

    #models = ['model_cnn.h5','drift_model.h5','drift_model_6.h5','drift_model_7.h5','drift_model_8.h5', 'drift_model_9.h5','drift_model_10.h5','drift_model_total.h5','drift_model_tun.h5']
    models = ['rf_model.skl']
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)

    model = joblib.load(models[0])
    precison = list()
    dates = list()  
    train = [df['2013-10-04':'2014-04-05']]  
    for date, group  in df.groupby(pd.Grouper(freq='6M')):
        y_pred = list()
        political = group[group.apply(lambda x: x['political']==True, axis=1)]['text_processed'].tolist()
        y_true = [1] * len(political)

        npolitical = group[group.apply(lambda x: x['political']==False, axis=1)]['text_processed'].tolist()
        y_true += [0] * len(npolitical)

        texts = political + npolitical
        for txt in texts:
            if is_political(txt):
                y_pred.append(1)
            else:
                y_pred.append(0)
        f1= f1_score(y_true, y_pred)
        precison.append(f1)
        dates.append(date)
        train.append(group)
        model = retrain(train)
        
    ax.plot(dates, precison, label= "classifier - 1")

    # # DDAL concept drift
    d_min = 999999;
    d_max = -999999;
    model = joblib.load(models[0])
    drift_class = drift.Drift(0.7,0.005,200)
    precison = list()
    dates = list()
    train = [df['2013-10-04':'2014-04-05']]  
    for date, group  in df.groupby(pd.Grouper(freq='6M')):
        y_pred = list()
        political = group[group.apply(lambda x: x['political']==True, axis=1)]['text_processed'].tolist()
        y_true = [1] * len(political)

        npolitical = group[group.apply(lambda x: x['political']==False, axis=1)]['text_processed'].tolist()
        y_true += [0] * len(npolitical)

        texts = political + npolitical
        for txt in texts:
            if is_political(txt):
                y_pred.append(1)
            else:
                y_pred.append(0)
        f1= f1_score(y_true, y_pred)
        precison.append(f1)
        dates.append(date)

        group['posteriori'] = group.apply(lambda x: np.max(model.predict_proba(gen_data(' '.join(x['text_processed']))), axis=1), axis=1)
        drift, d_min, d_max = drift_class.uncertainty_density(d_min, d_max, group.posteriori)
        if drift:
            try:
                print('DRIFT !!')
                d_min = 999999;
                d_max = -999999;
                group  = group.sample(frac=0.5)
                #group  = group[group.posteriori < 0.7]
                train.append(group)
                model = retrain(train)
            except Exception as e:
                print("Unexpected error: {}".format(e))

    ax.plot(dates, precison, label= "drift - 1")



    for name in models:
        model = joblib.load(name)
        precison = list()
        dates = list()
        for date, group  in df.groupby(pd.Grouper(freq='6M')):
            y_pred = list()
            political = group[group.apply(lambda x: x['political']==True, axis=1)]['text_processed'].tolist()
            y_true = [1] * len(political)

            npolitical = group[group.apply(lambda x: x['political']==False, axis=1)]['text_processed'].tolist()
            y_true += [0] * len(npolitical)

            texts = political + npolitical

            for txt in texts:
                if is_political(txt):
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            f1= f1_score(y_true, y_pred)
            precison.append(f1)
            dates.append(date)
        
        ax.plot(dates, precison, label= name)
    # with open('datas_3.txt') as fp:
    #     for line in fp:
    #         ax.axvline(x=line, ymin=0, ymax=1, color='b')

    ax.set_xlabel('Dates')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.ylim(0.7, 1)
    plt.xticks(rotation='30')
    plt.show()
    plt.clf()

