import pandas as pd
import numpy as np
import sklearn
import scipy
from skmultiflow.drift_detection import ADWIN, DDM, EDDM, PageHinkley
from sklearn.metrics import classification_report
import configparser
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.utils import shuffle
import gensim
class Drift:
    def __init__(self, theta, lamb, batch_size):
       self.theta = theta
       self.lamb = lamb
       self.batch_size = batch_size
    
    def uncertainty_density(self, d_min, d_max, data):
        #calculate density, presumming that data have the max posteriori 
        density = len(data[data < self.theta])/data.shape[0]
        d_min = density if density < d_min else d_min
        d_max = density if density > d_max else d_max
        drift = 1 if (d_max - d_min) > self.lamb else 0
        return (drift, d_min, d_max)
    
    def ks_drift(self, data, prev_data):
        dist = data[data < self.theta]
        dist2 = prev_data[prev_data < self.theta]
        return (scipy.stats.ks_2samp(dist.values, dist2.values)[1] < 0.05)

    def retrain(self,train):
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
    #df = pd.read_pickle(dir_data + 'trainning/trainning.pck')
    #df = pd.read_pickle(dir_data + 'validation/validation.pck')
    df = pd.read_pickle(dir_data + 'validation/validation_drift.pck')
    df = df.sort_index()

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_data + 'vector_representation/cbow_s100.txt',
                                                                     binary=False,
                                                                     unicode_errors="ignore")

    train = [df['2013-10-04':'2014-04-05']]

    # params for DDAL concept drift
    drift_class = Drift(0.7,0.005,200)

    model = joblib.load('rf_model.skl')
        
    # # DDAL concept drift
    d_min = 999999;
    d_max = -999999;
    for r in range(0, df.shape[0], drift_class.batch_size):
        group = df[r:(r + drift_class.batch_size)]
        group['posteriori'] = group.apply(lambda x: np.max(model.predict_proba(gen_data(' '.join(x['text_processed']))), axis=1), axis=1)
        drift, d_min, d_max = drift_class.uncertainty_density(d_min, d_max, group.posteriori)
        if drift:
            try:
                print(d_min)
                print(d_max)
                print('DRIFT !!')
                d_min = 999999;
                d_max = -999999;
                group  = group.sample(frac=0.5)
                #group  = group[group.posteriori < 0.7]
                train.append(group)
            except Exception as e:
                print("Unexpected error: {}".format(e))
    model = drift_class.retrain(train)
    # KS concept drift
    # for r in range(0, df.shape[0], drift_class.batch_size):
    #     group = df[r:(r + drift_class.batch_size)]
    #     group['posteriori'] = group.apply(lambda x: np.max(pc.is_political_prob(' '.join(x['text_processed'])), axis=1), axis=1)
    #     if r != 0:
    #         prev_data = df[(r - drift_class.batch_size):r]
    #         prev_data['posteriori'] = prev_data.apply(lambda x: np.max(pc.is_political_prob(' '.join(x['text_processed'])), axis=1), axis=1)
    #     else:
    #         prev_data = group
    #     drift = drift_class.ks_drift(group.posteriori,prev_data.posteriori)
    #     if drift:
    #         datas.append((group[:1].index[0],group[-1:].index[0]))
    #         print('DRIFT !!')
    #         drift_class.retrain(pc, group)

    # Adwin concept drift
    # adwin = ADWIN(delta=0.0002)
    # for i, text in enumerate(df.text_processed.values):
    #     adwin.add_element(np.max(pc.is_political_prob(' '.join(text)), axis=1))
    #     if adwin.detected_change():
    #         datas.append(df.iloc[i-1:i].index[0])
    #         print('DRIFT !!')
    #         drift_class.retrain(pc, df.iloc[i-30:i])

    # DDM concept drift
    # ddm = DDM()
    # for i, text in enumerate(df.text_processed.values):
    #     ddm.add_element(np.max(pc.is_political_prob(' '.join(text)), axis=1))
    #     if ddm.detected_change():
    #         datas.append(df.iloc[i-1:i].index[0])
    #         print('DRIFT !!')
    #         drift_class.retrain(pc, df.iloc[i-200:i])

    # eddm = EDDM()
    # for i, text in enumerate(df.text_processed.values):
    #     eddm.add_element(np.max(pc.is_political_prob(' '.join(text)), axis=1))
    #     if eddm.detected_change():
    #         datas.append(df.iloc[i-1:i].index[0])
    #         print('DRIFT !!')
    #         drift_class.retrain(pc, df.iloc[i-200:i])

    # ph = PageHinkley()
    # for i, text in enumerate(df.text_processed.values):
    #     ph.add_element(np.max(pc.is_political_prob(' '.join(text)), axis=1))
    #     if ph.detected_change():
    #         datas.append(df.iloc[i-1:i].index[0])
    #         print('DRIFT !!')
    #         drift_class.retrain(pc, df.iloc[i-200:i])

    # save drift dates
    # print(datas)
    # with open('datas.txt', 'w') as f:
    #     for line in datas:
    #         f.write("%s\n" % line)
    
    # f.close()

    # # save qtd instaces to label
    # with open('qtd.txt', 'a') as f:
    #     for line in qtd:
    #         f.write("%s,%s,%s\n" % line)

    # f.close()

    # save drift trainned model
    joblib.dump(model,"rf_model_7.skl")
    