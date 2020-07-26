
from load_model import PoliticalClassification
from cnn_update import train_CNN, gen_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, load_model
import pandas as pd
import numpy as np
import sklearn
import scipy
from skmultiflow.drift_detection import ADWIN, DDM, EDDM, PageHinkley
from sklearn.metrics import classification_report
import configparser

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

    def retrain(self, pc, df_retrain):
        # set all but dense layers untrainable
        # for layer in pc.model.layers[:-1]:
        #     layer.trainable=False
        x = list()
        # generate vectors of words
        for text in df_retrain.text_processed.tolist():
            seq = list()
            for word in text:
                seq.append(pc.vocab.get(word, pc.vocab['UNK']))
            x.append(seq)
        # true label
        y = df_retrain.political.tolist()
        data = pad_sequences(x, maxlen=28)
        y = np.array(y)
        # compute balance among classes
        class_weights = {}
        for cw in range(len(set(y))):
            class_weights[cw] = np.where(y == cw)[0].shape[
                0]/float(len(y))
        # update model
        pc.model.fit(data,to_categorical(y,num_classes=2),epochs=20,class_weight=class_weights)

def save_data(datas):
    with open(dir_data + 'coleta/datas.txt', 'a+') as f:
        f.write("%s\n" % datas)
    f.close()

if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_data = path['dir_data']
    dir_model = path['dir_model']
    #df = pd.read_pickle(dir_data + 'trainning/trainning.pck')
    #df = pd.read_pickle(dir_data + 'validation/validation.pck')
    #df = pd.read_pickle(dir_data + 'validation/validation_drift.pck')
    df = pd.read_pickle(dir_data + 'coleta/df_dep_tweets_2020_pol.pck')
    df.sort_index(inplace=True)
    df = df['10-2015':].copy()

    # params for DDAL concept drift
    #drift_class = Drift(0.7,0.005,500)
    drift_class = Drift(0.7,0.2,500)
        
    pc = PoliticalClassification('model_cnn.h5','model_cnn.npy',28)
    datas = list()
    qtd = list()

    # # DDAL concept drift
    d_min = 999999;
    d_max = -999999;
    for r in range(0, df.shape[0], drift_class.batch_size):
        group = df.iloc[r:(r + drift_class.batch_size)]
        group['posteriori'] = group.text_processed.apply(lambda x: np.max(pc.is_political_prob(' '.join(x)), axis=1))
        drift, d_min, d_max = drift_class.uncertainty_density(d_min, d_max, group.posteriori)
        if drift:
            try:
                save_data(group[:1].index[0])
                print(d_min)
                print(d_max)
                print('DRIFT !!')
                group['political'] = None
                group['text_processed'] = group.text_processed.str.join(' ')
                group.to_csv(dir_data+'coleta/drift_%r.csv'% r)
                input("Press enter to continue after label")
                df_train = pd.read_csv(dir_data+'coleta/drift_%r.csv'% r, parse_dates=True, index_col='created_at')
                d_min = 999999;
                d_max = -999999;
                df_train['text_processed'] = df_train.text_processed.str.split()
                drift_class.retrain(pc, df_train)
                pc.model.save(dir_model+"drift_model_drift_%r.h5"%r)
            except Exception as e:
                print("Unexpected error: {}".format(e))
    

    # # # DDAL concept drift
    # d_min = 999999;
    # d_max = -999999;
    # for r in range(0, df.shape[0], drift_class.batch_size):
    #     group = df[r:(r + drift_class.batch_size)]
    #     group['posteriori'] = group.apply(lambda x: np.max(pc.is_political_prob(' '.join(x['text_processed'])), axis=1), axis=1)
    #     drift, d_min, d_max = drift_class.uncertainty_density(d_min, d_max, group.posteriori)
    #     if drift:
    #         try:
    #             datas.append(group[:1].index[0])
    #             print(d_min)
    #             print(d_max)
    #             print('DRIFT !!')
    #             d_min = 999999;
    #             d_max = -999999;
    #             group  = group.sample(frac=0.5)
    #             #group  = group[group.posteriori < 0.7]
    #             qtd.append(((len(group)), len(group[group.political==True]), len(group[group.political==False])))
    #             drift_class.retrain(pc, group)
    #         except Exception as e:
    #             print("Unexpected error: {}".format(e))
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
    pc.model.save(dir_model+"drift_model_drift.h5")
    