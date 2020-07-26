import matplotlib
matplotlib.use('MacOSX')
import numpy as np
from sklearn.metrics import  f1_score
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
import configparser
from keras.utils import to_categorical
import matplotlib.dates as mdates

if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_data = path['dir_data']
    dir_model = path['dir_model']
    vocab = np.load('model_cnn.npy', allow_pickle=True).item()
    df = pd.read_pickle(dir_data + 'validation/validation_drift.pck')
    df = df.sort_index()

    models = ['model_cnn.h5','drift_model_drift_159500.h5','drift_model_drift_426000.h5','drift_model_drift_543000.h5','drift_model_drift_1390500.h5']
    #models = ['model_cnn.h5']
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    for name in models:
        model = load_model(dir_model+name)
        precison = list()
        dates = list()
        for date, group  in df.groupby(pd.Grouper(freq='3M')):
            print(date)
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
    with open(dir_data+'coleta/datas.txt') as fp:
        for line in fp:
            ax.axvline(x=line, ymin=0, ymax=1, color='b')

    ax.set_xlabel('Dates')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.ylim(0.4, 1)
    plt.xticks(rotation='30')
    plt.show()
    plt.clf()

