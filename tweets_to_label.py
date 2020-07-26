import pandas as pd

df = pd.read_pickle('df_dep_tweets_classified.pkl')
df_tmp = df['2016':'2018']
df_tmp.drop(df_tmp[df_tmp.text_processed==""].index, inplace=True)
y = ['2016','2017','2018']

df_m = df_tmp.groupby([pd.Grouper(freq='Y'),pd.Grouper(freq='M')]).count().reset_index(level=0)
values_m = list()
for i in y:
    values_m.append(np.ceil(((df_m[i].text/df_m[i].text.sum())*500).values).astype('int'))

tweets_m = list()
for i, year in enumerate(y):
    for  x,(m,g) in enumerate(df_tmp[year].groupby(pd.Grouper(freq='m'))):
        tweets_m.append(g.sample(n=values_m[i][x], random_state=1))

df_label = pd.concat(tweets_m)

df_label.text_processed.to_excel('df_label.xlsx')
df_label.to_pickle('df_label.pck')

