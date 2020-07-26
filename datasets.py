import pandas as pd
import json
from text_processor import TextProcessor

def load_tweets(file):
    with open(file, 'r') as f:
        tweets = (json.loads(line) for i, line in enumerate(f.readlines()))
    return tweets


if __name__ == "__main__":

    file = "/Users/lucasso 1/Documents/validation/nao_politicos.json"

    data = {'favorites': [], 'user_id': [], 'text': [],
            'retweets': [], 'created_at': [],
            'tweet_id': [], 'user_screen_name':[]}
    
    for t in load_tweets(file):
        data['user_id'].append(t['user_id'])
        data['favorites'].append(t['favorites'])
        data['text'].append(t['text'])
        data['retweets'].append(t['retweets'])
        data['created_at'].append(t['created_at'])
        data['tweet_id'].append(t['tweet_id'])
        data['user_screen_name'].append(t['user_screen_name'])

    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'], unit='ms')
    df = df.set_index('created_at')
    df = df.sort_index(ascending=True)

    tp = TextProcessor()
    df['text_processed'] = tp.text_process(df.text.tolist(), hashtags=True)
    df['political'] = 0
    file = file.replace('json', 'pck')
    df.to_pickle(file)