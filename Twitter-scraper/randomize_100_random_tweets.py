import json
import random
tweets = []

with open('100_tweets.json') as f:
    data = json.load(f)

for internal_list in data:
    for internal_dict in internal_list:
        if 'note_tweet' in internal_dict:
            internal_dict['text'] = internal_dict['note_tweet']['text']
        tweets.append(internal_dict['text'])

random.shuffle(tweets)

with open('100_tweets_text_only.json', 'w', encoding='UTF-8') as f:
    json.dump(tweets, f, ensure_ascii=False, indent=4)