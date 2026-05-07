from xdk import Client
import json
import os
from dotenv import load_dotenv
load_dotenv()
politicians = [41778159, 99354836, 17422867, 35603755, 155507136, 125581701, 1703029690137513984, 118084079, 105161244, 49278232, 25893416, 272656097, 403931173, 231772399, 118667870, 2361754832]
n_per_person = 7
client = Client(bearer_token=os.getenv('BEARER_TOKEN'))
fields_to_include = ['text', 'note_tweet']
file = '100_tweets.json'
tweets = []


for uid in politicians:
    for post in client.users.get_posts(id=uid, max_results=n_per_person,
                                       tweet_fields=fields_to_include,
                                       exclude=['retweets']):
        if post.data and len(post.data) > 0:
            tweets.append(post.data)
            break

with open (file, 'w', encoding='UTF-8') as f:
    json.dump(tweets, f, ensure_ascii=False, indent=4)