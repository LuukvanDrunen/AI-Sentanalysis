from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification, AutoModelForCausalLM, AutoModelForSequenceClassification
import os
import csv
import json
import argparse

# model_name = "sytnaxerror/corona-tweet-dutch_social-test2"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


parser = argparse.ArgumentParser(description="Script to test multiple models on parsed Tweets")
parser.add_argument("--input", type=str, required=True)
# d = {model_name: {n of sentence: pipe output}}
d = {}
tweet_urls = []
# file = './19-04-Geert-parsed.json'

args = parser.parse_args()


if __name__ == '__main__':
    file = args.input

csv_filename = f"{file.split('/')[-1].split('.')[0]}-multiple_text_performance.csv"
# file = "/mnt/c/Users/luukv/OneDrive - Universiteit Utrecht/Universiteit Utrecht/Thesis/Twitter-scraper/14-04-Geert-parsed-text-only.json"
# file = '/home/luukvandrunen/OneDrive/Universiteit Utrecht/Thesis/Twitter-scraper/14-04-Geert-parsed-text-only.json'


# pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

# print(pipe("De 100 jarige man die terug kwam om de wereld te reden is een waardige opvolger van de 100 jarige man die uit het raam klom en verdween. Ook dit maal laat Allan zien dat een 100 jarige niet altijd maar achter de geraniums zit. Samen met zijn vriend Jullius komt Allan weer in allerlei bizarre situaties terecht, zo beland hij met een luchtballon midden in de oceaan waar hij opgepikt wordt door een Noord-Koreaans militair schip en moet hij ontsnappen uit een nucleaire fabriek in Noord Korea. De 100 jarige man gaat wel met zijn tijd mee, dit maal heeft Allan een tablet gekregen waarop hij het nieuws kan lezen. De actuele politiek speelt dan ook een belangrijke rol in dit 2e boek over Allan. Zo heeft hij onder andere ontmoetingen met Donald Trump, Angela Merkel, en Kim Jong-Un. En natuurlijk word alles geschreven met een lekkere dosis humor waardoor er geregeld een glimlach op mijn gezicht verscheen. Voor de lezers die genoten hebben van het eerste deel van de 100 jarige man is dit zeker een aanrader."))

models = ['sytnaxerror/robbert-large-with-dbrd-plus-imdb-mac', 'sytnaxerror/robbert-large-with-toxi-text', 'sytnaxerror/robbert-L-dbrd-imdb-insurance', 'sytnaxerror/robbert-L-dbrd-imdb-clapDataset']


with open(file) as f:
    data = json.load(f)

# for i in data['data']:
#     print(i['text'])

# for i in range(len(data['data'])):
    
#     print(data['data'][i]['text'])

for i in models:
    model_name = i
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print(f"This is model: {model_name}")
    d[model_name] = {}
    for x in range(len(data['data'])):
        # d[model_name] = {i : pipe(data['data'][i]['text'])[0]}
        # d.update({model_name: {i: pipe(data['data'][i]['text'])[0]}})
        d[model_name][x] = pipe(data['data'][x]['text'])[0]
        tweet_urls.append(f"https://x.com/{data['data'][x]['author_id']}/status/{data['data'][x]['conversation_id']}")
    # d[model_name] = pipe("Deze faalhaas van het CDA @MinisterAenM Bart van den Brink moet opstappen en wegwezen. Al moeten we bij dwang iedere dag honderd moties van wantrouwen tegen hem indienen, we zullen hem geen seconde met rust laten. Dit is verraad.")[0]
    # print(pipe("Deze faalhaas van het CDA @MinisterAenM Bart van den Brink moet opstappen en wegwezen. Al moeten we bij dwang iedere dag honderd moties van wantrouwen tegen hem indienen, we zullen hem geen seconde met rust laten. Dit is verraad."))

print(d)


# Collect all tweet indices
all_indices = sorted(set(idx for model_data in d.values() for idx in model_data))

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Header: tweet_index, then label+score columns per model
    header = ['tweet_index', 'tweet_url']
    for model_name in d:
        short_name = model_name.split('/')[-1]  # cleaner column names
        header += [f'{short_name}_label', f'{short_name}_score']
    writer.writerow(header)
    
    # One row per tweet
    for idx in all_indices:
        row = [idx]
        row.append(tweet_urls[idx])
        for model_name in d:
            result = d[model_name].get(idx, {})
            # row += [result]
            row += [result.get('label', ''), result.get('score', '')]
        writer.writerow(row)

print(f"Saved to {csv_filename}")