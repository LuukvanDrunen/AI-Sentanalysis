from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification, AutoModelForCausalLM, AutoModelForSequenceClassification
import os
import csv
import json
import argparse

parser = argparse.ArgumentParser(description="Script to test multiple models on parsed Tweets")
parser.add_argument("--input", type=str, required=True)
d = {}
tweet_urls = []

args = parser.parse_args()

if __name__ == '__main__':
    file = args.input

csv_filename = f"{file.split('/')[-1].split('.')[0]}-multiple_text_performance.csv"

models = ['sytnaxerror/robbert-large-with-dbrd-plus-imdb-mac', 'sytnaxerror/robbert-large-with-toxi-text', 'sytnaxerror/robbert-L-dbrd-imdb-insurance', 'sytnaxerror/robbert-L-dbrd-imdb-clapDataset']

with open(file) as f:
    data = json.load(f)

for i in models:
    model_name = i
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print(f"This is model: {model_name}")
    d[model_name] = {}
    for x in range(len(data['data'])):
        d[model_name][x] = pipe(data['data'][x]['text'])[0]
        tweet_urls.append(f"https://x.com/{data['data'][x]['author_id']}/status/{data['data'][x]['conversation_id']}")
print(d)


all_indices = sorted(set(idx for model_data in d.values() for idx in model_data))

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    

    header = ['tweet_index', 'tweet_url']
    for model_name in d:
        short_name = model_name.split('/')[-1]  # cleaner column names
        header += [f'{short_name}_label', f'{short_name}_score']
    writer.writerow(header)
    

    for idx in all_indices:
        row = [idx, tweet_urls[idx]]
        for model_name in d:
            result = d[model_name].get(idx, {})
            row += [result.get('label', ''), result.get('score', '')]
        writer.writerow(row)

print(f"Saved to {csv_filename}")