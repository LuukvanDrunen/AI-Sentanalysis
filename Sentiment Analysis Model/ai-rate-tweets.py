from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification, AutoModelForCausalLM, AutoModelForSequenceClassification
import json
import csv
import re

text_only_file = '100_tweets_text_only.json'

csv_filename = 'ai_labeled.csv'
with open(text_only_file) as f:
    data = json.load(f)

model = AutoModelForSequenceClassification.from_pretrained("sytnaxerror/robbert-L-dbrd-imdb")
tokenizer = AutoTokenizer.from_pretrained("sytnaxerror/robbert-L-dbrd-imdb")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

labeled_data = pipe(data)
final_dict = {}

for i in range(len(data)):
    final_dict[data[i]] = int(re.search(r'\d+', labeled_data[i]['label']).group())


def save():
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        for key, value in final_dict.items():
            writer.writerow([key, value])

save()