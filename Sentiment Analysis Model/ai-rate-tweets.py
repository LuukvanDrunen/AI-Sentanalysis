from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification, AutoModelForCausalLM, AutoModelForSequenceClassification
import json
import csv
import re

# text_only_file = '../Results/Geert Wilders/Final-Wilders-Results.json'
text_only_file = '/home/luukvandrunen/OneDrive/Universiteit Utrecht/AI-Sentanalysis/Results/Geert Wilders/Part1-Final-Wilders.json'

csv_filename = 'Wilders_part1_labeled.csv'
with open(text_only_file) as f:
    data = json.load(f)
data = list(data)

model = AutoModelForSequenceClassification.from_pretrained("/home/luukvandrunen/.cache/huggingface/hub/models--sytnaxerror--robbert-L-dbrd-imdb-clap/snapshots/d070221cafb2b11bbfba1405431f8056efa8a479")
tokenizer = AutoTokenizer.from_pretrained("/home/luukvandrunen/.cache/huggingface/hub/models--sytnaxerror--robbert-L-dbrd-imdb-clap/snapshots/d070221cafb2b11bbfba1405431f8056efa8a479")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

labeled_data = pipe(data)
final_dict = {}

for i in range(len(data)):
    final_dict[data[i]['text']] = int(re.search(r'\d+', labeled_data[i]['label']).group())

# for i in data:
#     final_dict[i['text']] = int(re.search(r'\d+', labeled_data[i]['label']).group())

def save():
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        for key, value in final_dict.items():
            writer.writerow([key, value])

save()