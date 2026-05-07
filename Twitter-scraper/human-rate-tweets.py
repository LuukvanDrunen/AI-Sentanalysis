import json
import csv
csv_filename = 'gold_label.csv'
with open('100_tweets_text_only.json') as f:
    data = json.load(f)

final_dict = {}

def save():
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        for key, value in final_dict.items():
            writer.writerow([key, value])

for i in data:
    try:
        label = input(f'{i} \n')
        final_dict[i] = int(label)
    finally:
        save()



save()