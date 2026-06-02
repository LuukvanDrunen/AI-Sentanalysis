import json

with open('/home/luukvandrunen/OneDrive/Universiteit Utrecht/AI-Sentanalysis/Results/Geert Wilders/Wilders-Last-Decade-parsed-text-only.json') as f:
    input_data = json.load(f)

for i in input_data:
    if i['text'] == "" or i['text'].startswith('RT '):
        del i['text']
    # if i['text'].startswith('RT '):
    #     del i['text']

with open('Final-Wilders-Results.json', 'w') as f:
    json.dump(input_data, f, ensure_ascii=False)