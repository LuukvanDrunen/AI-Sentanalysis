import json

with open('/mnt/c/Users/luukv/OneDrive - Universiteit Utrecht/Universiteit Utrecht/AI-Sentanalysis/Results/Henri Bontenbal/Bontenbal-Last-Decade-parsed-text-only.json') as f:
    input_data = json.load(f)

for i in input_data:
    if i['text'] == "" or i['text'].startswith('RT '):
        del i['text']
    # if i['text'].startswith('RT '):
    #     del i['text']

def remove_empty_elements(d):
    """recursively remove empty lists, empty dicts, or None elements from a dictionary"""

    def empty(x):
        return x is None or x == {} or x == []

    if not isinstance(d, (dict, list)):
        return d
    elif isinstance(d, list):
        return [v for v in (remove_empty_elements(v) for v in d) if not empty(v)]
    else:
        return {k: v for k, v in ((k, remove_empty_elements(v)) for k, v in d.items()) if not empty(v)}
    
input_data = remove_empty_elements(input_data)

with open('Final-Bontenbal-Results.json', 'w') as f:
    json.dump(input_data, f, ensure_ascii=False)
    
