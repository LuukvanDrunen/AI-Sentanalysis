import json
import argparse
import re
import os
import copy

parser = argparse.ArgumentParser(description="JSON Parser")
parser.add_argument("--inputfile", type=str, required=True)
save = True
parser.add_argument("--textonly", type=bool)
parser.add_argument('--lang', type=bool)
args = parser.parse_args()

def parse(file, save, text, lang):
    non_dutch = []
    index_to_remove = []
    save_file = str(file).replace('.json', '-parsed.json')
    count = len([name for name in os.listdir('.') if os.path.isfile(name)])
    with open(file) as f:
        data = json.load(f)
    for i in data['data']:
        if 'note_tweet' in i:
            i['text'] = i['note_tweet']['text']
    if lang:
        for i in data['data']:
            if i['lang'] != 'nl':
                non_dutch.append(copy.deepcopy(i))
                index_to_remove.append(data['data'].index(i))
        for i in sorted(index_to_remove, reverse=True):
            del data['data'][i]
    if text:
        for i in non_dutch:
            for x in list(i):
                if x == 'text':
                    continue
                else: 
                    del i[x]
        for u in list(data):
            if u == 'data':
                continue
            else:
                data.pop(u)
        for i in list(data['data']):
            for x in list(i):
                if x == 'text':
                    continue
                else:
                    del i[x]
            i['text'] = re.sub(r'(?is)https://.+', '', i['text'])
    else:
        for i in list(data['data']):
            for x in list(i):
                if x == 'text' or x == 'media_metadata' or x == 'author_id' or x == 'public_metrics' or x == 'entities' or x == 'lang' or x == 'conversation_id' or x == 'id':
                    continue
                else:
                    del i[x]
            i['text'] = re.sub(r'(?is)https://.+', '', i['text'])
    if save:
        if lang:
            if text:
                lang_file = str(file).replace('.json', '-parsed-non-dutch-text-only.json')
                with open(lang_file, 'w') as f:
                    json.dump(non_dutch, f, ensure_ascii=False)
            else:
                lang_file = str(file).replace('.json', '-parsed-non-dutch.json')
                with open(lang_file, 'w') as f:
                    json.dump(non_dutch, f, ensure_ascii=False)
        if text:
            save_file = str(file).replace('.json', '-parsed-text-only.json')
        try:
            with open(save_file, "x") as f:
                json.dump(data, f, ensure_ascii=False)
        except FileExistsError:
            with open(str(save_file).replace('.json', f'-{str(count)}.json'), "x") as f:
                json.dump(data, f, ensure_ascii=False)

if __name__ == '__main__':
    parse(args.inputfile, save, args.textonly, args.lang)