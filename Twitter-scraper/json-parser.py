import json
import argparse
import re
import os
import copy

parser = argparse.ArgumentParser(description="JSON Parser")
parser.add_argument("--inputfile", type=str, required=True)
parser.add_argument("--textonly", type=bool)
parser.add_argument('--lang', type=bool)
args = parser.parse_args()

fields_to_keep = ['text', 'media_metadata', 'author_id', 'public_metrics', 'entities', 'lang', 'conversation_id', 'id']

def parse(file, text, lang):
    non_dutch = []
    index_to_remove = []
    save_file = str(file).replace('.json', '-parsed.json')
    count = len([name for name in os.listdir('.') if os.path.isfile(name)])
    with open(file) as f:
        input_data = json.load(f)
    for i in input_data: #Make sure that if note_tweet field is present that this value is saved under the text key
        if 'note_tweet' in i:
            i['text'] = i['note_tweet']['text']
    if lang:
        for i in input_data:
            if i['lang'] != 'nl':
                non_dutch.append(copy.deepcopy(i)) #Copy every non-dutch Post
                index_to_remove.append(input_data.index(i)) #Save index of non-dutch Posts that need to be removed
        for i in sorted(index_to_remove, reverse=True):
            del input_data[i]
        for dictionary in non_dutch: #Only keep the text field of all the non-Dutch Posts that we might need later
            for key in list(dictionary):
                if key == 'text':
                    continue
                else:
                    del dictionary[key]
    if text:
        # for key in list(input_data): #Remove every non-data field
        #     if key == 'data' or key == 'includes':
        #         continue
        #     else:
        #         input_data.pop(key)
        for dictionary in input_data: #Only keep the text field and remove every other field
            for key in list(dictionary):
                if key == 'text':
                    continue
                else:
                    del dictionary[key]
            dictionary['text'] = re.sub(r'(?is)https://.+', '', dictionary['text']) #remove the Post URL from the text field
    else: #If we don't want to parse only on text
        for dictionary in input_data:
            for key in list(dictionary):
                if key in fields_to_keep:
                    continue
                else:
                    del dictionary[key]
            dictionary['text'] = re.sub(r'(?is)https://.+', '', dictionary['text']) #remove the Post URL from the text field
    if lang:
        if text:
            lang_and_text_only_file = str(file).replace('.json', '-parsed-non-dutch-text-only.json')
            with open(lang_and_text_only_file, 'w') as f:
                json.dump(non_dutch, f, ensure_ascii=False)
        else:
            lang_file = str(file).replace('.json', '-parsed-non-dutch.json')
            with open(lang_file, 'w') as f:
                json.dump(non_dutch, f, ensure_ascii=False)
    if text:
        save_file = str(file).replace('.json', '-parsed-text-only.json')
    try:
        with open(save_file, "x") as f:
            json.dump(input_data, f, ensure_ascii=False)
    except FileExistsError:
        with open(str(save_file).replace('.json', f'-{str(count)}.json'), "x") as f:
            json.dump(input_data, f, ensure_ascii=False)

if __name__ == '__main__':
    parse(args.inputfile, args.textonly, args.lang)