from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification, AutoModelForCausalLM, AutoModelForSequenceClassification

# model_name = "sytnaxerror/corona-tweet-dutch_social-test2"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

# print(pipe("De 100 jarige man die terug kwam om de wereld te reden is een waardige opvolger van de 100 jarige man die uit het raam klom en verdween. Ook dit maal laat Allan zien dat een 100 jarige niet altijd maar achter de geraniums zit. Samen met zijn vriend Jullius komt Allan weer in allerlei bizarre situaties terecht, zo beland hij met een luchtballon midden in de oceaan waar hij opgepikt wordt door een Noord-Koreaans militair schip en moet hij ontsnappen uit een nucleaire fabriek in Noord Korea. De 100 jarige man gaat wel met zijn tijd mee, dit maal heeft Allan een tablet gekregen waarop hij het nieuws kan lezen. De actuele politiek speelt dan ook een belangrijke rol in dit 2e boek over Allan. Zo heeft hij onder andere ontmoetingen met Donald Trump, Angela Merkel, en Kim Jong-Un. En natuurlijk word alles geschreven met een lekkere dosis humor waardoor er geregeld een glimlach op mijn gezicht verscheen. Voor de lezers die genoten hebben van het eerste deel van de 100 jarige man is dit zeker een aanrader."))

models = ['sytnaxerror/yhavinga-imdb_dutch', 'sytnaxerror/benjaminvdb-dbrd', 'sytnaxerror/corona-tweet-dutch_social-test2', 'sytnaxerror/corona-tweet-dutch_social-test3', 'sytnaxerror/corona-tweet-dutch_social-test4', 'sytnaxerror/corona-tweet-dutch_social-test5', 'clapAI/roberta-large-multilingual-sentiment', 'sytnaxerror/robbert-large-with-imdb', 'sytnaxerror/robbert-large-with-dbrd-mac', 'sytnaxerror/robbert-large-with-dbrd-colab', 'sytnaxerror/xlm-roberta-large-with-concat-pc', 'sytnaxerror/xlm-roberta-large-with-concat-mac', 'sytnaxerror/robbert-large-with-dbrd-plus-imdb-mac', 'sytnaxerror/robbert-large-with-toxi-text']


for i in models:
    model_name = i
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print(f"This is model: {model_name}")
    print(pipe("Meteen allemaal oppakken en ons land uitzetten! Wat doet een D66-Kamerlid met illegalen?"))