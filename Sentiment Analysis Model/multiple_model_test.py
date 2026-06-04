from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification, AutoModelForCausalLM, AutoModelForSequenceClassification

models = ['sytnaxerror/yhavinga-imdb_dutch', 'sytnaxerror/benjaminvdb-dbrd', 'sytnaxerror/corona-tweet-dutch_social-test2', 'sytnaxerror/corona-tweet-dutch_social-test3', 'sytnaxerror/corona-tweet-dutch_social-test4', 'sytnaxerror/corona-tweet-dutch_social-test5', 'clapAI/roberta-large-multilingual-sentiment', 'sytnaxerror/robbert-large-with-imdb', 'sytnaxerror/robbert-large-with-dbrd-mac', 'sytnaxerror/robbert-large-with-dbrd-colab', 'sytnaxerror/xlm-roberta-large-with-concat-pc', 'sytnaxerror/xlm-roberta-large-with-concat-mac', 'sytnaxerror/robbert-large-with-dbrd-plus-imdb-mac', 'sytnaxerror/robbert-large-with-toxi-text']

post = "Meteen allemaal oppakken en ons land uitzetten! Wat doet een D66-Kamerlid met illegalen?"

for i in models:
    model_name = i
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print(f"This is model: {model_name}")
    print(pipe(post))