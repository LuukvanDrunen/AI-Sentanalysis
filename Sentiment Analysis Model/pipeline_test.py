from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification, AutoModelForCausalLM, AutoModelForSequenceClassification

d = {}

model_name = "sytnaxerror/robbert-large-with-dbrd-plus-imdb-mac"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

print(pipe("Deze faalhaas van het CDA @MinisterAenM Bart van den Brink moet opstappen en wegwezen. Al moeten we bij dwang iedere dag honderd moties van wantrouwen tegen hem indienen, we zullen hem geen seconde met rust laten. Dit is verraad.")[0])