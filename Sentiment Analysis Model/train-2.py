from datasets import load_dataset, ClassLabel, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import Trainer
from evaluate import load
from huggingface_hub import login

login()

ds1 = load_dataset('benjaminvdb/dbrd')
ds2 = load_dataset('yhavinga/imdb_dutch')
ds3 = concatenate_datasets([ds1['train'], ds2['train']])
ds4 = concatenate_datasets([ds1['test'], ds2['test']])

# model_name = "clapAI/roberta-large-multilingual-sentiment"
model_name = "DTAI-KULeuven/robbert-2023-dutch-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

ds3 = ds3.map(tokenize, batched=True)
ds4 = ds4.map(tokenize, batched=True)
# dataset = dataset.map(tokenize, batched=True)

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    # output_dir=f"{str(dataset_name).replace('/', '-')}",           # Directory for saving model checkpoints
    output_dir="large-robbert-with-concat-dataset",
    eval_strategy="epoch",     # Evaluate at the end of each epoch
    save_strategy="epoch",
    learning_rate=5e-5,              # Start with a small learning rate
    per_device_train_batch_size=16,  # Batch size per GPU
    per_device_eval_batch_size=16,
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Regularization
    save_total_limit=2,              # Limit checkpoints to save space
    load_best_model_at_end=True,     # Automatically load the best checkpoint
    logging_dir="./logs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    fp16=True                        # Enable mixed precision for faster training
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=ds3,
    eval_dataset=ds4,
    processing_class=tokenizer,
    data_collator=data_collator,        # Efficient batching
    compute_metrics=compute_metrics     # Custom metric
)

trainer.train()

trainer.push_to_hub()