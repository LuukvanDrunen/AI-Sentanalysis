from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import Trainer
from evaluate import load

metric = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model_name = "DTAI-KULeuven/robbert-2023-dutch-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset('benjaminvdb/dbrd', split='train')

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

# label_mapping = {"negative": 0, "positive": 1}
# dataset = dataset.map(lambda x: {"label": label_mapping[x["label"]]})

dataset.nam

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
tokenized_dataset = dataset.train_test_split(test_size=0.1)

print(tokenized_dataset["train"][0])


# training_args = TrainingArguments(
#     output_dir="roberta-finetuned",
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#     gradient_checkpointing=True,
#     bf16=True,
#     learning_rate=2e-5,
#     logging_steps=10,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )

training_args = TrainingArguments(
    output_dir="./results",           # Directory for saving model checkpoints
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

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     processing_class=tokenizer,
#     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )

trainer = Trainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,        # Efficient batching
    compute_metrics=compute_metrics     # Custom metric
)

trainer.train()