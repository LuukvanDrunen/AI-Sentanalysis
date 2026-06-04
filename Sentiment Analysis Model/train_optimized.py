from datasets import load_dataset, ClassLabel, concatenate_datasets, features
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import Trainer
from evaluate import load
import torch

from huggingface_hub import login

login()

model = AutoModelForSequenceClassification.from_pretrained(
    'sytnaxerror/robbert-large-with-dbrd-plus-imdb-mac',
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained('sytnaxerror/robbert-large-with-dbrd-plus-imdb-mac')

ds1 = load_dataset('clapAI/MultiLingualSentiment')


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=False,        # ← was 'max_length'; DataCollatorWithPadding handles padding per-batch
        truncation=True,
        max_length=256,
    )


ds1['train'] = ds1['train'].filter(lambda x: x['label'] != 'neutral', num_proc=22)
ds1['validation'] = ds1['validation'].filter(lambda x: x['label'] != 'neutral', num_proc=22)
ds1['test'] = ds1['test'].filter(lambda x: x['label'] != 'neutral', num_proc=22)
ds1['train'] = ds1['train'].class_encode_column('label')
ds1['validation'] = ds1['validation'].class_encode_column('label')
ds1['test'] = ds1['test'].class_encode_column('label')

ds1 = ds1.map(
    tokenize,
    batched=True,
    num_proc=22,
)


training_args = TrainingArguments(
    output_dir="robbert-L-dbrd-imdb-clapDataset",

    # ── Precision ──────────────────────────────────────────────────────────────
    bf16=True,
    tf32=True,
    bf16_full_eval=True,

    # ── Batch size & accumulation ──────────────────────────────────────────────
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=8,
    train_sampling_strategy="group_by_length",

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optim="adamw_torch_fused",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",

    # ── Memory & speed ────────────────────────────────────────────────────────
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    torch_compile=True,
    torch_compile_backend="inductor",
    torch_compile_mode="reduce-overhead",
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_drop_last=True,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=4,

    # ── DDP ───────────────────────────────────────────────────────────────────
    ddp_find_unused_parameters=False,

    # ── Eval & checkpointing ──────────────────────────────────────────────────
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    batch_eval_metrics=True,

    # ── Logging ───────────────────────────────────────────────────────────────
    logging_dir="./logs",
    logging_steps=50,
    report_to="tensorboard",
)


import numpy as np
import evaluate

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred, compute_result=False):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy_metric.add_batch(predictions=predictions, references=labels)
    f1_metric.add_batch(predictions=predictions, references=labels)

    if compute_result:
        return {
            **accuracy_metric.compute(),
            **f1_metric.compute(average="weighted"),
        }


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds1['train'],
    eval_dataset=ds1['validation'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.push_to_hub()

