if __name__ == "__main__":    
    from datasets import load_dataset, ClassLabel, concatenate_datasets, features
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSequenceClassification
    from transformers import Trainer
    from evaluate import load
    import torch

    from huggingface_hub import login

    login()

    # ⚡ flash_attention_2 gives ~2-3x speedup on attention vs sdpa
    # pip install flash-attn --no-build-isolation
    model = AutoModelForSequenceClassification.from_pretrained(
        'sytnaxerror/robbert-large-with-dbrd-plus-imdb-mac',
        attn_implementation="flash_attention_2",  # upgrade from sdpa
        torch_dtype=torch.bfloat16,               # load weights in bf16 directly
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

    # ds1 = ds1.remove_columns('lang')
    # ds1 = ds1.rename_column('is_toxic', 'label')
    for i in ds1:
        ds1[i] = ds1[i].filter(lambda x: x['label'] != 'neutral',num_proc=22)
        ds1[i] = ds1[i].class_encode_column('label')

    ds1 = ds1.map(
        tokenize,
        batched=True,
        num_proc=22,   # ← parallel tokenization; set to your CPU core count
    )

    training_args = TrainingArguments(
        output_dir="robbert-L-dbrd-imdb-clapDataset",

        # ── Precision ──────────────────────────────────────────────────────────────
        bf16=True,                          # Native BF16 on Ampere (A10); more stable than fp16
        tf32=True,                          # Free ~20% speedup on Ampere matmuls, no quality loss

        # ── Batch size & accumulation ──────────────────────────────────────────────
        per_device_train_batch_size=32,     # Safe for RoBERTa-large on 24 GB A10
        per_device_eval_batch_size=128,      # No optimizer states during eval → can go larger
        gradient_accumulation_steps=8,      # Effective batch = 32 × 2 GPUs × 8 = 512
        train_sampling_strategy="group_by_length",

        # ── Optimizer & scheduler ─────────────────────────────────────────────────
        optim="adamw_torch_fused",          # Fused AdamW built into PyTorch ≥2.0; no APEX needed
        learning_rate=2e-5,                 # Scale down slightly from 5e-5 for large models
        weight_decay=0.01,
        warmup_ratio=0.06,                  # ~6% of steps for warm-up; stabilizes early training
        lr_scheduler_type="cosine",         # Cosine decay outperforms linear for most finetunes

        # ── Memory & speed ────────────────────────────────────────────────────────
        gradient_checkpointing=True,        # Trades compute for memory; essential for large models
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for torch.compile compat
        torch_compile=True,
        torch_compile_backend="inductor",   # Best backend for A10 (Triton-based)
        # torch_compile_mode="reduce-overhead",  # ← reduces repeated recompilation from dynamic shapes
        torch_compile_mode="max-autotune",
        dataloader_num_workers=8,           # 4 per GPU is a safe ceiling; avoids CPU contention
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,

        # ── DDP ───────────────────────────────────────────────────────────────────
        ddp_find_unused_parameters=False,   # Must be False with gradient_checkpointing + DDP

        # ── Eval & checkpointing ──────────────────────────────────────────────────
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16_full_eval=True,

        # ── Logging ───────────────────────────────────────────────────────────────
        logging_dir="./logs",
        logging_steps=50,
        report_to="tensorboard",
    )


    # import numpy as np
    # import evaluate
    
    # accuracy_metric = evaluate.load("accuracy")
    # f1_metric = evaluate.load("f1")
    
    # def compute_metrics(eval_pred, compute_result=False):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    
    #     # Accumulate batch results
    #     accuracy_metric.add_batch(predictions=predictions, references=labels)
    #     f1_metric.add_batch(predictions=predictions, references=labels)
    
    #     # Only compute final summary on the last batch
    #     if compute_result:
    #         return {
    #             **accuracy_metric.compute(),
    #             **f1_metric.compute(average="weighted"),
    #         }

    import numpy as np
    import evaluate

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            **accuracy_metric.compute(predictions=predictions, references=labels),
            **f1_metric.compute(predictions=predictions, references=labels, average="weighted"),
        }

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8   # Aligns sequence lengths to tensor core boundaries on Ampere (BF16)
    )

    trainer = Trainer(
        model=model,                        # Pre-trained BERT model
        args=training_args,                 # Training arguments
        train_dataset=ds1['train'],
        eval_dataset=ds1['validation'],
        processing_class=tokenizer,
        data_collator=data_collator,        # Efficient batching
        compute_metrics=compute_metrics     # Custom metric
    )

    trainer.train()

    trainer.push_to_hub()