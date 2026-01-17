import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from log import logger


def self_training(exp_id, model_id="Qwen/Qwen2-7B-Instruct", ft_prefix="graph") -> str:
    """Fine-tune Qwen model locally"""
    
    finetuning_file = Path(f"outputs/finetuning_jsonl/finetune-{exp_id}.jsonl")
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(finetuning_file), split="train")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    def preprocess_function(examples):
        texts = [f"{ex['messages'][0]['content']}\n{ex['messages'][1]['content']}" for ex in examples["messages"]]
        tokenized = tokenizer(texts, truncation=True, max_length=2048)
        return tokenized
    
    dataset = dataset.map(preprocess_function, batched=True)
    
    output_dir = f"checkpoint/finetuned_{ft_prefix}_v{exp_id+1}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    ft_model_id = f"{model_id}-finetuned-{ft_prefix}-v{exp_id+1}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Fine-tuned model saved to: %s", output_dir)
    return ft_model_id