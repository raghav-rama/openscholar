#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_length: int = 2048
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    output_dir: str = "models/generator"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    

class InstructionDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        data_type: str = "all"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            all_data = json.load(f)
            
        if data_type == "all":
            self.data = all_data
        else:
            self.data = [d for d in all_data if d.get("source", "").startswith(data_type)]
            
        logger.info(f"Loaded {len(self.data)} examples of type: {data_type}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction = item['instruction']
        response = item['response']
        
        if 'citations' in item and item['citations']:
            citations_text = "\n\nCitations:\n"
            for cit in item['citations']:
                citations_text += f"[{cit['number']}] {cit.get('paper_title', 'Unknown')}\n"
            response += citations_text
            
        if self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = f"User: {instruction}\n\nAssistant: {response}"
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = encoding["input_ids"].clone()
        
        user_token_id = self.tokenizer.encode("User:", add_special_tokens=False)[0]
        assistant_token_id = self.tokenizer.encode("Assistant:", add_special_tokens=False)[0]
        
        assistant_start = None
        for i, token_id in enumerate(encoding["input_ids"][0]):
            if token_id == assistant_token_id:
                assistant_start = i
                break
                
        if assistant_start is not None:
            labels[0, :assistant_start] = -100
            
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
        

class GeneratorTrainer:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading tokenizer and model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_kwargs = {
            "torch_dtype": torch.float16 if config.fp16 else torch.float32,
            "device_map": "auto",
            "use_cache": not config.gradient_checkpointing
        }
        
        if config.bf16 and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
    def train(
        self,
        train_dataset: InstructionDataset,
        eval_dataset: Optional[InstructionDataset] = None
    ):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to="wandb" if wandb.run is not None else "none",
            ddp_find_unused_parameters=False,
            group_by_length=True,
            dataloader_num_workers=4
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        trainer.train()
        
        trainer.save_model(f"{self.config.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/final_model")
        
        if self.config.use_lora:
            self.model.save_pretrained(f"{self.config.output_dir}/lora_adapter")
            
            logger.info("Merging LoRA weights with base model...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(f"{self.config.output_dir}/merged_model")
            self.tokenizer.save_pretrained(f"{self.config.output_dir}/merged_model")
            

def generate_retriever_training_data(papers_path: str, output_path: str, num_samples: int = 10000):
    with open(papers_path, 'r') as f:
        papers = json.load(f)
        
    training_data = []
    
    for _ in tqdm(range(num_samples), desc="Generating retriever training data"):
        paper = np.random.choice(papers)
        
        if 'abstract' not in paper or not paper['abstract']:
            continue
            
        query_templates = [
            "What are the main contributions of research on {topic}?",
            "Summarize recent advances in {topic}",
            "What methods are used for {topic}?",
            "What are the applications of {topic}?",
            "What challenges exist in {topic} research?"
        ]
        
        words = paper['abstract'].split()
        topic = " ".join(np.random.choice(words, size=min(3, len(words)), replace=False))
        
        query = np.random.choice(query_templates).format(topic=topic)
        
        positive_passage = f"{paper['title']}\n\n{paper['abstract']}"
        
        negative_papers = np.random.choice(papers, size=5)
        negative_passages = []
        for neg_paper in negative_papers:
            if neg_paper['id'] != paper['id'] and 'abstract' in neg_paper:
                negative_passages.append(f"{neg_paper['title']}\n\n{neg_paper['abstract']}")
                
        training_data.append({
            "query": query,
            "positive_passage": positive_passage,
            "negative_passages": negative_passages
        })
        
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
        
    logger.info(f"Generated {len(training_data)} training examples for retriever")
    

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/generator")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--data_type", type=str, default="all", choices=["all", "openscholar", "general"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    
    args = parser.parse_args()
    
    if args.use_wandb:
        wandb.init(project="openscholar-generator")
        
    config = GeneratorConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_lora=not args.no_lora
    )
    
    trainer = GeneratorTrainer(config)
    
    train_dataset = InstructionDataset(
        args.train_data,
        trainer.tokenizer,
        max_length=config.max_length,
        data_type=args.data_type
    )
    
    eval_dataset = None
    if args.eval_data:
        eval_dataset = InstructionDataset(
            args.eval_data,
            trainer.tokenizer,
            max_length=config.max_length,
            data_type=args.data_type
        )
        
    trainer.train(train_dataset, eval_dataset)
    

if __name__ == "__main__":
    main()