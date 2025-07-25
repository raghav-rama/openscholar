#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    model_name: str = "facebook/contriever"
    max_length: int = 512
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    output_dir: str = "models/retriever"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    

class RetrieverDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        is_training: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        query = item['query']
        positive_passage = item['positive_passage']
        negative_passages = item.get('negative_passages', [])
        
        query_encoding = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            positive_passage,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        output = {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
        }
        
        if self.is_training and negative_passages:
            negative_passage = np.random.choice(negative_passages)
            negative_encoding = self.tokenizer(
                negative_passage,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            output['negative_input_ids'] = negative_encoding['input_ids'].squeeze()
            output['negative_attention_mask'] = negative_encoding['attention_mask'].squeeze()
            
        return output
        

class ContrieverTrainer:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        
    def compute_embeddings(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        embeddings = outputs.last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
        return embeddings
        
    def compute_loss(self, batch):
        query_embeddings = self.compute_embeddings(
            batch['query_input_ids'],
            batch['query_attention_mask']
        )
        
        positive_embeddings = self.compute_embeddings(
            batch['positive_input_ids'],
            batch['positive_attention_mask']
        )
        
        scores = torch.matmul(query_embeddings, positive_embeddings.T)
        
        if 'negative_input_ids' in batch:
            negative_embeddings = self.compute_embeddings(
                batch['negative_input_ids'],
                batch['negative_attention_mask']
            )
            
            negative_scores = torch.matmul(query_embeddings, negative_embeddings.T)
            scores = torch.cat([scores, negative_scores], dim=1)
            
        labels = torch.arange(len(query_embeddings)).to(self.device)
        loss = torch.nn.functional.cross_entropy(scores, labels)
        
        return loss
        
    def train(
        self,
        train_dataset: RetrieverDataset,
        eval_dataset: Optional[RetrieverDataset] = None
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )
            
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        if wandb.run is not None:
            wandb.init(project="openscholar-retriever", config=self.config.__dict__)
            
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss = self.compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": loss.item() * self.config.gradient_accumulation_steps,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/global_step": global_step
                        })
                        
                total_loss += loss.item()
                global_step += 1
                
                progress_bar.set_postfix({"loss": loss.item()})
                
                if global_step % self.config.eval_steps == 0 and eval_loader:
                    eval_loss = self.evaluate(eval_loader)
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model(f"{self.config.output_dir}/best_model")
                        
                    self.model.train()
                    
                if global_step % self.config.save_steps == 0:
                    self.save_model(f"{self.config.output_dir}/checkpoint-{global_step}")
                    
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
            
        self.save_model(f"{self.config.output_dir}/final_model")
        
    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(eval_loader)
        
        if wandb.run is not None:
            wandb.log({"eval/loss": avg_loss})
            
        logger.info(f"Evaluation loss: {avg_loss:.4f}")
        return avg_loss
        
    def save_model(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        logger.info(f"Model saved to {output_dir}")
        

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="facebook/contriever")
    parser.add_argument("--output_dir", type=str, default="models/retriever")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    if args.use_wandb:
        wandb.init(project="openscholar-retriever")
        
    config = RetrieverConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    trainer = ContrieverTrainer(config)
    
    train_dataset = RetrieverDataset(
        args.train_data,
        trainer.tokenizer,
        max_length=config.max_length,
        is_training=True
    )
    
    eval_dataset = None
    if args.eval_data:
        eval_dataset = RetrieverDataset(
            args.eval_data,
            trainer.tokenizer,
            max_length=config.max_length,
            is_training=False
        )
        
    trainer.train(train_dataset, eval_dataset)
    

if __name__ == "__main__":
    main()