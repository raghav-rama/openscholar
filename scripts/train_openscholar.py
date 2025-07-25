#!/usr/bin/env python3
import os
import sys
import json
import logging
from pathlib import Path
import argparse
from typing import List, Dict
import subprocess

sys.path.append(str(Path(__file__).parent.parent / "src"))

from openscholar import (
    OpenScholarDataStore,
    OpenScholarRetriever,
    OpenScholarReranker,
    OpenScholarGenerator,
    OpenScholarPipeline
)
from openscholar.data_generation import SyntheticDataGenerator
from openscholar.datastore import Paper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenScholarTrainingPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_datastore(self):
        logger.info("Preparing OpenScholar DataStore...")
        
        datastore = OpenScholarDataStore(
            data_dir=self.config['datastore']['path'],
            index_type=self.config['datastore']['index_type'],
            embedding_dim=self.config['datastore']['embedding_dim']
        )
        
        papers_path = self.config['datastore']['papers_path']
        if Path(papers_path).exists():
            logger.info(f"Loading papers from {papers_path}")
            with open(papers_path, 'r') as f:
                papers_data = json.load(f)
                
            for paper_data in papers_data[:self.config['datastore'].get('max_papers', 10000)]:
                paper = Paper(
                    paper_id=paper_data['id'],
                    title=paper_data['title'],
                    abstract=paper_data.get('abstract', ''),
                    authors=paper_data.get('authors', []),
                    year=paper_data.get('year', 0),
                    venue=paper_data.get('venue', ''),
                    citations=paper_data.get('citations', 0),
                    url=paper_data.get('url', ''),
                    full_text=paper_data.get('full_text', paper_data.get('abstract', ''))
                )
                datastore.add_paper(paper)
                
            datastore.save()
            logger.info(f"DataStore prepared with {len(datastore.papers)} papers")
        else:
            logger.warning(f"Papers file not found: {papers_path}")
            
        return datastore
        
    def generate_synthetic_data(self, datastore: OpenScholarDataStore):
        logger.info("Generating synthetic training data...")
        
        retriever = OpenScholarRetriever(
            model_name=self.config['retriever']['model_name']
        )
        reranker = OpenScholarReranker(
            model_name=self.config['reranker']['model_name']
        )
        generator = OpenScholarGenerator(
            model_name=self.config['generator']['model_name']
        )
        
        pipeline = OpenScholarPipeline(
            datastore=datastore,
            retriever=retriever,
            reranker=reranker,
            generator=generator
        )
        
        data_generator = SyntheticDataGenerator(
            pipeline=pipeline,
            model_name=self.config['data_generation']['model_name']
        )
        
        papers_list = list(datastore.papers.values())
        top_papers = sorted(
            papers_list,
            key=lambda p: p.citations,
            reverse=True
        )[:self.config['data_generation']['num_papers']]
        
        queries = data_generator.generate_queries_from_abstracts(
            top_papers,
            queries_per_paper=self.config['data_generation']['queries_per_paper']
        )
        
        synthetic_data = data_generator.generate_synthetic_responses(
            queries,
            save_intermediate=True,
            output_dir=str(self.output_dir / "synthetic_data")
        )
        
        mixed_data = data_generator.filter_and_mix_data(
            synthetic_data,
            synthetic_ratio=self.config['data_generation']['synthetic_ratio']
        )
        
        train_path = self.output_dir / "train_data.json"
        with open(train_path, 'w') as f:
            json.dump(mixed_data, f, indent=2)
            
        logger.info(f"Generated {len(mixed_data)} training examples")
        return train_path
        
    def train_retriever(self, train_data_path: Path):
        logger.info("Training retriever model...")
        
        retriever_script = Path(__file__).parent / "train_retriever.py"
        
        cmd = [
            "python", str(retriever_script),
            "--train_data", str(train_data_path),
            "--model_name", self.config['retriever']['model_name'],
            "--output_dir", str(self.output_dir / "retriever"),
            "--num_epochs", str(self.config['retriever']['num_epochs']),
            "--batch_size", str(self.config['retriever']['batch_size']),
            "--learning_rate", str(self.config['retriever']['learning_rate'])
        ]
        
        if self.config.get('use_wandb', False):
            cmd.append("--use_wandb")
            
        subprocess.run(cmd, check=True)
        logger.info("Retriever training completed")
        
    def train_generator(self, train_data_path: Path):
        logger.info("Training generator model...")
        
        generator_script = Path(__file__).parent / "train_generator.py"
        
        cmd = [
            "python", str(generator_script),
            "--train_data", str(train_data_path),
            "--model_name", self.config['generator']['model_name'],
            "--output_dir", str(self.output_dir / "generator"),
            "--num_epochs", str(self.config['generator']['num_epochs']),
            "--batch_size", str(self.config['generator']['batch_size']),
            "--learning_rate", str(self.config['generator']['learning_rate'])
        ]
        
        if self.config.get('use_wandb', False):
            cmd.append("--use_wandb")
            
        if not self.config['generator'].get('use_lora', True):
            cmd.append("--no_lora")
            
        subprocess.run(cmd, check=True)
        logger.info("Generator training completed")
        
    def run(self):
        logger.info("Starting OpenScholar training pipeline...")
        
        if self.config.get('prepare_datastore', True):
            datastore = self.prepare_datastore()
        else:
            datastore = OpenScholarDataStore(data_dir=self.config['datastore']['path'])
            datastore.load()
            
        if self.config.get('generate_data', True):
            train_data_path = self.generate_synthetic_data(datastore)
        else:
            train_data_path = Path(self.config['data_path'])
            
        if self.config.get('train_retriever', True):
            self.train_retriever(train_data_path)
            
        if self.config.get('train_generator', True):
            self.train_generator(train_data_path)
            
        logger.info("OpenScholar training pipeline completed!")
        

def create_default_config():
    config = {
        "output_dir": "outputs/openscholar_training",
        "use_wandb": False,
        "prepare_datastore": True,
        "generate_data": True,
        "train_retriever": True,
        "train_generator": True,
        "datastore": {
            "path": "data/openscholar_datastore",
            "papers_path": "data/papers.json",
            "index_type": "HNSW",
            "embedding_dim": 768,
            "max_papers": 10000
        },
        "data_generation": {
            "model_name": "meta-llama/Llama-3.1-70B-Instruct",
            "num_papers": 1000,
            "queries_per_paper": 3,
            "synthetic_ratio": 0.5
        },
        "retriever": {
            "model_name": "facebook/contriever",
            "num_epochs": 3,
            "batch_size": 32,
            "learning_rate": 1e-5
        },
        "reranker": {
            "model_name": "BAAI/bge-reranker-large"
        },
        "generator": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "use_lora": True
        }
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Train OpenScholar models")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--create_config",
        type=str,
        help="Create a default configuration file at the specified path"
    )
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        with open(args.create_config, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created default configuration at {args.create_config}")
        return
        
    if not args.config:
        parser.error("Please provide --config or use --create_config")
        
    pipeline = OpenScholarTrainingPipeline(args.config)
    pipeline.run()
    

if __name__ == "__main__":
    main()