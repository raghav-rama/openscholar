import json
import random
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import asyncio
from dataclasses import dataclass, asdict
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .pipeline import OpenScholarPipeline, OpenScholarResponse
from .datastore import Paper

logger = logging.getLogger(__name__)


@dataclass
class SyntheticQuery:
    query_id: str
    query: str
    source_abstract: str
    source_paper_id: str
    query_type: str


@dataclass
class SyntheticDataPoint:
    query_id: str
    query: str
    response: str
    citations: List[Dict]
    feedback_history: List[Dict]
    quality_scores: Dict[str, float]
    data_type: str


class SyntheticDataGenerator:
    def __init__(
        self,
        pipeline: OpenScholarPipeline,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        device: str = None,
        num_workers: int = 4,
        min_citation_count: int = 10,
        min_quality_score: float = 4.5
    ):
        self.pipeline = pipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.min_citation_count = min_citation_count
        self.min_quality_score = min_quality_score
        
        logger.info(f"Loading model for data generation: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
    def generate_queries_from_abstracts(
        self,
        papers: List[Paper],
        queries_per_paper: int = 3
    ) -> List[SyntheticQuery]:
        queries = []
        
        query_types = [
            "comprehensive_review",
            "method_comparison",
            "recent_advances",
            "applications",
            "challenges_and_limitations",
            "future_directions"
        ]
        
        for paper in tqdm(papers, desc="Generating queries"):
            if paper.citations < self.min_citation_count:
                continue
                
            for i in range(queries_per_paper):
                query_type = random.choice(query_types)
                query_text = self._generate_query(paper.abstract, query_type)
                
                if query_text:
                    query = SyntheticQuery(
                        query_id=f"{paper.paper_id}_q{i}",
                        query=query_text,
                        source_abstract=paper.abstract,
                        source_paper_id=paper.paper_id,
                        query_type=query_type
                    )
                    queries.append(query)
                    
        return queries
        
    def _generate_query(self, abstract: str, query_type: str) -> str:
        prompts = {
            "comprehensive_review": "Generate a literature review question that would require synthesizing multiple papers related to this abstract:",
            "method_comparison": "Generate a question comparing different methods or approaches in the field described by this abstract:",
            "recent_advances": "Generate a question about recent advances or developments in the area covered by this abstract:",
            "applications": "Generate a question about practical applications of the research described in this abstract:",
            "challenges_and_limitations": "Generate a question about challenges and limitations in the field described by this abstract:",
            "future_directions": "Generate a question about future research directions based on this abstract:"
        }
        
        prompt = f"""{prompts.get(query_type, prompts["comprehensive_review"])}

Abstract: {abstract[:500]}...

Question (one sentence, ending with ?):"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.95,
                do_sample=True
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        question = response.strip().split('\n')[0].strip()
        if question.endswith('?'):
            return question
        return None
        
    def generate_synthetic_responses(
        self,
        queries: List[SyntheticQuery],
        save_intermediate: bool = True,
        output_dir: str = "data/synthetic"
    ) -> List[SyntheticDataPoint]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        synthetic_data = []
        
        for query in tqdm(queries, desc="Generating responses"):
            try:
                response = self.pipeline.generate(query.query)
                
                initial_response = response.response
                initial_citations = response.citations.copy()
                
                quality_scores = self._evaluate_response_quality(response)
                
                if all(score >= self.min_quality_score for score in quality_scores.values()):
                    data_point = SyntheticDataPoint(
                        query_id=query.query_id,
                        query=query.query,
                        response=response.response,
                        citations=response.citations,
                        feedback_history=response.feedback_history,
                        quality_scores=quality_scores,
                        data_type="answer_generation"
                    )
                    synthetic_data.append(data_point)
                    
                    if response.feedback_history:
                        for i, feedback in enumerate(response.feedback_history):
                            feedback_data = SyntheticDataPoint(
                                query_id=f"{query.query_id}_f{i}",
                                query=initial_response,
                                response=json.dumps(feedback),
                                citations=initial_citations,
                                feedback_history=[],
                                quality_scores=quality_scores,
                                data_type="feedback_generation"
                            )
                            synthetic_data.append(feedback_data)
                            
                if save_intermediate and len(synthetic_data) % 100 == 0:
                    self._save_batch(synthetic_data[-100:], output_dir)
                    
            except Exception as e:
                logger.error(f"Error generating response for query {query.query_id}: {e}")
                continue
                
        if save_intermediate:
            self._save_batch(synthetic_data, output_dir, final=True)
            
        return synthetic_data
        
    def _evaluate_response_quality(
        self,
        response: OpenScholarResponse
    ) -> Dict[str, float]:
        prompt = f"""Evaluate the following scientific literature review response on a scale of 1-5 for each criterion:

Query: {response.query}

Response:
{response.response}

Citations: {len(response.citations)}

Evaluation Criteria:
1. Factual Precision and Citation Accuracy (1-5)
2. Organization and Coherence (1-5)

Provide scores in the format:
Factual Precision: X/5
Organization: X/5"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                top_p=0.95
            )
            
        eval_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        scores = {
            "factual_precision": 3.0,
            "organization": 3.0
        }
        
        import re
        factual_match = re.search(r'Factual Precision:\s*(\d+(?:\.\d+)?)/5', eval_text)
        org_match = re.search(r'Organization:\s*(\d+(?:\.\d+)?)/5', eval_text)
        
        if factual_match:
            scores["factual_precision"] = float(factual_match.group(1))
        if org_match:
            scores["organization"] = float(org_match.group(1))
            
        return scores
        
    def filter_and_mix_data(
        self,
        synthetic_data: List[SyntheticDataPoint],
        general_instruction_data: Optional[List[Dict]] = None,
        scientific_instruction_data: Optional[List[Dict]] = None,
        synthetic_ratio: float = 0.5
    ) -> List[Dict]:
        answer_data = [d for d in synthetic_data if d.data_type == "answer_generation"]
        feedback_data = [d for d in synthetic_data if d.data_type == "feedback_generation"]
        
        filtered_answer_data = self._pairwise_filter(answer_data)
        
        mixed_data = []
        
        for data_point in filtered_answer_data:
            mixed_data.append({
                "instruction": data_point.query,
                "response": data_point.response,
                "citations": data_point.citations,
                "source": "openscholar_synthetic"
            })
            
        for data_point in feedback_data[:len(filtered_answer_data)]:
            mixed_data.append({
                "instruction": f"Generate feedback for this response: {data_point.query}",
                "response": data_point.response,
                "source": "openscholar_feedback"
            })
            
        if general_instruction_data:
            num_general = int(len(mixed_data) * (1 - synthetic_ratio) / synthetic_ratio)
            mixed_data.extend(random.sample(general_instruction_data, min(num_general, len(general_instruction_data))))
            
        if scientific_instruction_data:
            num_scientific = int(len(mixed_data) * 0.2)
            mixed_data.extend(random.sample(scientific_instruction_data, min(num_scientific, len(scientific_instruction_data))))
            
        random.shuffle(mixed_data)
        return mixed_data
        
    def _pairwise_filter(
        self,
        data_points: List[SyntheticDataPoint]
    ) -> List[SyntheticDataPoint]:
        filtered = []
        
        for dp in data_points:
            if dp.feedback_history:
                initial_version = SyntheticDataPoint(
                    query_id=dp.query_id + "_initial",
                    query=dp.query,
                    response=dp.response.split('\n\n')[0] if '\n\n' in dp.response else dp.response,
                    citations=dp.citations[:len(dp.citations)//2] if dp.citations else [],
                    feedback_history=[],
                    quality_scores=dp.quality_scores,
                    data_type=dp.data_type
                )
                
                winner = self._compare_responses(initial_version, dp)
                filtered.append(winner)
            else:
                filtered.append(dp)
                
        return filtered
        
    def _compare_responses(
        self,
        response1: SyntheticDataPoint,
        response2: SyntheticDataPoint
    ) -> SyntheticDataPoint:
        avg_score1 = np.mean(list(response1.quality_scores.values()))
        avg_score2 = np.mean(list(response2.quality_scores.values()))
        
        return response2 if avg_score2 > avg_score1 else response1
        
    def _save_batch(
        self,
        data: List[SyntheticDataPoint],
        output_dir: Path,
        final: bool = False
    ):
        timestamp = "final" if final else str(len(list(output_dir.glob("batch_*.json"))))
        filename = output_dir / f"batch_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump([asdict(d) for d in data], f, indent=2)
            
        logger.info(f"Saved {len(data)} data points to {filename}")