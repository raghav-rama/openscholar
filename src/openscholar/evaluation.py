import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ScholarQAExample:
    query_id: str
    query: str
    domain: str
    expert_answer: str
    expert_citations: List[Dict]
    metadata: Dict


@dataclass
class EvaluationResult:
    query_id: str
    metrics: Dict[str, float]
    generated_answer: str
    generated_citations: List[Dict]
    expert_answer: str
    expert_citations: List[Dict]


class ScholarQABench:
    def __init__(
        self,
        data_path: str,
        domains: Optional[List[str]] = None
    ):
        self.data = self._load_data(data_path)
        
        if domains:
            self.data = [d for d in self.data if d.domain in domains]
            
        logger.info(f"Loaded {len(self.data)} examples from ScholarQABench")
        
    def _load_data(self, data_path: str) -> List[ScholarQAExample]:
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
            
        examples = []
        for item in raw_data:
            example = ScholarQAExample(
                query_id=item['query_id'],
                query=item['query'],
                domain=item['domain'],
                expert_answer=item['expert_answer'],
                expert_citations=item.get('expert_citations', []),
                metadata=item.get('metadata', {})
            )
            examples.append(example)
            
        return examples
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def get_queries(self) -> List[str]:
        return [ex.query for ex in self.data]
        
    def get_by_domain(self, domain: str) -> List[ScholarQAExample]:
        return [ex for ex in self.data if ex.domain == domain]


class OpenScholarEvaluator:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model: str = "microsoft/deberta-large-mnli",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model, device=self.device)
        
        logger.info(f"Loading NLI model: {nli_model}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        self.nli_model.to(self.device)
        self.nli_model.eval()
        
    def evaluate_single(
        self,
        generated_answer: str,
        generated_citations: List[Dict],
        expert_answer: str,
        expert_citations: List[Dict],
        query: str
    ) -> Dict[str, float]:
        metrics = {}
        
        metrics['citation_accuracy'] = self._evaluate_citation_accuracy(
            generated_answer, generated_citations
        )
        
        metrics['citation_relevance'] = self._evaluate_citation_relevance(
            generated_citations, expert_citations
        )
        
        metrics['factual_correctness'] = self._evaluate_factual_correctness(
            generated_answer, expert_answer
        )
        
        metrics['coverage'] = self._evaluate_coverage(
            generated_answer, expert_answer
        )
        
        metrics['organization'] = self._evaluate_organization(
            generated_answer
        )
        
        metrics['relevance'] = self._evaluate_relevance(
            generated_answer, query
        )
        
        metrics['overall'] = np.mean(list(metrics.values()))
        
        return metrics
        
    def _evaluate_citation_accuracy(
        self,
        answer: str,
        citations: List[Dict]
    ) -> float:
        citation_pattern = r'\[(\d+)\]'
        cited_numbers = set(re.findall(citation_pattern, answer))
        
        provided_numbers = set(str(c.get('number', '')) for c in citations)
        
        if not cited_numbers:
            return 0.0 if "based on" in answer.lower() else 1.0
            
        valid_citations = cited_numbers.intersection(provided_numbers)
        accuracy = len(valid_citations) / len(cited_numbers)
        
        return accuracy
        
    def _evaluate_citation_relevance(
        self,
        generated_citations: List[Dict],
        expert_citations: List[Dict]
    ) -> float:
        if not generated_citations or not expert_citations:
            return 0.0
            
        gen_texts = [c.get('text', '') for c in generated_citations]
        exp_texts = [c.get('text', '') for c in expert_citations]
        
        gen_embeddings = self.embed_model.encode(gen_texts)
        exp_embeddings = self.embed_model.encode(exp_texts)
        
        similarity_matrix = util.cos_sim(gen_embeddings, exp_embeddings)
        
        max_similarities = torch.max(similarity_matrix, dim=1)[0]
        relevance_score = torch.mean(max_similarities).item()
        
        return relevance_score
        
    def _evaluate_factual_correctness(
        self,
        generated: str,
        expert: str
    ) -> float:
        gen_sentences = self._split_into_sentences(generated)
        exp_sentences = self._split_into_sentences(expert)
        
        if not gen_sentences:
            return 0.0
            
        entailment_scores = []
        
        for gen_sent in gen_sentences[:10]:
            max_score = 0.0
            
            for exp_sent in exp_sentences:
                score = self._compute_entailment(exp_sent, gen_sent)
                max_score = max(max_score, score)
                
            entailment_scores.append(max_score)
            
        return np.mean(entailment_scores) if entailment_scores else 0.0
        
    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
        entailment_prob = probs[0][0].item()
        contradiction_prob = probs[0][2].item()
        
        score = entailment_prob - contradiction_prob
        return (score + 1) / 2
        
    def _evaluate_coverage(
        self,
        generated: str,
        expert: str
    ) -> float:
        gen_embedding = self.embed_model.encode(generated)
        exp_embedding = self.embed_model.encode(expert)
        
        similarity = util.cos_sim(gen_embedding, exp_embedding).item()
        
        gen_length = len(generated.split())
        exp_length = len(expert.split())
        length_ratio = min(gen_length / exp_length, 2.0) / 2.0
        
        coverage_score = 0.7 * similarity + 0.3 * length_ratio
        
        return coverage_score
        
    def _evaluate_organization(self, answer: str) -> float:
        paragraphs = answer.split('\n\n')
        
        if len(paragraphs) < 2:
            return 0.5
            
        has_intro = any(
            keyword in paragraphs[0].lower()[:100]
            for keyword in ['research', 'studies', 'literature', 'work']
        )
        
        has_citations = bool(re.findall(r'\[\d+\]', answer))
        
        avg_paragraph_length = np.mean([len(p.split()) for p in paragraphs])
        good_paragraph_length = 50 <= avg_paragraph_length <= 200
        
        transition_words = [
            'however', 'moreover', 'furthermore', 'additionally',
            'in contrast', 'similarly', 'consequently', 'therefore'
        ]
        has_transitions = any(
            word in answer.lower()
            for word in transition_words
        )
        
        score = 0.0
        if has_intro:
            score += 0.3
        if has_citations:
            score += 0.3
        if good_paragraph_length:
            score += 0.2
        if has_transitions:
            score += 0.2
            
        return score
        
    def _evaluate_relevance(self, answer: str, query: str) -> float:
        answer_embedding = self.embed_model.encode(answer)
        query_embedding = self.embed_model.encode(query)
        
        relevance = util.cos_sim(answer_embedding, query_embedding).item()
        
        query_keywords = set(query.lower().split())
        answer_keywords = set(answer.lower().split())
        keyword_overlap = len(query_keywords.intersection(answer_keywords)) / len(query_keywords)
        
        final_relevance = 0.7 * relevance + 0.3 * keyword_overlap
        
        return final_relevance
        
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
    def evaluate_dataset(
        self,
        predictions: List[Dict],
        benchmark: ScholarQABench,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        results = []
        
        for pred in tqdm(predictions, desc="Evaluating"):
            example = next(
                (ex for ex in benchmark.data if ex.query_id == pred['query_id']),
                None
            )
            
            if not example:
                continue
                
            metrics = self.evaluate_single(
                generated_answer=pred['response'],
                generated_citations=pred.get('citations', []),
                expert_answer=example.expert_answer,
                expert_citations=example.expert_citations,
                query=example.query
            )
            
            result = EvaluationResult(
                query_id=example.query_id,
                metrics=metrics,
                generated_answer=pred['response'],
                generated_citations=pred.get('citations', []),
                expert_answer=example.expert_answer,
                expert_citations=example.expert_citations
            )
            
            results.append(result)
            
        aggregate_metrics = self._aggregate_metrics(results)
        
        if save_path:
            self._save_results(results, aggregate_metrics, save_path)
            
        return aggregate_metrics
        
    def _aggregate_metrics(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        all_metrics = defaultdict(list)
        
        for result in results:
            for metric, value in result.metrics.items():
                all_metrics[metric].append(value)
                
        aggregate = {}
        for metric, values in all_metrics.items():
            aggregate[f"{metric}_mean"] = np.mean(values)
            aggregate[f"{metric}_std"] = np.std(values)
            
        return aggregate
        
    def _save_results(
        self,
        results: List[EvaluationResult],
        aggregate_metrics: Dict[str, float],
        save_path: str
    ):
        output_data = {
            "aggregate_metrics": aggregate_metrics,
            "individual_results": [asdict(r) for r in results]
        }
        
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Evaluation results saved to {save_path}")