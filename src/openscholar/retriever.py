import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import logging
from dataclasses import dataclass
import asyncio
import aiohttp
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    passage_id: str
    text: str
    score: float
    paper_id: str
    paper_title: str
    paper_year: int
    paper_citations: int


class OpenScholarRetriever:
    def __init__(
        self,
        model_name: str = "facebook/contriever",
        device: str = None,
        batch_size: int = 32,
        use_semantic_scholar: bool = True,
        use_web_search: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_semantic_scholar = use_semantic_scholar
        self.use_web_search = use_web_search
        
        logger.info(f"Loading retriever model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        embeddings = []
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings) if embeddings else np.array([])
        
    def encode_passages(self, passages: List[str]) -> np.ndarray:
        return self.encode_queries(passages)
        
    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict]:
        keywords = self._extract_keywords(query)
        
        async with aiohttp.ClientSession() as session:
            results = []
            
            for keyword in keywords[:3]:
                url = f"https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": keyword,
                    "limit": limit,
                    "fields": "paperId,title,abstract,year,citationCount,authors,url"
                }
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            results.extend(data.get("data", []))
                except Exception as e:
                    logger.error(f"Error searching Semantic Scholar: {e}")
                    
            seen_ids = set()
            unique_results = []
            for paper in sorted(results, key=lambda x: x.get("citationCount", 0), reverse=True):
                if paper["paperId"] not in seen_ids:
                    seen_ids.add(paper["paperId"])
                    unique_results.append(paper)
                    
            return unique_results[:limit]
            
    def _extract_keywords(self, query: str) -> List[str]:
        prompt = f"Extract 3-5 key search terms from this query: {query}"
        
        keywords = []
        words = query.lower().split()
        
        technical_terms = [w for w in words if len(w) > 4 and w.isalpha()]
        keywords.extend(technical_terms[:3])
        
        if len(keywords) < 3:
            keywords.append(query[:50])
            
        return keywords
        
    async def search_web(
        self,
        query: str,
        limit: int = 10,
        domains: List[str] = ["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "aclanthology.org"]
    ) -> List[Dict]:
        results = []
        
        return results
        
    def retrieve(
        self,
        query: str,
        datastore,
        k: int = 100,
        filter_func: Optional[callable] = None
    ) -> List[RetrievalResult]:
        query_embedding = self.encode_queries([query])[0]
        
        datastore_results = datastore.search(query_embedding, k=k, filter_func=filter_func)
        
        results = []
        for passage, score in datastore_results:
            paper = datastore.papers.get(passage.paper_id)
            if paper:
                result = RetrievalResult(
                    passage_id=passage.passage_id,
                    text=passage.text,
                    score=score,
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    paper_year=paper.year,
                    paper_citations=paper.citations
                )
                results.append(result)
                
        if self.use_semantic_scholar:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ss_papers = loop.run_until_complete(self.search_semantic_scholar(query))
            loop.close()
            
            for paper in ss_papers[:10]:
                if paper.get("abstract"):
                    result = RetrievalResult(
                        passage_id=f"ss_{paper['paperId']}",
                        text=f"{paper['title']}\n\n{paper['abstract']}",
                        score=0.0,
                        paper_id=paper["paperId"],
                        paper_title=paper["title"],
                        paper_year=paper.get("year", 0),
                        paper_citations=paper.get("citationCount", 0)
                    )
                    results.append(result)
                    
        return results


class OpenScholarReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = None,
        max_length: int = 512,
        batch_size: int = 16
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        
        logger.info(f"Loading reranker model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 50,
        max_passages_per_paper: int = 3,
        use_citation_boost: bool = True
    ) -> List[RetrievalResult]:
        if not results:
            return []
            
        scores = []
        
        for i in range(0, len(results), self.batch_size):
            batch_results = results[i:i + self.batch_size]
            batch_scores = self._score_batch(query, batch_results)
            scores.extend(batch_scores)
            
        for i, (result, score) in enumerate(zip(results, scores)):
            if use_citation_boost and result.paper_citations > 0:
                citation_boost = np.log(1 + result.paper_citations) / 10
                score = score + citation_boost
                
            scores[i] = score
            
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        final_results = []
        paper_counts = {}
        
        for result, score in scored_results:
            paper_id = result.paper_id
            
            if paper_id not in paper_counts:
                paper_counts[paper_id] = 0
                
            if paper_counts[paper_id] < max_passages_per_paper:
                result.score = score
                final_results.append(result)
                paper_counts[paper_id] += 1
                
            if len(final_results) >= top_k:
                break
                
        return final_results
        
    def _score_batch(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[float]:
        pairs = [[query, result.text] for result in results]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
            
        return scores.tolist()