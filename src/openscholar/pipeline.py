import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .datastore import OpenScholarDataStore
from .retriever import OpenScholarRetriever, OpenScholarReranker, RetrievalResult
from .generator import OpenScholarGenerator, GenerationOutput, FeedbackType

logger = logging.getLogger(__name__)


@dataclass
class OpenScholarResponse:
    query: str
    response: str
    citations: List[Dict[str, Any]]
    iterations: int
    feedback_history: List[Dict[str, Any]]
    retrieved_passages: List[Dict[str, Any]]


class OpenScholarPipeline:
    def __init__(
        self,
        datastore: OpenScholarDataStore,
        retriever: Optional[OpenScholarRetriever] = None,
        reranker: Optional[OpenScholarReranker] = None,
        generator: Optional[OpenScholarGenerator] = None,
        initial_retrieval_k: int = 100,
        rerank_k: int = 50,
        final_k: int = 20,
        max_iterations: int = 3,
        enable_citation_verification: bool = True
    ):
        self.datastore = datastore
        self.retriever = retriever or OpenScholarRetriever()
        self.reranker = reranker or OpenScholarReranker()
        self.generator = generator or OpenScholarGenerator()
        
        self.initial_retrieval_k = initial_retrieval_k
        self.rerank_k = rerank_k
        self.final_k = final_k
        self.max_iterations = max_iterations
        self.enable_citation_verification = enable_citation_verification
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def __call__(self, query: str) -> OpenScholarResponse:
        return self.generate(query)
        
    def generate(self, query: str) -> OpenScholarResponse:
        logger.info(f"Processing query: {query[:100]}...")
        
        initial_results = self._retrieve_passages(query)
        
        reranked_results = self.reranker.rerank(
            query,
            initial_results,
            top_k=self.rerank_k
        )
        
        final_passages = self._prepare_passages_for_generation(
            reranked_results[:self.final_k]
        )
        
        current_output = self.generator.generate_initial_response(
            query,
            final_passages
        )
        
        all_retrieved_passages = final_passages.copy()
        feedback_history = []
        
        for iteration in range(self.max_iterations):
            feedback_list = self.generator.generate_feedback(
                query,
                current_output,
                all_retrieved_passages
            )
            
            if not feedback_list:
                logger.info(f"No feedback generated at iteration {iteration + 1}")
                break
                
            for feedback in feedback_list:
                feedback_dict = {
                    "iteration": iteration + 1,
                    "type": feedback.feedback_type.value,
                    "description": feedback.description,
                    "retrieval_query": feedback.retrieval_query
                }
                feedback_history.append(feedback_dict)
                
                additional_passages = None
                if feedback.feedback_type == FeedbackType.RETRIEVAL and feedback.retrieval_query:
                    logger.info(f"Performing additional retrieval: {feedback.retrieval_query}")
                    additional_results = self._retrieve_passages(
                        feedback.retrieval_query,
                        k=30
                    )
                    
                    reranked_additional = self.reranker.rerank(
                        feedback.retrieval_query,
                        additional_results,
                        top_k=10
                    )
                    
                    additional_passages = self._prepare_passages_for_generation(
                        reranked_additional
                    )
                    all_retrieved_passages.extend(additional_passages)
                    
                current_output = self.generator.incorporate_feedback(
                    query,
                    current_output,
                    feedback,
                    all_retrieved_passages,
                    additional_passages
                )
                
        if self.enable_citation_verification:
            logger.info("Verifying citations...")
            current_output = self.generator.verify_citations(
                current_output,
                all_retrieved_passages
            )
            
        return OpenScholarResponse(
            query=query,
            response=current_output.text,
            citations=current_output.citations,
            iterations=current_output.iteration + 1,
            feedback_history=feedback_history,
            retrieved_passages=all_retrieved_passages
        )
        
    def _retrieve_passages(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[RetrievalResult]:
        k = k or self.initial_retrieval_k
        
        return self.retriever.retrieve(
            query,
            self.datastore,
            k=k
        )
        
    def _prepare_passages_for_generation(
        self,
        results: List[RetrievalResult]
    ) -> List[Dict[str, Any]]:
        passages = []
        
        for result in results:
            passage = {
                "passage_id": result.passage_id,
                "text": result.text,
                "score": result.score,
                "paper_id": result.paper_id,
                "paper_title": result.paper_title,
                "paper_year": result.paper_year,
                "paper_citations": result.paper_citations
            }
            passages.append(passage)
            
        return passages
        
    async def generate_async(self, query: str) -> OpenScholarResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate, query)
        
    def batch_generate(
        self,
        queries: List[str],
        show_progress: bool = True
    ) -> List[OpenScholarResponse]:
        from tqdm import tqdm
        
        responses = []
        
        iterator = tqdm(queries, desc="Processing queries") if show_progress else queries
        
        for query in iterator:
            try:
                response = self.generate(query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing query '{query[:50]}...': {e}")
                responses.append(
                    OpenScholarResponse(
                        query=query,
                        response=f"Error: {str(e)}",
                        citations=[],
                        iterations=0,
                        feedback_history=[],
                        retrieved_passages=[]
                    )
                )
                
        return responses
        
    def save_response(
        self,
        response: OpenScholarResponse,
        output_path: str
    ):
        import json
        
        data = {
            "query": response.query,
            "response": response.response,
            "citations": response.citations,
            "iterations": response.iterations,
            "feedback_history": response.feedback_history,
            "num_passages": len(response.retrieved_passages),
            "passages": [
                {
                    "passage_id": p["passage_id"],
                    "paper_title": p["paper_title"],
                    "score": p["score"]
                }
                for p in response.retrieved_passages[:10]
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved response to {output_path}")