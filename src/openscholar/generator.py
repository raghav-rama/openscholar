import torch
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    COVERAGE = "coverage"
    ORGANIZATION = "organization"
    CITATION = "citation"
    CLARITY = "clarity"
    RETRIEVAL = "retrieval"


@dataclass
class Feedback:
    feedback_type: FeedbackType
    description: str
    retrieval_query: Optional[str] = None


@dataclass
class GenerationOutput:
    text: str
    citations: List[Dict[str, Any]]
    feedback: List[Feedback]
    iteration: int


class OpenScholarGenerator:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = None,
        max_length: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_iterations: int = 3,
        use_flash_attention: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.max_iterations = max_iterations
        
        logger.info(f"Loading generator model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None
        }
        
        if use_flash_attention and self.device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
    def generate_initial_response(
        self,
        query: str,
        retrieved_passages: List[Dict],
        max_new_tokens: int = 1024
    ) -> GenerationOutput:
        prompt = self._build_initial_prompt(query, retrieved_passages)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        text, citations = self._parse_response(response, retrieved_passages)
        
        return GenerationOutput(
            text=text,
            citations=citations,
            feedback=[],
            iteration=0
        )
        
    def generate_feedback(
        self,
        query: str,
        current_response: GenerationOutput,
        retrieved_passages: List[Dict]
    ) -> List[Feedback]:
        prompt = self._build_feedback_prompt(query, current_response, retrieved_passages)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - 512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        feedback_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return self._parse_feedback(feedback_text)
        
    def incorporate_feedback(
        self,
        query: str,
        current_response: GenerationOutput,
        feedback: Feedback,
        retrieved_passages: List[Dict],
        additional_passages: Optional[List[Dict]] = None
    ) -> GenerationOutput:
        all_passages = retrieved_passages
        if additional_passages:
            all_passages = retrieved_passages + additional_passages
            
        prompt = self._build_refinement_prompt(
            query, current_response, feedback, all_passages
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - 1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        text, citations = self._parse_response(response, all_passages)
        
        return GenerationOutput(
            text=text,
            citations=citations,
            feedback=current_response.feedback + [feedback],
            iteration=current_response.iteration + 1
        )
        
    def verify_citations(
        self,
        response: GenerationOutput,
        retrieved_passages: List[Dict]
    ) -> GenerationOutput:
        prompt = self._build_citation_verification_prompt(response, retrieved_passages)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - 512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        verification_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        updated_text = self._apply_citation_fixes(response.text, verification_text, retrieved_passages)
        _, updated_citations = self._parse_response(updated_text, retrieved_passages)
        
        return GenerationOutput(
            text=updated_text,
            citations=updated_citations,
            feedback=response.feedback,
            iteration=response.iteration
        )
        
    def _build_initial_prompt(self, query: str, passages: List[Dict]) -> str:
        passage_texts = []
        for i, p in enumerate(passages):
            passage_texts.append(f"[{i+1}] {p['text'][:500]}...")
            
        prompt = f"""You are a scientific literature review assistant. Answer the following query by synthesizing information from the provided passages. Include inline citations using [1], [2], etc.

Query: {query}

Passages:
{chr(10).join(passage_texts)}

Instructions:
1. Provide a comprehensive answer addressing the query
2. Use inline citations [n] to reference specific passages
3. Ensure all scientific claims are supported by citations
4. Organize your response in a clear, logical manner

Response:"""
        return prompt
        
    def _build_feedback_prompt(
        self,
        query: str,
        response: GenerationOutput,
        passages: List[Dict]
    ) -> str:
        prompt = f"""Review the following response and provide up to 3 specific feedback points for improvement.

Query: {query}

Current Response:
{response.text}

Available Passages: {len(passages)}

Provide feedback in the following format:
1. [TYPE: coverage/organization/citation/clarity/retrieval] Description of improvement needed
   QUERY: (if retrieval needed) Additional search query

Feedback:"""
        return prompt
        
    def _build_refinement_prompt(
        self,
        query: str,
        response: GenerationOutput,
        feedback: Feedback,
        passages: List[Dict]
    ) -> str:
        passage_texts = []
        for i, p in enumerate(passages):
            passage_texts.append(f"[{i+1}] {p['text'][:300]}...")
            
        prompt = f"""Improve the following response based on the feedback provided.

Query: {query}

Current Response:
{response.text}

Feedback:
{feedback.feedback_type.value}: {feedback.description}

Available Passages:
{chr(10).join(passage_texts)}

Instructions:
1. Address the specific feedback while maintaining existing quality
2. Ensure proper inline citations [n]
3. Keep the response comprehensive and well-organized

Improved Response:"""
        return prompt
        
    def _build_citation_verification_prompt(
        self,
        response: GenerationOutput,
        passages: List[Dict]
    ) -> str:
        prompt = f"""Verify that all scientific claims in the following response have proper citations.

Response:
{response.text}

Available Passages: {len(passages)}

For each claim that needs a citation, provide:
CLAIM: "exact text needing citation"
CITATION: [n]

Verification:"""
        return prompt
        
    def _parse_response(
        self,
        response_text: str,
        passages: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        citation_pattern = r'\[(\d+)\]'
        citations_found = re.findall(citation_pattern, response_text)
        
        citations = []
        for cit_num in set(citations_found):
            idx = int(cit_num) - 1
            if 0 <= idx < len(passages):
                citation = {
                    "number": int(cit_num),
                    "passage_id": passages[idx].get("passage_id", f"passage_{idx}"),
                    "text": passages[idx]["text"],
                    "paper_title": passages[idx].get("paper_title", "Unknown"),
                    "paper_id": passages[idx].get("paper_id", "unknown")
                }
                citations.append(citation)
                
        return response_text.strip(), citations
        
    def _parse_feedback(self, feedback_text: str) -> List[Feedback]:
        feedback_list = []
        
        lines = feedback_text.strip().split('\n')
        current_feedback = None
        
        for line in lines:
            type_match = re.match(r'\d+\.\s*\[TYPE:\s*(\w+)\]\s*(.+)', line, re.IGNORECASE)
            if type_match:
                feedback_type_str = type_match.group(1).lower()
                description = type_match.group(2).strip()
                
                try:
                    feedback_type = FeedbackType(feedback_type_str)
                except ValueError:
                    feedback_type = FeedbackType.COVERAGE
                    
                current_feedback = Feedback(
                    feedback_type=feedback_type,
                    description=description
                )
                feedback_list.append(current_feedback)
                
            elif line.strip().startswith("QUERY:") and current_feedback:
                query = line.replace("QUERY:", "").strip()
                current_feedback.retrieval_query = query
                
        return feedback_list[:3]
        
    def _apply_citation_fixes(
        self,
        text: str,
        verification_output: str,
        passages: List[Dict]
    ) -> str:
        lines = verification_output.strip().split('\n')
        
        updated_text = text
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                claim_line = lines[i]
                citation_line = lines[i + 1]
                
                if claim_line.startswith("CLAIM:"):
                    claim = claim_line.replace("CLAIM:", "").strip().strip('"')
                    
                    if citation_line.startswith("CITATION:"):
                        citation = citation_line.replace("CITATION:", "").strip()
                        
                        if claim in updated_text and citation not in updated_text[updated_text.find(claim):]:
                            updated_text = updated_text.replace(claim, f"{claim} {citation}")
                            
        return updated_text