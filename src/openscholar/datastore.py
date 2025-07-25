import os
import json
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from tqdm import tqdm
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    venue: str
    citations: int
    url: str
    full_text: Optional[str] = None
    

@dataclass
class Passage:
    passage_id: str
    paper_id: str
    text: str
    section: str
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None


class OpenScholarDataStore:
    def __init__(
        self,
        data_dir: str = "data/openscholar_datastore",
        index_type: str = "HNSW",
        embedding_dim: int = 768,
        chunk_size: int = 250,
        overlap: int = 50
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.papers = {}
        self.passages = {}
        self.index = None
        self.passage_ids = []
        
        self._init_index()
        
    def _init_index(self):
        if self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
    def add_paper(self, paper: Paper) -> List[str]:
        self.papers[paper.paper_id] = paper
        
        if paper.full_text:
            passages = self._split_into_passages(paper)
            passage_ids = []
            
            for passage in passages:
                self.passages[passage.passage_id] = passage
                passage_ids.append(passage.passage_id)
                
            return passage_ids
        return []
        
    def _split_into_passages(self, paper: Paper) -> List[Passage]:
        passages = []
        text = paper.full_text
        words = text.split()
        
        start_idx = 0
        passage_num = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            passage_text = " ".join(words[start_idx:end_idx])
            
            passage_text = f"{paper.title}\n\n{passage_text}"
            
            start_char = len(" ".join(words[:start_idx])) if start_idx > 0 else 0
            end_char = len(" ".join(words[:end_idx]))
            
            passage = Passage(
                passage_id=f"{paper.paper_id}_p{passage_num}",
                paper_id=paper.paper_id,
                text=passage_text,
                section="main",
                start_char=start_char,
                end_char=end_char
            )
            
            passages.append(passage)
            start_idx = end_idx - self.overlap
            passage_num += 1
            
        return passages
        
    def add_embeddings(self, passage_ids: List[str], embeddings: np.ndarray):
        assert len(passage_ids) == embeddings.shape[0]
        
        for i, passage_id in enumerate(passage_ids):
            if passage_id in self.passages:
                self.passages[passage_id].embedding = embeddings[i]
                
        valid_indices = [i for i, pid in enumerate(passage_ids) if pid in self.passages]
        valid_embeddings = embeddings[valid_indices]
        valid_ids = [passage_ids[i] for i in valid_indices]
        
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(valid_embeddings)
            
        self.index.add(valid_embeddings)
        self.passage_ids.extend(valid_ids)
        
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[Passage, float]]:
        if len(self.passage_ids) == 0:
            return []
            
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            min(k * 2, len(self.passage_ids))
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.passage_ids):
                continue
                
            passage_id = self.passage_ids[idx]
            passage = self.passages[passage_id]
            
            if filter_func and not filter_func(passage):
                continue
                
            results.append((passage, float(dist)))
            
            if len(results) >= k:
                break
                
        return results
        
    def save(self, save_dir: Optional[str] = None):
        save_dir = Path(save_dir or self.data_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "papers.pkl", "wb") as f:
            pickle.dump(self.papers, f)
            
        passages_without_embeddings = {
            pid: Passage(
                passage_id=p.passage_id,
                paper_id=p.paper_id,
                text=p.text,
                section=p.section,
                start_char=p.start_char,
                end_char=p.end_char,
                embedding=None
            )
            for pid, p in self.passages.items()
        }
        
        with open(save_dir / "passages.pkl", "wb") as f:
            pickle.dump(passages_without_embeddings, f)
            
        with open(save_dir / "passage_ids.pkl", "wb") as f:
            pickle.dump(self.passage_ids, f)
            
        if len(self.passage_ids) > 0:
            faiss.write_index(self.index, str(save_dir / "index.faiss"))
            
        embeddings = []
        for pid in self.passage_ids:
            if self.passages[pid].embedding is not None:
                embeddings.append(self.passages[pid].embedding)
                
        if embeddings:
            np.save(save_dir / "embeddings.npy", np.array(embeddings))
            
        metadata = {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "num_papers": len(self.papers),
            "num_passages": len(self.passages)
        }
        
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved datastore to {save_dir}")
        
    def load(self, load_dir: Optional[str] = None):
        load_dir = Path(load_dir or self.data_dir)
        
        if not load_dir.exists():
            logger.warning(f"Load directory {load_dir} does not exist")
            return
            
        with open(load_dir / "papers.pkl", "rb") as f:
            self.papers = pickle.load(f)
            
        with open(load_dir / "passages.pkl", "rb") as f:
            self.passages = pickle.load(f)
            
        with open(load_dir / "passage_ids.pkl", "rb") as f:
            self.passage_ids = pickle.load(f)
            
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        self.index_type = metadata["index_type"]
        self.embedding_dim = metadata["embedding_dim"]
        self.chunk_size = metadata["chunk_size"]
        self.overlap = metadata["overlap"]
        
        index_path = load_dir / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            
        embeddings_path = load_dir / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            for i, pid in enumerate(self.passage_ids):
                if i < len(embeddings):
                    self.passages[pid].embedding = embeddings[i]
                    
        logger.info(f"Loaded datastore from {load_dir}")
        logger.info(f"Papers: {len(self.papers)}, Passages: {len(self.passages)}")
        
    def get_stats(self) -> Dict:
        return {
            "num_papers": len(self.papers),
            "num_passages": len(self.passages),
            "num_indexed_passages": len(self.passage_ids),
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap
        }