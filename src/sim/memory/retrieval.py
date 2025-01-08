from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
from .core import MemoryEntry, AbstractMemory, StructuredMemory

class MemoryRetrieval:
    """Handles semantic memory retrieval across concrete and abstract memories"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_relevant_memories(self,
                            memory: StructuredMemory,
                            query: str,
                            include_concrete: bool = True,
                            include_abstract: bool = True,
                            threshold: float = 0.387,
                            max_results: int = 10) -> Dict[str, List[Union[MemoryEntry, AbstractMemory]]]:
        """
        Find memories relevant to query across both concrete and abstract memories
        Returns dict with 'concrete' and 'abstract' keys
        """
        query_embedding = self.embedding_model.encode(query)
        results = {'concrete': [], 'abstract': []}
        
        if include_concrete:
            concrete_memories = memory.get_all()
            for mem in concrete_memories:
                if mem.embedding is None:
                    mem.embedding = self.embedding_model.encode(mem.text)
                similarity = self._compute_similarity(query_embedding, mem.embedding)
                if similarity >= threshold:
                    results['concrete'].append((mem, similarity))
                    
        if include_abstract:
            abstract_memories = memory.get_recent_abstractions() + \
                             ([memory.get_active_abstraction()] if memory.get_active_abstraction() else [])
            for mem in abstract_memories:
                if mem.embedding is None:
                    mem.embedding = self.embedding_model.encode(mem.summary)
                similarity = self._compute_similarity(query_embedding, mem.embedding)
                if similarity >= threshold:
                    results['abstract'].append((mem, similarity))
        
        # Sort by similarity and limit results
        results['concrete'] = [m[0] for m in sorted(results['concrete'], 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:max_results]]
        results['abstract'] = [m[0] for m in sorted(results['abstract'], 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:max_results]]
        return results
    
    def find_related_abstractions(self,
                                memory: StructuredMemory,
                                concrete_memory: MemoryEntry,
                                threshold: float = 0.387) -> List[AbstractMemory]:
        """Find abstract memories related to a concrete memory"""
        if concrete_memory.embedding is None:
            concrete_memory.embedding = self.embedding_model.encode(concrete_memory.text)
            
        related = []
        abstractions = memory.get_recent_abstractions() + \
                      ([memory.get_active_abstraction()] if memory.get_active_abstraction() else [])
        
        for abstract in abstractions:
            if abstract.embedding is None:
                abstract.embedding = self.embedding_model.encode(abstract.summary)
            similarity = self._compute_similarity(concrete_memory.embedding, abstract.embedding)
            if similarity >= threshold:
                related.append(abstract)
                
        return related
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )