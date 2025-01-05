from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.Messages import UserMessage
from .core import MemoryEntry, StructuredMemory

class MemoryRetrieval:
    """Handles memory retrieval using semantic similarity"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_by_drive(self, 
                     memory: StructuredMemory, 
                     drive: str, 
                     threshold: float = 0.5,
                     max_memories: int = 10) -> List[MemoryEntry]:
        """
        Get memories relevant to any phrase in a drive
        
        Args:
            memory: StructuredMemory instance
            drive: Full drive text containing one or more phrases
            threshold: Minimum similarity score to include
            max_memories: Maximum memories to return
        """
        # Split drive into constituent phrases
        phrases = [p.strip() for p in drive.split('.')]
        
        # Get embeddings for all phrases
        phrase_embeddings = self.embedding_model.encode(phrases)
        
        # Calculate similarities between each memory and each phrase
        similarities = []
        for mem_entry in memory.get_all():
            if mem_entry.embedding is None:
                mem_entry.embedding = self.embedding_model.encode(mem_entry.text)
                
            # Get max similarity across all phrases in drive
            max_sim = max(
                np.dot(mem_entry.embedding, phrase_emb) / 
                (np.linalg.norm(mem_entry.embedding) * np.linalg.norm(phrase_emb))
                for phrase_emb in phrase_embeddings
            )
            
            if max_sim >= threshold:
                similarities.append((mem_entry, max_sim))
        
        # Sort by similarity and return top memories
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in similarities[:max_memories]]
    
    def get_by_context(self,
                       memory: StructuredMemory,
                       context: str,
                       threshold: float = 0.5,
                       max_memories: int = 10) -> List[MemoryEntry]:
        """
        Get memories relevant to a specific context/query
        
        Args:
            memory: StructuredMemory instance
            context: Query text to match against
            threshold: Minimum similarity score to include
            max_memories: Maximum memories to return
        """
        context_embedding = self.embedding_model.encode(context)
        
        similarities = []
        for mem_entry in memory.get_all():
            if mem_entry.embedding is None:
                mem_entry.embedding = self.embedding_model.encode(mem_entry.text)
                
            sim = np.dot(mem_entry.embedding, context_embedding) / (
                np.linalg.norm(mem_entry.embedding) * np.linalg.norm(context_embedding)
            )
            
            if sim >= threshold:
                similarities.append((mem_entry, sim))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in similarities[:max_memories]]
    
    def get_memories_by_priority(self,
                                memory: StructuredMemory,
                                drives: List[str],
                                max_per_drive: int = 5) -> Dict[str, List[MemoryEntry]]:
        """
        Get memories organized by drive priority
        
        Args:
            memory: StructuredMemory instance
            drives: List of drives in priority order
            max_per_drive: Maximum memories to return per drive
            
        Returns:
            Dict mapping drive text to list of relevant memories
        """
        result = {}
        for drive in drives:
            drive_memories = self.get_by_drive(memory, drive, max_memories=max_per_drive)
            result[drive] = drive_memories
        return result
    
    def get_recent_relevant(self,
                           memory: StructuredMemory,
                           query: str,
                           time_window: timedelta = timedelta(hours=24),
                           threshold: float = 0.5,
                           max_memories: int = 10) -> List[MemoryEntry]:
        """
        Get recent memories relevant to a query
        
        Args:
            memory: StructuredMemory instance
            query: Query text to match against
            time_window: How far back to look
            threshold: Minimum similarity score
            max_memories: Maximum memories to return
        """
        cutoff_time = datetime.now() - time_window
        recent_memories = [m for m in memory.get_all() if m.timestamp >= cutoff_time]
        
        if not recent_memories:
            return []
            
        query_embedding = self.embedding_model.encode(query)
        
        similarities = []
        for mem_entry in recent_memories:
            if mem_entry.embedding is None:
                mem_entry.embedding = self.embedding_model.encode(mem_entry.text)
                
            sim = np.dot(mem_entry.embedding, query_embedding) / (
                np.linalg.norm(mem_entry.embedding) * np.linalg.norm(query_embedding)
            )
            
            if sim >= threshold:
                similarities.append((mem_entry, sim))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in similarities[:max_memories]]