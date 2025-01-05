# memory/core.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""
    text: str
    timestamp: datetime
    importance: float  # 0-1 score
    confidence: float  # 0-1 score for classification confidence
    embedding: Optional[np.ndarray] = None
    related_memories: Set[int] = field(default_factory=set)
    memory_id: Optional[int] = None

class StructuredMemory:
    """Manages organized storage and retrieval of agent memories"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memories: List[MemoryEntry] = []
        self.memory_buffer: List[MemoryEntry] = []  # Recent memories pending consolidation
        self._next_id = 0
    
    def add_entry(self, entry: MemoryEntry) -> int:
        """Add new memory entry and compute its embedding"""
        if entry.embedding is None:
            entry.embedding = self.embedding_model.encode(entry.text)
        
        entry.memory_id = self._next_id
        self._next_id += 1
        
        self.memory_buffer.append(entry)
        self.memories.append(entry)
        return entry.memory_id
    
    def get_all(self) -> List[MemoryEntry]:
        """Get all memories"""
        return self.memories
        
    def get_recent(self, n: int) -> List[MemoryEntry]:
        """Get n most recent memories"""
        return sorted(self.memories, key=lambda x: x.timestamp, reverse=True)[:n]
    
    def get_by_id(self, memory_id: int) -> Optional[MemoryEntry]:
        """Get specific memory by ID"""
        for memory in self.memories:
            if memory.memory_id == memory_id:
                return memory
        return None
    
    def link_memories(self, id1: int, id2: int) -> bool:
        """Create bidirectional link between memories"""
        mem1 = self.get_by_id(id1)
        mem2 = self.get_by_id(id2)
        if mem1 and mem2:
            mem1.related_memories.add(id2)
            mem2.related_memories.add(id1)
            return True
        return False
    
    def get_by_drive(self, drive: str, limit: int = None) -> List[MemoryEntry]:
        """Get memories related to a specific drive"""
        # For now, simple text matching - could be enhanced with embeddings
        drive_memories = []
        for memory in self.get_all():
            if drive.lower() in memory.text.lower():
                drive_memories.append(memory)
                
        # Sort by importance and recency
        drive_memories.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        
        if limit:
            return drive_memories[:limit]
        return drive_memories