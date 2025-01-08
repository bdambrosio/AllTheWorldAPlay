# memory/core.py
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class MemoryEntry:
    """Single concrete memory entry with metadata"""
    text: str
    timestamp: datetime  # Now represents simulation time
    importance: float  # 0-1 score
    confidence: float  # 0-1 score for classification confidence
    embedding: Optional[np.ndarray] = None
    related_memories: Set[int] = field(default_factory=set)
    memory_id: Optional[int] = None

@dataclass 
class AbstractMemory:
    """Represents a period of related activity"""
    summary: str  # Description of the activity
    start_time: datetime
    end_time: Optional[datetime] = None  # None if still active
    instances: List[int] = field(default_factory=list)  # Memory IDs of concrete instances
    drive: Optional[str] = None  # Related drive if any
    embedding: Optional[np.ndarray] = None
    memory_id: Optional[int] = None
    is_active: bool = True

class StructuredMemory:
    """Manages organized storage and retrieval of agent memories"""
    
    def __init__(self, owner=None):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.concrete_memories: List[MemoryEntry] = []
        self.abstract_memories: List[AbstractMemory] = []
        self.current_abstract: Optional[AbstractMemory] = None
        self._next_id = 0
        self.pending_cleanup = []  # Track memories ready for cleanup
        self.owner = owner  # Reference to owning agent
        
    def add_entry(self, entry: MemoryEntry) -> int:
        """Add new concrete memory and update abstractions"""
        if entry.embedding is None:
            entry.embedding = self.embedding_model.encode(entry.text)
        
        entry.memory_id = self._next_id
        self._next_id += 1
        
        self.concrete_memories.append(entry)
        
        # Handle abstract memory updates
        self._update_abstractions(entry)
        
        return entry.memory_id
    
    def get_all(self) -> List[MemoryEntry]:
        """Get all memories"""
        return self.concrete_memories
        
    def get_recent(self, n: int) -> List[MemoryEntry]:
        """Get n most recent memories"""
        return sorted(self.concrete_memories, key=lambda x: x.timestamp, reverse=True)[:n]
    
    def get_by_id(self, memory_id: int) -> Optional[MemoryEntry]:
        """Get specific memory by ID"""
        for memory in self.concrete_memories:
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
    
    def get_abstractions(self, count=None):
        """Get abstract pattern memories"""
        memories = sorted(self.abstract_memories,
                        key=lambda x: x.importance,
                        reverse=True)
        return memories[:count] if count else memories
    
    def _update_abstractions(self, new_memory: MemoryEntry):
        """Update abstract memories based on new concrete memory"""
        
        # Find related drives for the new memory
        drives = self._find_drives_in_memory(new_memory.text)
        
        # If no current abstract memory, create one
        if not self.current_abstract:
            self.current_abstract = AbstractMemory(
                summary=new_memory.text,
                start_time=new_memory.timestamp,
                instances=[new_memory.memory_id],
                drive=drives[0] if drives else None
            )
            self.current_abstract.memory_id = self._next_id
            self._next_id += 1
            return

        # Check if new memory fits current abstraction
        similarity = self._compute_similarity(
            new_memory.embedding,
            self.current_abstract.embedding if self.current_abstract.embedding else 
                self.embedding_model.encode(self.current_abstract.summary)
        )
        
        # Additional drive-based check
        same_drive = (not drives and not self.current_abstract.drive) or \
                     (drives and self.current_abstract.drive in drives)
        
        if similarity >= 0.33 and same_drive:  # Memory fits both semantically and drive-wise
            # Add to current abstraction
            self.current_abstract.instances.append(new_memory.memory_id)
            # Update summary (could be more sophisticated)
            self.current_abstract.summary = self._update_summary(self.current_abstract)
        else:
            # Close current abstraction
            self.current_abstract.is_active = False
            self.current_abstract.end_time = new_memory.timestamp
            
            # Only keep it if it has enough instances
            if len(self.current_abstract.instances) >= 3:
                self.abstract_memories.append(self.current_abstract)
                # Mark concrete memories for cleanup
                self.pending_cleanup.extend(self.current_abstract.instances)
            
            # Start new abstraction
            self.current_abstract = AbstractMemory(
                summary=new_memory.text,
                start_time=new_memory.timestamp,
                instances=[new_memory.memory_id],
                drive=drives[0] if drives else None
            )
            self.current_abstract.memory_id = self._next_id
            self._next_id += 1

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def _update_summary(self, abstract: AbstractMemory) -> str:
        """Update abstract memory summary based on instances"""
        concrete_instances = [
            self.get_by_id(mid).text 
            for mid in abstract.instances
            if self.get_by_id(mid)
        ]
        
        if not concrete_instances:
            return abstract.summary
        
        # For now, just use the most recent instance's text
        # Could be enhanced with LLM-based summarization
        return concrete_instances[-1]

    def get_recent_abstractions(self, n: int = 5) -> List[AbstractMemory]:
        """Get n most recent abstract memories, including current if active"""
        memories = sorted(
            self.abstract_memories + ([self.current_abstract] if self.current_abstract else []),
            key=lambda x: x.start_time,
            reverse=True
        )
        return memories[:n]

    def get_active_abstraction(self) -> Optional[AbstractMemory]:
        """Get current ongoing abstract memory if any"""
        return self.current_abstract

    def get_abstractions_in_timeframe(self, 
                                    start: datetime, 
                                    end: Optional[datetime] = None) -> List[AbstractMemory]:
        """Get abstract memories that overlap with given timeframe"""
        end = end or datetime.now()
        return [
            mem for mem in self.abstract_memories + ([self.current_abstract] if self.current_abstract else [])
            if mem.start_time <= end and (mem.end_time is None or mem.start_time >= start)
        ]

    def get_abstractions_by_drive(self, drive: str, limit: Optional[int] = None) -> List[AbstractMemory]:
        """Get abstract memories related to a specific drive"""
        drive_memories = [
            mem for mem in self.abstract_memories + ([self.current_abstract] if self.current_abstract else [])
            if mem.drive == drive
        ]
        drive_memories.sort(key=lambda x: x.start_time, reverse=True)
        return drive_memories[:limit] if limit else drive_memories

    def _find_drives_in_memory(self, text: str) -> List[str]:
        """Helper to find drives mentioned in memory text"""
        # This would need access to the agent's drives
        # For now, return empty list - will need to be connected to agent's drives
        return []

@dataclass
class NarrativeSummary:
    """Maintains a coherent narrative of an agent's experiences"""
    
    # Core narrative components
    recent_events: str  # Last few hours in detail
    ongoing_activities: str  # Current goals and activities
    background: str  # Key past events and relationships
    
    # Metadata
    last_update: datetime  # Last narrative update time
    update_interval: timedelta = field(default_factory=lambda: timedelta(hours=4))
    
    # Supporting information
    key_relationships: Dict[str, str] = field(default_factory=dict)  # name: relationship description
    active_drives: List[str] = field(default_factory=list)  # Current motivating drives
    key_events: List[Dict] = field(default_factory=list)  # Important memories that shaped character
    
    def needs_update(self, current_time: datetime) -> bool:
        """Check if narrative needs updating based on simulation time"""
        return (current_time - self.last_update) >= self.update_interval
    
    def get_full_narrative(self) -> str:
        """Get complete narrative summary"""
        sections = []
        
        if self.recent_events:
            sections.append("Recent Events:\n" + self.recent_events)
            
        if self.ongoing_activities:
            sections.append("Current Activities:\n" + self.ongoing_activities)
            
        if self.background:
            sections.append("Background:\n" + self.background)
            
        if self.key_relationships:
            rel_text = "\n".join(f"- {name}: {desc}" for name, desc 
                               in self.key_relationships.items())
            sections.append("Key Relationships:\n" + rel_text)
            
        return "\n\n".join(sections)
    
    def get_summary(self, length: str = 'medium') -> str:
        """Get narrative summary of specified length (short/medium/long)"""
        if length == 'short':
            # Just recent events and current activities
            return "\n\n".join([self.recent_events, self.ongoing_activities])
        elif length == 'medium':
            # Include key relationships
            rel_text = "\n".join(f"- {name}: {desc}" for name, desc 
                               in self.key_relationships.items())
            return "\n\n".join([
                self.recent_events,
                self.ongoing_activities,
                "Relationships:\n" + rel_text
            ])
        else:
            # Full narrative
            return self.get_full_narrative()
    
    def initialize_relationships(self, valid_characters: List[str]):
        """Initialize relationships with valid character names"""
        # Keep only relationships with valid characters
        self.key_relationships = {
            name: desc for name, desc in self.key_relationships.items()
            if name in valid_characters
        }