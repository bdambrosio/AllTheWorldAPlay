# memory/core.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.agh import Character, Goal, Task  # Only imported during type checking
    from sim.cognitive.driveSignal import Drive
    from sim.cognitive.EmotionalStance import EmotionalStance
    from sim.cognitive.driveSignal import SignalCluster

# At module level
try:
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
except Exception as e:
    print(f"Warning: Could not load embedding model locally: {e}")
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    
    def to_string(self) -> str:
        """Convert memory entry to string representation"""
        return self.text
    
    def __eq__(self, other):
        """Equality based on unique memory_id"""
        if not isinstance(other, MemoryEntry):
            return False
        return self.memory_id == other.memory_id
    
    def __hash__(self):
        """Hash based on unique memory_id"""
        return hash(self.memory_id)

@dataclass 
class AbstractMemory:
    """Represents a period of related activity"""
    summary: str  # Description of the activity
    start_time: datetime
    end_time: Optional[datetime] = None  # None if still active
    instances: List[int] = field(default_factory=list)  # Memory IDs of concrete instances
    drive: Optional[Drive] = None  # Related drive if any
    embedding: Optional[np.ndarray] = None
    memory_id: Optional[int] = None
    is_active: bool = True
    importance: float = 1.0  # Added importance field for abstractions
    
    @property
    def timestamp(self) -> datetime:
        """Return end_time if available, otherwise start_time"""
        return self.end_time if self.end_time is not None else self.start_time
    
    def to_string(self) -> str:
        """Convert abstract memory to string representation"""
        return self.summary
    
    def __eq__(self, other):
        """Equality based on unique memory_id"""
        if not isinstance(other, AbstractMemory):
            return False
        return self.memory_id == other.memory_id
    
    def __hash__(self):
        """Hash based on unique memory_id"""
        return hash(self.memory_id)

# MemoryMix class removed - now using consistent List[Union[MemoryEntry, AbstractMemory]] return type

class StructuredMemory:
    """Manages organized storage and retrieval of agent memories"""
    
    def __init__(self, owner=None):
        self.embedding_model = _embedding_model
        self.concrete_memories: List[MemoryEntry] = []
        self.abstract_memories: List[AbstractMemory] = []
        self._next_id = 0
        self.pending_cleanup = []  # Track memories ready for cleanup
        self.owner = owner  # Reference to owning agent
        self.abstraction_threshold = 0.33  # Similarity threshold for adding to existing abstractions
        self.cleanup_age_hours = 24  # Age threshold for cleanup
        
    def add_entry(self, entry: MemoryEntry) -> int:
        """Add new concrete memory and update abstractions"""
        if entry.embedding is None:
            entry.embedding = self.embedding_model.encode(entry.text)
        
        entry.memory_id = self._next_id
        self._next_id += 1
        
        self.concrete_memories.append(entry)
        
        # Look for existing abstraction to add to
        matching_abstraction = self._find_matching_abstraction(entry)
        
        if matching_abstraction:
            # Add to existing abstraction
            matching_abstraction.instances.append(entry.memory_id)
            matching_abstraction.end_time = entry.timestamp  # Update end time
            matching_abstraction.summary = self._update_summary(matching_abstraction)
            # Update abstraction embedding
            matching_abstraction.embedding = self.embedding_model.encode(matching_abstraction.summary)
        else:
            # Create new singleton abstraction
            new_abstraction = AbstractMemory(
                summary=entry.text,
                start_time=entry.timestamp,
                instances=[entry.memory_id],
                drive=self._find_drives_in_memory(entry.text),
                importance=entry.importance
            )
            new_abstraction.memory_id = self._next_id
            new_abstraction.embedding = entry.embedding.copy()
            self._next_id += 1
            self.abstract_memories.append(new_abstraction)
        
        # Perform cleanup of old memories
        self._cleanup_old_memories()
        
        return entry.memory_id
    
    def _find_matching_abstraction(self, entry: MemoryEntry) -> Optional[AbstractMemory]:
        """Find existing abstraction that matches the new memory"""
        if not self.abstract_memories:
            return None
            
        best_match = None
        best_similarity = 0
        
        for abstraction in self.abstract_memories:
            if abstraction.embedding is None:
                abstraction.embedding = self.embedding_model.encode(abstraction.summary)
            
            similarity = self._compute_similarity(entry.embedding, abstraction.embedding)
            
            # Check drive compatibility
            entry_drives = self._find_drives_in_memory(entry.text)
            drive_compatible = (not entry_drives and not abstraction.drive) or \
                             (entry_drives and abstraction.drive in entry_drives)
            
            if similarity >= self.abstraction_threshold and drive_compatible and similarity > best_similarity:
                best_similarity = similarity
                best_match = abstraction
        
        return best_match
    
    def _cleanup_old_memories(self):
        """Remove old concrete memories that are part of abstractions, but preserve high-importance ones"""
        if not self.owner or not hasattr(self.owner, 'context'):
            return
            
        if self.owner.context is None or self.owner.context.simulation_time is None: # too early, skip
            return
        current_time = self.owner.context.simulation_time
        cutoff_time = current_time - timedelta(hours=self.cleanup_age_hours)
        
        # Find memories that are old and part of abstractions
        memories_to_remove = []
        abstracted_memory_ids = set()
        
        # Collect all memory IDs that are part of abstractions
        for abstraction in self.abstract_memories:
            abstracted_memory_ids.update(abstraction.instances)
        
        for memory in self.concrete_memories:
            # Only remove if: old AND part of abstraction AND not high importance
            if (memory.timestamp < cutoff_time and 
                memory.memory_id in abstracted_memory_ids and
                memory.importance <= 0.67):
                memories_to_remove.append(memory)
        
        # Remove old memories
        for memory in memories_to_remove:
            self.concrete_memories.remove(memory)
    
    def get_all(self) -> List[Union[MemoryEntry, AbstractMemory]]:
        """Get all memories (concrete and abstract)"""
        return self.concrete_memories + self.abstract_memories
        
    def get_recent(self, n: int) -> List[Union[MemoryEntry, AbstractMemory]]:
        """Get n most recent memories - returns both concrete and abstract"""
        # Combine and sort all memories by timestamp
        all_memories = []
        
        # Add concrete memories with their timestamps
        for mem in self.concrete_memories:
            all_memories.append((mem, mem.timestamp))
        
        # Add abstract memories with their start times
        for mem in self.abstract_memories:
            all_memories.append((mem, mem.start_time))
        
        # Sort by timestamp and return the most recent n
        all_memories.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in all_memories[:n]]
    
    def get_recent_concrete(self, n: int) -> List[MemoryEntry]:
        """Get n most recent concrete memories only (for backward compatibility)"""
        return sorted(self.concrete_memories, key=lambda x: x.timestamp, reverse=True)[:n]
    
    def get_by_id(self, memory_id: int) -> Optional[Union[MemoryEntry, AbstractMemory]]:
        """Get specific memory by ID (searches both concrete and abstract)"""
        # Search concrete memories first
        for memory in self.concrete_memories:
            if memory.memory_id == memory_id:
                return memory
        
        # Search abstract memories
        for memory in self.abstract_memories:
            if memory.memory_id == memory_id:
                return memory
        
        return None
        
    def link_memories(self, id1: int, id2: int) -> bool:
        """Create bidirectional link between memories"""
        mem1 = self.get_by_id(id1)
        mem2 = self.get_by_id(id2)
        if mem1 and mem2:
            # Only concrete memories have related_memories field
            if isinstance(mem1, MemoryEntry):
                mem1.related_memories.add(id2)
            if isinstance(mem2, MemoryEntry):
                mem2.related_memories.add(id1)
            return True
        return False
    
    def get_abstractions(self, count=None) -> List[AbstractMemory]:
        """Get abstract pattern memories (kept for backward compatibility)"""
        memories = sorted(self.abstract_memories,
                        key=lambda x: x.importance,
                        reverse=True)
        return memories[:count] if count else memories

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
        
        # For now, create a simple summary from recent instances
        # Could be enhanced with LLM-based summarization
        if len(concrete_instances) <= 2:
            return concrete_instances[-1]
        else:
            return f"Pattern of {len(concrete_instances)} related activities: {concrete_instances[-1]}"

    def get_recent_abstractions(self, n: int = 5) -> List[AbstractMemory]:
        """Get n most recent abstract memories (kept for backward compatibility)"""
        memories = sorted(self.abstract_memories, key=lambda x: x.start_time, reverse=True)
        return memories[:n]

    def get_active_abstraction(self) -> Optional[AbstractMemory]:
        """Get most recent abstraction (for backward compatibility)"""
        if self.abstract_memories:
            return sorted(self.abstract_memories, key=lambda x: x.start_time, reverse=True)[0]
        return None

    def get_abstractions_in_timeframe(self, 
                                    start: datetime, 
                                    end: Optional[datetime] = None) -> List[AbstractMemory]:
        """Get abstract memories that overlap with given timeframe (kept for backward compatibility)"""
        end = end or datetime.now()
        return [
            mem for mem in self.abstract_memories
            if mem.start_time <= end and (mem.end_time is None or mem.start_time >= start)
        ]

    def get_abstractions_by_drive(self, drive: str, limit: Optional[int] = None) -> List[AbstractMemory]:
        """Get abstract memories related to a specific drive (kept for backward compatibility)"""
        drive_memories = [
            mem for mem in self.abstract_memories
            if mem.drive == drive
        ]
        drive_memories.sort(key=lambda x: x.start_time, reverse=True)
        return drive_memories[:limit] if limit else drive_memories

    def get_by_criteria(self, 
                       drive: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       min_importance: Optional[float] = None,
                       memory_type: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Union[MemoryEntry, AbstractMemory]]:
        """Flexible memory retrieval with multiple criteria"""
        results = []
        
        # Filter concrete memories
        if memory_type != 'abstract':
            for mem in self.concrete_memories:
                if drive and hasattr(mem, 'drive') and mem.drive != drive:
                    continue
                if start_time and mem.timestamp < start_time:
                    continue
                if end_time and mem.timestamp > end_time:
                    continue
                if min_importance and mem.importance < min_importance:
                    continue
                results.append(mem)
        
        # Filter abstract memories
        if memory_type != 'concrete':
            for mem in self.abstract_memories:
                if drive and mem.drive != drive:
                    continue
                if start_time and mem.start_time < start_time:
                    continue
                if end_time and mem.start_time > end_time:
                    continue
                if min_importance and mem.importance < min_importance:
                    continue
                results.append(mem)
        
        # Sort by relevance (importance * recency)
        current_time = datetime.now()
        if self.owner and hasattr(self.owner, 'context'):
            current_time = self.owner.context.simulation_time
        
        def score_memory(mem):
            if isinstance(mem, MemoryEntry):
                age_hours = (current_time - mem.timestamp).total_seconds() / 3600
            else:
                age_hours = (current_time - mem.start_time).total_seconds() / 3600
            age_factor = max(0.1, 1.0 - (age_hours / 48))
            return mem.importance * age_factor
        
        results.sort(key=score_memory, reverse=True)
        return results[:limit] if limit else results

    def _find_drives_in_memory(self, text: str) -> Optional[str]:
        """Helper to find drives mentioned in memory text"""
        # This would need access to the agent's drives
        # For now, return None - will need to be connected to agent's drives
        return None

class MemoryRetrieval:
    """Handles semantic memory retrieval across concrete and abstract memories"""
    
    def __init__(self):
        self.embedding_model = _embedding_model  # Use shared model
    
    def _retrieve_by_embedding(self, 
                             memory: StructuredMemory,
                             search_embedding: np.ndarray,
                             threshold: float = 0.1,
                             max_results: int = 3) -> List[MemoryEntry]:
        """Core retrieval method using embedding similarity for concrete memories"""
        related_memories = []
        current_time = memory.owner.context.simulation_time if memory.owner and hasattr(memory.owner, 'context') else datetime.now()
        
        for mem in memory.concrete_memories:
            if mem.embedding is None:
                mem.embedding = self.embedding_model.encode(mem.text)
                
            similarity = self._compute_similarity(search_embedding, mem.embedding)
            if similarity >= threshold:
                age_hours = (current_time - mem.timestamp).total_seconds() / 3600
                age_factor = max(0.5, 1.0 - (age_hours / 24))
                score = similarity * mem.importance * age_factor
                related_memories.append((mem, score))
                
        sorted_memories = sorted(related_memories, key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in sorted_memories[:max_results]]
    
    def _retrieve_abstractions_by_embedding(self, 
                                           memory: StructuredMemory,
                                           search_embedding: np.ndarray,
                                           threshold: float = 0.1,
                                           max_results: int = 3) -> List[AbstractMemory]:
        """Core retrieval method using embedding similarity for abstract memories"""
        related_abstractions = []
        current_time = memory.owner.context.simulation_time if memory.owner and hasattr(memory.owner, 'context') else datetime.now()
        
        for abstract in memory.abstract_memories:
            if abstract.embedding is None:
                abstract.embedding = self.embedding_model.encode(abstract.summary)
                
            similarity = self._compute_similarity(search_embedding, abstract.embedding)
            if similarity >= threshold:
                age_hours = (current_time - abstract.start_time).total_seconds() / 3600
                age_factor = max(0.5, 1.0 - (age_hours / 48))  # Abstractions decay slower
                score = similarity * abstract.importance * age_factor
                related_abstractions.append((abstract, score))
                
        sorted_abstractions = sorted(related_abstractions, key=lambda x: x[1], reverse=True)
        return [abstract for abstract, _ in sorted_abstractions[:max_results]]

    def _retrieve_mixed_by_embedding(self, 
                                   memory: StructuredMemory,
                                   search_embedding: np.ndarray,
                                   threshold: float = 0.1,
                                   max_results: int = 6) -> List[Union[MemoryEntry, AbstractMemory]]:
        """Core retrieval method that returns both concrete and abstract memories"""
        # Get both types
        concrete = self._retrieve_by_embedding(memory, search_embedding, threshold, max_results//2)
        abstract = self._retrieve_abstractions_by_embedding(memory, search_embedding, threshold, max_results//2)
        
        # Combine and sort by relevance
        all_results = []
        current_time = memory.owner.context.simulation_time if memory.owner and hasattr(memory.owner, 'context') else datetime.now()
        
        for mem in concrete:
            age_hours = (current_time - mem.timestamp).total_seconds() / 3600
            age_factor = max(0.5, 1.0 - (age_hours / 24))
            similarity = self._compute_similarity(search_embedding, mem.embedding)
            score = similarity * mem.importance * age_factor
            all_results.append((mem, score))
        
        for mem in abstract:
            age_hours = (current_time - mem.start_time).total_seconds() / 3600
            age_factor = max(0.5, 1.0 - (age_hours / 48))
            similarity = self._compute_similarity(search_embedding, mem.embedding)
            score = similarity * mem.importance * age_factor
            all_results.append((mem, score))
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in all_results[:max_results]]

    def get_by_drive(self, 
                    memory: StructuredMemory, 
                    drive: Drive, 
                    threshold: float = 0.1,
                    max_results: int = 6) -> List[Union[MemoryEntry, AbstractMemory]]:
        """Get both concrete and abstract memories related to drive using embeddings"""
        if drive.embedding is None:
            drive.embedding = self.embedding_model.encode(drive.text)
        return self._retrieve_mixed_by_embedding(memory, drive.embedding, threshold, max_results)
    
    def get_concrete_by_drive(self, 
                             memory: StructuredMemory, 
                             drive: Drive, 
                             threshold: float = 0.1,
                             max_results: int = 3) -> List[MemoryEntry]:
        """Get concrete memories related to drive using embeddings (for backward compatibility)"""
        if drive.embedding is None:
            drive.embedding = self.embedding_model.encode(drive.text)
        return self._retrieve_by_embedding(memory, drive.embedding, threshold, max_results)

    def get_by_text(self,
                   memory: StructuredMemory,
                   search_text: str,
                   threshold: float = 0.1, 
                   max_results: int = 6) -> List[Union[MemoryEntry, AbstractMemory]]:
        """Get both concrete and abstract memories related to search text using embeddings"""
        search_embedding = self.embedding_model.encode(search_text)
        return self._retrieve_mixed_by_embedding(memory, search_embedding, threshold, max_results)
    
    def get_concrete_by_text(self,
                            memory: StructuredMemory,
                            search_text: str,
                            threshold: float = 0.1, 
                            max_results: int = 3) -> List[MemoryEntry]:
        """Get concrete memories related to search text using embeddings (for backward compatibility)"""
        search_embedding = self.embedding_model.encode(search_text)
        return self._retrieve_by_embedding(memory, search_embedding, threshold, max_results)

    def find_related_abstractions(self,
                                memory: StructuredMemory,
                                concrete_memory: MemoryEntry,
                                threshold: float = 0.387) -> List[AbstractMemory]:
        """Find abstract memories related to a concrete memory (kept for backward compatibility)"""
        if concrete_memory.embedding is None:
            concrete_memory.embedding = self.embedding_model.encode(concrete_memory.text)
            
        return self._retrieve_abstractions_by_embedding(memory, concrete_memory.embedding, threshold)
        
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

@dataclass
class NarrativeSummary:
    """Maintains a coherent narrative of an agent's experiences"""
    
    # Core narrative components
    recent_events: str  # Last few hours in detail
    ongoing_activities: str  # Current goals and activities
    
    # Metadata
    last_update: datetime  # Last narrative update time
    update_interval: timedelta = field(default_factory=lambda: timedelta(hours=0.5))
    
    # Supporting information
    focus_signalClusters: List[SignalCluster] = field(default_factory=list) # history of focusSignalClusters driving the narrative
    active_drives: List[Drive] = field(default_factory=list)  # Current motivating drives
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
                                    
        return "\n\n".join(sections)
    
    def get_summary(self, scope: str = None) -> str:
        """Get narrative summary of specified length (short/medium/long)"""
        return self.get_full_narrative()
        

    
