from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import hash_utils
from utils.Messages import UserMessage

# At module level
from utils.embedding_model import get_embedding_model
_embedding_model = get_embedding_model()

class SensoryMode(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory" 
    MOVEMENT = "movement"
    INTERNAL = "internal"
    THOUGHT = "thought"   # the character's own guesses and reasoning: remembered, but not events (drives.appraise)
    UNCLASSIFIED = "unclassified"

@dataclass
class InformationItem:
    item: str
    content: str
    permanence: str
    timestamp: datetime

    @staticmethod
    def from_hash(hash_item: str, timestamp: datetime) -> Optional["InformationItem"]:
        item = hash_utils.find('item', hash_item)
        content = hash_utils.find('content', hash_item)
        permanence = hash_utils.find('permanence', hash_item)
        if item and content and permanence:
            return InformationItem(item, content, permanence, timestamp)
        return None
    
@dataclass
class PerceptualInput:
    """Individual sensory input with metadata"""
    mode: SensoryMode
    content: str
    timestamp: datetime
    intensity: float = 1.0  # 0-1 scale
    confidence: float = 1.0
    source: Optional[str] = None

class PerceptualState:
    """Manages current perceptual state including sensory inputs and attention"""
    MAX_PER_MODE = 20
    
    def __init__(self, owner = None):
        self.owner = owner  # Reference to owning agent
        self.current_inputs: Dict[SensoryMode, List[PerceptualInput]] = {
            mode: [] for mode in SensoryMode
        }
        self.attention_focus: Optional[SensoryMode] = None
        self.attention_threshold = 0.0  # Minimum intensity to notice
        self.embedding_model = _embedding_model  # shared; was a fresh load per character
        self.information_items: List[InformationItem] = []
        
    def add_input(self, sensory_input: PerceptualInput, remember: bool = True) -> None:
        """Add new sensory input and manage attention"""
        # Only process inputs above attention threshold
        if sensory_input.intensity >= self.attention_threshold:
            inputs = self.current_inputs[sensory_input.mode]
            inputs.append(sensory_input)
            # 'current' percepts: keep a recent window per mode (this list was never pruned;
            # the full history lives in structured memory)
            del inputs[:-self.MAX_PER_MODE]
            
            # Update attention focus for high intensity inputs
            if (self.attention_focus is None and 
                sensory_input.intensity > 0.7):
                self.attention_focus = sensory_input.mode
                
            # Add to agent's memory if significant
            if self.owner and remember:
                self.owner.add_to_history(
                    f"{sensory_input.mode.value}: {sensory_input.content}"
                )
        return 
    # self.extract_information(sensory_input)
    
    def record_information_items_from_look(self, percept_hash: List[str], resources: List[str], characters: List[str]) -> List[InformationItem]:
        """Record the resources and characters in view. One item per (kind, name): a repeat
        sighting refreshes its timestamp (every look used to append and re-embed them all)."""
        now = self.owner.context.simulation_time
        known = {(i.item, i.content): i for i in self.information_items}
        items = []
        for kind, names in (('resource', resources), ('character', characters)):
            for name in names:
                item = known.get((kind, name))
                if item is None:
                    item = InformationItem(kind, name, 'permanent', now)
                    item.embedding = self.embedding_model.encode(item.content)
                    self.information_items.append(item)
                    known[(kind, name)] = item
                else:
                    item.timestamp = now
                items.append(item)
        return items

    def get_information_items(self, search_text: str, threshold: float = 0.6, max_results: int = 5) -> List[InformationItem]:
        """Get information items related to search text using embeddings"""
        if not search_text or not self.information_items:
            return []
            
        search_embedding = self.embedding_model.encode(search_text)
        current_time = self.owner.context.simulation_time
        decay_rate = 0.0001  # Adjust this value to control how quickly older items decay
        
        # Calculate time-weighted similarity scores
        scored_items = []
        for item in self.information_items:
            if item.embedding is None:
                item.embedding = self.embedding_model.encode(item.content)
            
            # Calculate age in seconds
            age_seconds = (current_time - item.timestamp).total_seconds()
            
            # Calculate base similarity
            similarity = self._compute_similarity(search_embedding, item.embedding)
            
            # Apply temporal decay
            time_weight = np.exp(-decay_rate * age_seconds)
            final_score = similarity * time_weight
            
            if final_score >= threshold:
                scored_items.append((final_score, item))
        
        # Sort by final score and return top results
        scored_items.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored_items[:max_results]]
        return results

    def get_current_percepts(self, mode: Optional[SensoryMode] = None, chronological: bool = False) -> List[PerceptualInput]:
        """Get current percepts, optionally filtered by mode"""
        if mode:
            return self.current_inputs[mode]
        
        all_percepts = []
        for mode_inputs in self.current_inputs.values():
            all_percepts.extend(mode_inputs)
        if chronological:
            return sorted(all_percepts, key=lambda x: x.timestamp, reverse=True)
        else:
            return all_percepts
        
    
    
    def get_focused_percepts(self) -> List[PerceptualInput]:
        """Get percepts for current attention focus"""
        if self.attention_focus:
            return self.current_inputs[self.attention_focus]
        return []
    
    def get_state_summary(self) -> str:
        """Generate summary of current perceptual state"""
        summary_parts = []
        
        for mode in SensoryMode:
            percepts = self.get_current_percepts(mode)
            if percepts:
                mode_summary = f"{mode.value}: "
                mode_summary += ", ".join(
                    f"{p.content} ({p.intensity:.1f})" 
                    for p in sorted(percepts, 
                                  key=lambda x: x.intensity, 
                                  reverse=True)[:3]
                )
                summary_parts.append(mode_summary)
                
        if self.attention_focus:
            summary_parts.append(f"Attention: {self.attention_focus.value}")
            
        return "\n".join(summary_parts)

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )