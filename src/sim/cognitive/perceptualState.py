from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import hash_utils
from src.utils.Messages import UserMessage

# At module level
try:
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
except Exception as e:
    print(f"Warning: Could not load embedding model locally: {e}")
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class SensoryMode(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory" 
    MOVEMENT = "movement"
    INTERNAL = "internal"
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
    
    def __init__(self, owner = None):
        self.owner = owner  # Reference to owning agent
        self.current_inputs: Dict[SensoryMode, List[PerceptualInput]] = {
            mode: [] for mode in SensoryMode
        }
        self.attention_focus: Optional[SensoryMode] = None
        self.attention_threshold = 0.0  # Minimum intensity to notice
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.information_items: List[InformationItem] = []
        
    def add_input(self, sensory_input: PerceptualInput) -> None:
        """Add new sensory input and manage attention"""
        # Only process inputs above attention threshold
        if sensory_input.intensity >= self.attention_threshold:
            self.current_inputs[sensory_input.mode].append(sensory_input)
            
            # Update attention focus for high intensity inputs
            if (self.attention_focus is None and 
                sensory_input.intensity > 0.7):
                self.attention_focus = sensory_input.mode
                
            # Add to agent's memory if significant
            if self.owner:
                self.owner.add_to_history(
                    f"{sensory_input.mode.value}: {sensory_input.content}"
                )
        return 
    # self.extract_information(sensory_input)
    
    def record_information_items_from_look(self, percept_hash: List[str], resources: List[str], characters: List[str]) -> List[InformationItem]:
        """Extract information items from perceptual hash - currently only resources and characters"""
        items = []
        for resource_name in resources:
            item = InformationItem('resource', resource_name, 'permanent', self.owner.context.simulation_time)
            item.embedding = self.embedding_model.encode(item.content)
            self.information_items.append(item)
        for character_name in characters:
            item = InformationItem('character', character_name, 'permanent', self.owner.context.simulation_time)
            item.embedding = self.embedding_model.encode(item.content)
            self.information_items.append(item)
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
    
    def clear_old_percepts(self, max_age_seconds: float = 300) -> None:
        """Remove percepts older than specified age"""
        current_time = self.owner.context.simulation_time
        for mode in SensoryMode:
            self.current_inputs[mode] = [
                p for p in self.current_inputs[mode]
                if (current_time - p.timestamp).total_seconds() <= max_age_seconds
            ]
            
    def shift_attention(self, mode: Optional[SensoryMode] = None) -> None:
        """Explicitly shift attention to specified mode or let attention flow naturally"""
        if mode:
            self.attention_focus = mode
        else:
            # Find mode with highest intensity recent input
            highest_intensity = 0
            for mode in SensoryMode:
                recent = self.get_current_percepts(mode)
                if recent and recent[0].intensity > highest_intensity:
                    highest_intensity = recent[0].intensity
                    self.attention_focus = mode
                    
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

    # everything below is unused junk, afaik


    def integrate_with_memory(self) -> None:
        """Integrate significant percepts into agent's memory system"""
        if not self.owner:
            return
            
        # Get all current percepts above memory threshold
        significant = [
            p for p in self.get_current_percepts()
            if p.intensity > 0.5
        ]
        
        # Group related percepts
        grouped = {}
        for percept in significant:
            if percept.source in grouped:
                grouped[percept.source].append(percept)
            else:
                grouped[percept.source] = [percept]
                
        # Create consolidated memories for groups
        for source, percepts in grouped.items():
            if len(percepts) > 1:
                # Combine related percepts into single memory
                combined = f"You perceive from {source}: " + ", ".join(
                    f"({p.mode.value}) {p.content}"
                    for p in sorted(percepts, key=lambda x: x.intensity, reverse=True)
                )
                self.owner.add_to_history(combined)
            else:
                # Single percept gets added individually
                p = percepts[0]
                self.owner.add_to_history(
                    f"You perceive from {p.source} ({p.mode.value}): {p.content}"
                )

    def recent_significant_change(self, 
                            time_window: float = 5.0,
                            intensity_threshold: float = 0.7) -> bool:
        """
        Check if there have been any significant perceptual changes recently.
    
        Args:
            time_window: Number of seconds to look back
            intensity_threshold: Minimum intensity to consider significant
        
        Returns:
            bool: True if significant changes detected
        """
        current_time = self.owner.context.simulation_time
        window_start = current_time - timedelta(seconds=time_window)
    
        # Check each sensory mode
        for mode in SensoryMode:
            recent_inputs = [
                input for input in self.current_inputs[mode]
                if input.timestamp >= window_start
            ]
        
            # Look for high intensity inputs
            for input in recent_inputs:
                if input.intensity >= intensity_threshold:
                    return True
                
            # Check for pattern changes
            if len(recent_inputs) >= 2:
                # Get average intensity before and after midpoint
                midpoint = len(recent_inputs) // 2
                early_avg = sum(i.intensity for i in recent_inputs[:midpoint]) / midpoint
                late_avg = sum(i.intensity for i in recent_inputs[midpoint:]) / (len(recent_inputs) - midpoint)
            
                # Check if intensity pattern changed significantly 
                if abs(late_avg - early_avg) >= intensity_threshold/2:
                    return True
    
        return False

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )