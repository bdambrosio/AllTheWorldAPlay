from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import hash_utils
from src.utils.Messages import UserMessage

# At module level
_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        self.extract_information(sensory_input)
    
    def extract_information(self, sensory_input: PerceptualInput) -> str:
        """Extract information from sensory input"""
        prompt = [UserMessage(content="""Extract the highest priority specific information items from the following sensory input:

{{$input}}

A specific information item is important if it is an actor or if it is relevant to current active signals.
Do NOT perform any inference, return only items that are directly stated in the input.
The current active signals are:

{{$signals}}


List found specific information items in priority order, highest first.
                              
A specific information item can be (following are examples, not exhaustive):
    1 a fact about the world or an item in it (e.g. 'these berries are poisonous')
    2 a description of the agent's own thoughts or feelings (e.g. 'John am hungry')
    3 a description of the actions the agent can or did take (e.g. 'Mary can eat the berries', 'I  did eat these berries')
    4 a change in actor inventory (e.g. 'John now has a map')
    5 a change in actor beliefs (e.g. 'John now knows that the berries are poisonous')
    6 a change in actor goals (e.g. 'John now wants to eat the berries')
    7 a change in another actor's beliefs (e.g. 'Mary now knows that the berries are poisonous')
    8 a change in another actor's goals (e.g. 'Mary now wants to eat the berries')
    9 a change in another actor's inventory (e.g. 'Mary now has a map')
    10 a change in another actor's actions (e.g. 'Mary is looking at the map')
    11 a change in another actor's thoughts or feelings (e.g. 'Mary is hungry')
    12 a change in the environment (e.g. 'the door is now open'), especially a change in proximity of other characters.

For each information item, provide the following information:
- item: the type of information item:
  - location: {{$name}} is at a location
  - goal: {{$name}} wants to do something
  - action: {{$name}} is doing something
  - inventory: {{$name}} has something or {{$name} no longer has something
  - knowledge: {{$name}} knows something
  - thought: {{$name}} is thinking about something
  - feeling: {{$name}} is feeling something
  - proximity: {{$name}} is near another actor or resource or is no longer near another actor or resource
- content: the content of the information item - the location, goal, action, inventory item, thought, feeling, or other actor or resource.
- permanence: how long the information item remains valid ('transient' or 'permanent'). Any information about direction, distance, or objects at current location is transient, since the actor can move. Similarly, time, temperature, and other changing conditions are transient.

Use the following hash-format for each information item.
Each item should begin with a #type tag, and should end with ## on a separate line as shown below:
be careful to insert line breaks only where shown, separating a value from the next tag:

#item fact / belief / goal / action / inventory / state
#content concise (max 10 words) description of the information item
#permanence transient / permanent
##

Respond only with the hash-formatted items as shown above.
Do not include any other introductory, explanatory, discursive, or formatting text.
End you response with
<end/>
"""
        )]
        ranked_signalClusters = self.owner.driveSignalManager.get_scored_clusters()
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order

        response = self.owner.llm.ask({"input": sensory_input.content, "name": self.owner.name, "signals": focus_signalClusters}, prompt, stops=["<end/>"], max_tokens=80)
        items = []
        hash_items = hash_utils.findall_forms(response)
        for hash_item in hash_items:
            item = InformationItem.from_hash(hash_item, sensory_input.timestamp)
            if item:
                # Add embedding to the item
                item.embedding = self.embedding_model.encode(item.content)
                self.information_items.append(item)
                if item.item == 'location':
                    pass
                    #self.owner.update_location(self.owner, item.content) #check for location change
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