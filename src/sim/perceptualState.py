from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import numpy as np

class SensoryMode(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory" 
    MOVEMENT = "movement"
    INTERNAL = "internal"

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
        self.attention_threshold = 0.3  # Minimum intensity to notice
        
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
            if self.owner and sensory_input.intensity > 0.5:
                self.owner.add_to_history(
                    f"You perceive ({sensory_input.mode.value}): {sensory_input.content}"
                )
    
    def get_current_percepts(self, mode: Optional[SensoryMode] = None) -> List[PerceptualInput]:
        """Get current percepts, optionally filtered by mode"""
        if mode:
            return self.current_inputs[mode]
        
        all_percepts = []
        for mode_inputs in self.current_inputs.values():
            all_percepts.extend(mode_inputs)
        return sorted(all_percepts, key=lambda x: x.timestamp, reverse=True)
    
    def get_focused_percepts(self) -> List[PerceptualInput]:
        """Get percepts for current attention focus"""
        if self.attention_focus:
            return self.current_inputs[self.attention_focus]
        return []
    
    def clear_old_percepts(self, max_age_seconds: float = 300) -> None:
        """Remove percepts older than specified age"""
        current_time = datetime.now()
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

