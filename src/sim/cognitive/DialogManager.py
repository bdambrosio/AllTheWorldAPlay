from datetime import datetime
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set
#from sim.agh import Character
from sim.memory.core import MemoryEntry
import utils.xml_utils as xml

@dataclass
class Dialog:
    
    def __init__(self, owner, other_actor):
        """Tracks the context of an ongoing dialog"""
        self.actor = owner  # character modeling this dialog, an Agh
        self.target = other_actor    # other character in dialog
        self.start_time: float = field(default_factory=time.time)
        self.turn_count: int = 0
        self.fatigue: float = 0.0  # Add fatigue counter
        self.fatigue_threshold: float = 5.0  # Configurable threshold
        self.interrupted_task: Optional[str] = None  # Task that was interrupted by dialog
        self.active: bool = False
        self.transcript: List[str] = []
    
    def activate(self, source=None):
        """Initialize a new dialog context"""
        if self.active:
            # Already in a dialog - could add logic here to handle multiple conversations
            return
            
        # Store current task if being interrupted, but don't try to stack dialogs
        #  we store interrupted task, but don't push dialog task here, read only on character
        self.active = True
        self.turn_count = 0

    def add_turn(self, speaker, message):
        """Record a new turn in the conversation and increase fatigue"""
        "We know both actors in the dialog, speaker better be one of them"
        "speaker is an Agh object, we need the name"
        self.turn_count += 1
        self.fatigue += 1.0  # Basic increment
        # Could be modified based on turn length, emotion, etc.
        if self.actor is not self.target:
            self.transcript.append(f"{speaker.name} says: {message}")
        else:
            self.transcript.append(f"...{message}...")

    def is_fatigued(self) -> bool:
        """Check if conversation has become fatiguing"""
        return self.fatigue >= self.fatigue_threshold
        
    def duration(self) -> float:
        """Get how long dialog has been active"""
        return time.time() - self.start_time
            
    def get_transcript(self, max_turns=10):
        return '\n'.join(self.transcript[-max_turns:])
        
    def get_current_dialog(self, max_turns=10):
        return '\n'.join(self.transcript[-min(max_turns, self.turn_count):])
    
    def deactivate_dialog(self):
        """End current dialog with fatigue consideration"""
        self.active = False
        # Store fatigue level for future interactions
        self.actor.memory_consolidator.update_cognitive_model(
            memory=self.actor.structured_memory,
            narrative=self.actor.narrative,
            knownActorManager=self.actor.actor_models,
            current_time=self.actor.context.simulation_time,
            character_desc=self.actor.character
        )
        
        # again, read only on actor, let actor manage focus task stack
