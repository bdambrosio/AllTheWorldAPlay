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
        self.active: bool = True
        self.transcript: List[str] = []
    
    def activate(self, source='dialog'):
        """Initialize a new dialog context"""
        if self.active:
            # Already in a dialog - could add logic here to handle multiple conversations
            return
            
        # Store current task if being interrupted
        self.turn_count = 0
        interrupted_task = None
        if self.actor.active_task.peek() not in (None, 'dialog', source):
            self.interrupted_task = self.actor.active_task.peek()
        # Push dialog task
        self.actor.active_task.push('dialog')

    def add_turn(self, speaker, message):
        """Record a new turn in the conversation and increase fatigue"""
        "We know both actors in the dialog, speaker better be one of them"
        "speaker is an Agh object, we need the name"
        self.turn_count += 1
        self.fatigue += 1.0  # Basic increment
        # Could be modified based on turn length, emotion, etc.
        self.transcript.append(f"{speaker.name} says: {message}")

    def is_fatigued(self) -> bool:
        """Check if conversation has become fatiguing"""
        return self.fatigue >= self.fatigue_threshold
        
    def duration(self) -> float:
        """Get how long dialog has been active"""
        return time.time() - self.start_time
            
    def get_transcript(self, max_turns=10):
        return '\n'.join(self.transcript[-max_turns:])
        
    def deactivate_dialog(self):
        """End current dialog with fatigue consideration"""
        self.active = False
        self.turn_count = 0
        self.fatigue = 0.0
        # Store fatigue level for future interactions
        self.actor.memory_consolidator.update_cognitive_model(
            memory=self.actor.structured_memory,
            narrative=self.actor.narrative,
            knownActorManager=self.actor.actor_models,
            current_time=self.actor.context.simulation_time,
            character_desc=self.actor.character
        )
        interrupted_task = self.interrupted_task
        self.actor.intentions = []
        
        if self.actor.active_task.peek() == 'dialog':
            self.actor.active_task.pop()
           
        if self.interrupted_task and self.interrupted_task != self.actor.active_task.peek():
            if self._is_task_still_relevant(self.interrupted_task):
                # If the interrupted task is not the current task, push it onto the stack 
                # to resume it after the dialog ends
                self.actor.active_task.push(self.interrupted_task)   
        elif self.interrupted_task and self.actor.active_task.peek() == self.interrupted_task:
            if not self._is_task_still_relevant(self.interrupted_task):
                # If the interrupted task is the current task, and no longer relevant, remove it
                self.actor.active_task.pop()
            
    def _is_task_still_relevant(self, task_name: str) -> bool:
        """Check if a task is still relevant and should be resumed"""
        # Look for task in current priorities
        for task in self.actor.priorities:
            if xml.find('<name>', task) == task_name:
                # Check if task is already satisfied using agent's satisfaction test
                termination_check = xml.find('<termination_check>', task)
                if termination_check:
                    satisfied = self.actor.test_priority_termination(
                        termination_check,
                        '',  # No new consequences to check
                        ''   # No world updates to check
                    )
                    if satisfied:
                        return False  # Task is complete, don't resume
                
                return True  # Task exists and isn't satisfied
            
        return False  # Task not found in current priorities
