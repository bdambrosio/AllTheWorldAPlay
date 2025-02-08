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
        self.actor = owner  # character modeling this dialog
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

    def add_turn(self, actor, message):
        """Record a new turn in the conversation and increase fatigue"""
        self.turn_count += 1
        self.fatigue += 1.0  # Basic increment
        # Could be modified based on turn length, emotion, etc.
        self.transcript.append(f"{actor.name}: {message}")

    def is_fatigued(self) -> bool:
        """Check if conversation has become fatiguing"""
        return self.fatigue >= self.fatigue_threshold
        
    def duration(self) -> float:
        """Get how long dialog has been active"""
        return time.time() - self.start_time
            
    def get_transcript(self):
        return self.transcript
        
    def deactivate_dialog(self):
        """End current dialog with fatigue consideration"""
        if not self.active:
            return
            
        # Store fatigue level for future interactions
        if self.target:
            entry = MemoryEntry(
                text=f"Conversation with {self.target.name} ended with fatigue level {self.fatigue}",
                importance=0.5,  # Default importance
                timestamp=datetime.now(),
                confidence=1.0
            )

            self.actor.structured_memory.add_entry(entry)

        self.actor.memory_consolidator.update_cognitive_model(
            memory=self.actor.structured_memory,
            narrative=self.actor.narrative,
            knownActorManager=self.actor.known_actors,
            current_time=self.actor.context.simulation_time,
            character_desc=self.actor.character
        )
        interrupted_task = self.interrupted_task
        if self.participants:

            for actor_name in self.participants:
                actor = self.actor.context.get_actor_by_name(actor_name)
                if actor and actor.dialog_manager:
                    if actor.dialog_manager.current_dialog:
                        actor.dialog_manager.current_dialog = None
                actor.intentions = []
        # a dialog with another character never ends! This is long term memory of verbal interactions with a character
        # self.current_dialog = None
        self.actor.intentions = []
        
        if self.actor.active_task.peek() == 'dialog':
            self.actor.active_task.pop()
           
        if self.interrupted_task and self._is_task_still_relevant(self.interrupted_task):
            # If the interrupted task is not the current task, push it onto the stack 
            # to resume it after the dialog ends
            if self.interrupted_task != self.actor.active_task.peek():
                self.actor.active_task.push(self.interrupted_task)   
        elif self.interrupted_task and self.actor.active_task.peek() == self.interrupted_task:
            # If the interrupted task is the current task, remove it since no longer relevant
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
