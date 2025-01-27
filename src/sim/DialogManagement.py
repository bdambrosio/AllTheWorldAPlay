import time
from dataclasses import dataclass, field
from typing import Optional, Set
import utils.xml_utils as xml

@dataclass
class DialogContext:
    """Tracks the context of an ongoing dialog"""
    initiator: str  # Name of agent who started dialog
    target: str    # Name of agent being spoken to
    start_time: float = field(default_factory=time.time)
    turn_count: int = 0
    fatigue: float = 0.0  # Add fatigue counter
    fatigue_threshold: float = 5.0  # Configurable threshold
    interrupted_task: Optional[str] = None  # Task that was interrupted by dialog
    participants: Set[str] = field(default_factory=set)
    
    def add_turn(self):
        """Record a new turn in the conversation and increase fatigue"""
        self.turn_count += 1
        self.fatigue += 1.0  # Basic increment
        # Could be modified based on turn length, emotion, etc.
        
    def is_fatigued(self) -> bool:
        """Check if conversation has become fatiguing"""
        return self.fatigue >= self.fatigue_threshold
        
    def duration(self) -> float:
        """Get how long dialog has been active"""
        return time.time() - self.start_time

class DialogManager:
    """Manages dialog state and interrupted tasks"""
    def __init__(self, agent):
        self.agent = agent
        self.current_dialog: Optional[DialogContext] = None
        
    def start_dialog(self, from_actor, to_actor, source='dialog'):
        """Initialize a new dialog context"""
        if self.current_dialog is not None:
            # Already in a dialog - could add logic here to handle multiple conversations
            return
            
        # Store current task if being interrupted
        interrupted_task = None
        if self.agent.active_task.peek() not in (None, 'dialog', source):
            interrupted_task = self.agent.active_task.peek()
            
        self.current_dialog = DialogContext(
            initiator=from_actor.name,
            target=to_actor.name,
            interrupted_task=interrupted_task
        )
        self.current_dialog.participants.add(from_actor.name)
        self.current_dialog.participants.add(to_actor.name)
        
        # Push dialog task
        self.agent.active_task.push('dialog')
        
    def add_turn(self):
        """Record a new turn in the current dialog"""
        if self.current_dialog:
            self.current_dialog.add_turn()
            
    def end_dialog(self):
        """End current dialog with fatigue consideration"""
        if not self.current_dialog:
            return
            
        # Store fatigue level for future interactions
        if self.current_dialog.target:
            self.agent.structured_memory.add_memory(
                f"Conversation with {self.current_dialog.target} ended with fatigue level {self.current_dialog.fatigue}"
            )
            
        interrupted_task = self.current_dialog.interrupted_task
        self.current_dialog = None
        
        if self.agent.active_task.peek() == 'dialog':
            self.agent.active_task.pop()
            
        if interrupted_task and self._is_task_still_relevant(interrupted_task):
            self.agent.active_task.push(interrupted_task)
            
    def _is_task_still_relevant(self, task_name: str) -> bool:
        """Check if a task is still relevant and should be resumed"""
        # Look for task in current priorities
        for task in self.agent.priorities:
            if xml.find('<Name>', task) == task_name:
                # Could add additional checks here (effectiveness, state changes, etc)
                return True
        return False

# Modified hear() method in Agh class
def hear(self, from_actor, message, source='dialog', respond=True):
    """Handle incoming messages with dialog context tracking"""
    if source == 'dialog':
        # Initialize or update dialog context
        if not hasattr(self, 'dialog_manager'):
            self.dialog_manager = DialogManager(self)
            
        if self.dialog_manager.current_dialog is None:
            self.dialog_manager.start_dialog(from_actor, self, source)
        
        self.dialog_manager.add_turn()
        
        # Basic dialog length check - can be made more sophisticated later
        if self.dialog_manager.current_dialog.turn_count > 1 and not self.always_respond:
            self.dialog_manager.end_dialog()
            return
            
    # Rest of existing hear() logic...
    if respond:
        response = self.generate_response(from_actor, message, source)
        if response:
            self.intentions.append(response)
            
def tell(self, to_actor, message, source='dialog', respond=True):
    """Initiate or continue dialog with dialog context tracking"""
    if source == 'dialog':
        if not hasattr(self, 'dialog_manager'):
            self.dialog_manager = DialogManager(self)
            
        if self.dialog_manager.current_dialog is None:
            self.dialog_manager.start_dialog(self, to_actor, source)
            
        self.dialog_manager.add_turn()
        
    self.acts(to_actor, 'Say', message, '', source)