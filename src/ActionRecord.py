from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import time

class Mode(Enum):
    THINK = "Think"
    SAY = "Say"
    DO = "Do" 
    MOVE = "Move"
    LOOK = "Look"

@dataclass
class StateSnapshot:
    """Snapshot of relevant state values before/after an action"""
    values: Dict[str, str]  # State term -> assessment value
    timestamp: float

@dataclass 
class ActionRecord:
    """Record of a single action and its immediate results"""
    # Basic action info
    mode: Mode
    action_text: str
    source_task: str  # Task that generated this action
    timestamp: float
    
    # Context and results
    target: Optional[str]  # Target of action if any
    context_feedback: str  # Response from Context
    energy_used: float    # Rough measure of effort (higher for physical actions)
    
    # State tracking
    state_before: StateSnapshot
    state_after: StateSnapshot
    
    # Effectiveness tracking
    helped_states: List[str]  # State terms that improved
    hindered_states: List[str]  # State terms that got worse
    
    @property
    def effort_score(self) -> float:
        """Calculate standardized effort score"""
        base_effort = {
            Mode.THINK: 1.0,
            Mode.SAY: 1.0,
            Mode.LOOK: 1.5,
            Mode.MOVE: 2.0,
            Mode.DO: 3.0
        }
        return base_effort[self.mode] * self.energy_used

class ActionMemory:
    """Manages action history and analysis"""
    def __init__(self):
        self.records: List[ActionRecord] = []
        self.task_sequences: Dict[str, List[ActionRecord]] = {}  # Task -> sequence of actions
        
    def add_record(self, record: ActionRecord):
        """Add a new action record and update sequences"""
        self.records.append(record)
        if record.source_task not in self.task_sequences:
            self.task_sequences[record.source_task] = []
        self.task_sequences[record.source_task].append(record)

    def get_task_effectiveness(self, task_name: str, window: int = 5) -> float:
        """Calculate recent effectiveness score for a task
        Returns score from 0.0 (ineffective) to 1.0 (highly effective)"""
        if task_name not in self.task_sequences:
            return 1.0  # New tasks start optimistic
            
        recent_actions = self.task_sequences[task_name][-window:]
        if not recent_actions:
            return 1.0
            
        # Count improved states vs effort invested
        total_improvements = sum(len(record.helped_states) for record in recent_actions)
        total_effort = sum(record.effort_score for record in recent_actions)
        
        if total_effort == 0:
            return 1.0
            
        return min(1.0, total_improvements / total_effort)

    def is_action_repetitive(self, mode: Mode, action_text: str, 
                           task_name: str, lookback: int = 3) -> bool:
        """Check if this would be a repetitive action"""
        if task_name not in self.task_sequences:
            return False
            
        recent = self.task_sequences[task_name][-lookback:]
        similar_actions = [r for r in recent 
                         if r.mode == mode and
                         self._similar_text(r.action_text, action_text)]
        
        # If we've done very similar actions recently and they haven't helped much
        if similar_actions:
            effectiveness = self.get_task_effectiveness(task_name)
            return effectiveness < 0.3  # Threshold for "not helping enough"
        
        return False
        
    def _similar_text(self, text1: str, text2: str) -> bool:
        """Simple text similarity check - could be enhanced with NLP"""
        # For now just check if core verbs/nouns overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1.intersection(words2)) / len(words1.union(words2)) > 0.5
   
    def compare_states(self, before: StateSnapshot, after: StateSnapshot) -> tuple[List[str], List[str]]:
        """Compare two state snapshots and return lists of improved and worsened states"""
        helped_states = []
        hindered_states = []
        
        value_map = {
            'very low': 1, 'low': 2, 'medium-low': 3,
            'medium': 4, 'medium-high': 5, 'high': 6, 'very high': 7
        }
        
        for term, after_value in after.values.items():
            before_value = before.values.get(term)
            if before_value is None:
                continue
                
            try:
                before_val = value_map.get(before_value.lower(), 4)  # Default to medium
                after_val = value_map.get(after_value.lower(), 4)
                
                if after_val < before_val:  # Lower values are better
                    helped_states.append(term)
                elif after_val > before_val:
                    hindered_states.append(term)
            except AttributeError:
                continue
                
        return helped_states, hindered_states

    def update_record_states(self, record: ActionRecord):
        """Update record with state change analysis"""
        helped, hindered = self.compare_states(record.state_before, record.state_after)
        record.helped_states = helped
        record.hindered_states = hindered
        
def create_action_record(agent, mode: Mode, action_text: str, 
                        task_name: str, target=None) -> ActionRecord:
    """Helper to create an ActionRecord from current agent state"""
    state_snapshot = StateSnapshot(
        values={term: info["state"] 
                for term, info in agent.state.items()},
        timestamp=time.time()
    )
    
    # Base energy costs could be adjusted based on agent condition
    base_energy = {
        Mode.THINK: 1.0,
        Mode.SAY: 1.0, 
        Mode.LOOK: 1.5,
        Mode.MOVE: 2.0,
        Mode.DO: 3.0
    }
    
    return ActionRecord(
        mode=mode,
        action_text=action_text,
        source_task=task_name,
        timestamp=time.time(),
        target=target,
        context_feedback="",  # Will be filled in after action
        energy_used=base_energy[mode],
        state_before=state_snapshot,
        state_after=None,  # Will be filled in after action
        helped_states=[],
        hindered_states=[]
    )