from typing import Optional
from sim.cognitive.state import StateSystem
from sim.cognitive.priority import PrioritySystem

class CognitiveProcessor:
    def __init__(self, llm, character_description):
        self.llm = llm
        self.character_description = character_description
        self.state_system = StateSystem(llm, character_description)
        self.priority_system = PrioritySystem(llm, character_description)

    def process_cognitive_update(self, cognitive_state, recent_memories, current_situation, step):
        """Process cognitive updates including state and priority changes"""
        # Update state based on new memories and situation
        new_state = self.state_system.generate_state(
            recent_memories,
            current_situation
        )
        
        # Update priorities based on new state
        new_priorities = self.priority_system.update_priorities(
            new_state,
            recent_memories,
            current_situation
        )
        
        return CognitiveState(
            state=new_state,
            active_priorities=new_priorities
        )

class CognitiveState:
    """Container for cognitive state information"""
    def __init__(self, state=None, active_priorities=None):
        self.state = state if state else {}
        self.active_priorities = active_priorities if active_priorities else [] 