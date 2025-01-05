# sim/cognitive/state.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from utils.Messages import UserMessage
from sim.memory.core import MemoryEntry, StructuredMemory
import utils.xml_utils as xml

@dataclass
class StateAssessment:
    """Single drive state assessment"""
    term: str           # Short term describing the state
    value: str         # Assessment value (high, medium-high, etc)
    trigger: str       # What triggered this state
    completion_check: str  # How to check if state is satisfied

class StateSystem:
    """Handles drive state assessment"""
    
    def __init__(self, llm, character):
        self.llm = llm
        self.character = character

    def generate_state(self,
                      drives: List[str],
                      situation: str,
                      memory: StructuredMemory) -> Dict[str, Dict[str, str]]:
        """Generate state assessment for all drives"""
        state = {}
        for drive in drives:
            # Get relevant memories for this drive
            drive_memories = memory.get_by_drive(drive, limit=3)
            assessment = self._assess_drive(drive, situation, drive_memories)
            if assessment:
                state[assessment.term] = {
                    "drive": drive,
                    "state": assessment.value,
                    "trigger": assessment.trigger,
                    "termination_check": assessment.completion_check
                }
        return state
    
    def _assess_drive(self, 
                     drive: str, 
                     situation: str, 
                     drive_memories: List[MemoryEntry]) -> Optional[StateAssessment]:
        """Assess current state for a drive using LLM"""
        
        # Format memories for LLM input
        formatted_memories = "\n".join([
            f"- [{mem.timestamp}] (importance: {mem.importance:.1f}): {mem.text}"
            for mem in drive_memories
        ])
        
        prompt = [UserMessage(content="""Analyze the current state of a basic Drive, given the Character, Situation, and relevant Memories below.

<Character>
{{$character}}
</Character>

<Drive>
{{$drive}}
</Drive>

<Situation>
{{$situation}}
</Situation>

<Memories>
{{$memories}}
</Memories>

Respond using this XML format:
<Term>brief term describing state</Term>
<Assessment>high|medium-high|medium|medium-low|low</Assessment>
<Trigger>what triggered this state</Trigger>
<Termination>how to check if state is resolved</Termination>
""")]

        variables = {
            "character": self.character,
            "drive": drive,
            "situation": situation,
            "memories": formatted_memories
        }

        response = self.llm.ask(variables, prompt, temp=0.1)
        
        try:
            term = xml.find('<Term>', response)
            assessment = xml.find('<Assessment>', response)
            trigger = xml.find('<Trigger>', response)
            termination = xml.find('<Termination>', response)
            
            if term and assessment and trigger and termination:
                return StateAssessment(
                    term=term.strip(),
                    value=assessment.strip().lower(),
                    trigger=trigger.strip(),
                    completion_check=termination.strip()
                )
        except Exception as e:
            print(f"Error parsing state assessment: {str(e)}")
            
        return None
    