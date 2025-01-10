# sim/cognitive/state.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
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
    timestamp: datetime = None  # When this assessment was made

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
        
        # Get simulation time from memory's owner
        current_time = memory.owner.context.simulation_time
        
        # Get recent concrete memories and abstractions
        recent_memories = memory.get_recent(5)
        recent_abstracts = memory.get_recent_abstractions(3)
        
        # Format memory context
        memory_context = self._format_memory_context(recent_memories, recent_abstracts)
        
        for drive in drives:
            assessment = self._assess_drive(
                drive=drive,
                situation=situation,
                memory_context=memory_context,
                memory=memory
            )
            
            if assessment:
                state[assessment.term] = {
                    "drive": drive,
                    "state": assessment.value,
                    "trigger": assessment.trigger,
                    "termination_check": assessment.completion_check,
                    "timestamp": assessment.timestamp or current_time
                }
        return state

    def _format_memory_context(self, memories: List[MemoryEntry], abstracts: List) -> str:
        """Format memories and abstractions for LLM context"""
        context = []
        
        if memories:
            context.append("Recent Events:")
            for mem in memories:
                context.append(f"- [{mem.timestamp}] {mem.text}")
                
        if abstracts:
            context.append("\nRecent Patterns:")
            for abs in abstracts:
                context.append(f"- {abs.summary}")
                
        return "\n".join(context)

    def _assess_drive(self, 
                     drive: str,
                     situation: str,
                     memory_context: str,
                     memory: StructuredMemory) -> Optional[StateAssessment]:
        """Assess current state for a drive using LLM"""
        
        prompt = [UserMessage(content="""Analyze the current state of a basic Drive, given the Character, Situation, and Memory Context below.

<Character>
{{$character}}
</Character>

<Drive>
{{$drive}}
</Drive>

<Situation>
{{$situation}}
</Situation>

<MemoryContext>
{{$memory_context}}
</MemoryContext>

Respond using this XML format:
<Term>brief term describing state</Term>
<Assessment>high|medium-high|medium|medium-low|low</Assessment>
<Trigger>what triggered this state</Trigger>
<Termination>how to check if state is resolved</Termination>
                              
End your response with:
</End>
""")]

        variables = {
            "character": self.character,
            "drive": drive,
            "situation": situation,
            "memory_context": memory_context
        }

        response = self.llm.ask(variables, prompt, temp=0.1, stops=['</End>'])
        
        try:
            term = xml.find('<Term>', response)
            assessment = xml.find('<Assessment>', response)
            trigger = xml.find('<Trigger>', response)
            termination = xml.find('<Termination>', response)
            
            if all([term, assessment, trigger, termination]):
                return StateAssessment(
                    term=term.strip(),
                    value=assessment.strip().lower(),
                    trigger=trigger.strip(),
                    completion_check=termination.strip(),
                    timestamp=memory.owner.context.simulation_time
                )
        except Exception as e:
            print(f"Error parsing state assessment: {str(e)}")
            
        return None
    