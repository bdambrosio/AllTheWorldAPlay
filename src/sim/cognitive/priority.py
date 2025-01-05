# sim/cognitive/priority.py
from typing import Dict, List
from utils.Messages import UserMessage
from sim.memory.core import MemoryEntry, StructuredMemory
import utils.xml_utils as xml

class PrioritySystem:
    """Handles task priorities and updates"""
    
    def __init__(self, llm, character):
        self.llm = llm
        self.character = character
        
    def update_priorities(self,
                         drives: List[str],
                         state: Dict[str, Dict[str, str]],
                         memory: StructuredMemory,
                         situation: str) -> List[str]:
        """Update task priorities based on current state"""        
        # Map state for LLM input
        mapped_state = self._map_state(state)
        
        # Get memories relevant to current drives
        relevant_memories = []
        for drive in drives:
            relevant_memories.extend(memory.get_by_drive(drive, limit=2))
        formatted_memories = self._format_memories(relevant_memories)
        
        # Generate new priorities through LLM
        new_priorities = self._generate_priority_tasks(
            drives,
            mapped_state,
            formatted_memories,
            situation
        )
        
        return new_priorities

    def _map_state(self, state: Dict) -> str:
        """Map state dict to string format for LLM"""
        mapped = []
        for key, item in state.items():
            trigger = item['drive']
            value = item['state']
            mapped.append(f"- '{key}: {trigger}', State: '{value}'")
        return "A 'State' of 'High' means the task is important or urgent\n" + '\n'.join(mapped)

    def _format_memories(self, memories: List[MemoryEntry]) -> str:
        """Format memories for LLM input"""
        formatted = []
        for memory in memories:
            formatted.append(f"- [{memory.timestamp}] (importance: {memory.importance:.1f}): {memory.text}")
        return '\n'.join(formatted)

    def _generate_priority_tasks(self,
                               drives: List[str],
                               state: str,
                               memories: str,
                               situation: str) -> List[str]:
        """Generate prioritized tasks using LLM"""
        prompt = [UserMessage(content="""Create a set of three short term plans given who you are, your Situation, your Stance, your Memory, and your recent RecentHistory as listed below. 

<Character>
{{$character}}
</Character>

<Drives>
{{$drives}}
</Drives>

<Situation>
{{$situation}}
</Situation>

<Stance>
{{$state}}
</Stance>

<RecentHistory>
{{$memories}}
</RecentHistory>

Respond using this XML format:

<Plan>
  <Name>Make Shelter</Name>
  <Steps>
    1 identify protected location
    2 gather sticks and branches
    3 build shelter
  </Steps>
  <Rationale>I am in a forest and need shelter for the night. sticks and branches are likely nearby</Rationale>
  <TerminationCheck>Shelter is complete</TerminationCheck>
</Plan>

Respond ONLY with your three highest priority plans using the above XML format.
Plans should be as specific as possible.
Rationale statements must be concise and limited to a few keywords or at most two terse sentences.""")]

        variables = {
            "character": self.character,
            "drives": "\n".join(drives),
            "situation": situation,
            "state": state,
            "memories": memories
        }

        response = self.llm.ask(variables, prompt, temp=0.6)
        
        # Extract all <Plan> sections
        tasks = []
        for plan in xml.findall('<Plan>', response):
            name = xml.find('<Name>', plan)
            steps = xml.find('<Steps>', plan)
            rationale = xml.find('<Rationale>', plan)
            termination = xml.find('<TerminationCheck>', plan)
            if all([name, steps, rationale, termination]):
                tasks.append(f"<Plan><Name>{name}</Name><Steps>{steps}</Steps><Rationale>{rationale}</Rationale><TerminationCheck>{termination}</TerminationCheck></Plan>")
            
        return tasks