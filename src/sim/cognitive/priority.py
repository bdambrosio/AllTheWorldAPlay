# sim/cognitive/priority.py
from typing import Dict, List
from utils.Messages import UserMessage
from sim.memory.core import MemoryEntry, StructuredMemory, Drive
import utils.xml_utils as xml
from datetime import datetime, timedelta

class PrioritySystem:
    """Handles task priorities and updates"""
    
    def __init__(self, llm, character):
        self.llm = llm
        self.character = character
        
    def _get_drive_memories(self, memory: StructuredMemory, drive: Drive) -> List[MemoryEntry]:
        """Get memories relevant to a drive for priority assessment"""
        return memory.owner.memory_retrieval.get_by_drive(
            memory=memory,
            drive=drive,
            threshold=0.1,
            max_results=5  # Limit to most relevant for priority decisions
        )

    def update_priorities(self, memory: StructuredMemory, current_state: Dict) -> List[str]:
        """Update task priorities based on state and memory"""
        drive_memories = {}
        
        # Get relevant memories for each drive
        for drive in memory.owner.drives:
            memories = self._get_drive_memories(memory, drive)
            if memories:
                drive_memories[drive] = memories
                
        # Generate tasks based on drive memories
        tasks = []
        for drive, memories in drive_memories.items():
            drive_tasks = self._generate_tasks(
                drive=drive,
                memories=memories,
                current_state=current_state
            )
            tasks.extend(drive_tasks)
            
        return tasks

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
Rationale statements must be concise and limited to a few keywords or at most two terse sentences.

End your response with:
</End>""")]

        variables = {
            "character": self.character,
            "drives": "\n".join(drives),
            "situation": situation,
            "state": state,
            "memories": memories
        }

        response = self.llm.ask(variables, prompt, temp=0.6, stops=['</End>'])
        
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