from datetime import datetime
from typing import List
from sim.memory.core import MemoryEntry, StructuredMemory, MemoryCategory
from sim.agh import Character

class Human(Character):
    def __init__(self, name, ui):
        super().__init__(name, "Human user representative")
        self.structured_memory = StructuredMemory()
        self.ui = ui
        self.priority_task = None
        self.active_task = None

    def hear(self, from_actor, message, source='dialog', respond=True):
        if from_actor.name != 'Watcher':
            # Don't display messages the user types!
            self.add_to_history(f"You hear {from_actor.name} say: {message}")
            self.ui.display(f"\n{from_actor.name}: {message}\n")
        if respond:
            response = self.generate_response(from_actor, message, source)
            if response:
                self.intentions.append(response)

    def tell(self, to_actor, message, source='inject', respond=True):
        if source == "init":
            # special case, reset context
            self.active_task = to_actor.name
        if source != 'inject':
            self.add_to_history(f'You say to {to_actor.name}: {message}')
        # user has no task management!
        self.acts(to_actor, 'Say', message, '', source)

    def add_to_history(self, message: str):
        """Add message to structured memory"""
        entry = MemoryEntry(
            text=message,
            category=self._determine_category(message),
            importance=0.5,
            timestamp=datetime.now(),
            justification="User interaction",
            related_drives=[],  # Humans don't use drives
            emotional_valence=0.0,
            confidence=1.0
        )
        self.structured_memory.add_entry(entry)
        self.new_memory_cnt += 1

    def format_history(self, n=2):
        """Get n most recent memories"""
        recent_memories = self.structured_memory.get_recent(n)
        return '\n\n'.join(memory.text for memory in recent_memories)

    def _determine_category(self, message: str) -> MemoryCategory:
        """Simple categorization for human messages"""
        message = message.lower()
        if "hear" in message or "say" in message:
            return MemoryCategory.SOCIAL
        elif "see" in message or "observe" in message:
            return MemoryCategory.ENVIRONMENTAL
        return MemoryCategory.CORE

    def _find_related_drives(self, message: str) -> List[str]:
        """Humans don't use drives"""
        return []

    def forward(self, step):
        """Humans don't need autonomous updates"""
        pass

    def clear_task_if_satisfied(self, task_xml, consequences, world_updates):
        """Humans don't use autonomous task management"""
        pass

    def acts(self, target, act_name, act_arg='', reason='', source=''):
        """Process human actions"""
        if act_name == 'Say':
            if source != 'inject':
                # Not user input - system generated
                self.show += f"\n{self.name}: '{act_arg}'"
            if target:
                target.hear(self, act_arg, source)
                    
    def inject(self, input_text):
        """Process user input from UI"""
        # Parse "Character name, message" format
        parts = input_text.split(',')
        if len(parts) != 2:
            print("Input must be in format: Character name, message")
            return
            
        target_name = parts[0].strip()
        message = parts[1].strip()
        
        # Find target character
        target = None
        for actor in self.context.actors:
            if actor.name.lower() == target_name.lower():
                target = actor
                break
                
        if target:
            self.tell(target, message)
        else:
            print(f"Character {target_name} not found")