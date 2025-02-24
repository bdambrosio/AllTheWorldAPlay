from datetime import datetime
from typing import List
from sim.memory.core import MemoryEntry, StructuredMemory
from sim.agh import Character, Stack
import asyncio

from PyQt5.QtWidgets import QDialog
class Human(Character):
    def __init__(self, name, character_description, ui=None):
        super().__init__(name, character_description)
        self.structured_memory = StructuredMemory()
        self.ui = ui

    def hear(self, from_actor, message, source='dialog', respond=True):
        pass

    def generate_response(self, from_actor, message, source):
        pass
    
    def tell(self, to_actor, message, source='inject', respond=True):
        pass
 
    def format_history(self, n=2):
            """Get n most recent memories"""
            recent_memories = self.structured_memory.get_recent(n)
            return '\n\n'.join(memory.text for memory in recent_memories)


    def add_to_history(self, message):
        """Add a message to the history"""
        pass

    def _find_related_drives(self, message: str) -> List[str]:
        """Humans don't use drives"""
        return []

    def forward(self, step):
        """Humans don't need autonomous updates"""
        pass

    def acts(self, target, act_name, act_arg='', reason='', source=''):
        pass                       
    def senses(self, sense_data = None):
        pass  

    def cognitive_cycle(self):
        pass

    async def inject(self, input_text):
        """Process user input from UI"""
        # Parse "Character name, message" format
        parts = input_text.split(',')
        if len(parts) < 2:
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
            await target.hear(self, message, source='dialog with watcher')
        else:
            print(f"Character {target_name} not found")