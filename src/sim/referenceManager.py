from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from sim.context import Context
    from sim.agh import Character

from utils.Messages import UserMessage
from utils.llm_api import LLM
from utils.hash_utils import find
import utils.xml_utils as xml


class ReferenceManager:
    def __init__(self,context,llm):
        # Simple dict of lists, indexed by character name
        self.context: Context = context
        self.relationships = {}
        self.llm = llm

    def declare_relationship(self, char1_name, relation, char2_name, reverse_relation):
        # Will add both:
        # char1 relation char2
        # char2 reverse_relation char1
        if char1_name not in self.relationships:
            self.relationships[char1_name] = []
        if char2_name not in self.relationships:
            self.relationships[char2_name] = []
        self.relationships[char1_name].append((relation, char2_name))
        self.relationships[char2_name].append((reverse_relation, char1_name))
        
    def get_relationships(self, char_name):
        return self.relationships.get(char_name, [])

    def get_all_relationships(self):
        relationships = []
        for char_name, relations in self.relationships.items():
            for relation, other_char in relations:
                relationships.append(f'{char_name} is {relation} {other_char}')
        return relationships
    
    def discover_relationships(self, addl_context=None):
        """Use LLM to identify potential relationships between entities from context"""
        prompt = [
            UserMessage(content="""
            Based on the current state and history, identify potential relationships between characters.
            
            Current state:
            {{$current_state}}
            {{$addl_context}}
            
            Characters:
            {{$characters}}
            
            For each pair of characters that appear to have a relationship, respond with:
            <relationship>
            <person1>name</person1>
            <person2>name</person2>
            <relation1to2>how person1 relates to person2 (e.g., "father")</relation1to2>
            <relation2to1>how person2 relates to person1 (e.g., "daughter")</relation2to1>
            </relationship>
            
            Provide multiple <relationship> entries if you identify multiple relationships.
            
            End with:
            </end>
            """)
        ]
        
        characters_text = "\n".join([f"{character.name}:\n{character.character}\n" 
                                    for character in self.context.actors+self.context.npcs])
        
        response = self.llm.ask({
            "current_state": self.context.current_state,
            "addl_context": addl_context if addl_context else '',
            "characters": characters_text
        }, prompt, stops=["</end>"])
        
        # Extract relationships
        relationship_blocks = xml.findall('<relationship>', response)
        
        for block in relationship_blocks:
            person1_name = xml.find('<person1>', block)
            person2_name = xml.find('<person2>', block)
            relation1to2 = xml.find('<relation1to2>', block)
            relation2to1 = xml.find('<relation2to1>', block)
            
            if person1_name and person2_name and relation1to2 and relation2to1:
                # Look up canonical IDs
                person1_id = self.context.resolve_character(person1_name)
                person2_id = self.context.resolve_character(person2_name)
                
                if person1_id and person2_id:
                    # Add relationship variations
                    if relation1to2 and relation2to1:
                        self.declare_relationship(person2_name, relation1to2, person1_name, relation2to1)
                    
    def resolve_reference_with_llm(self, reference_text:str) -> Optional[Character]:
        """
        Resolve a reference using known relationships and character descriptions
        
        Args:
            reference_text: Text reference to resolve (e.g. "Jean's father" or "the young hiker")
            
        Returns:
            str or None: Character name if resolved, None otherwise
        """
        # Build character context
        char_context = []
        for char in self.context.actors+self.context.npcs:  # Assumes method to get both actors/npcs
            if hasattr(char, 'reference_dscp'):
                char_context.append(f"{char.name} - {char.reference_dscp}")
        
        # Build relationship context
        relationship_context = []
        for char in self.relationships:
            for relation, other in self.relationships[char]:
                relationship_context.append(f"{char} is {relation} {other}")
        
        prompt = [
            UserMessage(content=f"""
Given these characters:
{chr(10).join(char_context)}

And these relationships:
{chr(10).join(relationship_context)}

Who is referred to by: "{reference_text}"?

Respond with just the character's name, or "unknown" if you cannot determine who is being referenced.
Do not include any introductory, explanatory, or formatting text in your response.

End your response with:
</end>
""")
        ]
        
        response = self.llm.ask({}, prompt, stops=["</end>"]).strip()
        
        # If response is a known character name, return it
        if response in self.relationships:
            return response
        character = self.context.get_actor_or_npc_by_name(response)
        if character:
            return character.name
        
        return None