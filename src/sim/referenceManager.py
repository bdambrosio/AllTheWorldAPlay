from sympy import re
from utils.Messages import UserMessage
from utils.llm_api import LLM
from utils.hash_utils import find
import utils.xml_utils as xml


class ReferenceManager:
    def __init__(self,context,llm):
        # Simple dict of lists, indexed by character name
        self.context = context
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


    def discover_relationships(self):
        """Use LLM to identify potential relationships between entities from context"""
        prompt = [
            UserMessage(content=f"""
            Based on the current state and history, identify potential relationships between characters.
            
            Current state:
            {{$current_state}}
            
            History:
            {{$history}}
            
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
        
        characters_text = "\n".join([f"{character.name}: {character.character}" 
                                    for character in self.actors])
        
        response = self.llm.ask({
            "current_state": self.current_state,
            "history": self.history(),
            "characters": characters_text
        }, prompt, stops=["</end>"])
        
        # Extract relationships
        relationship_blocks = re.findall(r'<relationship>(.*?)</relationship>', 
                                    response, re.DOTALL)
        
        for block in relationship_blocks:
            person1 = xml.find('<person1>', block)
            person2 = xml.find('<person2>', block)
            relation1to2 = xml.find('<relation1to2>', block)
            relation2to1 = xml.find('<relation2to1>', block)
            
            if person1 and person2 and relation1to2 and relation2to1:
                # Look up canonical IDs
                person1_id = self._get_entity_id_by_name(person1)
                person2_id = self._get_entity_id_by_name(person2)
                
                if person1_id and person2_id:
                    # Add relationship variations
                    if relation1to2:
                        self.entity_registry.add_variation(person2_id, person1.lower(), 
                                                        f"my {relation1to2}")
                        self.entity_registry.add_variation(person2_id, person1.lower(), 
                                                        f"{relation1to2}")
                    
                    if relation2to1:
                        self.entity_registry.add_variation(person1_id, person2.lower(), 
                                                        f"my {relation2to1}")
                        self.entity_registry.add_variation(person1_id, person2.lower(), 
                                                        f"{relation2to1}")

    def resolve_reference_with_llm(self, reference_text):
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
        
        return None