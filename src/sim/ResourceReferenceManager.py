from sympy import re
from utils.Messages import UserMessage
from utils.llm_api import LLM
from utils.hash_utils import find
import utils.xml_utils as xml


class ResourceReferenceManager:
    """ resolves references to resources in the context of the world map """
    def __init__(self,context,llm):
        # Simple dict of lists, indexed by character name
        self.context = context
        self.relationships = {}
        self.llm = llm

    def declare_relationship(self, character_name, relation, resource_name, reverse_relation):
        """ this manages reference relationships between characters and resources 
            for example, "interview location" and "Office#2" """
        if character_name not in self.relationships:
            self.relationships[character_name] = []
        if resource_name not in self.relationships:
            self.relationships[resource_name] = []
        self.relationships[character_name].append((relation, resource_name))
        self.relationships[resource_name].append((reverse_relation, character_name))
        
    def get_relationships(self, character_name):
        return self.relationships.get(character_name, [])

    def discover_relationships(self):
        """Use LLM to identify potential relationships between entities from context"""
        prompt = [
            UserMessage(content=f"""
            Based on the current state and history, identify potential relationships between resources.
            
            Current state:
            {{$current_state}}
            
            History:
            {{$history}}
            
            Resources:
            {{$resources}}
            
            For each pair of resources that appear to have a relationship, respond with:
            <relationship>
            <resource1>name</resource1>
            <resource2>name</resource2>
            <relation1to2>how resource1 relates to resource2 (e.g., "contains")</relation1to2>
            <relation2to1>how resource2 relates to resource1 (e.g., "contained in")</relation2to1>
            </relationship>
            
            Provide multiple <relationship> entries if you identify multiple relationships.
            
            End with:
            </end>
            """)
        ]
        
        resources_text = "\n".join([f"{resource.name}: {resource.description}" 
                                    for resource in self.context.map.resource_registry])
        
        response = self.llm.ask({
            "current_state": self.current_state,
            "history": self.history(),
            "resources": resources_text
        }, prompt, stops=["</end>"])
        
        # Extract relationships
        relationship_blocks = re.findall(r'<relationship>(.*?)</relationship>', 
                                    response, re.DOTALL)
        
        for block in relationship_blocks:
            resource1 = xml.find('<resource1>', block)
            resource2 = xml.find('<resource2>', block)
            relation1to2 = xml.find('<relation1to2>', block)
            relation2to1 = xml.find('<relation2to1>', block)
            
            if resource1 and resource2 and relation1to2 and relation2to1:
                # Look up canonical IDs
                resource1_id = self._get_entity_id_by_name(resource1)
                resource2_id = self._get_entity_id_by_name(resource2)
                
                if resource1_id and resource2_id:
                    # Add relationship variations
                    if relation1to2:
                        self.entity_registry.add_variation(resource2_id, resource1.lower(), 
                                                        f"my {relation1to2}")
                        self.entity_registry.add_variation(resource2_id, resource1.lower(), 
                                                        f"{relation1to2}")
                    
                    if relation2to1:
                        self.entity_registry.add_variation(resource1_id, resource2.lower(), 
                                                        f"my {relation2to1}")
                        self.entity_registry.add_variation(resource1_id, resource2.lower(), 
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