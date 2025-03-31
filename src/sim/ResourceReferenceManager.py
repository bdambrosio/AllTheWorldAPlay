from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Act, Character, Goal, Task
    from sim.context import Context  # Only imported during type checking
from sympy import re
from utils.Messages import UserMessage
from utils.llm_api import LLM
from utils.hash_utils import find
import utils.xml_utils as xml
from sim.prompt import ask

class ResourceReferenceManager:
    """ resolves references to resources in the context of the world map """
    def __init__(self,character, context,llm):
        # Simple dict of lists, indexed by character name
        self.context: Context = context
        self.character: Character = character
        self.relationships = {}
        self.llm = llm


    def resolve_reference_with_llm(self, reference_text:str):
        """ resolve a reference to a resource in the context of the world map """
        mission = f"""Your task is to resolve a reference to a resource in the world map.
Given information about the character referencing the resource, the reference text, the map of resources in the world,
the character's current location, and the character's stated intention to move the the reference text, determine the most likely resource that the character is referencing.
Return the name of the resource that the character is referencing. Return None if you cannot determine the resource. Do not include any introductory, explanatory, or formatting text in your response.
"""
        suffix = f"""

The reference text for the move destination :
{reference_text}

Character current location: {self.character.x()}, {self.character.y()}

List of all map resources: 

{'\n'.join(self.character.context.map.get_resource_list())}

Return the name of the resource that the character is referencing. 
- If the reference text is a resource name, return that resource name.
- If the reference text is a direction, return the nearest resource in the direction the character is moving.
- If the reference text is an action and includes a reference to a resource, return the resource name. For example, "go to the tree" should return the tree resource name.

Return None if you cannot determine the resource. Do not include any introductory, explanatory, or formatting text in your response.

End your response with:
<end/>
"""
        response = ask(self.character, mission, suffix, {}, 15)
        resource = self.context.map.get_resource_by_name(response.strip())
        if resource:
            return resource, response.strip()
        else:
            return None, resource.strip()