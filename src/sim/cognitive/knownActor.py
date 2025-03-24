from __future__ import annotations
from typing import Optional
from sim.cognitive.DialogManager import Dialog
from utils.Messages import UserMessage
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking

class KnownActor:
    def __init__(self, owner:Character, other_agh:Character):
        """An instance of a 'character internal model' of another character"""
        self.owner = owner
        self.actor_agh = other_agh
        self.canonical_name = other_agh.name
        self.name = other_agh.name
        self.visible = False
        self.goal = ''
        self.model = ''
        self.distance = 0
        self.dialog = Dialog(self.owner, self.actor_agh)
        self.relationship = f"No significant interactions with {self.canonical_name} yet. Relationship is neutral."

    def infer_goal(self, percept):
        """infer the goal of the actor based on a perceptual input"""
        prompt = [UserMessage(content=f"""Analyze the perceptual input below and infer the goal of the {self.canonical_name}, 
given your current relationship with them:

Current Relationship:
{self.relationship}

Perceptual Input:
{percept.mode}: {percept.content}

respond with a goal description of up to 6 tokens, no additional text.
Do not include any introductory, explanatory, discursive, formatting, or concluding text.
End with:
</end>
""")]
        response = self.owner.llm.ask({}, prompt, max_tokens=20, stops=["</end>"])
        if response:
            self.goal = response.strip()

    def format_transcript(self, include_transcript=False):
        return self.dialog.get_transcript(8) if include_transcript else ''
    
    def update_relationship(self, all_texts):
        """self is An instance of a model of another actor 
          updating beliefs about owner relationship to this actor"""
        char_memories = [text for text in all_texts if self.canonical_name.lower() in text.lower()]
        if char_memories is None or len(char_memories) == 0:
            return

        if self.owner.name is not self.actor_agh.name:
            prompt = [UserMessage(content=f"""Analyze relationship between these characters based on recent interactions.
Character: 
{self.owner.character}

Other Character: {self.actor_agh.name}

Previous Relationship Status:
{self.relationship}

Recent Interactions:
{chr(10).join(f"- {text}" for text in char_memories)}

Describe their current relationship in a brief statement that captures:
1. Character's perception of the other character's nature
2. Character's current emotional state towards the other character
3. Any recent changes in their relationship
4. Ongoing dynamics

Respond with a concise updated relationship description of up to 100 tokens, no additional text.
End with:
</end>
""")]
        else:
            prompt = [UserMessage(content=f"""Analyze self-model of this character based on recent thoughts.
Character: 
{self.owner.character}

Previous self-model:
{self.relationship}

Recent Thoughts:
{chr(10).join(f"- {text}" for text in char_memories)}

Describe their current self-model in a brief statement that captures:
1. Character's perception of his/her own nature
2. Character's current emotional comfort level with his/her own nature
3. Any recent changes in their self-model
4. Ongoing dynamics

Respond with a concise updated self-model description of up to 100 tokens, no additional text.
End with:
</end>
""")]

        new_relation = self.owner.llm.ask({}, prompt, max_tokens=200, stops=["</end>"])
        if new_relation:
            self.relationship = new_relation.strip()

class KnownActorManager:
    def __init__(self, owner, context):
        self.owner = owner
        self.context = context
        self.known_actors = {}
        self.resolution_cache = {}  # reference_text -> (character, canonical_name)

    def create_character(self, reference_text):
        """Create a new character - used only after resolve_reference fails to find an existing character"""
        actor_name = reference_text.strip().capitalize()
        actor_agh = self.context.get_npc_by_name(actor_name, self.owner.mapAgent.x, self.owner.mapAgent.y, create_if_missing=True)
        self.known_actors[actor_name] = KnownActor(self.owner, actor_agh)
        self.resolution_cache[actor_name] = (actor_agh, actor_name)
        return actor_agh, actor_name
        
    def resolve_character(self, reference_text):
        """
        Resolve a reference to a character, using cache first then falling back to context  
        Args:
            reference_text: Text reference to resolve 
        Returns:
            tuple: (character, canonical_name) or (None, None) if unresolved
        """
        # Normalize reference text
        if reference_text is None or reference_text == '' or reference_text.lower() == 'none':
            return None, None
        reference_text = reference_text.strip().capitalize()
        # Check cache first
        if reference_text in self.resolution_cache:
            return self.resolution_cache[reference_text]
            
        # Check if it's a direct reference to a known actor
        if reference_text in self.known_actors:
            result = (self.known_actors[reference_text].actor_agh, reference_text)
            self.resolution_cache[reference_text] = result
            return result
            
        # Check if it's a reference to a character in the context
        result = self.context.resolve_character(reference_text)
        #if result:
        self.resolution_cache[reference_text] = result # cache everything to avoid repeated lookups
        if result:
            return result
            
        # Cache miss, return None
        return (None, None)

    def names(self):
        return [actor.canonical_name for actor in self.known_actors.values()]
    
    def actor_models(self):
        return self.known_actors.values()
    
    def add_actor_model(self, actor_name):
        actor_name = actor_name.strip().capitalize()
        actor_agh,_ = self.context.resolve_character(actor_name)
        if actor_agh:
            # Always store under both names to be safe
            self.known_actors[actor_agh.name] = KnownActor(self.owner, actor_agh, )
            self.known_actors[actor_name] = self.known_actors[actor_agh.name]
        else:
            print(f"{actor_name} not found")
        return self.known_actors.get(actor_name)

    def get_actor_model(self, actor_name: str, create_if_missing: bool=False) -> Optional[KnownActor]:
        actor_name = actor_name.strip().capitalize()
        if actor_name not in self.known_actors:
            if create_if_missing:
                print(f"{self.owner.name} creating model for {actor_name}")
                self.add_actor_model(actor_name)
            else:
                    return None
        return self.known_actors[actor_name] if actor_name in self.known_actors else None

    def set_all_actors_visible(self):
        for actor in self.known_actors:
            self.known_actors[actor.canonical_name].visible = True

    def set_all_actors_invisible(self):
        for actor in self.known_actors.values():
            actor.visible = False

    def update_all_relationships(self, all_texts):
        valid_chars = [a.name for a in self.context.actors if a.name != self.owner.name] + [a.name for a in self.context.npcs]
        all_chars = set(valid_chars) & set(self.names())
        for actor in self.known_actors.values():
            #actor is a KnownActor instance, a model of the owner's relationship to this actor
            actor.update_relationship(all_texts)

    def get_known_relationships(self, include_transcript=False):
        relationships = {}
        for actor in self.known_actors.values():
            if actor == self:
                continue
            relationship = actor.relationship
            transcript = actor.format_transcript(include_transcript)
            relationships[actor.canonical_name] = relationship + '\n\n' + transcript
        return relationships

    def format_relationships(self, include_transcript=False):
        relationships = self.get_known_relationships(include_transcript)
        return '\n\n'.join([f"{name}:\n {relationship}" for name, relationship in relationships.items()])

    def resolve_or_create_character(self, reference_text):
        """
        Resolve a reference to a character, creating one if it doesn't exist
        
        Args:
            reference_text: Text reference to resolve
            
        Returns:
            tuple: (character, canonical_name) - guaranteed to not be None
        """
        # Try to resolve first
        if not reference_text or reference_text == '' or reference_text.lower() == 'none':
            return None, None
        
        reference_text = reference_text.strip().capitalize()
        resource, canonical_name = self.resolve_resource(reference_text)
        if resource: # resource found, not a character - expansion to come
            return None, None
        
        character, canonical_name = self.resolve_character(reference_text)
        if not character:
            # Use owner's current location for the new character
            character, canonical_name = self.create_character(reference_text)
        
        return character, canonical_name
    
    def resolve_resource(self, reference_text):
        """
        Resolve a reference to a resource
        Args:
            reference_text: Text reference to resolve
        Returns:
            tuple: (resource, canonical_name) or (None, None) if unresolved
        """
        # Try to resolve first
        if not reference_text or reference_text == '' or reference_text.lower() == 'none':
            return None, None
        resource = self.context.map.resource_registry.get(reference_text)
        if resource:
            return resource, reference_text
        return None, None