from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from venv import create
from sim.cognitive.DialogManager import Dialog
from src.utils import hash_utils
from utils.Messages import UserMessage
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking

class KnownActor:
    def __init__(self, owner:Character, other_agh:Character):
        """An instance of a 'character internal model' of another character"""
        self.owner: Character = owner
        self.actor_agh: Character = other_agh
        self.canonical_name: str = other_agh.name
        self.name: str = other_agh.name
        self.visible: bool = False
        self.goal: str = ''
        self.distance: int = 0
        self.dialog: Dialog = Dialog(self.owner, self.actor_agh)
        self.relationship: str = f"No significant interactions with {self.canonical_name} yet. Relationship is neutral."
        self.tensions: List[str] = []
        self.recent_relationship_update: bool = False

    def to_json(self):
        """Convert KnownActor state to JSON-serializable dict, excluding runtime references"""
        return {
            'canonical_name': self.canonical_name,
            'name': self.name,
            'visible': self.visible,
            'goal': self.goal,
            'model': self.model,
            'distance': self.distance,
            'relationship': self.relationship,
            'dialog': self.dialog.to_json() if self.dialog else None,
            'tensions': self.tensions
        }
    
    @classmethod
    def from_json(cls, data, owner, kam):
        """Create a new KnownActor instance from JSON data using KnownActorManager to resolve characters"""
        # Resolve the actor_agh using the KnownActorManager
        actor_agh, _ = kam.resolve_character(data['name'])
        if not actor_agh:
            return None
            
        # Create new instance
        known_actor = cls(owner, actor_agh)
        
        # Restore state
        known_actor.canonical_name = data['canonical_name']
        known_actor.name = data['name']
        known_actor.visible = data['visible']
        known_actor.goal = data['goal']
        known_actor.distance = data['distance']
        known_actor.relationship = data['relationship']
        known_actor.tensions = data['tensions']
        # Restore dialog if it exists, passing the resolved characters
        if data['dialog']:
            known_actor.dialog = Dialog.from_json(data['dialog'], owner, actor_agh)
            
        return known_actor

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
        response = self.owner.llm.ask({}, prompt, tag='KnownActor.infer_goal', max_tokens=20, stops=["</end>"])
        if response:
            self.goal = response.strip()

    def format_transcript(self, include_transcript=False):
        return self.dialog.get_transcript(8) if include_transcript else ''
    
    def update_relationship(self, all_texts, use_all_texts=False):      
        """self is An instance of a model of another actor 
          updating beliefs about owner relationship to this actor"""

        if use_all_texts:
            char_memories = all_texts
        else:
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
1. Character's perception of the other character's nature: predictability, honesty, trustworthiness, reliability.
2. Character's assessment of the other character's capabilities and motivations.
3. Character's assessment of other character's emotional state.
4. Character's trust level for the other character - are their goals and drives aligned with yours, will they act in your interest?
5. Character's assessment of evidence for recent changes in relationship with other character?

Respond with a concise updated relationship description of up to 160 tokens plus a single True / False assement of whether the self-model has changed significantly"
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#Change True / False
#Relationship updated relationship paragraph of up to 160 tokens with no newline characters
##

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

Respond with a concise updated self-model description of up to 160 tokens plus a single True / False assement of whether the self-model has changed significantly"
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#Change True / False
#Relationship updated self-model paragraph of up to 160 tokens with no newline characters
##

End with:
</end>
""")]

        new_relation = self.owner.llm.ask({}, prompt, tag='KnownActor.update_relationship', max_tokens=200, stops=["</end>"])
        if new_relation:
            self.relationship = hash_utils.find('Relationship', new_relation)   
            self.clear_tensions()
            if not self.recent_relationship_update:
                self.recent_relationship_update = hash_utils.find('Change', new_relation).strip() == 'True'
        return self.relationship

    def short_update_relationship(self, all_texts, use_all_texts=False):      
        """self is An instance of a model of another actor 
          updating beliefs about owner relationship to this actor
          this is a quick update, not a full analysis, following a dialog turn"""
        if use_all_texts:
            char_memories = all_texts
        else:
            char_memories = [text for text in all_texts if self.canonical_name.lower() in text.lower()]
        if char_memories is None or len(char_memories) == 0:
            return

        if self.owner.name is not self.actor_agh.name:
            prompt = [UserMessage(content=f"""Analyze relationship between these characters based on recent interactions.
Character: 
{self.owner.character}

Other Character: {self.actor_agh.name}

Previous Relationship Estimate:
{self.relationship}

Recent Interactions:
{chr(10).join(f"- {text}" for text in char_memories)}

Create a short (6-10 word) update to the relationship estimate above that captures any changes in:
1. Character's perception of the other character's nature
2. Character's current emotional state towards the other character
3. Character's trust level for the other character - are their goals and drives aligned with yours, will they act in your interest?
4. Any recent changes in their relationship with you?

Respond with a concise change in the relationship estimate of up to 10 words. Do not include any, reasoning, introductory, explanatory or markdown formatting.
If there are no changes, no response is needed.
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

Create a short (6-10 word) update to the relationship estimate above that captures any changes in:
1. Character's perception of his/her own nature
2. Character's current emotional comfort level with his/her own nature
3. Any recent changes in their self-model
4. Ongoing dynamics            

Respond with a concise change in the self-model description of up to 10 words. Do not include any, reasoning, introductory, explanatory or markdown formatting.
If there are no changes, no response is needed.
End with:
</end>
""")]

        relation_update = self.owner.llm.ask({}, prompt, tag='KnownActor.update_relationship', max_tokens=24, stops=["</end>"])
        if relation_update and relation_update.strip() != '':
            self.relationship +=relation_update.strip()
            self.recent_relationship_update = True


    def add_tension(self, item):
        """add an item to the relationship"""
        # tensions field is not used, but keep it for now
        self.tensions.append(item)
        self.relationship += f'\n{item}'
        
    def get_tensions(self):
        return '\n'.join(self.tensions)
    
    def clear_tensions(self):
        self.tensions = []

    def get_relationship(self):
        """return the relationship of the actor to the owner, including tensions and any relationships this actor has with other characters
        """
        return self.relationship + self.get_tensions() + '\n'+'\n'.join([f"{rel} {other}" for rel, other in self.owner.context.reference_manager.get_relationships(self.actor_agh.name)])

class KnownActorManager:
    def __init__(self, owner, context):
        self.owner = owner
        self.context = context
        self.known_actors: Dict[str, KnownActor] = {}
        self.resolution_cache: Dict[str, Tuple[Character, str]] = {}  # reference_text -> (character, canonical_name)

    def create_character(self, reference_text):
        """create a new character - used only after resolve_reference fails to find an existing character"""
        actor_name = reference_text.strip().capitalize()
        actor_agh = self.context.get_npc_by_name(actor_name, description=reference_text, x=self.owner.mapAgent.x, y=self.owner.mapAgent.y, create_if_missing=True)
        self.known_actors[actor_name] = KnownActor(self.owner, actor_agh)
        self.resolution_cache[actor_name] = (actor_agh, actor_name)
        if self.context.current_scene:
            # we are in a scene, so char should have a goal
            actor_agh.instantiate_narrative_goal(self.context.current_scene['post_narrative'])
            actor_agh.focus_goal = actor_agh.goals[0]
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
    
    def known(self, name):
        if name.strip().capitalize() in self.known_actors:
            return True
        return False
        
    def actor_models(self):
        return self.known_actors.values()
    
    def add_actor_model(self, actor_name):
        actor_name = actor_name.strip().capitalize()
        actor_agh,_ = self.context.resolve_character(actor_name)
        if actor_agh and (actor_agh.__class__.__name__ == 'NarrativeCharacter' or actor_agh.__class__.__name__ == 'Character'):
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
        actor = self.known_actors[actor_name]

        return actor

    def set_all_actors_visible(self):
        for actor in self.known_actors:
            self.known_actors[actor.canonical_name].visible = True

    def set_all_actors_invisible(self):
        for actor in self.known_actors.values():
            actor.visible = False

    def update_all_relationships(self, all_texts, char_names=None):
        if char_names is None:
            valid_chars = [a.name for a in self.context.actors if a.name != self.owner.name] + [a.name for a in self.context.npcs]
            char_names = set(valid_chars) & set(self.names())
        for actor in self.known_actors.values():
            if actor.canonical_name in char_names:
                #actor is a KnownActor instance, a model of the owner's relationship to this actor
                actor.update_relationship(all_texts)

    def get_known_relationships(self, include_transcript=False):
        relationships = {}
        for actor in self.known_actors.values():
            if actor == self:
                continue
            relationship = actor.get_relationship()
            transcript = actor.format_transcript(include_transcript)
            relationships[actor.canonical_name] = relationship + '\n\n' + transcript
        return relationships

    def format_relationships(self, include_transcript=False):
        relationships = self.get_known_relationships(include_transcript)
        if relationships:
            return '\n\n'.join([f"{name}:\n {relationship}" for name, relationship in relationships.items()])
        else:
            return 'None to report'

    def get_known_actor_relationship(self, actor_name: str) -> str:
        """
        Get the known actor relationship for the specified actor if it exists.
        Returns an empty string if no relationship exists.
        
        Args:
            actor_name: Name of the actor to get the model for
            
        Returns:
            str: The known actor model if it exists, empty string otherwise
        """
        actor_name = actor_name.strip().capitalize()
        if actor_name in self.known_actors:
            return self.known_actors[actor_name].relationship
        return ''

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

    def to_json(self):
        """Convert manager state to JSON-serializable dict"""
        return {
            'known_actors': {
                name: actor.to_json() 
                for name, actor in self.known_actors.items()
            },
            'resolution_cache': {
                key: (value[0].name, value[1])  # Store just the names
                for key, value in self.resolution_cache.items()
            }
        }

    def save_to_file(self, filepath):
        """Save state to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_json(), f)

    def dialogs(self):
        """return a list of dialogs"""
        return [f'{actor.canonical_name}:\n {actor.dialog.get_transcript()}\n' for actor in self.known_actors.values() if actor.dialog]

    @classmethod
    def from_json(cls, data, owner, context):
        """Create new instance from JSON data"""
        manager = cls(owner, context)
        
        # Restore known_actors
        for name, actor_data in data['known_actors'].items():
            actor = KnownActor.from_json(actor_data, owner, manager)
            if actor:
                manager.known_actors[name] = actor
                
        # Restore resolution_cache
        for key, (char_name, canon_name) in data['resolution_cache'].items():
            char, _ = manager.resolve_or_create_character(char_name)
            if char:
                manager.resolution_cache[key] = (char, canon_name)
                
        return manager

    @classmethod
    def load_from_file(cls, filepath, owner, context):
        """Load state from file"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_json(data, owner, context)
    
    def get_dialog_transcripts(self, max_turns=10):
        """return a list of lines fromdialog transcripts"""
        transcripts = []
        for actor in self.known_actors.values():
            if actor is not self and actor.dialog:
                transcript = actor.dialog.get_transcript(max_turns)
                if transcript:
                    transcripts.extend([f'#Name: {actor.canonical_name}']+transcript.split('\n')+['##'])
        return transcripts