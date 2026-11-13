from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from sim.cognitive.DialogManager import Dialog
from utils import hash_utils
from sim.trace import trace
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
        if owner is other_agh or owner.name == other_agh.name:
            # self-model: the "no interactions yet" seed below made no sense for oneself
            self._relationship_base = "Self-model not yet formed."
        else:
            self._relationship_base = f"No significant interactions with {self.canonical_name} yet. Relationship is neutral."
        # short updates (and tensions) between full rewrites; capped so the text, which is
        # embedded in prompts, can't grow without bound
        self._relationship_notes: List[str] = []
        self.relationship: str = self._relationship_base
        self.tensions: List[str] = []
        self.recent_relationship_update: bool = False

    def to_json(self):
        """Convert KnownActor state to JSON-serializable dict, excluding runtime references"""
        return {
            'canonical_name': self.canonical_name,
            'name': self.name,
            'visible': self.visible,
            'goal': self.goal,
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
        known_actor._relationship_base = data['relationship']
        known_actor._relationship_notes = []
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
            target_goal = getattr(self.actor_agh, 'focus_goal', None)
            trace.event('inferred_goal', observer=self.owner.name, target=self.canonical_name, inferred=self.goal,
                        actual=target_goal.short_string() if target_goal else None)

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

        if self.owner.name != self.actor_agh.name:
            prompt = [UserMessage(content=f"""Analyze relationship between these characters based on recent interactions.
Character: 
{self.owner.character}

Other Character: {self.actor_agh.name}

Previous Relationship Status:
{self.relationship}

Recent Interactions:
{chr(10).join(f"- {text}" for text in char_memories)}

Say how {self.owner.name} now stands with {self.actor_agh.name}, as {self.owner.name} sees it: what has passed between them, what {self.owner.name} believes {self.actor_agh.name} wants, and what {self.owner.name} now thinks of them.
Write in the third person, in plain sentences, 60 words at most. Say only what the interactions listed show; what has not been shown is not known, and is left out. Keep what still holds from the previous status, and drop what the interactions have overtaken. No list of qualities, and no verdicts the interactions do not bear out.

Respond with that, plus a single True / False assessment of whether the relationship has changed significantly.
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#Change True / False
#Relationship how they now stand, 60 words at most, with no newline characters
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
        rewritten = hash_utils.find('Relationship', new_relation) if new_relation else ''
        if rewritten:  # a reply without #Relationship used to wipe the relationship to ''
            self._relationship_base = rewritten.strip()
            self._relationship_notes = []
            self._rebuild_relationship()
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

        if self.owner.name != self.actor_agh.name:
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

        relation_update = self.owner.llm.ask({}, prompt, tag='KnownActor.update_relationship', max_tokens=40, stops=["</end>"])
        if relation_update and relation_update.strip() != '':
            self._add_relationship_note(relation_update.strip())
            self.recent_relationship_update = True

    MAX_RELATIONSHIP_NOTES = 5

    def _add_relationship_note(self, note):
        self._relationship_notes.append(note)
        del self._relationship_notes[:-self.MAX_RELATIONSHIP_NOTES]
        self._rebuild_relationship()

    def _rebuild_relationship(self):
        self.relationship = '\n'.join([self._relationship_base] + self._relationship_notes)


    def add_tension(self, item):
        """add an item to the relationship"""
        # tensions field is not used, but keep it for now
        self.tensions.append(item)
        self._add_relationship_note(item)
        
    def get_tensions(self):
        return '\n'.join(self.tensions)
    
    def clear_tensions(self):
        self.tensions = []

    def get_relationship(self):
        """What the owner knows and thinks of this person, in the owner's own terms: what they
        are to it, how they look, what can be seen of them now, what has passed between the two
        and what the owner makes of it, and what the owner takes them to be after.

        Everything here is the owner's own: its eyes, its talk with them (update_relationship),
        what it has watched them do (infer_goal). It used to carry the act's tension points,
        which are the director's ('Samantha suspects Joe ...' in Joe's notes on Samantha, before
        they had spoken), and every relationship the other person has with anyone (so each
        character knew whose mistress the widow was)."""
        owner, other = self.owner, self.actor_agh
        if owner is other or owner.name == other.name:
            return self.relationship
        lines = []
        try:
            mine = [str(rel) for rel, whom in owner.context.reference_manager.get_relationships(other.name)
                    if str(getattr(whom, 'name', whom)).lower() == owner.name.lower()]
            if mine:
                lines.append(f"What they are to you: {', '.join(mine)}.")
            looks = owner.context._image_description(other) if getattr(other, 'character', None) else ''
            if looks:
                lines.append(f"How they look: {looks.rstrip('. ')}.")
            if self.visible and getattr(other, 'body', None) is not None:
                lines.append(f"As you see them now: {other.body.visible()}.")
        except Exception:
            pass
        lines.append(self.relationship)
        if self.goal:
            lines.append(f"What you take them to be after: {self.goal}")
        return '\n'.join(l for l in lines if l)

class KnownActorManager:
    def __init__(self, owner, context):
        self.owner = owner
        self.context = context
        # one model per character, keyed by the character's own name; other spellings that
        # resolved to it live in aliases (storing each model under both names made every
        # iteration over known_actors -- prompts' name lists, relationship updates, dialogs,
        # the Explore panel -- see that character twice)
        self.known_actors: Dict[str, KnownActor] = {}
        self.aliases: Dict[str, str] = {}  # normalised other name -> key in known_actors
        self.resolution_cache: Dict[str, Tuple[Character, str]] = {}  # reference_text -> (character, canonical_name)

    def _model(self, name) -> Optional[KnownActor]:
        """The model for a name or one of its aliases, or None."""
        if not name:
            return None
        name = name.strip()
        for candidate in (name, name.capitalize()):
            if candidate in self.known_actors:
                return self.known_actors[candidate]
        key = self.aliases.get(name.capitalize())
        return self.known_actors.get(key) if key else None

    def _store(self, alias, actor_agh) -> KnownActor:
        """The model for actor_agh (created if new), with alias pointing at it."""
        model = self.known_actors.get(actor_agh.name)
        if model is None:
            model = self.known_actors[actor_agh.name] = KnownActor(self.owner, actor_agh)
        if alias and alias != actor_agh.name:
            self.aliases[alias] = actor_agh.name
        return model

    def create_character(self, reference_text):
        """create a new character - used only after resolve_reference fails to find an existing character"""
        actor_name = reference_text.strip().capitalize()
        actor_agh = self.context.get_npc_by_name(actor_name, description=reference_text, x=self.owner.mapAgent.x, y=self.owner.mapAgent.y, create_if_missing=True)
        self._store(actor_name, actor_agh)
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
        model = self._model(reference_text)
        if model is not None:
            result = (model.actor_agh, reference_text)
            self.resolution_cache[reference_text] = result
            return result
            
        # Check if it's a reference to a character in the context
        result = self.context.resolve_character(reference_text)
        if result and result[0] is not None:
            # cache hits only: caching misses hid characters created later under that name
            self.resolution_cache[reference_text] = result
            return result
        return (None, None)

    def names(self):
        return [actor.canonical_name for actor in self.known_actors.values()]
    
    def known(self, name):
        return self._model(name) is not None
        
    def actor_models(self):
        return self.known_actors.values()
    
    def add_actor_model(self, actor_name):
        actor_name = actor_name.strip().capitalize()
        actor_agh,_ = self.context.resolve_character(actor_name)
        if actor_agh and (actor_agh.__class__.__name__ == 'NarrativeCharacter' or actor_agh.__class__.__name__ == 'Character'):
            return self._store(actor_name, actor_agh)
        print(f"{actor_name} not found")
        return None

    def get_actor_model(self, actor_name: str, create_if_missing: bool=False) -> Optional[KnownActor]:
        model = self._model(actor_name)
        if model is None and create_if_missing:
            print(f"{self.owner.name} creating model for {actor_name.strip().capitalize()}")
            model = self.add_actor_model(actor_name)  # None when there is no such character
        return model

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
        cast = self.context.actors + self.context.extras if self.context is not None else []
        for actor in self.known_actors.values():
            # the people of the play and anyone spoken with. Not the owner's model of itself
            # (it held Joe's own lines back to him as thoughts), nor someone only inferred
            # ('Cave1_owner', a paragraph on a person nobody had met, in every prompt)
            if actor.actor_agh is self.owner or not (actor.actor_agh in cast or actor.dialog.transcript):
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
        model = self._model(actor_name)
        return model.relationship if model is not None else ''

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
            'aliases': dict(self.aliases),
            'resolution_cache': {
                key: (value[0].name, value[1])  # Store just the names
                for key, value in self.resolution_cache.items()
            }
        }

    def save_to_file(self, filepath):
        """Save state to file"""
        import json
        # serialize first: opening for write before a failing to_json() truncated the file
        data = json.dumps(self.to_json())
        with open(filepath, 'w') as f:
            f.write(data)

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
                # files saved before aliases existed hold the same actor under two names
                key = actor.canonical_name or name
                manager.known_actors.setdefault(key, actor)
                if name != key:
                    manager.aliases[name] = key
        manager.aliases.update(data.get('aliases', {}))
                
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