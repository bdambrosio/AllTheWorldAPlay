from sim.cognitive.DialogManager import Dialog
from utils.Messages import UserMessage

class KnownActor:
    def __init__(self, actor, manager):
        """An instance of a self-model of another actor"""
        self.manager = manager
        self.actor = actor
        self.name = actor.name
        self.visible = False
        self.goal = ''
        self.model = ''
        self.distance = 0
        self.dialog = Dialog(self.manager.owner, actor)
        self.relationship = f"No significant interactions with {actor.name} yet. Relationship is neutral."

    def infer_goal(self, percept):
        """infer the goal of the actor based on a perceptual input"""
        prompt = [UserMessage(content=f"""Analyze the perceptual input below and infer the goal of the {self.name}, 
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
        response = self.manager.owner.llm.ask({}, prompt, max_tokens=20, stops=["</end>"])
        if response:
            self.goal = response.strip()

    def format_transcript(self, include_transcript=False):
        return self.dialog.get_transcript(10) if include_transcript else ''
    
    def update_relationship(self, all_texts):
        """self is An instance of a model of another actor 
          updating beliefs about owner relationship to this actor"""
        char_memories = [text for text in all_texts if self.name.lower() in text.lower()]
        if char_memories is None or len(char_memories) == 0:
            return
        prompt = [UserMessage(content=f"""Analyze the relationship between these characters based on recent interactions.

Character: 
{self.manager.owner.character}

Other Character: {self.actor.name}

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

        new_relation = self.manager.owner.llm.ask({}, prompt, max_tokens=200, stops=["</end>"])
        if new_relation:
            self.relationship = new_relation.strip()

class KnownActorManager:
    def __init__(self, owner_agh, context):
        self.owner = owner_agh
        self.context = context
        self.known_actors = {}

    def names(self):
        return [actor.name for actor in self.known_actors.values()]
    
    def actor_models(self):
        return self.known_actors.values()
    
    def add_actor_model(self, actor_name):
        actor_agh = self.context.get_actor_by_name(actor_name)
        if actor_agh and actor_agh.name not in self.known_actors:
            self.known_actors[actor_agh.name] = KnownActor(actor_agh, self)
        return self.known_actors[actor_name]

    def get_actor_model(self, actor_name, create_if_missing=False):
        if actor_name not in self.known_actors:
            if create_if_missing:
                print(f"{self.owner.name} creating model for {actor_name}")
                self.add_actor_model(actor_name)
            else:
                    return None
        return self.known_actors[actor_name] if actor_name in self.known_actors else None

    def set_all_actors_visible(self):
        for actor in self.known_actors:
            self.known_actors[actor.name].visible = True

    def set_all_actors_invisible(self):
        for actor in self.known_actors.values():
            actor.visible = False

    def update_all_relationships(self, all_texts):
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
            relationships[actor.name] = relationship + '\n\n' + transcript
        return relationships

    def format_relationships(self, include_transcript=False):
        relationships = self.get_known_relationships(include_transcript)
        return '\n\n'.join([f"{name}:\n {relationship}" for name, relationship in relationships.items()])