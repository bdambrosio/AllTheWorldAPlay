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

    def update_relationship(self, all_texts):
        """self is An instance of a model of another actor 
          updating beliefs about owner relationship to this actor"""
        char_memories = [text for text in all_texts if self.name.lower() in text.lower()]
        if char_memories is None or len(char_memories) == 0:
            return
        prompt = [UserMessage(content=f"""Analyze the relationship between these characters based on recent interactions.

Character: 
{self.manager.owner.character}

Other Characters: {self.actor.name}

Previous Relationship Status:
{self.relationship}

Recent Interactions:
{chr(10).join(f"- {text}" for text in char_memories)}

Describe their current relationship in a brief statement that captures:
1. The nature of their connection
2. Current emotional state
3. Any recent changes
4. Ongoing dynamics

Respond with a concise updated relationship description of up to 100 tokens, no additional text.
End with:
</end>
""")]

        new_relation = self.manager.owner.llm.ask({}, prompt, max_tokens=200, stops=["</end>"])
        if new_relation:
            self.relationship = new_relation.strip()

class KnownActorManager:
    def __init__(self, owner, context):
        self.owner = owner
        self.context = context
        self.known_actors = {}

    def names(self):
        return [actor.name for actor in self.known_actors.values()]
    
    def values(self):
        return self.known_actors.values()
    
    def add_actor_model(self, actor_name):
        actor = self.context.get_actor_by_name(actor_name)
        if actor and actor.name not in self.known_actors:
            self.known_actors[actor.name] = KnownActor(actor, self)

    def get_actor_model(self, actor_name, create_if_missing=False):
        if actor_name not in self.known_actors:
            if create_if_missing:
                self.add_actor_model(actor_name)
            else:
                return None
        return self.known_actors[actor_name]

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

    def get_known_relationships(self):
        relationships = {}
        for actor in self.known_actors.values():
            relationships[actor.name] = actor.relationship
        return relationships
