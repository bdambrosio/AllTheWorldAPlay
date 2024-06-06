import json
import traceback
import random
import llm_api
from utils.Messages import UserMessage
import utils.xml_utils as xml

class Context():
    def __init__(self, actors, situation, step='4 hours'):
        self.initial_state = situation
        self.current_state = situation
        self.actors = actors
        for actor in self.actors:
            actor.context = self
        self.step = step  # amount of time to step per scene update
        self.name = 'World'
        self.llm = None

    def load(self, dir):
        try:
            with open(dir / 'scenario.json', 'r') as s:
                scenario = json.load(s)
                if 'name' in scenario.keys():
                    print(f" {scenario['name']} found")
                self.initial_state = scenario['initial_state']
                self.current_state = scenario['current_state']
            for actor in self.actors:
                actor.load(dir)
        except FileNotFoundError as e:
            print(str(e))
        pass

    def save(self, dir, name):
        try:
            # first save world state
            with open(dir / 'scenario.json', 'w') as s:
                scenario = {'name': name, 'initial_state': self.initial_state, 'current_state': self.current_state}
                json.dump(scenario, s, indent=4)

            # now save actor states
            for actor in self.actors:
                actor.save(dir / str(actor.name + '.json'))

        except FileNotFoundError as e:
            print(str(e))

    def history(self):
        try:
            if self.actors is None or self.actors[0] is None:
                return ''
            if self.actors[0].history is None:
                return ''
            history = '\n\n'.join(self.actors[0].history)
            hs = history.split('\n')
            hs_renamed = [self.actors[0].name + s[3:] if s.startswith('You') else s for s in hs]
            history = '\n'.join(hs_renamed)
        except Exception as e:
            traceback.print_exc()
        return history

    def image(self, filepath, image_generator='tti_serve'):
        try:
            state = '. '.join(self.current_state.split('.')[:2])
            characters = '\n' + '.\n'.join(
                [entity.name + ' is ' + entity.character.split('.')[0][8:] for entity in self.actors])
            prompt = state + characters
            # print(f'calling generate_dalle_image\n{prompt}')
            if image_generator == 'tti_serve':
                llm_api.generate_image(f"""wide-view photorealistic. {prompt}""", size='512x512', filepath=filepath)
            else:
                llm_api.generate_dalle_image(f"""wide-view photorealistic. {prompt}""", size='512x512',
                                             filepath=filepath)
        except Exception as e:
            traceback.print_exc()
        return filepath

    def get_actor_by_name(self, name):
        for actor in self.actors:
            if actor.name == name:
                return actor
        return None

    def world_updates_from_act_consequences(self, consequences):
        prompt = [UserMessage(content="""Given the following immediate effects of an action on the environment, generate zero to two concise sentences to add to the following state description.
It may be there are no significant updates to report.
Limit your changes to the consequences for elements in the existing state or new elements added to the state.
Most important are those consequences that might activate or inactive tasks or intentions by actors.

<ActionEffects>
{{$consequences}}
</ActionEffects>

<Environment>
{{$state}}
</Environment>

Your response should be concise, and only include only statements about changes to the existing Environment.
Do NOT repeat elements of the existing Environment, respond only with significant changes.
Do NOT repeat as an update items already present at the end of the Environment statement.
Your updates should be dispassionate. 

<Updates>
concise statement of significant changes to Environment, if any.
</Updates>

Include ONLY the concise updated state description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
End your response with:
<END>""")
                  ]

        response = self.llm.ask({"consequences": consequences, "state": self.current_state},
                                prompt, temp=0.5, stops=['<END>'], max_tokens=60)
        updates = xml.find('<Updates>', response)
        if updates is not None:
            self.current_state += '\n' + updates
        else:
            updates = ''
        return updates

    def do(self, actor, action):
        prompt = [UserMessage(content="""You are simulating a dynamic world. 
Your task is to determine the result of {{$name}} performing the following action:

<Action>
{{$action}}
</Action>

in the current situation: 

<Situation>
{{$state}}
</Situation>

given {{$name}} basic drives are:

<Drives>
{{$drives}}
</Drives> 

Respond with the observable result.
Respond ONLY with the observable immediate effects of the above Action on the environment and characters.
It is usually not necessary or desirable to repeat the above action statement in your response.
Format your response as one or more simple declarative sentences.
Include in your response:
- changes in the physical environment, e.g. 'the door opens', 'the rock falls',...
- sensory inputs, e.g. {{$name}} 'sees ...', 'hears ...', 
- changes in {{$name}}'s possessions (e.g. {{$name}} 'gains ... ',  'loses ... ', / ... / )
- changes in {{$name})'s or other actor's state (e.g., {{$name}} 'becomes tired' / 'is injured' / ... /).
- specific information acquired by {{$name}}. State the actual knowledge acquired, not merely a description of the type or form.
Do NOT extend the scenario with any follow on actions or effects.
Be extremely terse when reporting character emotional state, only report the most significant emotional state changes.
Be concise!
Do not include any Introductory, explanatory, or discursive text.
End your response with:
<END>
""")]
        history = self.history()
        consequences = self.llm.ask({"name": actor.name, "action": action, "drives": actor.drives,
                                     "state": self.current_state}, prompt, temp=0.7, stops=['END'], max_tokens=300)

        if consequences.endswith('<'):
            consequences = consequences[:-1]
        world_updates = self.world_updates_from_act_consequences(consequences)
        print(f'\nContext Do consequences:\n {consequences}')
        print(f' Context Do world_update:\n {world_updates}\n')
        return consequences, world_updates

    def senses(self, sense_data='', ui_task_queue=None):
        # since at the moment there are only two chars, each with complete dialog, we can take it from either.
        # not really true, since each is ignorant of the others thoughts
        history = self.history()
        if random.randint(1, 7) == 1:
            event = """
Include a event occurence consistent with the PreviousState below, such as appearance of a new object, 
natural event such as weather (if outdoors), communication event such as email arrival (if devices available to receive such), etc.

===Examples===
PreviousState:
Apartment

History:
worry about replacement

Event:
Annie receives an email directed to her personally from an unknown agent.

-----

PreviousState:
Open forest 

History:
Safety, hunger

Event:
Joe finds a sharp object that can be used as a tool.
-----

===End Examples===

"""
        else:
            event = ""

        prompt = [UserMessage(content="""You are a dynamic world. Your task is to update the environmemt description. 
Include day/night cycles and weather patterns. 
Update location and other physical situation characteristics as indicated in the History.
Your response should be concise, and only include only an update of the physical situation.
        """ + event + """
Your situation description should be dispassionate, 
and should begin with a brief description of the current physical space suitable for a text-to-image generator. 
The situation as of {{$step}} ago was:

<PreviousSituation>
{{$situation}}
</PreviousSituation> 

In the interim, the characters in the world had the following interactions:

<History>
{{$history}}
</History>

All actions performed by actors since the last situation update are including in the above History.
Do not include in your updated situation any actions not listed above.
Include characters in your response only with respect to the effects of their above actions on the situation.

Respond using the following XML format:

<Situation>
Sentence describing physical space, suitable for image generator,
Updated State description of about 300 words
</Situation>

Respond with an updated situation description reflecting a time passage of {{$step}} and the environmental changes that have occurred.
Include ONLY the concise updated situation description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
.
Limit your total response to about 330 words
End your response with:
END""")]

        response = self.llm.ask({"situation": self.current_state, 'history': history, 'step': self.step}, prompt,
                                temp=0.6, stops=['END'], max_tokens=500)
        new_situation = find('<Situation>', response)
        if new_situation is not None:
            updates = self.world_updates_from_act_consequences(new_situation)
            self.current_state = new_situation
            self.show = '\n-----scene-----\n' + new_situation
        # self.current_state += '\n'+updates
        print(f'World updates:\n{updates}')
        for actor in self.actors:
            actor.add_to_history(f"you notice {updates}\n")
            actor.forward(self.step)  # forward three hours and update history, etc
        return response
