import json
import traceback
import random
from sim import map
from utils import llm_api
from utils.Messages import UserMessage
import utils.xml_utils as xml
from datetime import datetime, timedelta


class Context():
    def __init__(self, actors, situation, step='4 hours', mapContext=True):
        self.initial_state = situation
        self.current_state = situation
        self.actors = actors
        self.map = map.WorldMap(60, 60)
        for actor in self.actors:
            #place all actors in the world
            actor.context = self
            if mapContext:
                actor.mapAgent = map.Agent(30, 30, self.map, actor.name)
            for actor in self.actors:
                if actor.mapAgent != None:
                    actor.look() # provide initial local view
            # Initialize relationships with valid character names
            if hasattr(actor, 'narrative'):
                valid_names = [a.name for a in self.actors if a != actor]
                actor.narrative.initialize_relationships(valid_names)
        self.step = step  # amount of time to step per scene update
        self.name = 'World'
        self.llm = None
        self.simulation_time = datetime.now()  # Starting time
        self.time_step = step  # Amount to advance each step
        # Add new fields for UI independence
        self.state_listeners = []
        self.output_buffer = []
        self.widget_refs = {}  # Keep track of widget references for PyQt UI

    def set_llm(self, llm):
        self.llm = llm
        for actor in self.actors:
            actor.set_llm(llm)

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
        """Get combined history from all actors using structured memory"""
        history = []
        
        for actor in self.actors:
            # Get recent memories from structured memory
            recent_memories = actor.structured_memory.get_recent(5)
            for memory in recent_memories:
                history.append(f"{actor.name}: {memory.text}")
                
        return '\n'.join(history) if history else ""

    def generate_image_description(self):
        return "wide-view photorealistic style. "+self.current_state
        
        
    def image(self, filepath, image_generator='tti_serve'):
        try:
            state = '. '.join(self.current_state.split('.')[:2])
            characters = '\n' + '.\n'.join(
                [entity.name + ' is ' + entity.character.split('.')[0][8:] for entity in self.actors])
            prompt = state + characters
            # print(f'calling generate_dalle_image\n{prompt}')
            if image_generator == 'tti_serve':
                filepath = llm_api.generate_image(self.llm, f"""wide-view photorealistic style. {prompt}""", size='512x512', filepath=filepath)
            else:
                filepath = llm_api.generate_dalle_image(f"""wide-view photorealistic style. {prompt}""", size='512x512',
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
        """ This needs overhaul to integrate and maintain consistency with world map."""
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
Use the following XML format:
<Updates>
concise statement(s) of significant changes to Environment, if any, one per line.
</Updates>

Include ONLY the concise updated state description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
End your response with:
</End>""")
                  ]

        response = self.llm.ask({"consequences": consequences, "state": self.current_state},
                                prompt, temp=0.5, stops=['</End>'], max_tokens=60)
        updates = xml.find('<Updates>', response)
        if updates is not None:
            self.current_state += '\n' + updates
        else:
            updates = ''
        return updates

    def do(self, actor, action):
        """ This is the world determining the effects of an actor action"""
        prompt = [UserMessage(content="""You are simulating a dynamic world. 
Your task is to determine the result of {{$name}} performing the following action:

<Action>
{{$action}}
</Action>

in the current situation: 

<Situation>
{{$state}}
</Situation>

given {{$name}} local map is:

<LocalMap>
{{$local_map}}
</LocalMap

Respond with the observable result.
Respond ONLY with the observable immediate effects of the above Action on the environment and characters.
It is usually not necessary or desirable to repeat the above action statement in your response.
Observable result must be consistent with information provided in the LocalMap.
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
<STOP>
""")]
        history = self.history()
        local_map = actor.mapAgent.get_detailed_visibility_description()
        local_map = xml.format_xml(local_map)
        consequences = self.llm.ask({"name": actor.name, "action": action, "local_map": local_map,
                                     "state": self.current_state}, prompt, temp=0.7, stops=['<STOP>'], max_tokens=300)

        if consequences.endswith('<'):
            consequences = consequences[:-1]
        world_updates = self.world_updates_from_act_consequences(consequences)
        print(f'\nContext Do consequences:\n {consequences}')
        print(f' Context Do world_update:\n {world_updates}\n')
        return consequences, world_updates

    def senses(self, sense_data='', ui_task_queue=None):
        """ This is where the world advances the timeline in the scenario """
        # Debug prints
        for actor in self.actors:
            print(f"Actor {actor.name} type: {type(actor)}")
            print(f"Actor {actor.name} has cognitive_processor: {hasattr(actor, 'cognitive_processor')}")

        # since at the moment there are only two chars, each with complete dialog, we can take it from either.
        # not really true, since each is ignorant of the others thoughts
        history = self.history()
        if self.step == 'static': # static world, nothing changes!
            prompt = [UserMessage(content="""You are a static world. Your task is to update the environment description. 
            Update location and other physical situation characteristics as indicated in the History
            Your response should be concise, and only include only an update of the physical situation.
            Introduce new elements only when specifically appearing in History
            Do NOT add description not present in the PreviousState or History below.
            Your situation description should be dispassionate, 
            and should begin with a brief description of the current physical space suitable for a text-to-image generator. 
            The previous state was:

            <PreviousState>
            {{$situation}}
            </PreviousState> 

            In the interim, the characters in the world had the following interactions:

            <History>
            {{$history}}
            </History>

            All actions performed by actors since the last situation update are including in the above History.
            Do not include in your updated situation any actions not listed above.
            Do not include any comments on actors or their actions. Only report the resulting world state

            Respond using the following XML format:

            <Situation>
            Sentence describing physical space, suitable for image generator,
            Updated State description of about 300 words
            </Situation>

            Respond with an updated world state description reflecting a time passage of {{$step}} and the environmental changes that have occurred.
            Include ONLY the concise updated situation description in your response. 
            Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
            .
            Limit your total response to about 330 words
            End your response with:
            END""")]

            response = self.llm.ask({"situation": self.current_state, 'history': history, 'step': self.step}, prompt,
                                    temp=0.6, stops=['END'], max_tokens=500)
            new_situation = xml.find('<Situation>', response)
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
            self.show = '\n-----scene-----\n' + self.current_state
            return

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

        prompt = [UserMessage(content="""You are a dynamic world. Your task is to update the environment description. 
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
Sentence describing physical space, suitable for image generator.
Updated State description of about 300 words
</Situation>

Respond with an updated world state description reflecting a time passage of {{$step}} and the environmental changes that have occurred.
Include ONLY the concise updated situation description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
.
Limit your total response to about 330 words
Ensure your response is surrounded with <Situation> and </Situation> tags as shown above.
End your response with:
<End/>""")]

        response = self.llm.ask({"situation": self.current_state, 'history': history, 'step': self.step}, prompt,
                                temp=0.6, stops=['<End/>'], max_tokens=700)
        new_situation = xml.find('<Situation>', response)
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

    def advance_time(self):
        """Advance simulation clock by time_step"""
        if isinstance(self.time_step, str):
            # Parse "4 hours" etc
            amount, unit = self.time_step.split()
            delta = timedelta(**{unit: int(amount)})
        else:
            delta = self.time_step
        self.simulation_time += delta
        return self.simulation_time

    def add_state_listener(self, listener):
        """Allow UIs to register for updates"""
        self.state_listeners.append(listener)

    def _notify_listeners(self, update_type, data=None):
        """Notify all listeners of state changes"""
        for listener in self.state_listeners:
            listener(update_type, data)

    # Modify existing display method
    def display_output(self, text):
        """UI-independent output handling"""
        self.output_buffer.append({
            'text': text,
            'timestamp': self.simulation_time
        })
        self._notify_listeners('output', {'text': text})

    # Modify existing display update
    def update_display(self):
        """UI-independent state update"""
        state = {
            'characters': self.characters,
            'simulation_time': self.simulation_time,
            'running': self.running,
            'paused': self.paused
        }
        self._notify_listeners('state_update', state)

    def set_widget(self, entity, widget):
        """Maintain widget references for PyQt UI"""
        self.widget_refs[entity] = widget
        entity.widget = widget

    def get_widget(self, entity):
        """Get widget reference for entity"""
        return self.widget_refs.get(entity)

    def to_json(self):
        """Return JSON-serializable dict of context state"""
        return {
            'show': self.current_state,
            'image': self.image('worldsim.png')
        }
