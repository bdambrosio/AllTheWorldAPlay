from __future__ import annotations
import json
import traceback
import random
from queue import Queue
from typing import TYPE_CHECKING
from sim import map
from utils import hash_utils, llm_api
from utils.Messages import UserMessage
import utils.xml_utils as xml
from datetime import datetime, timedelta
import utils.choice as choice
from typing import List
import asyncio

if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking

class Context():
    def __init__(self, characters, description, scenario_module, server_name=None):
        """Initialize a context with characters and world description
        
        Args:
            characters: List of Character objects
            description: Text description of the world/setting
            scenario_module: Module containing scenario types and rules
            server_name: Optional server name for image generation
        """
        self.characters = characters
        self.description = description
        self.scenario_module = scenario_module
        self.server_name = server_name
        # Initialize characters in world
        for character in characters:
            character.context = self
            # Additional character initialization as needed

        self.initial_state = description
        self.current_state = description
        self.actors: List[Character] = characters
        self.npcs = [] # created as needed
        self.map = map.WorldMap(60, 60, scenario_module)
        self.step = '0 hours'  # amount of time to step per scene update
        self.name = 'World'
        self.llm = llm_api.LLM(server_name)
        self.simulation_time = datetime.now()  # Starting time
        self.time_step = '0 hours'  # Amount to advance each step
        # Add new fields for UI independence
        self.state_listeners = []
        self.output_buffer = []
        self.widget_refs = {}  # Keep track of widget references for PyQt UI
        self.force_sense = False # force full sense for all actors
        self.message_queue = Queue()  # Queue for messages to be sent to the UI
        self.choice_response = asyncio.Queue()  # Queue for receiving choice responses from UI
        self.current_actor_index = 0  # Add this line to track position in actors list
        self.show = ''
        self.simulation_time = self.extract_simulation_time(description)
        for resource_id, resource in self.map.resource_registry.items():
            has_owner = self.check_resource_has_npc(resource)
            if has_owner:
                owner:Character = self.get_npc_by_name(resource['name']+'_owner', x=resource['location'][0], y=resource['location'][1], create_if_missing=True)
                resource['properties']['owner'] = owner.mapAgent

        
        for actor in self.actors + self.npcs:
            #place all actors in the world
            actor.set_context(self)
            actor.mapAgent = map.Agent(actor.init_x, actor.init_y, self.map, actor.name)
            # Initialize relationships with valid character names
            if hasattr(actor, 'narrative'):
                valid_names = [a.name for a in self.actors if a != actor]
        for actor in self.actors:
            actor.driveSignalManager.analyze_text(actor.character, actor.drives, self.simulation_time)
            actor.driveSignalManager.analyze_text(self.current_state, actor.drives, self.simulation_time)
            actor.look()
            actor.driveSignalManager.recluster() # recluster drive signals after actor initialization
            #actor.generate_goal_alternatives()
            #actor.generate_task_alternatives() # don't have focus task yet
            actor.wakeup = False

    def extract_simulation_time(self, situation):
        # Extract simulation time from situation description
        prompt = [UserMessage(content="""You are a simulation time extractor.
Your task is to extract the exact time of day and datefrom the following situation description:

<situation>
{{$situation}}
</situation>

Respond with the simulation time in a format that can be parsed by the datetime.parse function, assuming reasonable defaults for missing components based on the context of the situation.
Respond with two lines in exactly this format:
TIME: HH:MM AM/PM
DATE: Month DD

If either piece of information is not explicitly stated in the text, make a reasonable inference based on context clues (e.g., "early morning" suggests morning time, "soft light" might suggest spring or summer). 
If absolutely no information is available for either field, use "unknown" for that field.
""")]
        response = self.llm.ask({"situation": situation}, prompt, temp=0.5, max_tokens=10)
        lines = response.strip().split('\n')
        time_str = None
        date_str = None
        
        # Extract time and date from response
        for line in lines:
            if line.startswith('TIME:'):
                time_str = line.replace('TIME:', '').strip()
            elif line.startswith('DATE:'):
                date_str = line.replace('DATE:', '').strip()
    
        # Handle unknown values
        if not time_str or time_str == 'unknown':
            time_str = '12:00 PM'  # Default to noon
        if not date_str or date_str == 'unknown':
            date_str = 'January 1'  # Default to January 1
        
        # Combine date and time strings
        datetime_str = f"{date_str} {time_str}"
    
        # Use current year as default
        current_year = datetime.now().year
    
        try:
            # Parse the combined string
            dt = datetime.strptime(f"{datetime_str} {current_year}", "%B %d %I:%M %p %Y")
            return dt
        except ValueError as e:
            print(f"Error parsing datetime: {e}")
            return None


    def to_json(self):
        return {
            'name': self.name,
            'initial_state': self.initial_state,
            'current_state': self.current_state,
            'current_state': self.current_state,
            'actors': [actor.to_shallow_json() for actor in self.actors],
            'npcs': [npc.to_shallow_json() for npc in self.npcs],
            #'map': self.map.to_json(),
            'server_name': self.server_name,
            'simulation_time': self.simulation_time.isoformat(),
            'time_step': self.time_step,
            'step': self.step,
            'current_actor_index': self.current_actor_index,
        }

    def set_llm(self, llm):
        self.llm = llm
        for actor in self.actors:
            actor.set_llm(llm)
            actor.last_sense_time = self.simulation_time

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
            state = '. '.join(self.current_state.split('.')[:3])
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
        """Helper to find actor by name"""
        for actor in self.actors:
            if actor.name == name or actor.name in name:
                return actor
        return None

    def plausible_npc(self, name):
        """Check if a name is plausible for an NPC"""
        return name.lower() in ['viewer','father', 'mother', 'sister', 'brother', 'husband', 'wife', 'friend', 'neighbor',  'stranger']

    def get_npc_by_name(self, name, x=20, y=20, create_if_missing=False):
        """Helper to find NPC by name"""
        for actor in self.npcs:
            if actor.name == name or actor.name in name:
                return actor
        # create a new NPC
        if create_if_missing: #and self.plausible_npc(name):
            from sim.agh import Character
            npc = Character(name, character_description=f'{name} is a non-player character', server_name=self.llm.server_name)
            npc.x = x
            npc.y = y
            npc.set_context(self)
            npc.llm = self.llm
            self.npcs.append(npc)
            return npc
        return None

    def resolve_reference(self, actor: Character, reference, create_if_missing=False):
        """Resolve a reference to an actor. This is assumed to be a simple reference, not a complete phrase. e.g. John, Father, old man Henry, etc."""
        "If you have a complete sentence, use Character.say_target to indentify the target reference."
        if reference is None or reference == '' or reference.lower() == 'none':
            return None
        referenced_actor = self.get_actor_by_name(reference)
        if referenced_actor is None:
            referenced_actor = self.get_npc_by_name(reference, x=actor.mapAgent.x, y=actor.mapAgent.y, create_if_missing=create_if_missing)
        return referenced_actor

    
    def world_updates_from_act_consequences(self, consequences):
        """ This needs overhaul to integrate and maintain consistency with world map."""
        prompt = [UserMessage(content="""Given the following immediate effects of an action on the environment, generate zero to two concise sentences to add to the following state description.
It may be there are no significant updates to report.
Limit your changes to the consequences for elements in the existing state or new elements added to the state.
Most important are those consequences that might activate or inactive tasks or actions by actors.

<actionEffects>
{{$consequences}}
</actionEffects>

<environment>
{{$state}}
</environment>

Your response should be concise, and only include only statements about changes to the existing Environment.
Do NOT repeat elements of the existing Environment, respond only with significant changes.
Do NOT repeat as an update items already present at the end of the Environment statement.
Your updates should be dispassionate. 
Use the following XML format:
<updates>
concise statement(s) of significant changes to Environment, if any, one per line.
</updates>

Include ONLY the concise updated state description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
End your response with:
</end>""")
                  ]

        response = self.llm.ask({"consequences": consequences, "state": self.current_state},
                                prompt, temp=0.5, stops=['</end>'], max_tokens=60)
        updates = xml.find('<updates>', response)
        if updates is not None:
            self.current_state += '\n' + updates.strip()
        else:
            updates = ''
        return updates

    def character_updates_from_act_consequences(self, consequences, actor):
        """ This needs overhaul to integrate and maintain consistency with world map."""
        prompt = [UserMessage(content="""Given the following immediate effects of an action on the environment, generate zero to two concise sentences to add to the actor's state description.
It may be there are no significant updates to report.
Limit your changes to the consequences for elements in the existing state or new elements added to the state.
Most important are those consequences that might activate or inactive tasks or actions by actors.

<actionEffects>
{{$consequences}}
</actionEffects>

<environment>
{{$state}}
</environment>
                              
<actor state>
{{$actor_state}}
</actor state>

Your response should be concise, and only include only statements about changes to the actor's state.
Do NOT repeat elements of the existing actor's state, respond only with significant changes.
Do NOT repeat as an update items already present at the end of the actor state statement.
Your updates should be dispassionate. 
Use the following XML format:
<updates>
concise statement(s) of significant changes to Environment, if any, one per line.
</updates>

Include ONLY the concise updated state description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
End your response with:
</end>""")
                  ]

        response = self.llm.ask({"consequences": consequences, "actor_state": actor.narrative.get_summary('medium'), "state": self.current_state},
                                prompt, temp=0.5, stops=['</end>'], max_tokens=60)
        updates = xml.find('<updates>', response)
        if updates is not None:
            self.current_state += '\n' + updates.strip()
        else:
            updates = ''
        return updates
    
    def do(self, actor, action):
        """ This is the world determining the effects of an actor action"""
        prompt = [UserMessage(content="""You are simulating a dynamic world. 
Your task is to determine the result of {{$name}} performing the following action:

<action>
{{$action}}
</action>

in the current situation: 

<situation>
{{$state}}
</situation>

given {{$name}} local map is:

<localMap>
{{$local_map}}
</localMap>

And character {{$name}} is:

<character>
{{$character}}
</character>

with current situation:

<situation>
{{$narrative}}
</situation>

Respond with the observable result.
Respond ONLY with the observable immediate effects of the above Action on the environment and characters.
Be careful to include any effects on the state of {{$name}} in your response.
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
<end/>
""")]
        history = self.history()
        local_map = actor.mapAgent.get_detailed_visibility_description()
        local_map = xml.format_xml(local_map)
        consequences = self.llm.ask({"name": actor.name, "action": action, "local_map": local_map,
                                     "state": self.current_state, "character": actor.character, "narrative":  actor.narrative.get_summary('medium')}, prompt, temp=0.7, stops=['<end/>'], max_tokens=300)

        if consequences.endswith('<'):
            consequences = consequences[:-1]
        world_updates = self.world_updates_from_act_consequences(consequences)
        character_updates = self.character_updates_from_act_consequences(consequences, actor)   
        print(f'\nContext Do consequences:\n {consequences}')
        print(f' Context Do world_update:\n {world_updates}\n')
        return consequences, world_updates, character_updates


    async def update(self, local_only=False):

        history = self.history()

        event = ""
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

        prompt = [UserMessage(content="""You are a dynamic world. Your task is to update the environment description. 
Include day/night cycles and weather patterns. It is now {{$time}}.
Update location and other physical situation characteristics as indicated in the History.
Your response should be concise, and only include only an update of the physical situation.
        """ + event + """
Your situation description should be dispassionate, 
and should begin with a brief description of the current physical space suitable for a text-to-image generator. 
The situation previously was:

<previousSituation>
{{$situation}}
</previousSituation> 

In the interim, the characters in the world had the following interactions:

<history>
{{$history}}
</history>

All actions performed by actors since the last situation update are including in the above History.
Do not include in your updated situation any actions not listed above.
Include characters in your response only with respect to the effects of their above actions on the situation.

Respond using the following XML format:

<situation>
Sentence describing physical space, suitable for image generator.
Updated State description of about 300 words
</situation>

Respond with an updated world state description reflecting a time passage of {{$step}} and the environmental changes that have occurred.
Include ONLY the concise updated situation description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
.
Limit your total response to about 270 words
Ensure your response is surrounded with <situation> and </situation> tags as shown above.
End your response with:
<end/>""")]

        response = self.llm.ask({"situation": self.current_state, 'history': history, 'step': self.step, 'time': self.simulation_time.isoformat()}, prompt,
                                temp=0.6, stops=['<end/>'], max_tokens=450)
        new_situation = xml.find('<situation>', response)       
        # Debug prints
        self.message_queue.put({'name':self.name, 'text':f'\n\n-----scene----- {self.simulation_time.isoformat()}\n'})
        await asyncio.sleep(0.1)
         
        if new_situation is None:
            return
        self.current_state = new_situation
        self.show = new_situation
        self.message_queue.put({'name':self.name, 'text':self.show})
        self.show = '' # has been added to message queue!
        await asyncio.sleep(0.1)
        if local_only:
            return

        updates = self.world_updates_from_act_consequences(new_situation)
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
            'show': ' \n\n'+self.current_state,
            'image': self.image('worldsim.png')
        }

    def format_tasks(self, tasks, labels):
        task_list = []
        for task, label in zip(tasks, labels):
            task_text = f'{label} - {hash_utils.find("name", task)} ({hash_utils.find("description", task)}), {hash_utils.find("reason", task)}; ' 
            task_dscp= task_text + f' Needs: {hash_utils.find("needs", task)}; committed: {hash_utils.find("committed", task)}'
            task_actor_names = hash_utils.find('actors', task).split(',')
            task_memories = set()  
            for actor_name in task_actor_names:
                actor = self.get_actor_by_name(actor_name.strip())
                if actor is None:
                    print(f'\n** Context format_tasks: Actor {actor_name.strip()} not found**')
                    continue
                else:
                    memories = actor.memory_retrieval.get_by_text(
                        memory=actor.structured_memory,
                        search_text=task_text,
                        threshold=0.1,
                        max_results=5
                    )
                # More explicit memory text collection
                for memory in memories:
                    task_memories.add(memory.text)  # Add each memory text individually

            task_memories = '\n\t'.join(task_memories)
            task_list.append(task_dscp + '\n\t' + task_memories)
        return '\n\n'.join(task_list)
    

    def map_actor(self, actor):
        mapped_actor = f"""<actor>
    <name>{actor.name}</name>
    <character>
        {actor.character.replace('\n', ' ')}
    </character>
    <goals>
    {'\n'+'\n        '.join([actor.goals[goal]['name'] for goal in actor.goals])}
    </goals>
    <tasks>
        {'\n        '.join([hash_utils.find('name', task) for task in actor.tasks])}   
    </tasks>
    <memories>
        {'\n        '.join([memory.text for memory in actor.structured_memory.get_recent(6)])}
    </memories>
</actor>"""
        return mapped_actor

    def map_actors(self):
        mapped_actors = '\n'.join([self.map_actor(actor) for actor in self.actors])
        return mapped_actors

    def choose(self, task_choices):
        if len(task_choices) == 1:
            return task_choices[0]
        prompt = [UserMessage(content="""The task is to order the execution of a set of task options, listed under <tasks> below.
Your current situation is:

<situation>
{{$situation}}
</situation>

the actors you are managing include:
                              
<actors>
{{$actors}}
</actors>


Your task options are provided in the labelled list below.
Labels are Greek letters chosen from {Alpha, Beta, Gamma, Delta, Epsilon, etc}. Not all letters are used.

<tasks>
{{$tasks}}
</tasks>

Please:
1. Reason through the importance, urgency, dependencies, and strengths and weaknesses of the task options
2. Order committed tasks early, especially if they are important and urgent or needed by other committed tasks.
3. Reason carefully about dependencies among task options, timing of task options, and the order of execution.
4. Compare them against your current goals and drives with respect to your memory and perception of your current situation
5. Reason in keeping with your character. 
6. Assign an execution order to each task option, ranging from 1 (execute as soon as possible) to {{$num_options}} (execute last), 

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#task\n#label label of chosen task\n#order execution order (an int, 1-{{$num_options}})\n##
#task\n#label ...\n#order ...\n##
...
##

Review to ensure the assigned execution order is consistent with the task option dependencies, urgency, and importance.
Respond only with the above hash-formatted information, 
    instantiated with the selected task label from the Task list and execution order determined by your reasoning.
Do not include any introductory, explanatory, or discursive text, 
End your response with:
</end>
"""
                              )]

        labels = ['Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega']
        random.shuffle(labels)
        # Change from dict to set for collecting memory texts
                
        print(f'\nWorld Choose: {'\n - '+'\n - '.join([hash_utils.find('name', task) for task in task_choices])}')
        response = self.llm.ask({
            "situation": self.current_state,
            "actors": self.map_actors(), 
            "tasks": self.format_tasks(task_choices, labels[:len(task_choices)]),
            "num_options": len(task_choices)
        }, prompt, temp=0.0, stops=['</end>'], max_tokens=150)
        # print(f'sense\n{response}\n')
        index = -1
        ordering = hash_utils.findall('task', response)
        min_order = 1000
        task_to_execute = None
        best_label = None
        ordering = hash_utils.findall('task', response)
        pairs = [(hash_utils.find('label', item).strip(), hash_utils.find('order', item).strip()) for item in ordering]
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        label_choice = choice.exp_weighted_choice(sorted_pairs, 0.75)
        task_to_execute = task_choices[labels.index(label_choice[0])]
        print(f'  Chosen label: {label_choice} task: {task_to_execute.replace('\n', '; ')}')
        return task_to_execute
        
    def next_act(self):
        """Pick next actor in round-robin fashion and return it"""
        if not self.actors:
            return None
        
        # first see if any committed goals without needs
        committed_tasks = []
        for actor in self.actors:
            for task in actor.tasks:
                # if task has a need, and the need is in the actor's tasks(ie, not yet complete), skip
                if hash_utils.find('committed', task)=='True':
                    #do we have to check for a needs task across all actors? O rmaybe joint tasks sb on all participants tasks list?
                    if hash_utils.find('needs', task)=="" or not actor.get_task(hash_utils.find('needs', task)):
                        committed_tasks.append(task)
        if len(committed_tasks) == 0:
            # oh oh, something went wrong, see if any committed tasks without needs
            for actor in self.actors:
                for task in actor.tasks:
                    if hash_utils.find('committed', task)=='True':
                        committed_tasks.append(task)
        if committed_tasks:
            next_task = self.choose(committed_tasks)
            print(f'\nWorld Next Act: {next_task.replace('\n', '; ')}')
            actor_names = hash_utils.find('actors', next_task)
            if actor_names:
                actor_names = actor_names.strip().split(', ')
                actors = [self.get_actor_by_name(actor_name.strip()) for actor_name in actor_names]
                if actors:
                    for actor in actors:
                        actor.next_task = next_task
                    return next_task, actors #agh.acts will need to handle multiple actors
                
        # else Get next actor
        actor = self.actors[self.current_actor_index]
        # Update index for next time
        self.current_actor_index = (self.current_actor_index + 1) % len(self.actors)
        
        actor.next_task=None
        return None, [actor]

    def check_resource_has_npc(self, resource):
        """Check if a resource type should have an NPC"""
        for allocation in self.map._resource_rules['allocations']:
            if allocation['resource_type'] == resource['type']:
                return allocation.get('has_npc', False)
        return False
