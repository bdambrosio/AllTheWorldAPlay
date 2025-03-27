from __future__ import annotations
import json
import traceback
import random
from queue import Queue
from typing import TYPE_CHECKING
from sim import map
from src.sim.referenceManager import ReferenceManager
from utils import hash_utils, llm_api
from utils.Messages import UserMessage, SystemMessage
import utils.xml_utils as xml
from datetime import datetime, timedelta
import utils.choice as choice
from typing import List
import asyncio
from sim.prompt import ask as default_ask
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
        self.transcript = [] #message queue history
        self.choice_response = asyncio.Queue()  # Queue for receiving choice responses from UI
        self.current_actor_index = 0  # Add this line to track position in actors list
        self.show = ''
        self.simulation_time = self.extract_simulation_time(description)
        self.reference_manager = ReferenceManager(self, self.llm)
        for resource_id, resource in self.map.resource_registry.items():
            has_owner = self.check_resource_has_npc(resource)
            if has_owner:
                owner:Character = self.get_npc_by_name(resource['name']+'_owner', description=f'{resource["name"]}_owner owns {resource["name"]} ', x=resource['location'][0], y=resource['location'][1], create_if_missing=True)
                resource['properties']['owner'] = owner.mapAgent
                self.reference_manager.declare_relationship(resource['name'], 'owned by', owner.name, 'owner of')

        self.last_consequences = '' # for world updates from recent acts
        self.last_updates = '' # for world updates from recent acts
        self.last_update_time = self.simulation_time
        
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
            return datetime.now()


    def to_save_json(self):
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
            if actor.name == name:
                return actor
        return None

    def plausible_npc(self, name):
        """Check if a name is plausible for an NPC"""
        return name.lower() in ['Viewer','father', 'mother', 'sister', 'brother', 'husband', 'wife', 'friend', 'neighbor',  'stranger']

    def get_npc_by_name(self, name, description=None, x=20, y=20, create_if_missing=False):
        """Helper to find NPC by name"""
        name = name.strip().capitalize()
        for actor in self.npcs:
            if actor.name == name:
                return actor
        # create a new NPC
        if create_if_missing: #and self.plausible_npc(name):
            from sim.agh import Character
            npc = Character(name, character_description=description if description else f'{name} is a non-player character', init_x=x, init_y=y, server_name=self.llm.server_name)
            npc.x = x
            npc.y = y
            npc.set_context(self)
            npc.llm = self.llm
            map_agent = self.map.get_agent(name)
            if map_agent is not None:
                npc.mapAgent = map_agent
            else:
                npc.mapAgent = map.Agent(x, y, self.map, npc.name)
            self.npcs.append(npc)
            return npc
        return None

    def get_actor_or_npc_by_name(self, name):
        """Helper to find actor or NPC by name"""
        for actor in self.actors:
            if actor.name == name:
                return actor
        for npc in self.npcs:
            if npc.name == name:
                return npc
        return None # not found

    def resolve_character(self, reference_text):
        """
        Resolve a reference to a character from either actors or NPCs
        
        Args:
            speaker: Character making the reference
            reference_text: Text to resolve into a character reference
            
        Returns:
            tuple: (character, canonical_name) or (None, None) if unresolved
        """
        # Normalize reference text
        reference_text = reference_text.strip().capitalize()
        
        # Check active actors first
        for actor in self.actors:
            if actor.name == reference_text:
                return (actor, reference_text)
            
        # Then check NPCs
        for npc in self.npcs:
            if npc.name == reference_text:
                return (npc, reference_text)
        
        canonical_name = self.reference_manager.resolve_reference_with_llm(reference_text)
        if canonical_name:
            return (self.get_actor_or_npc_by_name(canonical_name), reference_text)
        
        return (None, None)

    
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
        self.last_consequences = consequences
        character_updates = self.character_updates_from_act_consequences(consequences, actor)   
        self.last_updates = character_updates
        print(f'\nContext Do consequences:\n {consequences}')
        print(f' Context Do world_update:\n {world_updates}\n')
        return consequences, world_updates, character_updates


    async def update(self, local_only=False):

        history = self.history()

        event = ""
        if not local_only and random.randint(1, 7) == 1:
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
Updated State description of about 200 words
</situation>

Respond with an updated world state description of about 200 words reflecting the current time and the environmental changes that have occurred.
Include ONLY the updated situation description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
.
Limit your total response to about 200 words
Ensure your response is surrounded with <situation> and </situation> tags as shown above.
End your response with:
<end/>""")]

        response = self.llm.ask({"situation": self.current_state, 'history': history, 'step': self.step, 'time': self.simulation_time.isoformat()}, prompt,
                                temp=0.6, stops=['<end/>'], max_tokens=270)
        new_situation = xml.find('<situation>', response)       
        # Debug prints
        if not local_only or self.simulation_time - self.last_update_time > timedelta(hours=3):
            self.message_queue.put({'name':self.name, 'text':f'\n\n-----scene----- {self.simulation_time.isoformat()}\n'})
            self.transcript.append(f'\n\n-----scene----- {self.simulation_time.isoformat()}\n')
            await asyncio.sleep(0.1)
         
        if new_situation is None:
            return
        self.current_state = new_situation
        if local_only or self.simulation_time - self.last_update_time > timedelta(hours=3):
            self.last_update_time = self.simulation_time
            return
        self.show = new_situation
        self.message_queue.put({'name':self.name, 'text':self.show})
        self.transcript.append(f'{self.show}')
        self.show = '' # has been added to message queue!
        await asyncio.sleep(0.1)

        updates = self.world_updates_from_act_consequences(new_situation)
        # self.current_state += '\n'+updates
        print(f'World updates:\n{updates}')
        for actor in self.actors:
            actor.add_to_history(f"you notice {updates}\n")
            #actor.forward(self.step)  # forward three hours and update history, etc
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


    def check_resource_has_npc(self, resource):
        """Check if a resource type should have an NPC"""
        for allocation in self.map._resource_rules['allocations']:
            if allocation['resource_type'] == resource['type']:
                return allocation.get('has_npc', False)
        return False

    def map_actor(self, actor):
        mapped_actor = f"""{actor.name}: {actor.focus_goal.to_string() if actor.focus_goal else ""}\n   {actor.focus_task.peek().to_string() if actor.focus_task.peek() else ""}\n  {actor.focus_action.to_string() if actor.focus_action else ""}"""
        return mapped_actor+'\n  Remaining tasks:\n    '+'\n    '.join([task.to_string() for task in actor.tasks])

    def map_actors(self):
        mapped_actors = []
        for actor in self.actors:
            mapped_actors.append(self.map_actor(actor))
        return '\n'.join(mapped_actors)

    async def choose_delay(self):
        """Choose a delay for the next cognitive cycle"""
        prompt = [SystemMessage(content="""You are a delay chooser.
Your task is to choose a delay for the next cognitive cycle.
The delay asyncio.shield be a number of hours from now  .
The delay should be between 0 and 12 hours.
The delay should be a multiple of 0.5 hours.

Following is a record of the current situation and recent events, followed by a transcript of the recent scene.
Use these to choose a delay that is appropriate for the next cognitive cycle. 
For example, if it is currently evening and the characters are awaiting an event the next day, a delay of 12 hours or until morning would be appropriate
If one or more characters are engaged in an urgent activity, a delay of 0.1 hours to 0.5 hours, or even 0.0 hours, might be appropriate.
Characters engaging in repetitive Thinking or planning might be a strong indicator they are passing time waiting for something to happen, and so a longer delay might be appropriate.
In all cases, the delay should be chosen to move the timeline to the next significant event or activity.

Use the following method to choose the delay:

1. Determine the current time of day
2. Identify the nearest deadline for significant events or activities that are currently occurring or expected to occur soon
3. Base delay is the elapsed time between now and the nearest deadline
4. Review the tasks in actor task lists to identify any tasks that need to be completed by the above nearest deadline
5. Include implicit tasks such as sleeping, eating, etc.
6. Final delay is Base delay minus remaining-task elapsed time.
                                
Do NOT report any of the above steps in your response.
Respond only with the nearest deadline, the elapsed time needed for remaining tasks, and the delay in hours, to the nearest 0.1 hours.

"""),
UserMessage(content="""

Situation
{{$situation}}
            
Actors
{{$actors}}
            

Transcript of recent activity, observations, and events
{{$transcript}}

Respond only with the nearest deadline, the elapsed time needed for remaining tasks, and the delay in hours, to the nearest 0.1 hours. Do not include units (e.g. 'hours')
use the following hash-format for your response:

#delay delay to nearest 0.1 hours
#deadline time of nearest deadline to the nearest 0.1 hours
#tasks elapsed time needed for remaining tasks to the nearest 0.1 hours
##

            
Do not include any introductory, explanatory, or discursive text, just the delay in the above format.
End your response with:
</end>
"""
)]
        if  self.last_update_time < self.simulation_time - timedelta(hours=1):
            await self.update(local_only=True) 
            self.last_update_time = self.simulation_time# update the world to get the latest situation
        
        delay = self.llm.ask({"situation":self.current_state, "actors":self.map_actors(), "transcript":'\n'.join(self.transcript[-20:])}, prompt, temp=0.4, stops=['</end>'], max_tokens=20)

        try:
            delay_str = hash_utils.find('delay', delay)
            delay_f = float(delay_str.strip())
            return delay_f
        except Exception as e:
            print(f'Error choosing delay: {e}')
            return 0.0
