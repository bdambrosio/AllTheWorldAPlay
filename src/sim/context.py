from __future__ import annotations
from asyncio.log import logger
import copy
import json
import os
from pathlib import Path
import traceback
import random
import logging
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, cast
import sim.map as map
from sim.referenceManager import ReferenceManager
from sim.cognitive.EmotionalStance import EmotionalStance
from sim.narrativeCharacter import NarrativeCharacter
from src.utils.VoiceService import VoiceService
from utils import hash_utils, llm_api
from utils.Messages import UserMessage, SystemMessage
import utils.xml_utils as xml
from datetime import datetime, timedelta
import utils.choice as choice
from typing import List
import asyncio
from prompt import ask as default_ask
if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking
    from sim.character_dataclasses import Act, Task, Goal


logger = logging.getLogger('simulation_core')

def run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # If already in an event loop, use create_task and wait
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)
        

class Context():
    def __init__(self, characters, description, scenario_module, npcs=None, server_name=None, model_name=None):
        """Initialize a context with characters and world description
        
        Args:
            characters: List of Character objects
            description: Text description of the world/setting
            scenario_module: Module containing scenario types and rules
            server_name: Optional server name for image generation
        """
        # Initialize logging
        
        self.characters = characters
        self.description = description
        self.scenario_module = scenario_module
        self.server_name = server_name
        self.model_name = model_name
        # Initialize characters in world
        for character in characters:
            character.context = self
            # Additional character initialization as needed

        self.initial_state = description
        self.current_state = description
        self.actors: List[Character] = characters
        self.npcs = npcs if npcs else [] # created as needed
        self.map = map.WorldMap(60, 60, scenario_module)
        self.step = False  # Boolean step indicator from simulation server
        self.run = False # Boolean run indicator from simulation server
        self.name = ''
        self.llm = llm_api.LLM(server_name, model_name)
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
        self.voice_service = VoiceService()
        self.voice_service.set_provider('elevenlabs')
        try:
            api_key = os.getenv('ELEVENLABS_API_KEY')
            if not api_key:
                print("Please set ELEVENLABS_API_KEY environment variable and restart for voiced output")
            self.voice_service.set_api_key('elevenlabs', api_key)
        except Exception as e:
            print(f"Error setting up voice service: {e}")
            self.voice_service.set_provider('coqui')

        voices = self.voice_service.get_voices()
 
        for resource_id, resource in self.map.resource_registry.items():
            has_owner = self.check_resource_has_npc(resource)
            if has_owner:
                owner:Character = self.get_npc_by_name(resource['name']+'_owner'.capitalize(), description=f'{resource["name"]}_owner owns {resource["name"]} ', x=resource['location'][0], y=resource['location'][1], create_if_missing=True)
                resource['properties']['owner'] = owner.mapAgent
                self.reference_manager.declare_relationship(resource['name'], 'owned by', owner.name, 'owner of')

        self.last_consequences = '' # for world updates from recent acts
        self.last_updates = '' # for world updates from recent acts
        self.last_update_time = self.simulation_time
        self.narrative = None
        self.current_scene = None
        self.scene_pre_narrative = '' # setting up the scene
        self.scene_post_narrative = '' # dominant theme of the scene, concluding note
        self.play_file = None
        self.map_file = None

        for actor in self.actors + self.npcs:
            #place all actors in the world. this can be overridden by "H.mapAgent.move_to_resource('Office#1')" AFTER context is created.
            print(f"Context initializing{actor.name}")
            actor.set_context(self)
            print(f"Context initialized{actor.name} speech_stylizer: {actor.speech_stylizer}")
            if actor.mapAgent is None:
                actor.mapAgent = map.Agent(actor.init_x, actor.init_y, self.map, actor.name)
            else:
                actor.mapAgent.x = actor.init_x
                actor.mapAgent.y = actor.init_y
            # Initialize relationships with valid character names. This was for knownActor relationships, I think?
            if hasattr(actor, 'narrative'):
                valid_names = [a.name for a in self.actors if a != actor]

        self.reference_manager.discover_relationships()
        for actor in self.actors + self.npcs:
            print(f"Context checking{actor.name} speech_stylizer: {actor.speech_stylizer}")
            #actor.driveSignalManager.analyze_text(actor.character, actor.drives, self.simulation_time)
            actor.driveSignalManager.analyze_text(self.current_state, actor.drives, self.simulation_time)
            actor.look()
            actor.driveSignalManager.recluster() # recluster drive signals after actor initialization
            #actor.generate_goal_alternatives()
            #actor.generate_task_alternatives() # don't have focus task yet
            actor.wakeup = False

        self.voice_service = VoiceService()
        self.voice_service.set_provider('elevenlabs')
        try:
            api_key = os.getenv('ELEVENLABS_API_KEY')
            if not api_key:
                print("Please set ELEVENLABS_API_KEY environment variable and restart for voiced output")
            self.voice_service.set_api_key('elevenlabs', api_key)
        except Exception as e:
            print(f"Error setting up voice service: {e}")
            self.voice_service.set_provider('coqui')

        
    async def start(self):
        voices = self.voice_service.get_voices()
        #print(f"Available voices: {voices}")
        self.message_queue.put({'name':'', 'text':"play context initialized, creating characters...", 
                                'elevenlabs_params': json.dumps({'voice_id': voices[0]['voice_id'], 'stability':0.5, 'similarityBoost':0.5})})
        await asyncio.sleep(0.1)

    def repair_json(self, response, error):
        """Repair JSON if it is invalid"""
        prompt = [UserMessage(content="""You are a JSON repair tool.
Your task is to repair the following JSON:

<json>
{{$json}}
</json> 

The reported error is:
{{$error}}

Respond with the repaired JSON string. Make sure the string is in a format that can be parsed by the json.loads function. No commentary, no code fences.
""")]
        response = self.llm.ask({"json": response, "error": error}, prompt, tag='repair_json', temp=0.2, max_tokens=3500)
        try:
            return json.loads(response.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            print(f'Error parsing JSON: {e}')
            return None

    def reserialize_narrative_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the narrative JSON structure and returns (is_valid, error_message)
        """
        # Check top-level structure
        if not isinstance(json_data, dict):
            return False, "Root must be a JSON object"
        reserialized = copy.deepcopy(json_data)
        for act in reserialized["acts"]:
            if "scenes" not in act or not isinstance(act["scenes"], list):
                continue
            for scene in act["scenes"]:
                if isinstance(scene["time"], datetime):
                    scene["time"] = scene["time"].isoformat()
                else:
                    print(f'Invalid time format in scene {scene["scene_number"]}: {scene["time"]}')
        return reserialized

    def reserialize_act_to_string(self, act: Dict[str, Any]) -> str:
        """Reserialize the act to a string"""
        serialized_str = self.reserialize_narrative_json({"acts":[act]})
        return json.dumps(serialized_str['acts'][0], indent=2)

    def extract_simulation_time(self, situation):
        # Extract simulation time from situation description
        prompt = [UserMessage(content="""You are a simulation time extractor.
Your task is to extract the exact time of day and datefrom the following situation description:

<situation>
{{$situation}}
</situation>

Respond with the simulation time in a isoformat string that can be parsed by the datetime.parse function. Supply missing components using reasonable defaults based on the context of the situation.
Respond with two lines in exactly this format:

#year yyyy, e.g. 2001
#month mm, e.g. 05
#day dd, e.g. 15
#hour hh, e.g. 15
#ampm am/pm, e.g. pm


If any piece of information is not explicitly stated in the text, make a reasonable inference based on context clues (e.g., "early morning" suggests morning time, "soft light" might suggest spring or summer). 
If absolutely no information is available for any field, use "unknown" for that field. However, always make best effort to supply a specific value consistent with the context.
""")]
        response = self.llm.ask({"situation": situation}, prompt, tag='Context.extract_simulation_time', temp=0.5, max_tokens=20)
        try:
            year = hash_utils.find('year', response)
            if year is None or not year.strip().isdigit():
                year = datetime.now().year
            else:
                year = int(year.strip())
            month = hash_utils.find('month', response)
            if month is None or not month.strip().isdigit():
                month = datetime.now().month
            else:
                month = int(month.strip())
            day = hash_utils.find('day', response)
            if day is None or not day.isdigit():
                day = datetime.now().day
            else:
                day = int(day.strip())
            hour = hash_utils.find('hour', response)
            if hour is None or not hour.isdigit():
                hour = datetime.now().hour
            else:
                hour = int(hour.strip())
            ampm = hash_utils.find('ampm', response)
            if ampm is None or ampm.strip() == 'unknown':
                ampm = 'AM' if hour < 12 else 'PM'
            return datetime(year, month, day, hour, 0, 0, tzinfo=datetime.now().tzinfo)
        except Exception as e:  
            print(f"Error parsing datetime: {e}")
            logger.error(f"Error parsing datetime: {e}\n {response}")
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

    def act_image(self, actor: Character, act: Act, consequences:str = '', filepath='worldsim.png', image_generator='tti_serve'):
        try:
            state = '. '.join(self.current_state.split('.')[:4])
            characters = '\n' + '.\n'.join(
                [entity.name + ' is ' + entity.character.split('.')[0][8:] for entity in self.actors])
            actor_dscp_str = actor.character.strip().split('.')[0].strip()
            actor_dscp_str.replace(actor.name, '')
            act_str=f'. {actor.name}. {actor_dscp_str}'
            if act:
                act_str += f', {act.mode}: {act.action}'
            else:
                act_str += '. '
            prompt = act_str + consequences + state + characters
            # print(f'calling generate_dalle_image\n{prompt}')
            if image_generator == 'tti_serve':
                filepath = llm_api.generate_image(self.llm, f"""wide-view photorealistic style. {prompt}""", size='512x512', filepath=filepath)
            else:
                filepath = llm_api.generate_dalle_image(f"""wide-view photorealistic style. {prompt}""", size='512x512', filepath=filepath)
        except Exception as e:
            traceback.print_exc()
        return filepath
    
    def to_act_image_json(self, actor:Character, act:Act, consequences:str):
        """Return JSON-serializable dict of context state"""
        return {
            'show': ' \n\n'+self.current_state,
            'image': self.act_image(actor, act, consequences, 'worldsim.png')
        }

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
            npc = NarrativeCharacter(name.capitalize(), character_description=description if description else f'{name} is a non-player character', init_x=x, init_y=y, server_name=self.llm.server_name)
            npc.set_context(self)
            npc.llm = self.llm
            map_agent = self.map.get_agent(name)
            if map_agent is not None:
                npc.mapAgent = map_agent
                npc.mapAgent.x = x
                npc.mapAgent.y = y
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
        first_name = reference_text.strip().split(' ')[0]
        for actor in self.actors:
            if actor.name == reference_text or actor.name == first_name:
                return (actor, reference_text)
            
        # Then check NPCs
        for npc in self.npcs:
            if npc.name == reference_text or npc.name == first_name:
                return (npc, reference_text)
        
        character_name = self.reference_manager.resolve_reference_with_llm(reference_text)
        if character_name:
            return (self.get_actor_or_npc_by_name(character_name), character_name)
        
        return (None, None)

    
    def resolve_resource(self, reference_text):
        """
        Resolve a reference to a resource from the map
        
        Args:
            speaker: Character making the reference
            reference_text: Text to resolve into a resource reference
            
        Returns:
            tuple: (resource, canonical_name) or (None, None) if unresolved
        """
        # Normalize reference text
        reference_text = reference_text.strip().capitalize()
        
        # Check active resources first
        for resource in self.map.resource_registry:
            if resource.name == reference_text:
                return (resource, reference_text)
                    
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
                                prompt, tag='Context.world_updates_from_act_consequences', temp=0.5, stops=['</end>'], max_tokens=60)
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
                                prompt, tag='Context.character_updates_from_act_consequences', temp=0.5, stops=['</end>'], max_tokens=60)
        updates = xml.find('<updates>', response)
        if updates is not None:
            self.current_state += '\n' + updates.strip()
        else:
            updates = ''
        return updates
    
    async def do(self, actor, act):
        """ This is the world determining the effects of an actor action"""
        action = act.action
        target = act.target
        duration = act.duration
        mode = act.mode
        source = act.source
        target = act.target
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

Respond with the observable result. Target a sentence or two at most as your response length.
Respond ONLY with the observable immediate effects of the above Action on the environment and characters.
Include any effects on the physical, mental, or emotional state of {{$name}} in your response, even if not observable to other characters.
Do not repeat the above action statement in your response.
Observable result must be consistent with information provided in the LocalMap.
Format your response as one or more simple declarative sentences.
Include in your response:
- changes in the physical environment, e.g. 'the door opens', 'the rock falls',...
- sensory inputs, e.g. {{$name}} 'sees ...', 'hears ...', 
- changes in {{$name}}'s possessions (e.g. {{$name}} 'gains ... ',  'loses ... ', / ... / )
- changes in {{$name})'s or other actor's physical, mental, or emotional state (e.g., {{$name}} 'becomes tired' / 'is injured' / ... /).
- specific information acquired by {{$name}}. State the actual knowledge acquired, not merely a description of the type or form (e.g. {{$name}} learns that ... or {{$name}} discovers that ...).
Do NOT extend the scenario with any follow on actions or effects.
Be  terse when reporting character emotional state, only report the most significant emotional state changes.

Do not include any Introductory, explanatory, or discursive text.
End your response with:
</end>
""")]
        history = self.history()
        local_map = actor.mapAgent.get_detailed_visibility_description()
        local_map = xml.format_xml(local_map)
        consequences = self.llm.ask({"name": actor.name, "action": action, "local_map": local_map,
                                     "state": self.current_state, "character": actor.character, "narrative":  actor.narrative.get_summary('medium')}, 
                                     prompt, tag='Context.do', temp=0.7, stops=['</end>'], max_tokens=300)

        if consequences.endswith('<'):
            consequences = consequences[:-1]
        world_updates = self.world_updates_from_act_consequences(consequences)
        self.last_consequences = consequences
        character_updates = self.character_updates_from_act_consequences(consequences, actor)   
        self.last_updates = character_updates
        print(f'\nContext Do consequences:\n {consequences}')
        print(f' Context Do world_update:\n {world_updates}\n')
        self.message_queue.put({'name':self.name, 'text':f'world_update', 'data':self.to_act_image_json(actor, act, consequences)})
        await asyncio.sleep(0.1)
        return consequences, world_updates, character_updates


    async def update(self, narrative='', local_only=False):

        history = self.history()

        event = ""
        if not local_only and random.randint(1, 7) == -1:
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


{{$narrative}}
                              
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
</end>""")]

        narrative_insert = ''
        if narrative:
            narrative_insert = f"""The scene has changed to:

$narrative
            
Ensure your response reflects this change.
"""
        response = self.llm.ask({"narrative": narrative_insert, "situation": self.current_state, 'history': history, 'time': self.simulation_time.isoformat()}, prompt,
                                tag='Context.update', temp=0.6, stops=['</end>'], max_tokens=270)
        new_situation = xml.find('<situation>', response)       
        # Debug prints
        if not local_only:
            self.message_queue.put({'name':'', 'text':f'\n\n----- situation update -----{self.simulation_time.isoformat()}\n'})
            self.transcript.append(f'\n\n----- situation update ----- {self.simulation_time.isoformat()}\n')
            await asyncio.sleep(0.1)
         
        if new_situation is None:
            return
        self.current_state = new_situation
        if local_only:
            self.last_update_time = self.simulation_time
            return
        self.show = new_situation
        self.message_queue.put({'name':'', 'text':self.show})
        self.transcript.append(f'{self.show}')
        self.show = '' # has been added to message queue!
        self.message_queue.put({'name':self.name, 'text':f'world_update', 'data':self.to_json()})
        await asyncio.sleep(0.1)

        await asyncio.sleep(0.1)

        updates = self.world_updates_from_act_consequences(new_situation)
        # self.current_state += '\n'+updates
        print(f'World updates:\n{updates}')
        for actor in self.actors:
            actor.add_to_history(f"you notice {updates}\n")
            actor.update()  # update history, etc
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
        {'\n        '.join([hash_utils.find('name', task) for task in actor.focus_goal.task_plan]) if actor.focus_goal else ''}   
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
        return mapped_actor+'\n  Remaining tasks:\n    '+'\n    '.join([task.to_string() for task in actor.focus_goal.task_plan]) if actor.focus_goal else ''

    def map_actors(self):
        mapped_actors = []
        for actor in self.actors:
            mapped_actors.append(self.map_actor(actor))
        return '\n'.join(mapped_actors)

        
    def compute_task_plan_limits(self, scene):
        """Compute the task plan limits for an actor for a scene"""
        characters_in_scene = scene.get('characters', ['a','b'])
        try:
            character_task_limit = scene.get('task_budget', int(1.3*len(characters_in_scene)))/len(characters_in_scene)
        except Exception as e:
            print(f'Error computing task plan limits: {e}')
            character_task_limit = 1
        return int(character_task_limit)


    def establish_tension_points(self, act):
        """Establish tension points for an act"""
        if 'tension_points' not in act or not isinstance(act['tension_points'], list):
            return
        for tension_point in act['tension_points']:
            try:
                character_names = tension_point['characters']
                issue = tension_point['issue']
                resolution_requirement = tension_point['resolution_requirement']
                characters: List[NarrativeCharacter] = []
                for character_name in character_names:
                    character, _ = self.resolve_character(character_name)
                    if character is None:
                        print(f'Character {character_name} not found in act {act["act_title"]}')
                        continue
                    characters.append(character)
                for character in characters:
                    for other_character in characters:
                        if other_character != character or len(characters) == 1:
                            character.add_perceptual_input(f'Issue: {issue} {"with "+ other_character.name if len(characters) > 1 else ""}')
                            character.actor_models.get_actor_model(other_character.name, create_if_missing=True).add_tension(issue)
            except Exception as e:
                print(f'Error establishing tension points: {e}')

    async def run_scene(self, scene):
        """Run a scene"""
        print(f'Running scene: {scene["scene_title"]}')
        self.message_queue.put({'name':self.name, 'text':f' -----scene----- {scene["scene_title"]}\n    Setting: {scene["location"]} at {scene["time"]}'})
        if type(scene['time']) == str:
            try:
                self.simulation_time = datetime.fromisoformat(scene['time'])
            except Exception as e:
                print(f'Error parsing scene time: {e}')
        else:
            self.simulation_time = scene['time']
        await asyncio.sleep(0.1)
        self.current_scene = scene
        if scene.get('pre_narrative'):
            self.current_state += '\n\n'+scene['pre_narrative']
            self.scene_pre_narrative = scene['pre_narrative']
            self.message_queue.put({'name':self.name, 'text':f'    {scene["pre_narrative"]}'})
            await asyncio.sleep(0.1)
        if scene.get('post_narrative'):
            self.scene_post_narrative = scene['post_narrative']

        #construct a list of characters in the scene in the order in which they appear
        characters_in_scene: List[Character] = []
        for character_name in scene['action_order']:
            character_name = character_name.capitalize().strip()
            character = self.get_actor_by_name(character_name)
            if character is None:
                print(f'Character {character_name} not found in scene {scene["scene_title"]}')
                character = self.get_npc_by_name(character_name, create_if_missing=True)
            if character_name == 'Context':
                continue
            if character not in characters_in_scene:
                characters_in_scene.append(character)

        # establish character locations and goals for scene
        location = scene['location']
        x,y = self.map.random_location_by_terrain(location)
        characters_at_location = []
        scene_goals = {}
        #set characters in scene at scene location
        for character in characters_in_scene:
            character.mapAgent.x = x
            character.mapAgent.y = y
            characters_at_location.append(character)
            character.current_scene = scene

        #now that all characters are in place, establish their goals
        for character in characters_in_scene:
            character.look() # important to do this after everyone is in place.
            try: # can fail if invented character name mismatch for some reason
                goal_text = scene['characters'][character.name.capitalize()]['goal']
            except Exception as e:
                print(f'Error getting goal for {character.name}: {e}')
                goal_text = ''
            # instatiate narrative goal sets goals and focus goal as side effects
            scene_goals[character.name] = character.instantiate_narrative_goal(goal_text)

        # ok, actors - live!
        characters_finished_tasks = []
        while len(characters_in_scene) > len(characters_finished_tasks):
            for character in characters_in_scene:
                if character in characters_finished_tasks:
                    continue
                elif character.focus_goal is None and (not character.goals  or len(character.goals) == 0):
                    characters_finished_tasks.append(character)
                    continue
                else:
                    self.message_queue.put({'name':'', 'text':f''})
                    await character.cognitive_cycle(narrative=True)
        if self.current_scene.get('post_narrative'):
            self.current_state += '\n'+self.current_scene['post_narrative']
            for character in characters_in_scene:
                character.add_perceptual_input(self.current_scene['post_narrative'], mode = 'internal')
            self.message_queue.put({'name':'\n', 'text':f' ----scene wrapup: {self.current_scene["post_narrative"]}\n'})
            await asyncio.sleep(0.1)

        scene_duration = self.current_scene.get('duration', 0)
        if scene_duration > 0:
            self.simulation_time += timedelta(minutes=scene_duration)
        await self.update(local_only=True)


    async def run_narrative_act(self, act):
        """Run a narrative"""
        self.establish_tension_points(act)
        scenes = act.get('scenes', [])
        if len(scenes) == 0:
            logger.error(f'No scenes in act: {act["act_title"]}')
            return
        print(f'Running act: {act["act_title"]}')
        self.message_queue.put({'name':'', 'text':f'\n---ACT----- {act["act_title"]}'})
        await asyncio.sleep(0.1)
        for scene in act['scenes']:
            while self.step is False and  self.run is False:
                await asyncio.sleep(0.5)
            print(f'Running scene: {scene["scene_title"]}')
            await self.run_scene(scene)
            self.step = False
        await self.update()
                
    async def create_character_narratives(self, play_file, map_file):
        """called from the simulation server to create narratives for all characters"""
        self.play_file = play_file
        self.map_file = map_file
        """Create narratives for all characters,
            then share them with each other (selectively),
            then update them with the latest shared information"""
        for character in cast(List[NarrativeCharacter], self.actors):
            self.message_queue.put({'name':character.name, 'text':f'---- creating narrative -----'})
            character.write_narrative(play_file, map_file)
            await asyncio.sleep(0.1)
        for character in cast(List[NarrativeCharacter], self.actors):
            self.message_queue.put({'name':character.name, 'text':f'---- initial coordination -----'})
            await character.share_narrative()
            await asyncio.sleep(0.1)
        for character in cast(List[NarrativeCharacter], self.actors):
            self.message_queue.put({'name':character.name, 'text':f'----- updating narrative -----'})
            character.update_narrative_from_shared_info()
            logger.info(f'\n{character.name}\n{json.dumps(character.reserialize_narrative_json(character.plan), indent=2)}')
            await asyncio.sleep(0.1)
        self.message_queue.put({'name':self.name, 'text':f'----- {play_file} character narratives created -----'})
        await asyncio.sleep(0.1)

    async def integrate_narratives(self, character_narrative_blocks):
        """Integrate narratives into a single coherent narrative for a single act
        character_narrative_blocks is a list of lists, each containing a character and their planned next play act
        """
        # reformat character_narrative_blocks into a list of character plans for the prompt
        character_plans = []
        for character_block in character_narrative_blocks:
            character = character_block[0]
            character_narrative = character.reserialize_act_to_string(character_block[1])
            character_plans.append(f'{character.name}\n{character_narrative}\n')
        character_plans = '\n'.join(character_plans)

        character_backgrounds = []
        for character in cast(List[NarrativeCharacter], self.actors):
            drives = '\n'.join([f'{d.id}: {d.text}; activation: {d.activation:.2f}' for d in character.drives])
            narrative = character.narrative.get_summary('medium')
            surroundings = character.look_percept
            memories = '\n'.join(memory.text for memory in character.structured_memory.get_recent(8))
            emotionalState = EmotionalStance.from_signalClusters(character.driveSignalManager.clusters, character)        
            emotional_stance = emotionalState.to_definition()
            character_backgrounds.append(f'{character.name}\n{drives}\n{narrative}\n{surroundings}\n{memories}\n{emotional_stance}\n')
        character_backgrounds = '\n'.join(character_backgrounds)

        prompt = [UserMessage(content="""You are a skilled playwright and narrative integrator.
You are given a list of plans (play outlines), one from each of the characters in the play. These plans were created by each character independently.
Your job is to integrate them into a single coherent plan across all characters.
A plan is a JSON object containing a sequence of acts, each with a sequence of scenes (think of a play).
Resolve any staging conflicts resulting from overlapping scenes with overlapping time signatures and overlapping characters (aka actors in the plans)
In integrating scenes within an act across characters, pay special attention to the pre- and post-narratives, as well as the characters's goals, to create a coherent scene. 
It may be necessary for the integrated narrative to have more scenes than any of the original plans.
Do not attempt to resolve conflicts that result from conflicting character drives or goals. Rather, integrate acts and scenes in a way that provides dramatic tension and allows characters to resolve it among themselves.
The integrated plan should be a single JSON object with a sequence of acts, each with a sequence of scenes. 
The overall integrated plan should include as much of the original content as possible, ie, if one character's plan has an act or scene that is not in the others, it's narrative intent should be included in the integrated plan.

Here are the original, independent, plans for each character:
{{$plans}}

In the performance it is currently {{$time}}. Use this information to generate a time-appropriate sequence of scenes for your integrated Act.
The following additional information about the state of the performance will be useful in integrating the acts.

{{$backgrounds}}

Respond with the updated act, using the following format:

```json
updated act
```

Act format is as follows:

An Act is a single JSON document that outlines a short-term plan for yourself
###  Structure
Return exactly one JSON object with these keys:

- "act_number" (int, copied from the original act)  
- "act_title"   (string, copied from the original act or rewritten as appropriate)  
- "act_description" (string, short description of the act, focusing on it's dramatic tension and how it fits into the overall narrative arc)
- "act_goals" {"primary": "primary goal", "secondary": "secondary goal"}
- "act_pre_state": (string, description of the situation / goals / tensions before the act starts)
- "act_post_state": (string, description of the situation / goals / tensions after the act ends)
- "tension_points": [
    {"characters": ["<Name>", ...], "issue": (string, concise description of the issue), "resolution_requirement": (string, "partial" / "complete")}
    ...
  ]
- "scenes"      (array) 

Each **scene** object must have:
{ "scene_number": int, // sequential within the play 
 "scene_title": string, // concise descriptor 
 "location": string, // pick from resource or terrain names in the map file
 "time": YYYY-MM-DDTHH:MM:SS, // the start time of the scene, in ISO 8601 format
 "duration": mm, // the duration of the scene, in minutes
 "characters": { "<Name>": { "goal": "<one-line playable goal>" }, … }, 
 "action_order": [ "<Name>", … ], // 2-4 beats max, list only characters present 
 "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors.", 
 "post_narrative": "Short prose (≤20 words) summarising end state and what emotional residue or new tension lingers." 
 // OPTIONAL: 
 "task_budget": 4 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the scene to avoid rambling or repetition. 
 }

Never break JSON validity.  Ensure the returned form is parseable by the json.loads function.
Keep the JSON human-readable (indent 2 spaces).

Return **only** the JSON.  No commentary, no code fences.
End your response with </end>
""")]
        self.message_queue.put({'name':self.name, 'text':f'------ integrating narratives -----'})
        await asyncio.sleep(0.1)
        response = self.llm.ask(
                              {"time": self.simulation_time.isoformat(), "plans": character_plans, "backgrounds": character_backgrounds},
                              prompt, max_tokens=4000, stops=['</end>'], tag='integrate_narratives')
        try:
            updated_narrative_plan = None
            if not response:
                return None
            response = response.replace("```json", "").replace("```", "").strip()
            updated_narrative_plan = json.loads(response)
            logger.info(f'Integrated act: {self.reserialize_act_to_string(updated_narrative_plan)}')
        except Exception as e:
            print(f'Error parsing updated act: {e}')
            updated_narrative_plan = self.repair_json(response, e)
        if updated_narrative_plan is not None:
            self.validate_dates_in_plan(updated_narrative_plan)
        self.narrative = updated_narrative_plan
        self.message_queue.put({'name':self.name, 'text':f'------ integrated narratives -----'})
        await asyncio.sleep(0.1)
        return updated_narrative_plan

    def validate_dates_in_plan(self, narrative_plan):
        """Validate and adjust scene dates to ensure they are after simulation time and advance temporally.
        
        Args:
            narrative_plan: The narrative plan containing acts and scenes
        """
        if narrative_plan and 'acts' in narrative_plan:
            acts = narrative_plan['acts']
        elif type(narrative_plan) == dict:
            acts = [narrative_plan]
        elif type(narrative_plan) == list:
            acts = narrative_plan
        else:
            return
            
        prev_scene_time = self.simulation_time
        prev_duration = 0
        for act in acts:
            if 'scenes' not in act:
                continue
            for scene in act['scenes']:
                if 'time' not in scene:
                    continue
                # Parse scene time if it's a string
                if isinstance(scene['time'], str):
                    try:
                        scene_time = datetime.fromisoformat(scene['time'])
                    except Exception as e:
                        print(f'Error parsing scene time: {e}')
                        scene_time = self.simulation_time
                else:
                    scene_time = scene['time']
                # If scene time is before simulation time or previous scene, advance it
                while scene_time <= prev_scene_time:
                    # Advance by one day while preserving time of day
                    scene_time = scene_time + timedelta(days=1)
                # Update scene time and previous scene time
                scene['time'] = scene_time
                if scene_time < prev_scene_time+timedelta(minutes=prev_duration):
                    scene_time = prev_scene_time+timedelta(minutes=prev_duration)
                    scene['time'] = scene_time
                duration = scene.get('duration', 0)
                if type(duration) == int and duration > 0:
                    prev_duration = duration
                else:
                    prev_duration = 0
                prev_scene_time = scene_time


    async def run_integrated_narrative(self):
        """Called from simulation, on input individual character narratives have been created, but not an integrated overall narrative
        Integrates on an act by act basis, assuming that, since characters had a chance to share their narratives, actwise integration is coherent"""
        # get length of longest character narrative plan
        integrated_act_count = 0
        for character in cast(List[NarrativeCharacter], self.actors):
            character_narrative_act_count = len(character.plan['acts'])
            if character_narrative_act_count > integrated_act_count:
                integrated_act_count = character_narrative_act_count

        try:
            for i in range(integrated_act_count):
                previous_act = None
                character_narrative_blocks = []
                for character in cast(List[NarrativeCharacter], self.actors):
                    if i < len(character.plan['acts']):
                        updated_act = character.replan_narrative_act(character.plan['acts'][i], previous_act)
                        character_narrative_blocks.append([character, updated_act])  
                next_act = await self.integrate_narratives(character_narrative_blocks)
                if next_act is None:
                    logger.error('No act to run')
                    return
                await self.run_narrative_act(next_act)
                previous_act = next_act
        except Exception as e:
            logger.error(f'Error running integrated narrative: {e}')
            traceback.print_exc()
        return


    async def step_act(self):
        """Step through the act - only execute one scene, main job is to find the next scene to run"""
        next_scene_time = self.simulation_time + timedelta(hours=1000)
        next_character = None
        next_scene = None
        characters_finished_act = []
        for character in cast(List[NarrativeCharacter], self.actors):
            if character in characters_finished_act:
                continue
            else:
                scene = character.current_scene
                if scene is not None:
                    if scene.get('time') < next_scene_time:
                        next_scene_time = scene.get('time')
                        next_character = character
                        next_scene = scene
                else:
                    characters_finished_act.append(character)
        return next_character, next_scene

