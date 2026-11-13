from __future__ import annotations
import collections
import copy
import json
import os
from pathlib import Path
import traceback
import random
import logging
import re
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, cast
import sim.map as map
from sim.cognitive import drives as drive_model
from sim.cognitive.drives import Drive
from sim.referenceManager import ReferenceManager
from sim.cognitive.EmotionalStance import EmotionalStance
from sim.narrativeCharacter import NarrativeCharacter
from sim.memory.core import NarrativeSummary
from utils.VoiceService import VoiceService
from utils import hash_utils, llm_api, json_reply
from utils.Messages import UserMessage, SystemMessage
import utils.xml_utils as xml
from datetime import datetime, timedelta
import utils.choice as choice
from typing import List
import asyncio
import time
from sim.prompt import ask as default_ask
from sim.trace import trace
import sim.survival as survival
import sim.stage as stage
import sim.record as record
from sim.screenplay import Screenplay
from sim import ablation
from sim.character_dataclasses import Stack, Act, Task, Goal, CentralNarrative, datetime_handler, parse_duration

if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking


logger = logging.getLogger('simulation_core')

# World images: one render costs seconds of GPU time, and look/do acts come every few seconds.
WORLD_IMAGE_SECS = float(os.environ.get('WORLD_IMAGE_SECS', '60'))
WORLD_IMAGE_FILES = 8


WORLD_IMAGE_SIZE = '768x768'   # shown at about 400 px; was 512


class Context():
    def __init__(self, characters, description, scenario_module, extras=[], server_name=None, model_name=None,
                 start_time=None):
        """Initialize a context with characters and world description
        
        Args:
            characters: List of Character objects
            description: Text description of the world/setting
            scenario_module: Module containing scenario types and rules
            server_name: Optional server name for image generation
        """
        # Initialize logging
        
        #self.characters = characters
        self.description = description
        self.scenario_module = scenario_module
        self.server_name = server_name
        self.model_name = model_name
        # Initialize characters in world
        for character in characters+extras:
            character.context = self
            # Additional character initialization as needed

        self.screenplay = Screenplay()   # what the company has settled above the scene level (sim/screenplay.py)
        self.initial_state = description
        self.current_state = description
        llm_api.set_preamble(lambda: record.cast_note(self))   # every call of this play knows who is he and who is she
        self.actors: List[Character] = characters
        self.npcs = []
        self.extras = extras # created as needed
        self.map = map.WorldMap(60, 60, scenario_module)
        self.step = False  # Boolean step indicator from simulation server
        self.run = False # Boolean run indicator from simulation server
        self.name = ''
        self.llm = llm_api.LLM(server_name, model_name)
        self.llm.owner = 'Context'  # for the run trace (sim.trace)
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
        # A play states when it starts (start_time: a datetime or ISO string). Without one the
        # date and hour are guessed from the description by the LLM.
        if start_time is not None:
            self.simulation_time = start_time if isinstance(start_time, datetime) else datetime.fromisoformat(start_time)
        else:
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
                owner.placeholder = True   # the map's stand-in for an owner, not a person of the play
                resource['properties']['owner'] = owner.mapAgent
                self.reference_manager.declare_relationship(resource['name'], 'owned by', owner.name, 'owner of')

        self.last_consequences = '' # for world updates from recent acts
        self.last_updates = '' # for world updates from recent acts
        self.last_update_time = self.simulation_time
        self.current_scene = None
        # world-image throttle (see act_image_json): last render time, scene and file index
        self._world_image_at = 0.0
        self._world_image_scene = None
        self._world_image_n = 0
        self.scene_pre_narrative = '' # setting up the scene
        self.scene_post_narrative = '' # dominant theme of the scene, concluding note
        self.play_file = None
        self.map_file = None
        self.narrative_level= False
        self.scene_integrated_task_plan: List[Dict[str, NarrativeCharacter, Task]] = []
        self.scene_task_budget = 0  # plan length at scene start (place_commitment, run_scene)
        # concrete physical events (acts with visible results, travel, hazards), newest last; fed to
        # dialog and act prompts so characters respond to what actually happens (add_event)
        self.recent_events = collections.deque(maxlen=16)
        self.scene_events = []   # this scene's events (scene wrap-up, outcome check)
        self.act_scene_wrapups = []
        self.act_events = []     # this act's events
        self.scene_integrated_task_plan_index = 0 # using explicit index allows cog cycle to insert tasks!
        self.current_act = None
        self.embedding_model = None
        
        # Initialize narrative staleness detection
        from sim.narrative_staleness import NarrativeStalnessDetector
        self.staleness_detector = NarrativeStalnessDetector(self, window_size=5)
        self.previous_interventions = []

        for actor in self.actors + self.extras + self.npcs:
            #place all actors in the world. this can be overridden by "H.mapAgent.move_to_resource('Office#1')" AFTER context is created.
            print(f"Context initializing {actor.name}")
            actor.set_context(self)
            print(f"Context initialized {actor.name} speech_stylizer: {actor.speech_stylizer}")
            # centre stage unless the character was given a position (a play may then move it:
            # "H.mapAgent.move_to_resource('Office1')" after the context is created)
            x = self.map.stage_centre[0] if actor.init_x is None else actor.init_x
            y = self.map.stage_centre[1] if actor.init_y is None else actor.init_y
            if actor.mapAgent is None:
                actor.mapAgent = map.Agent(x, y, self.map, actor.name)
            else:
                actor.mapAgent.x, actor.mapAgent.y = x, y
            # Initialize relationships with valid character names. This was for knownActor relationships, I think?
            if hasattr(actor, 'narrative'):
                valid_names = [a.name for a in self.actors+self.extras if a != actor]

        self.reference_manager.discover_relationships()
        for actor in self.actors + self.extras + self.npcs:
            print(f"Context checking{actor.name} speech_stylizer: {actor.speech_stylizer}")
            actor.look()
            #actor.generate_goal_alternatives()
            #actor.generate_task_alternatives() # don't have focus task yet
            actor.wakeup = False
        # (the voice service set up above was rebuilt here, refetching the voice list)
        survival.attach(self)  # bodies, when the scenario defines survival (plays/scenarios/forest.py)

        
    def parse_scenario(self, scenario_lines: List[str]) -> dict:
        """
        Parse a scenario file and extract setting, characters, and drives.
        
        Args:
            scenario_lines: List of lines from the scenario file
            
        Returns:
            Dictionary with setting, characters list containing name, description, drives
        """
        if isinstance(scenario_lines, list):
            content = '\n'.join(scenario_lines)
        else:
            content = scenario_lines
        
        # Extract setting from Context creation
        context_pattern = r'W = context\.Context\(\[[^\]]+\],\s*"""(.*?)"""\s*,'
        context_match = re.search(context_pattern, content, re.DOTALL)
        setting = context_match.group(1).strip() if context_match else ""
        
        characters = []
        
        # Find all NarrativeCharacter definitions - improved pattern to handle line boundaries
        char_pattern = r'(\w+)\s*=\s*NarrativeCharacter\(\s*"([^"]+)"\s*,\s*"""([^"]+?)"""\s*,'
        char_matches = re.finditer(char_pattern, content, re.DOTALL)
        
        for char_match in char_matches:
            var_name = char_match.group(1)
            char_name = char_match.group(2)
            char_description = char_match.group(3).strip()
            
            # Find drives for this character - ensure we match the exact variable name
            drives_pattern = rf'\b{re.escape(var_name)}\.set_drives\(\[\s*((?:"[^"]*",?\s*)+)\]\)'
            drives_match = re.search(drives_pattern, content, re.DOTALL)
            
            drives = ""
            if drives_match:
                drives_text = drives_match.group(1)
                # Extract individual drive strings
                drive_strings = re.findall(r'"([^"]*)"', drives_text)
                drives = '. '.join(drive_strings)
            
            characters.append({
                "name": char_name,
                "description": char_description,
                "drives": drives
            })
        
        return {
            "setting": setting,
            "characters": characters
        }

    def summarize_scenario(self) -> dict:
        """
        Extract setting, characters, and drives from the already-parsed Context data.
        
        Returns:
            Dictionary with setting, characters list containing name, description, drives
        """
        # Setting is stored in current_state
        setting = self.current_state if hasattr(self, 'current_state') else ""
        
        characters = []
        
        # Extract characters from actors and extras
        all_characters = []
        if hasattr(self, 'actors'):
            all_characters.extend(self.actors)
        if hasattr(self, 'extras'):
            all_characters.extend(self.extras)
        
        for character in all_characters:
            # Get character drives as text strings
            if hasattr(character, 'drives') and character.drives:
                # Each drive has a .text attribute
                drive_texts = [drive.text for drive in character.drives]
                drives = '. '.join(drive_texts)
            else:
                drives = ""
            
            characters.append({
                "name": character.name,
                "description": character.character,  # Character description is stored in .character
                "drives": drives
            })
        
        return {
            "setting": setting,
            "characters": characters
        }

    def summarize_map(self) -> str:
        """The physical world for the planning prompts: the kinds of terrain, and the named
        places near the cast with what each offers (sim/stage.py)."""
        return f"Terrain: {', '.join(self.map.terrain_types.__members__)}\n{stage.gazetteer(self)}"


    # The screenplay's parts under the names the prompts have always read them by.
    @property
    def central_narrative(self):
        """The play's central dramatic question, as the playwright wrote it."""
        return self.screenplay.question_text

    @central_narrative.setter
    def central_narrative(self, text):
        self.screenplay.question['text'] = text or ''

    @property
    def episode(self):
        """What this episode settles and leaves open (frame_episode)."""
        return self.screenplay.frame

    @episode.setter
    def episode(self, frame):
        self.screenplay.frame = frame or {}

    @property
    def act_frame(self):
        """The focus, stakes and roles of the act now planned or played ({} before the first)."""
        return self.screenplay.act.focus if self.screenplay.act else {}

    @property
    def act_central_narrative(self):
        """The dramatic framework of the act now planned or played ('' before the first)."""
        return self.screenplay.act.framework if self.screenplay.act else ''

    @property
    def current_state(self):
        """The Situation that prompts read. In a world with mechanics it is rendered from the
        world (sim/record.py) each time it is read: the setting as the play gives it, where
        things stand now, and the notes of the scene in progress (its heading, set-up, arrivals,
        dangers). Elsewhere it is the free text it always was."""
        if survival.config(self) and getattr(self, 'actors', None):
            lately = ' '.join(w.split(' Assessed outcome:')[0] for w in getattr(self, 'act_scene_wrapups', [])[-2:])
            value = f'{self.initial_state.strip()}\n\n{record.situation(self, lately)}\n{self._state_notes}'
        else:
            value = self._state_base + self._state_notes
        self._state_read = value
        return value

    @current_state.setter
    def current_state(self, value):
        read = getattr(self, '_state_read', None)
        if read and value.startswith(read):   # '+=': a note on the scene in progress
            self._state_notes += value[len(read):]
            self._state_read = None
        else:
            self._state_base, self._state_notes = value, ''

    MAX_JUMPS_IN_ACT = 1   # times a scene other than an act's first may move the play on

    def move_on(self, target):
        """The play moves on to `target`, off stage (survival.pass_offstage: needs are met, not
        raised; a danger left standing is over). The cast is told how long it has been and what
        time it is, so that nobody carries on from the morning at dusk. Returns the hours passed."""
        gap = target - self.simulation_time
        hours = gap.total_seconds() / 3600
        if hours <= 0:
            return 0.0
        met = survival.pass_offstage(self, gap)
        self.simulation_time = target
        if hours >= survival.REAL_JUMP_HOURS:
            told = (f"About {round(hours)} hours have passed. It is now {stage.clock(self)}."
                    + (" In that time you ate, drank and rested as you could." if met else ''))
            for character in self.actors + self.extras:
                character.add_perceptual_input(told, percept=False, mode='internal')
            self.add_event(f'About {round(hours)} hours pass')
            trace.event('time_jump', hours=round(hours, 1), to=stage.clock(self, date=True), needs_met=sorted(met))
            self._hours_moved = getattr(self, '_hours_moved', 0.0) + hours
        return hours

    def _may_move_on(self, opens_act):
        return opens_act or getattr(self, '_act_jumps', 0) < self.MAX_JUMPS_IN_ACT

    def strangers(self, *characters, land=True):
        """The play's cast (or those named) begin as strangers: none knows another's name until it
        is said in its hearing (record.speech reports a name given). Called from a play file. A
        character's actor still sees the names, to aim an act; its notes say which the character
        does not yet know, and a line that uses one is said again.
        They are strangers to the country too, unless land=False: a character then knows only
        the places it has seen. What lies at the others reaches its actor as stage directions
        (survival.available_actions part='unseen'), not the character."""
        people = list(characters) or list(self.actors)
        for character in people:
            character.unknown_names = {other.name for other in people if other is not character}
            character.unknown_land = bool(land)
        trace.event('strangers', cast=[c.name for c in people])

    def strangers_note(self):
        """For the pre-scene step: who does not yet know whose name."""
        lines = [f"{c.name} does not yet know the name of: {', '.join(sorted(c.unknown_names))}."
                 for c in self.actors if getattr(c, 'unknown_names', None)]
        new = [c.name for c in self.actors if getattr(c, 'unknown_land', False)]
        if new:
            lines.append(f"{', '.join(new)}: new to this country, knowing only the places each has seen. A set-up or goal may send "
                         f"them looking for water or shelter; it does not have them know where an unseen place lies.")
        return ('These people met as strangers. ' + ' '.join(lines)) if lines else ''

    def _hazard_claim_stands(self, words, scene=None):
        """Is a 'presence' finding that names one of this world's dangers a real one? Speaking of
        a danger (feared, or past) claims nothing. Having it arrive does, unless it is the danger
        already on the cast or one this scene may bring on (survival.brought_on): a play has
        one danger, and a second that a plan writes in is written out again."""
        named = [h for h in (survival.config(self) or {}).get('hazards', []) if h['name'].lower() in words.lower()]
        if not named:
            return True   # not one of the world's dangers: an ordinary finding
        active = (getattr(self, 'active_threat', None) or {}).get('hazard')
        bringing = survival.brought_on(self, scene) if scene is not None else None
        if active in named or bringing in named:
            return False
        return any(survival.arrives(h, words) for h in named) and (survival.hazard_spent(self) or scene is not None)

    async def start(self):
        voices = self.voice_service.get_voices()
        message = {'name':'', 'text':"play context initialized, creating characters..."}
        if voices:  # empty without an ElevenLabs key; voices[0] made plays unloadable then
            message['elevenlabs_params'] = json.dumps({'voice_id': voices[0]['voice_id'], 'stability':0.5, 'similarityBoost':0.5}, default=datetime_handler)
        self.message_queue.put(message)
        await asyncio.sleep(1.0)  # Give simulation.py time to process and forward

    def format_character_for_context_ask(self, character:NarrativeCharacter):
        recent_memories = character.structured_memory.get_recent(8)
        memory_text = '\n\t\t'.join(memory.to_string() for memory in recent_memories)
        goal_history = '\n\t\t'.join([f'{g.name} - {g.description}; {g.completion_statement}' for g in character.goal_history]) if character.goal_history else f'None to report'
        return f'\n\n##Name: {character.name}' + \
            f'\n\tDescription: {character.character}' + \
            (f'\n\tBody, as the world has it (what was only said did not happen): {character.body.summary()}' if getattr(character, 'body', None) is not None else '') + \
            (f'\n\tPlanned for this scene but NOT done: {", ".join(stage.pending_business(character))}' if stage.pending_business(character) else '') + \
            f'\n\tDrives (most pressing first):\n'+drive_model.drive_lines(character) + \
            f'\n\tNarrative: {character.narrative.get_summary('medium')}'+ \
            f'\n\tGoal_history: {goal_history}' + \
            f'\n\tMemories: {memory_text}' + \
            f'\n\tEmotional_stance: {character.emotionalStance.to_definition()}\n#'


    async def context_ask(self, system_prompt:str=None, prefix:str=None, suffix:str=None, addl_bindings:dict={}, max_tokens:int=100, log:bool=True, tag:str=''):
        prompt = []
        if system_prompt:
            prompt.append(SystemMessage(content=system_prompt))
        # (prefix was accepted and silently dropped: check_post_state_ambiguity's whole task
        # framing lived there)
        prompt.append(UserMessage(content=(prefix + '\n' if prefix else '') + """    

#Central Narrative Focus
{{$central_narrative}}
##
                     
#Situation 
{{$situation}}
##

#Actors
{{$actors}}
##

"""))
        if suffix:
            prompt.append(UserMessage(content=suffix+'\n\nend your response with </end>'))
        

        bindings = {"central_narrative":self.central_narrative,
                    "situation":self.current_state,
                    "actors":'\n'.join([self.format_character_for_context_ask(a) for a in self.actors+self.extras]),}
        for key, value in addl_bindings.items():
            bindings[key]=value
        
        response = self.llm.ask(bindings, prompt, tag=tag, max_tokens=max_tokens, stops=['</end>'], log=log)
        return response
    
    def repair_json(self, response, error):
        """Repair JSON if it is invalid"""

        if not response.startswith('{')and '{' in response:
            start = response.find('{')
            end = response.rfind('}')
            response = response[start:end+1]

        # Remove newlines that are outside of string values
        in_string = False
        result = []
        i = 0
        while i < len(response):
            if response[i] == '"' and (i == 0 or response[i-1] != '\\'):
                in_string = not in_string
            if not in_string and response[i] == '\n':
                i += 1
                continue
            result.append(response[i])
            i += 1
        response = ''.join(result)

        # Find first complete JSON form
        brace_count = 0
        json_end = 0
        for i, char in enumerate(response):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        if json_end > 0:
            response = response[:json_end]

        try:
            return json.loads(response)
        except Exception as e:
            error = e

        # Ok, ask llm
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

    def reserialize_scene_time(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the narrative JSON structure and returns (is_valid, error_message)
        """
        # Check top-level structure
        if not isinstance(scene, dict):
            return False, "Root must be a JSON object"
        reserialized = copy.deepcopy(scene)
        when = scene.get("time")
        if not isinstance(when, str):   # a string is already serialized; a missing time is now
            reserialized["time"] = (when if isinstance(when, datetime) else self.simulation_time).isoformat()
        return reserialized
    
    def reserialize_acts_times(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
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
                # a plan's scene times are not what sets the clock (run_scene), and a rewritten
                # act may leave one out: that ended a play at the head of its third act
                when = scene.get("time")
                if isinstance(when, datetime):
                    scene["time"] = when.isoformat()
                elif not isinstance(when, str):
                    scene["time"] = self.simulation_time.isoformat()
        return reserialized

    def reserialize_act_times(self, act: Dict[str, Any]) -> str:
        """Reserialize the act to a string"""
        serialized_str = self.reserialize_acts_times({"acts":[act]})
        return json.dumps(serialized_str['acts'][0], indent=2, default=datetime_handler)

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
        response = self.llm.ask({"situation": situation}, prompt, tag='Context.extract_simulation_time', temp=0.5, max_tokens=60)
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
            # 20 tokens cut the reply off before the hour, and 'pm' was read but never applied:
            # a late-afternoon play started at 1 in the morning of the current year
            ampm = (hash_utils.find('ampm', response) or '').strip().lower()
            if ampm == 'pm' and hour < 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
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
            'actors': [actor.to_shallow_json() for actor in self.actors],
            'extras': [extra.to_shallow_json() for extra in self.extras],
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
        for actor in self.actors+self.extras:
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
        
        for actor in self.actors+self.extras:
            # Get recent memories from structured memory
            recent_memories = actor.structured_memory.get_recent(5)
            for memory in recent_memories:
                history.append(f"{actor.name}: {memory.to_string()}")
                
        return '\n'.join(history) if history else ""

    def generate_image_description(self):
        return "wide-view photorealistic style. "+self.current_state
        
        
    @staticmethod
    def _image_description(entity):
        """First sentence of a character description without its lead-in: [8:] assumed every
        description starts 'You are ', so 'Joe, a healthy young man' became 'ealthy young man'."""
        first = entity.character.split('.')[0].strip()
        first = re.sub(r'^(you are|i am)\s+', '', first, flags=re.IGNORECASE)
        if first.lower().startswith(entity.name.lower()):
            first = first[len(entity.name):]
        return first.lstrip(' ,:;-').strip()

    def image(self, filepath, image_generator='tti_serve'):
        try:
            prompt = stage.picture(self)   # the stage as it stands: place, hour, who is there
            if prompt is None:
                state = '. '.join(self.current_state.split('.')[:3])
                characters = '\n' + '.\n'.join(
                    [entity.name + ' is ' + self._image_description(entity) for entity in self.actors])
                prompt = f'{stage.style(self)}. Wide view. {state}{characters}'
            trace.event('image', of='world', prompt=prompt)
            filepath = llm_api.generate_image(self.llm, prompt, size=WORLD_IMAGE_SIZE, filepath=filepath)
        except Exception as e:
            traceback.print_exc()
        return filepath

    def act_image(self, actor: Character, act: Act, consequences:str = '', filepath='worldsim.png', image_generator='tti_serve'):
        try:
            # the same picture of the stage, with the deed in front. (It was the actor's
            # description, the raw act, the settled facts with their body notes and place ids,
            # then the opening of the Situation.)
            deed = re.sub(rf'^\s*{re.escape(actor.name)}\s+', '', (act.action if act else '') or '').strip()
            prompt = stage.picture(self, doing=deed if act is not None and str(act.mode).split('.')[-1] in ('Do', 'Move') else '')
            if prompt is None:
                state = '. '.join(self.current_state.split('.')[:4])
                prompt = f"{stage.style(self)}. Wide view. {self._image_description(actor)}, {deed}. {state}"
            trace.event('image', of='act', prompt=prompt)
            filepath = llm_api.generate_image(self.llm, prompt, size=WORLD_IMAGE_SIZE, filepath=filepath)
        except Exception as e:
            traceback.print_exc()
        return filepath
    
    def _next_world_image_file(self):
        """Rotating world-image filenames: a path already queued to the UI is not overwritten
        by the next render before simulation.py reads it, and the image dir does not grow."""
        self._world_image_n = (self._world_image_n + 1) % WORLD_IMAGE_FILES
        return f'worldsim_{self._world_image_n}.png'

    def _world_image_due(self):
        """A new world image at most every WORLD_IMAGE_SECS, or when the scene changes."""
        scene_changed = self.current_scene is not self._world_image_scene
        return scene_changed or time.monotonic() - self._world_image_at >= WORLD_IMAGE_SECS

    async def act_image_json(self, actor:Character, act:Act, consequences:str):
        """world_update payload after an act. 'image' is a file path, or None when throttled
        (simulation.py then resends the cached image). Rendering runs off the event loop."""
        image = None
        if self._world_image_due():
            self._world_image_at = time.monotonic()
            self._world_image_scene = self.current_scene
            image = await asyncio.to_thread(self.act_image, actor, act, consequences, self._next_world_image_file())
        return {'show': ' \n\n'+self.current_state, 'image': image}

    async def world_image_json(self):
        """world_update payload with a fresh image (play load, situation updates), off the event loop."""
        self._world_image_at = time.monotonic()
        self._world_image_scene = self.current_scene
        image = await asyncio.to_thread(self.image, self._next_world_image_file())
        return {'show': ' \n\n'+self.current_state, 'image': image}

    def get_actor_by_name(self, name):
        """Helper to find actor by name"""
        for actor in self.actors+self.extras:
            if actor.name == name:
                return actor
        return None

    def set_resource_owner(self, resource_id, owner_name, description=None):
        """Make `owner_name` the owner of a resource: a cast member, or an NPC created here (a
        play names the real owner, e.g. a character's father, in place of the '<Resource>_owner'
        NPC that __init__ gives every owned resource type). Returns the owner."""
        resource = self.map.resource_registry[resource_id]
        owner = next((a for a in self.actors + self.extras if a.name == owner_name), None)
        if owner is None:
            x, y = resource['location']
            owner = self.get_npc_by_name(owner_name, description=description or f'{owner_name} owns {resource_id}',
                                         x=x, y=y, create_if_missing=True)
        previous = resource['properties'].get('owner')
        previous_name = getattr(previous, 'name', None)
        if previous_name and previous_name != owner.name:
            relations = self.reference_manager.relationships
            relations.pop(previous_name, None)
            for name in list(relations):
                relations[name] = [(rel, other) for rel, other in relations[name] if other != previous_name]
            if not any(r['properties'].get('owner') is previous for rid, r in self.map.resource_registry.items()
                       if rid != resource_id):
                self.npcs = [n for n in self.npcs if n.name != previous_name]
                if previous in self.map.agents:
                    self.map.agents.remove(previous)
        resource['properties']['owner'] = owner.mapAgent
        self.reference_manager.declare_relationship(resource_id, 'owned by', owner.name, 'owner of')
        survival.give_body(self, owner)   # an owner with a part to play can hold things and be dealt with
        return owner

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
        for actor in self.actors+self.extras:
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
        for actor in self.actors+self.extras+self.npcs:
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

Your response should be concise, and only include only statements about changes to existing Environment.
Do NOT report changes to the description (e.g. the time is not listed), only changes to the situation being described.
Do NOT repeat elements of the existing Environment, respond only with significant changes to the Environment.
Do NOT repeat as an update items already present at the end of the Environment statement.
Your updates should be dispassionate. 
Use the following XML format:
<updates>
concise statement(s) of significant changes to Environment, if any, one per line.
</updates>

Include ONLY the concise updated state description in your response. Be concise, limit your response to about 30 words.
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

Include ONLY the concise updated state description in your response. Limit your response to about 20 words.
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
    
    async def look(self, actor, act):
        """ This is the world determining the effects of an actor action"""
        action = act.action
        target = act.target
        duration = act.duration
        mode = act.mode
        source = act.source
        prompt = [UserMessage(content="""You are simulating a dynamic world. 
Your task is to determine the information gained by {{$name}} scanning visually with the following intent:

<action>
{{$action}}, {{$target}}
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
                              
and recent history:
<history>
{{$history}}
</history>

The body of each character present, as it is:
<bodies>
{{$bodies}}
</bodies>
What each is carrying:
<holdings>
{{$holdings}}
</holdings>
It is now {{$clock}}.
Everyone in the play, and whether each is he or she: {{$cast}}.
Looking changes nobody's body and nothing anyone holds: report no bodily condition, possession or time of day that the above does not show.
Danger here now: {{$danger}}. Report no animal, person or danger that is not listed here or on the map.
{{$rejected}}

Respond with the information result. Target a sentence or two at most as your response length.
If the look shows {{$name}} nothing it did not already know (no movement, no change, the same trees), respond with the single word: nothing
Respond ONLY with the immediate information gain by the character.
Include any effects on the physical, mental, or emotional state of {{$name}} in your response, even if not observable to other characters.
Do not repeat the above action statement in your response.
Information gained must be consistent with information provided in the LocalMap.
Format your response as one or more simple declarative sentences.
Include in your response:
- changes in the physical environment, e.g. '{{$name}} sees the door open', ...
- sensory inputs, e.g. {{$name}} 'sees ...', 'hears ...', 
- changes in {{$name}}'s or other actor's physical, mental, or emotional state (e.g., {{$name}} 'becomes tired' / 'is injured' / ... /).
- specific information acquired by {{$name}}. State the actual knowledge acquired, not merely a description of the type or form (e.g. {{$name}} learns that ... or {{$name}} discovers that ...).
Do NOT extend the scenario with any follow on actions or effects.
Be terse, only report the most significant  state changes. Limit your response to 70 words.

Do not include any Introductory, explanatory, or discursive text.
End your response with:
</end>
""")]
        history = self.history()
        local_map = actor.mapAgent.get_detailed_visibility_description()
        local_map = xml.format_xml(local_map)
        inputs = {"name": actor.name, "action": action, "local_map": local_map, "target": target,
                  "history": "\n".join([m.to_string() for m in actor.structured_memory.get_recent(16)]),
                  "holdings": survival.holdings(actor) or 'nothing the world keeps track of', "clock": stage.clock(self),
                  "danger": record.danger(actor) or 'none', "cast": record.cast(self),
                  "bodies": '; '.join(f'{c.name}: {c.body.summary(public=True)}' for c in [actor] + actor.bystanders()
                                      if getattr(c, 'body', None) is not None) or 'not tracked',
                  "state": self.current_state, "character": actor.character, "narrative":  actor.narrative.get_summary('medium')}
        narrate = lambda rejected='': self.llm.ask({**inputs, "rejected": rejected}, prompt, tag='Context.look', temp=0.7,
                                                   stops=['</end>'], max_tokens=300)
        nothing = lambda text: '' if re.fullmatch(r'\W*nothing\W*', (text or '').strip(), re.I) else text
        settled = record.settle(self, actor, f'looks: {action}', None, nothing(narrate()), lambda note: nothing(narrate(note)), kind='look') or ''
        consequences = str(settled)   # what was seen; a look settles nothing in the world

        if consequences.endswith('<'):
            consequences = consequences[:-1]
        #world_updates = self.world_updates_from_act_consequences(consequences)
        self.last_consequences = consequences
        #character_updates = self.character_updates_from_act_consequences(consequences, actor)   
        #self.last_updates = character_updates
        print(f'\nContext Do consequences:\n {consequences}')
        #print(f' Context Do world_update:\n {world_updates}\n')
        self.message_queue.put({'name':self.name, 'text':f'world_update', 'data':await self.act_image_json(actor, act, consequences)})
        await asyncio.sleep(0.1)
        return consequences

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

These facts about the result are already settled and must not be contradicted:
<facts>
{{$facts}}
</facts>

The body of each character present, as it is:
<bodies>
{{$bodies}}
</bodies>
Give nobody a bodily condition (a fever, a wound, thirst quenched, hunger) that Bodies and the facts do not show.

What each character present is carrying:
<holdings>
{{$holdings}}
</holdings>
These things change hands only as the facts above say. If the action takes or moves one of them otherwise, report the attempt and leave the thing with its holder.

What has happened lately:
{{$events}}

It is now {{$clock}}. Light, dark and the time of day in your response must fit that.
Danger here now: {{$danger}}. A danger is driven off, strikes or leaves only as the facts say; otherwise it is still there, or there is none.
What is left at places nearby that have been drawn on: {{$stock}}. Say nothing else about how much is left anywhere.

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

The characters present are: {{$present}}. Nobody else is here.
Everyone in the play, and whether each is he or she: {{$cast}}.

You are the stage, not a character: report what the action physically brings about, and nothing more.
Respond in exactly two lines:
#seen <the action as someone standing there saw it, in one short plain sentence>
#result <what the action brought about that #seen does not already say, as someone standing there could see or hear it: a door that opens, water that is gone, a thing found, gained or lost, an answer that does not come. One or two simple sentences, 40 words at most. If it brought about nothing more (a look, a gesture, a stance, a walk, something put down or pushed across), write: nothing. That a thing stays as it was (still knotted, still in a hand, undisturbed) is not a result: write nothing>
The audience reads the character's own words for the action, then only your #result.
The result must be consistent with the LocalMap and with the facts above.
Include, when they apply:
- changes in the physical environment, e.g. 'the door opens', 'the rock falls',...
- what the facts say {{$name}} gained, lost, found or read, in the words of someone watching
Do NOT:
- give {{$name}} or anyone a thing, a wound, an illness or a need met that the facts, Bodies and holdings do not show; if the action uses something {{$name}} does not have, show only what is seen of the try (a hand that finds nothing, a gesture at an empty place), and never say what anyone has or lacks, or where a thing came from
- say what any other character feels, thinks, decides or does in reply: their reactions are theirs to play
- have the action succeed at controlling another character (a snatch, a shove, a forced signature): report the attempt and what is physically true so far; the other character answers it
- introduce objects, documents, people or facts that the action and the situation did not already contain
- extend the scenario with any follow-on actions or effects
- explain a rule or why something cannot be, or end with what did not change ('Nothing else changes', 'nothing else moved'): say only what is seen
- mention 'the facts', 'Bodies' or these instructions: you describe a world, not a ruling
{{$rejected}}
Do not include any Introductory, explanatory, or discursive text.
End your response with:
</end>
""")]
        # survival mechanics (sim/survival.py) settle eating, drinking, shelter, threats ...;
        # the narration describes them rather than inventing other outcomes
        facts = survival.resolve_do(self, actor, action, affordance=getattr(act, 'affordance', None),
                                    targets=act.target if isinstance(act.target, list) else None)
        history = self.history()
        local_map = actor.mapAgent.get_detailed_visibility_description()
        local_map = xml.format_xml(local_map)
        inputs = {"name": actor.name, "action": action, "local_map": local_map,
                  "state": self.current_state, "character": actor.character, "narrative":  actor.narrative.get_summary('medium'),
                  "present": ', '.join(c.name for c in [actor] + actor.bystanders()), "cast": record.cast(self),
                  "holdings": survival.holdings(actor) or 'nothing the world keeps track of',
                  "clock": stage.clock(self), "stock": survival.stock(actor) or 'nothing has been used up',
                  "danger": record.danger(actor) or 'none', "events": self.recent_events_text(6),
                  "bodies": '; '.join(f'{c.name}: {c.body.summary(public=True)}' for c in [actor] + actor.bystanders()
                                      if getattr(c, 'body', None) is not None) or 'not tracked',
                  "facts": facts or 'none'}
        # The narrator may tell the deed again under #seen, which nobody reads; only #result is
        # shown, checked and remembered. Told only 'do not repeat the action', it retold it in more
        # words ('Hu slides the pen across the desk ... in the morning light') after the actor's own.
        def narrate(rejected=''):
            reply = self.llm.ask({**inputs, "rejected": rejected}, prompt, tag='Context.do', temp=0.7,
                                 stops=['</end>'], max_tokens=300) or ''
            found = re.search(r'#result\s*:?\s*(.*)', reply, re.S)
            return found.group(1).strip() if found else re.sub(r'^\s*#seen[^\n]*\n?', '', reply).strip()
        # the narration is held to the world's record before it is shown, remembered or built on
        # 'nothing', or a sentence that only says nothing came of it ('He does not find or gain anything new.')
        empty = r"[^.!?]*\b(does not|doesn't|did not) (find|gain)\b[^.!?]*[.!?]?|[^.!?]*\b(finds?|gains?|learns?) nothing( new)?\b[^.!?]*[.!?]?"
        nothing = lambda text: '' if re.fullmatch(r'\W*nothing\W*', (text or '').strip(), re.I) else re.sub(empty, '', text or '').strip()
        settled = record.settle(self, actor, action, facts, nothing(narrate()), lambda note: nothing(narrate(note)), kind='do')
        # what settling made real or refused is part of what happened: the actor is told it, and
        # a deed that rested on something refused was a try (Character.acts shows it so)
        actor.settled_notes = list(getattr(settled, 'facts', ())) + list(getattr(settled, 'refused', ()))
        actor.deed_refused = bool(getattr(settled, 'tainted', False))
        consequences = ' '.join(filter(None, [facts or ''] + actor.settled_notes + [settled or '']))
        if consequences.endswith('<'):
            consequences = consequences[:-1]
        # What the deed did is the settled fact and the admitted narration. (Two more LLM calls
        # summarised the narration into 'updates' that were appended to the situation everyone
        # reads: that is how a thing a narrator once wrote became true from then on.)
        world_updates = re.sub(survival.BODY_NOTE, '', consequences).strip()
        self.last_consequences = consequences
        character_updates = ''
        self.last_updates = character_updates
        print(f'\nContext Do consequences:\n {consequences}')
        print(f' Context Do world_update:\n {world_updates}\n')
        self.message_queue.put({'name':self.name, 'text':f'world_update', 'data':await self.act_image_json(actor, act, consequences)})
        await asyncio.sleep(0.1)
        return consequences, world_updates, character_updates

    def identify_state_changes(self, previous_state, new_state):
        """Identify changes in the state description"""
        prompt = [UserMessage(content="""You are a skilled setting analyst. Identify the changes from the previousState and newState below:
<previousState>
{{$previous_state}}
</previousState>
         
<newState>
{{$new_state}}
</newState>

Respond with one or two concise sentences describing the changes. Include only updates in the situation being described, not changes to the description itself.
All phrasing should directly state the updated situation. For example, rather than report:
"The calendar with the interview reminder is no longer mentioned, and Alex's emotional state is described as groggy"
report:
"Alex is groggy"
Respond only with the text describing the changes.
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
End your response with:
</end>
""")]
        response = self.llm.ask({"previous_state": previous_state, "new_state": new_state}, prompt, tag='Context.identify_state_changes', temp=0.6, stops=['</end>'], max_tokens=100)
        return response

    async def update(self, local_only=False, changes_only=False):
        """Rewrite the situation description from what has happened since the last update."""
        if local_only and survival.config(self):
            # the situation is rendered from the world (current_state): nothing to rewrite
            self.last_update_time = self.simulation_time
            return
        history = self.history()

        prompt = [UserMessage(content="""You are a dynamic world. Your task is to update the environment description. 
Include day/night cycles and weather patterns. It is now {{$time}}.

Update location and other physical situation characteristics as indicated in the History.
Your response should be concise, and only include only an update of the physical situation.

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
Updated State description of up to 125 words
</situation>

Respond with an updated world state description of at most 125 words reflecting the current time and the environmental changes that have occurred.
Include ONLY the updated situation description in your response. Do not include changes of the description (e.g., 'more detailed'), only changes to the situation being described.
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
.
Limit your total response to 125 words at most.
Ensure your response is surrounded with <situation> and </situation> tags as shown above.
End your response with:
</end>""")]

        response = self.llm.ask({"situation": self.current_state, 'history': history, 'time': self.simulation_time.isoformat()}, prompt,
                                tag='Context.update', temp=0.6, stops=['</end>'], max_tokens=270)
        new_situation = xml.find('<situation>', response)       
        # Debug prints
        if not local_only and not changes_only:
            self.message_queue.put({'name':'', 'text':f'\n\n----- situation update -----{self.simulation_time.isoformat()}'})
            self.transcript.append(f'\n\n----- situation update ----- {self.simulation_time.isoformat()}\n')
            await asyncio.sleep(0.1)
         
        if new_situation is None:
            return
        previous_state = self.current_state
        self.current_state = new_situation
        if local_only:
            self.last_update_time = self.simulation_time
            return
        changes = self.identify_state_changes(previous_state, new_situation)
        if not changes_only:
            self.show = new_situation
            self.message_queue.put({'name':'', 'text':self.show})
            self.transcript.append(f'{self.show}')
        else:
            self.message_queue.put({'name':'', 'text':changes})
            self.transcript.append(f'{new_situation}')

        self.show = '' # has been added to message queue!
        self.message_queue.put({'name':self.name, 'text':f'world_update', 'data':await self.world_image_json()})
        await asyncio.sleep(0.1)

        await asyncio.sleep(0.1)

        updates = self.world_updates_from_act_consequences(new_situation)
        # self.current_state += '\n'+updates
        print(f'World updates:\n{updates}')
        for actor in self.actors+self.extras:
            actor.add_to_history(f"you notice {updates}\n")
            actor.update()  # update history, etc
            
        # Record character states for staleness detection - moved to run_scene
        #if hasattr(self, 'staleness_detector'):
        #    # Run staleness analysis periodically
        #    try:
        #        staleness_analysis = await self.staleness_detector.analyze_staleness(changes)
        #        if staleness_analysis and self.staleness_detector.should_trigger_intervention(staleness_analysis):
        #            await self.inject_staleness_intervention(staleness_analysis)
        #    except Exception as e:
        #        traceback.print_exc()
        #        print(f"Staleness detection error: {e}")
                
        return response


    async def inject_staleness_intervention(self, analysis: Dict[str, Any]):
        """Inject an intervention to combat narrative staleness"""
        intervention_type = analysis.get('intervention_type', 'environmental')
        intervention_description = analysis.get('intervention_description', 'A sudden change occurs')
        staleness_score = analysis.get('staleness_score', 0)
        primary_factors = analysis.get('primary_factors', [])
        
        print(f"\n🎭 STALENESS DETECTED (Score: {staleness_score})")
        print(f"   Factors: {', '.join(primary_factors)}")
        print(f"   Injecting {intervention_type} intervention: {intervention_description}")
        # The director may bring new things in, in the world's terms (sim/record.py). What cannot
        # be put in those terms is taken out before anyone hears of it; an intervention that was
        # nothing else is not made at all (it used to reach every character as a percept, and
        # the next scene was rewritten around it).
        cast = [c for c in self.actors if getattr(c, 'body', None) is not None]
        if survival.config(self) and cast and ablation.enabled('narration_gate'):
            hazards = [h['name'].lower() for h in (survival.config(self) or {}).get('hazards', [])]
            newcomers = bool(analysis.get('entities', {}).get('new_character'))
            found = record.check(self, cast[0], 'The director brings an event into the play.', None,
                                 intervention_description, people=cast) or []
            claims = [(k, w) for k, w in found if k != 'place' and not (k == 'presence' and (newcomers or (
                any(h in w.lower() for h in hazards) and not self._hazard_claim_stands(w))))]
            made, refused, remaining = record.admit(self, cast[0], claims, intervention_description, 'director', people=cast)
            trace.event('intervention_check', made=made, refused=[f'{k}: {w}' for k, w in remaining], text=intervention_description)
            if remaining:
                # an intervention is one event: with a part of it refused (the wounded dog), what is
                # left ('stopping to tend the animal') stands on nothing, so none of it is made
                return None

        for character in analysis['entities']['new_character']:
            #<new_character name="Storm Chaser" 
            # description="A mysterious outsider caught in the storm, bringing urgent news." 
            # motivation="To seek shelter and reveal critical information." 
            # drives="Revelation"/>
            character_name = map.normalize_name_for_enum_lookup(character['name'])
            character_description = character['description'] + '. '+character['motivation']
            new_character = self.get_npc_by_name(character_name, character_description, x=0, y=0, create_if_missing=True)
            new_character.drives = [Drive(drive.strip()) for drive in character['motivation'].split('.')]
            self.extras.append(new_character)

        for state_change in analysis['entities']['state_change']:
            print(f"State change: {state_change['name']} - {state_change['drive']} - {state_change['value']}")
            # the drive is named by id, text or a paraphrase ('love_and_belonging'); it was looked
            # up by exact text, so no intervention ever changed a drive
            character, _ = self.resolve_character(state_change['name'])
            drive = drive_model.nudge(character, state_change['drive'], up='-' not in state_change['value']) if character else None
            if drive is None:
                logger.warning(f"staleness intervention: no drive '{state_change['drive']}' for {state_change['name']}")

        # Inject the intervention into the world state
        self.current_state += f"\n\n{intervention_description}"
            
        # Notify all characters about the intervention
        for actor in self.actors + self.extras:
            actor.add_perceptual_input(intervention_description, mode='environmental')
            
        # a director's note: it reaches the characters (above) and the trace, not the audience
        trace.event('staleness_intervention', type=intervention_type, text=intervention_description)
            
        await asyncio.sleep(0.1)
        self.previous_interventions.append(intervention_description)
        return intervention_description
    
    async def generate_environmental_intervention(self):
        """Generate an environmental intervention event"""
        # Get current location context
        location_context = ""
        for actor in self.actors:
            if hasattr(actor, 'mapAgent'):
                terrain = actor.mapAgent.world.patches[actor.mapAgent.x][actor.mapAgent.y].terrain_type
                location_context = f"Characters are in {terrain.name.lower()} terrain"
                break
        
        environmental_events = [
            f"The weather suddenly changes, bringing unexpected {random.choice(['rain', 'wind', 'fog', 'sunlight'])}",
            f"A distant {random.choice(['sound', 'light', 'smoke', 'movement'])} catches everyone's attention",
            f"The ground {random.choice(['trembles slightly', 'shifts', 'reveals something hidden'])}",
            f"An unexpected {random.choice(['door opens', 'path appears', 'structure becomes visible'])}",
            f"The environment {random.choice(['grows quieter', 'becomes more active', 'changes temperature'])}"
        ]
        
        return random.choice(environmental_events)
    
    async def generate_character_intervention(self):
        """Generate a character-based intervention event"""
        character_events = [
            f"A stranger approaches the area, their intentions unclear",
            f"One of the characters suddenly remembers something important",
            f"Someone receives unexpected news or information",
            f"A character experiences a sudden emotional shift",
            f"An urgent need or problem suddenly arises for someone"
        ]
        
        return random.choice(character_events)
    
    async def generate_resource_intervention(self):
        """Generate a resource-based intervention event"""
        resource_events = [
            f"Something valuable is discovered nearby",
            f"An important tool or resource becomes available",
            f"A useful object is found in an unexpected place",
            f"Access to a needed resource suddenly becomes possible",
            f"A new opportunity or option becomes apparent"
        ]
        
        return random.choice(resource_events)

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

    def set_widget(self, entity, widget):
        """Maintain widget references for PyQt UI"""
        self.widget_refs[entity] = widget
        entity.widget = widget

    def get_widget(self, entity):
        """Get widget reference for entity"""
        return self.widget_refs.get(entity)


    def format_tasks(self, tasks, labels):
        task_list = []
        for task, label in zip(tasks, labels):
            task_text = f'{label} - {hash_utils.find("name", task)} ({hash_utils.find("description", task)}), {hash_utils.find("reason", task)}; ' 
            task_dscp= task_text + f' Needs: {hash_utils.find("needs", task)}; committed: {hash_utils.find("committed", task)}'
            task_actor_names = hash_utils.find('actors', task).split(',')
            task_memories = set()  
            for actor_name in task_actor_names:
                actor, actor_name = self.resolve_character(actor_name.strip())
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
                    task_memories.add(memory.to_string())  # Add each memory text individually

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
        {'\n        '.join([memory.to_string() for memory in actor.structured_memory.get_recent(6)])}
    </memories>
</actor>"""
        return mapped_actor


    def check_resource_has_npc(self, resource):
        """Check if a resource type should have an NPC"""
        for allocation in self.map._resource_rules['allocations']:
            if allocation['resource_type'] == resource['type']:
                return allocation.get('has_npc', False)
        return False

    def add_dynamic_resource(self, resource_type_name: str, description: str = "", 
                           terrain_weights: dict = None, requires_property: bool = False,
                           count: int = 1, auto_place: bool = True) -> list:
        """Add a new dynamic resource type and optionally place instances.
        
        This is the main interface for the staleness detection system to add new resources.
        
        Args:
            resource_type_name: Name of the new resource type (e.g., "ArtInstallation")
            description: Description of the resource
            terrain_weights: Dict mapping terrain names to placement weights (e.g., {"Plaza": 2.0})
            requires_property: Whether the resource requires a property to be placed
            count: Number of instances to place
            auto_place: Whether to automatically place the resource instances
            
        Returns:
            List of placed resource IDs if auto_place=True, empty list otherwise
        """
        # Add the resource type to the registry
        resource_type = self.map.add_dynamic_resource_type(resource_type_name, description)
        
        placed_resources = []
        if auto_place and count > 0:
            # Place the resources on the map
            placed_resources = self.map.place_dynamic_resource(
                resource_type_name=resource_type_name,
                description=description,
                terrain_weights=terrain_weights,
                requires_property=requires_property,
                count=count
            )
            
            # Log the placement
            if placed_resources:
                print(f"Dynamic resource system: Placed {len(placed_resources)} {resource_type_name}(s)")
                for resource_id in placed_resources:
                    resource = self.map.resource_registry[resource_id]
                    location = resource['location']
                    print(f"  {resource['name']} at ({location[0]}, {location[1]})")
            else:
                print(f"Dynamic resource system: Failed to place {resource_type_name} - no suitable locations found")
        
        return placed_resources

    def map_actor(self, actor):
        mapped_actor = f"""{actor.name}: {actor.focus_goal.to_string() if actor.focus_goal else ""}\n   {actor.focus_task.peek().to_string() if actor.focus_task.peek() else ""}\n  {actor.focus_action.to_string() if actor.focus_action else ""}"""
        return mapped_actor+'\n  Remaining tasks:\n    '+'\n    '.join([task.to_string() for task in actor.focus_goal.task_plan]) if actor.focus_goal else ''

    def map_actors(self):
        mapped_actors = []
        for actor in self.actors:
            mapped_actors.append(self.map_actor(actor))
        return '\n'.join(mapped_actors)

        
    def compute_task_plan_limits(self, character, scene):
        """Compute the task plan limits for an actor for a scene"""
        action_order = scene.get('action_order', ['a','b'])
        if character.name not in action_order:
            return 1
        #beats_in_scene = len(action_order)
        my_beats = action_order.count(character.name)
        try:
            #character_task_limit = scene.get('task_budget', int(1.5*len(beats_in_scene)))/len(beats_in_scene)
            character_task_limit = my_beats
        except Exception as e:
            print(f'Error computing task plan limits: {e}')
            character_task_limit = 1
        return int(min(3, character_task_limit))


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
                # What is at issue in an act is the director's to know. It was written into each
                # listed character's memory and into what it thought of the other ('Issue:
                # Samantha suspects Joe ... with Samantha' was Joe's own memory before they had
                # spoken). Scene goals carry it to the actors, in their notes.
                trace.event('tension_point', characters=[c.name for c in characters], issue=issue,
                            resolution_requirement=resolution_requirement)
            except Exception as e:
                print(f'Error establishing tension points: {e}')

    async def reground_scene(self, scene, cast, opens_act=False):
        """Fit a planned scene to where the play now stands. An act's scenes are all written
        before any is played, so a later scene can assume a world that has moved on: 'March to
        the Spring' opened with the cast at the spring, and scenes walked them back to a place
        they had left so the plan could happen there. One LLM call just before the scene keeps
        its purpose and goals and corrects place, set-up and ending in place. Returns True when
        the scene's purpose has already happened and it should not be played.
        The result is then held to the world's record like any narration (sim/record.py): a plan
        is where an injury nobody has, or a cache of supplies the world does not hold, used to
        come in, and everything after played along. opens_act: the scene keeps its planned place
        and is not skipped."""
        prompt = [UserMessage(content="""You are the stage manager of an improvised play. The next scene was planned before the earlier scenes of this act were played. Fit it to where things actually stand, keeping its dramatic purpose.

<planned_scene>
{{$scene}}
</planned_scene>

<where_things_stand>
Everyone in the play, and whether each is he or she: {{$cast}}.
It is now {{$clock}}.
{{$places}}
Bodies and what each carries: {{$bodies}}
Dangers this world has: {{$hazards}}. Now: {{$danger}}.
{{$clues}}
What has happened lately:
{{$events}}
</where_things_stand>

Rules:
- location: "same" unless the scene's action cannot happen where the cast now is. Never send them back to a place they have just left only because the plan named it.
- A scene is played in one place with its cast together. The set-up does not send them apart or off on separate errands; if the plan has them part and meet again, set the scene where they meet.
- The set-up must start from what is true now: nobody arrives where they already are, wants what their body already has (see Bodies), or lacks what they already hold.
- when: first settle what part of the day the scene is played in. {{$moving}}
- Then write the set-up and ending for the time the scene is played in, and make the title fit it. Do not state the time of day in the set-up unless it matters to what happens: the heading gives it.
- The scene is played in a world that keeps count. It may use what is there: the places listed, what the cast carries, and the dangers named above (a set-up that brings one on makes it real). It may also bring in something new, which is then real for the rest of the play, so bring in only what the plan truly needs: an object, a source of water, food or shelter. No animal or danger other than those named. Do not write as already so what Bodies and the list do not show (nobody is 'still injured' who is unhurt).
- The world, not you, decides what happens to bodies and how a danger ends, and it decides when the scene is played. The set-up may bring a danger on; neither set-up, ending nor goals say that it strikes, whom it hurts, that it is driven off, or that anyone is wounded, healed or fed. Where the plan says so, keep what the scene is for and leave that out.
- If the scene turns on a thing being found, read, shown or handed over (a note, a tool, a mark on a wall), the set-up puts that thing in the scene and says plainly where it lies, so that it is there to be found: "Under the bedding lies a torn page of handwriting." A thing only hinted at ("something left behind") is not in the world, and nobody can find it.
- The ending is the turn the scene should reach between the characters: something chosen, refused, admitted or found out, or how they now stand with each other, shown in what they do. Nobody announces a feeling or a choice: 'he says aloud that he stays' is not an ending; his staying is. It is not an event of the body or of the danger.
- If a danger is on the cast now (see Now), the scene opens with it and is about meeting it; do not skip the scene.
- If everything the scene was for has already happened, skip it.
- Otherwise change as little as you can.
{{$rejected}}

Respond with these lines only:
#skip yes or no
#when same, or one of: dawn, morning, midday, afternoon, dusk, night
#location same, or the exact name of a place listed above
#title the scene's title: as planned, unless it names a time of day or a thing that is no longer so
#setup the scene's set-up, one or two sentences, true to where things stand
#ending what the scene should arrive at, one sentence
#goal <character name>: that character's goal in the scene (one line for each character)
#unknown <character name>: one thing that character does NOT know as the scene opens, of what the set-up, ending or goals speak of
</end>

On #unknown: a character knows what it has lived (see what has happened lately) and what concerns itself. It does not know another's name before it has been told it, nor another's condition, memories, intentions or secrets, nor what is going to happen, until these are shown or said in its presence. How the scene will turn out is not one of these: nobody knows that, and it needs no line. Give a line for each such thing the scene turns on, as many lines as are needed, and none for a character who is missing nothing. These go to the actors, so that each plays what its character has still to find out.
{{$strangers}}
""")]
        if opens_act and not survival.config(self):
            return False   # no world to hold an opening scene to
        planned = {k: scene.get(k) for k in ('scene_title', 'location', 'when', 'pre_narrative', 'post_narrative', 'characters', 'business', 'tempo')}
        # not while a danger is on the cast: it would be left standing, hours behind them
        may_move = self._may_move_on(opens_act) and not getattr(self, 'active_threat', None)
        moving = ('"same" means now, and is right for most scenes. Name a later part of the day only when what the scene is for depends on it '
                  '(its title, set-up or ending call for night, or dusk, or dawn, and it is not that now): the play then moves on to the next such time, '
                  'hours pass off stage, and the cast is taken to have eaten, drunk and rested in them, so the scene is not about that.'
                  if may_move else
                  'It must be "same": the play has already moved on in this act and is played now. Where the plan calls for another time of day, '
                  'keep the scene\'s purpose and write it for now.')
        hazards = [h['name'].lower() for h in (survival.config(self) or {}).get('hazards', [])]
        threat = getattr(self, 'active_threat', None)
        inputs = {'scene': json.dumps(planned, indent=1, default=datetime_handler),
                  'places': stage.gazetteer(self), 'clock': stage.clock(self), 'moving': moving, 'cast': record.cast(self),
                  'strangers': self.strangers_note(),
                  'clues': self.screenplay.clue_note(),
                  'bodies': '; '.join(f'{c.name}: {c.body.summary()}' for c in cast if getattr(c, 'body', None) is not None) or 'not tracked',
                  'hazards': ((', '.join(hazards) or 'none') if not survival.hazard_spent(self) or threat else
                              'none may come on now: this play has had its danger (it may be spoken of as past)'),
                  'danger': threat['hazard']['description'] if threat else 'none',
                  'events': (' '.join(self.act_scene_wrapups[-2:]) + '\n' + self.recent_events_text(6)).strip()}
        checked = bool(survival.config(self)) and bool(cast) and ablation.enabled('narration_gate')
        changed, claims, note, brought = {}, [], '', []
        for attempt in range(2):
            reply = self.llm.ask({**inputs, 'rejected': note}, prompt, tag='reground_scene', temp=0.2, stops=['</end>'], max_tokens=460) or ''
            find = lambda tag: (hash_utils.find(tag, reply) or '').strip()
            if find('skip').lower().startswith('y') and not opens_act:
                return True
            if attempt == 0 and may_move:
                # the scene's time is established before anything else is settled about it: the
                # set-up was written for that time, and the check that follows reads the clock
                target = stage.next_time(self.simulation_time, find('when'))
                if target is not None and self.move_on(target) >= survival.REAL_JUMP_HOURS:
                    self._act_jumps = getattr(self, '_act_jumps', 0) + (0 if opens_act else 1)
                    changed['when'] = scene['when'] = stage.part_of(self.simulation_time)
                    inputs.update(clock=stage.clock(self), moving='It must be "same": the play has just moved on to this time.',
                                  bodies='; '.join(f'{c.name}: {c.body.summary()}' for c in cast if getattr(c, 'body', None) is not None) or 'not tracked')
            for key, value in (('location', '' if opens_act else find('location')), ('pre_narrative', find('setup')),
                               ('post_narrative', find('ending')), ('scene_title', find('title').strip('"\''))):
                if value and value != scene.get(key):
                    changed[key] = scene[key] = value
            for name, goal in re.findall(r'#goal\s+([^:\n]+):\s*(.+)', reply):
                for key, entry in (scene.get('characters') or {}).items():
                    if key.lower() == name.strip().lower() and isinstance(entry, dict) and goal.strip():
                        entry['goal'] = goal.strip()
            # what each character has still to find out: a note for its actor (Character.director_notes)
            for entry in (scene.get('characters') or {}).values():
                if isinstance(entry, dict):
                    entry['unknown'] = []
            for name, missing in re.findall(r'#unknown\s+([^:\n]+):\s*(.+)', reply):
                for key, entry in (scene.get('characters') or {}).items():
                    if key.lower() == name.strip().lower() and isinstance(entry, dict) and missing.strip():
                        entry['unknown'].append(missing.strip())
            if not checked:
                break
            goals = ' '.join(f"{name} is to: {(entry or {}).get('goal', '')}" for name, entry in (scene.get('characters') or {}).items())
            told = f"{scene.get('pre_narrative', '')}\nThe scene is to end: {scene.get('post_narrative', '')}\n{goals}"
            # what lies at the scene's place is there to be written of, though the cast has not
            # walked there yet (the cup planted at the spring was refused from the spring scene's
            # set-up, and Joe asked who carved a cup the audience had not been shown)
            target = stage.resolve(self, scene.get('location'), stage.here(cast[0]))
            there = [p['name'] for hidden in (False, True) for p in survival.lying(self, target[0], hidden=hidden)] if target else []
            found = record.check(self, cast[0], f"A scene is set at {scene.get('location') or 'the same place'}; the cast walks there if it is elsewhere.",
                                 f"Lying at {target[1]}, where the scene is set: {', '.join(there)}" if there else None,
                                 told, people=cast) or []
            # where the scene is set is the plan's to say, and a danger of this world that the
            # set-up brings on is made real (survival.stage_hazard)
            claims = [(kind, words) for kind, words in found
                      if kind != 'place' and not (kind == 'presence' and not self._hazard_claim_stands(words, scene))]
            if not claims:
                break
            # a plan may bring new things into the world: what can be put in the world's terms is
            # made real where the scene is set (the injury is then in the body, the pool on the map)
            target = stage.resolve(self, scene.get('location'), stage.here(cast[0]))
            made, refused = [], []
            if attempt == 0:   # once: the rewrite that follows is only to take out what was refused
                made, refused, claims = record.admit(self, cast[0], claims, told, 'director', people=cast,
                                                     at=target[0] if target else None)
            brought += made
            if not claims:
                break
            note = record.rejection(claims, made, refused)
        if claims:
            # Still there after one rewrite: it goes from everything the scene hands on (set-up,
            # ending and goals), not only from what the audience reads. A goal that was nothing
            # but the refused thing becomes the scene's plain purpose.
            for key in ('pre_narrative', 'post_narrative'):
                kept = record.strip(scene.get(key, ''), claims)
                if kept != scene.get(key, ''):
                    changed[key] = scene[key] = kept
            for name, entry in (scene.get('characters') or {}).items():
                if isinstance(entry, dict) and entry.get('goal'):
                    # a goal that was nothing but the refused thing: the character's part in
                    # this act, which is what it is there for (it used to become a line that
                    # asked nothing of anyone: 'Meet what this scene brings as it is')
                    role = next((r for n, r in ((self.act_frame or {}).get('roles') or {}).items() if n.lower() == str(name).lower()), '')
                    entry['goal'] = (record.strip(entry['goal'], claims) or role
                                     or f"Meet what this scene brings ('{scene.get('scene_title', '')}') as it is.")
        if checked:
            trace.event('scene_check', title=scene.get('scene_title'),
                        outcome='cut' if claims else 'rewritten' if note else 'admitted' if brought else 'ok',
                        claims=[f'{k}: {w}' for k, w in claims], made=brought, note=note.strip())
        if changed:
            trace.event('scene_regrounded', title=scene.get('scene_title'), **changed)
        return False

    async def run_scene(self, scene, act=None, opens_act=False):
        """Run a scene. The cast stays where the previous scene left it unless this scene names
        another place (sim/stage.py); opens_act: the first scene of an act may jump there."""
        print(f'Running scene: {scene["scene_title"]}')
        new_scene = await self.request_scene_choice(scene)
        if new_scene is None:
            logger.error(f'No scene to run: {scene["scene_title"]}')
            return
        scene = new_scene
        await asyncio.sleep(0.1)
        # A scene starts when the one before it ended, unless it needs another part of the day:
        # then the play moves on to it (stage.next_time), which the pre-scene step settles
        # (reground_scene, '#when'). The clock times a plan writes are start-plus-durations
        # arithmetic ('The Pact of the Night', set-up 'Night approaches', stamped 07:20) and are
        # not used, except that a time hours ahead on a scene with no 'when' is a jump it asked for.
        self._hours_moved = 0.0
        planned = self.validate_scene_time(scene)
        if (stage.part_named(scene.get('when')) is None
                and (planned - self.simulation_time).total_seconds() >= survival.REAL_JUMP_HOURS * 3600):
            if opens_act and not survival.config(self):
                self.move_on(planned)   # no pre-scene step in a world without mechanics: the jump is made here
            else:
                # asked of the pre-scene step, which moves the play on only for a scene that is
                # played (a scene moved the clock seven hours, was skipped as already done, and
                # the next act opened at dusk without a word)
                scene['when'] = stage.part_of(planned) + (' next day' if planned.date() > self.simulation_time.date() else '')
        survival.pass_offstage(self, timedelta(0))
        self.current_scene = scene
        self.scene_events = []
        self._scene_transcript_start = len(self.transcript)

        #construct a list of characters in the scene in the order in which they appear
        characters_in_scene: List[Character] = []
        for character_name in scene['action_order']:
            if character_name == 'Context':
                continue
            character_name = character_name.capitalize().strip()
            character, canonical_character_name = self.resolve_character(character_name)
            if character is None:
                print(f'Character {character_name} not found in scene {scene["scene_title"]}')
                character = self.get_npc_by_name(character_name, create_if_missing=True)
            else:
                character_name = canonical_character_name
            if character not in characters_in_scene:
                characters_in_scene.append(character)

        # rarely, the director brings on one of the world's dangers; the scene is then fitted to it
        survival.director_hazard(self, characters_in_scene, (act or {}).get('act_number'))
        if await self.reground_scene(scene, characters_in_scene, opens_act):
            trace.event('scene_skipped', title=scene.get('scene_title'))
            self.current_scene = None
            return ''

        # the cast gathers at the scene's place (walking there, unless the scene opens an act);
        # the heading names the place the scene is actually played ('same' said nothing)
        scene_start = scene["time"] = self.simulation_time   # as established: where the last scene left the clock, or moved on
        hours_between = self._hours_moved
        arrival = stage.set_scene(self, scene, characters_in_scene, opens_act)
        survival.stage_hazard(self, scene, characters_in_scene)   # a beast the scene's set-up brings on is a real one
        # The heading and the set-up name the same place: when the set-up names one place and the
        # cast is standing at it, that is where the scene is ('Setting: the apple tree' headed a
        # set-up that began 'At the spring': both were two steps away).
        named = set(stage.places_named(self, scene.get('pre_narrative') or ''))
        if len(named) == 1 and characters_in_scene and getattr(characters_in_scene[0], 'mapAgent', None) is not None:
            rid = next(iter(named))
            if stage._distance(stage.here(characters_in_scene[0]), self.map.resource_registry[rid]['location']) <= stage.AT_PLACE_TILES:
                scene['place'] = rid
        heading = f'{scene["scene_title"]}\n    Setting: {scene["place"]}, {stage.clock(self, date=True)}'   # after the walk there
        self._state_notes = ''   # the situation's notes are those of the scene in progress
        if hours_between >= 2:   # the day went from half past eight to seven in the evening without a word
            self.message_queue.put({'name': '', 'text': f"    {'A day and more passes' if hours_between >= 30 else f'About {round(hours_between)} hours pass'}."})
        self.message_queue.put({'name':self.name, 'text':f' -----scene----- {heading}'})
        self.current_state += f'\n{heading}'
        await asyncio.sleep(0.1)
        trace.event('scene_start', title=scene.get('scene_title'), location=scene.get('location'),
                    pre_narrative=scene.get('pre_narrative'), planned_post=scene.get('post_narrative'),
                    goals={name: (c or {}).get('goal') for name, c in (scene.get('characters') or {}).items()},
                    action_order=scene.get('action_order'))
        self.scene_pre_narrative = ''
        if scene.get('pre_narrative'):
            self.current_state += '\n\n'+scene['pre_narrative']
            self.scene_pre_narrative = scene['pre_narrative']
            self.message_queue.put({'name':self.name, 'text':f'    {scene["pre_narrative"]}'})
            # The planned set-up is the scene's opening as written. (An LLM "situation update" here
            # paraphrased it and put characters in the scene who were not: 3 calls per scene.)
            self.message_queue.put({'name':self.name, 'text':'world_update', 'data':await self.world_image_json()})
            await asyncio.sleep(0)
        if arrival:
            self.add_event(arrival)
            self.current_state += '\n' + arrival
            self.message_queue.put({'name': self.name, 'text': f'    {arrival}'})

        if scene.get('post_narrative'):
            self.scene_post_narrative = scene['post_narrative']

        scene_goals = {}
        for character in characters_in_scene:
            character.current_scene = scene
            character.scene_deeds = []   # the world's actions it performs this scene (survival._record)
        self.held_out = []   # things held out to someone in this scene, theirs to take (survival._prop_do)

        # A choice the scene turns on is made before the scene is played and shapes that
        # character's goal in it. (It was asked for after the scene, from the plan, so Samantha
        # 'decided' to follow Joe to a spring she had already reached.) The choice and its
        # reason are the character's own: they are not lines of the play.
        for character in characters_in_scene:
            planned = next((v for k, v in (scene.get('characters') or {}).items() if str(k).lower() == character.name.lower()), None) or {}
            character.direction = {'title': scene.get('scene_title', ''), 'place': scene.get('place', ''),
                                   'setup': scene.get('pre_narrative', ''), 'ending': scene.get('post_narrative', ''),
                                   'goal': planned.get('goal', '') if isinstance(planned, dict) else '',
                                   'decision': None, 'unknown': list(planned.get('unknown') or []) if isinstance(planned, dict) else []}
        decided = {}
        for decision in await self.check_decision_required(act, scene) or []:
            character, character_name = self.resolve_character(decision.get('character', ''))
            if character is None or character not in characters_in_scene or character.name in decided:
                continue   # one choice for a character in a scene
            choice_dict = await character.decide(decision, act=act, scene=scene)
            if choice_dict and choice_dict.get('choice'):
                choice, reason = choice_dict['choice'], choice_dict.get('reason', '')
                drive_model.denied(character, choice_dict.get('drive_denied'), choice)
                # a choice the character will come to in the scene: a note for the actor, not a
                # memory of having already made it ('I have decided: go with Joe to find water',
                # remembered by a woman who had not yet spoken to him)
                character.direction['decision'] = (choice, reason)
                decided[character.name] = choice
                trace.event('decision', character=character.name, choices=decision.get('choices'), choice=choice, reason=reason)

        #now that all characters are in place, establish their goals
        for character in characters_in_scene:
            # memory and the character's own account of where things stand are brought up to date
            # before its goal and plan are made from them (this ran after, so goals and plans, and
            # the Explore panel, read an account one scene old)
            character.update(now=True)
            character.look('Look', act_arg='', reason=f'{scene["pre_narrative"]}') # important to do this after everyone is in place.
            if character.name.lower() == 'interviewer':
                print(f'{character.name} {scene["pre_narrative"]}')

            # the goal the notes were just built with (it was looked up a second time here, by
            # another spelling of the name, and could come back empty)
            goal_text = character.direction.get('goal') or ''
            # (a choice made before the scene was added here as 'You have decided: ...', and the
            # character's goal was then built from the choice, not from the plan; the choice is
            # in the actor's notes, as something the character comes to)
            business = stage.pending_business(character)
            if business:
                goal_text = f"{goal_text} In this scene you must actually do (a Do, not talk): {', '.join(business)}."
            # instatiate narrative goal sets goals and focus goal as side effects
            scene_goals[character.name] = character.instantiate_narrative_goal(goal_text)
            await character.request_goal_choice(scene_goals[character.name], narrative=True)
            self.message_queue.put({'name':character.name, 'text':'character_update', 'data':character.to_json()})
            await asyncio.sleep(0.4)
            # now generate initial task plan
            if character.focus_goal:
                await character.generate_task_plan(character.focus_goal, ntasks=(scene['action_order']).count(character.name)+1)
            else:
                # instantiate_narrative_goal returns no goal when its LLM call fails; that raised
                # ValueError here and ended the whole narrative
                logger.warning(f'{character.name}: no scene goal (LLM failure?), no task plan this scene')
                self.message_queue.put({'name':'', 'text':f'{character.name} has no goal for this scene'})
            self.message_queue.put({'name':character.name, 'text':'character_update', 'data':await character.to_json_with_image()})
            self.message_queue.put({'name':character.name, 'text':f'character_detail', 'data':character.get_explorer_state()})
            await asyncio.sleep(0.4)

        # ok, actors - live!
        scene_integrated_task_plan = await self.integrate_task_plans(scene)
        self.scene_history = []
        self.scene_integrated_task_plan_index = 0 # using explicit index allows cog cycle to insert tasks!
        self.scene_task_budget = len(self.scene_integrated_task_plan)  # see place_commitment
        # where the scene stood when its tasks were planned: a task is looked at again
        # (refresh_task) only in the light of what has happened since
        self._scene_plan_mark = (len(self.scene_events), len(self.transcript))
        trace.event('scene_plan', entries=[{'actor': e['actor'].name, 'task_id': e['task'].id, 'task': e['task'].name}
                                           for e in self.scene_integrated_task_plan])
        while self.scene_integrated_task_plan_index < len(self.scene_integrated_task_plan):
            scene_task = self.scene_integrated_task_plan[self.scene_integrated_task_plan_index]
            pending = len(self.scene_integrated_task_plan) - self.scene_integrated_task_plan_index
            if scene_task.get('optional') and pending > self.scene_task_budget:
                # an optional (RELEVANT) commitment, and the scene is already over budget
                logger.info(f"optional commitment skipped ({pending} pending > budget {self.scene_task_budget}): "
                            f"{scene_task['actor'].name} {scene_task['task'].name}")
                trace.event('optional_skipped', character=scene_task['actor'].name, task_id=scene_task['task'].id)
                self.scene_integrated_task_plan_index += 1
                continue
            character = scene_task['actor']
            if stage.off_stage(character):   # it has left: what it had planned for here is not done here
                trace.event('task_skipped', character=character.name, task=scene_task['task'].name, why='left the scene')
                self.scene_integrated_task_plan_index += 1
                continue
            character_name = character.name
            task = scene_task['task']
            character.focus_goal = scene_task['goal']
            character.focus_goal.task_plan = [task]
            # refresh task to ensure it is up to date with the latest information
            task = character.refresh_task(task, self.scene_integrated_task_plan, 
                                          final_task=self.scene_integrated_task_plan_index == len(self.  scene_integrated_task_plan)-1)
            #await character.request_task_choice([task]) #not needed, cog_cycle will do it
            character.focus_task = Stack()
            character.focus_task.push(task)
            trace.event('scene_task', character=character.name, task_id=task.id, task=task.name,
                        planned_task_id=scene_task['task'].id, index=self.scene_integrated_task_plan_index,
                        inserted=scene_task.get('inserted_at') is not None, optional=bool(scene_task.get('optional')))
            self.scene_history.append(f'{character.name} {task.to_string()}') # should this be after cognitive cycle? But what if spontaneous subtasks are generated?
            #self.message_queue.put({'name':'', 'text':f''})
            await asyncio.sleep(0.2)
            while self.step is False and  self.run is False:
                await asyncio.sleep(0.5)
            await character.step_task()
            survival.hazard_tick(self)
            self.message_queue.put({'name':character.name, 'text':f'character_detail', 'data':character.get_explorer_state()})
            self.step = False

            await asyncio.sleep(1.0)
            self.scene_integrated_task_plan_index += 1
            # a scene between people is over when they have gone: the one left behind went on
            # trying to bed down beside a woman who had walked out (bellona)
            still_here = [c for c in characters_in_scene if not stage.off_stage(c)]
            if len(characters_in_scene) > 1 and len(still_here) <= 1:
                trace.event('scene_emptied', left=[c.name for c in characters_in_scene if stage.off_stage(c)],
                            remaining=[c.name for c in still_here])
                break
        
        # The scene's planned business is done before the scene closes: what the plan and the
        # talk both assume happened (Joe said 'my turn' at the spring and never drank) happens.
        # Each does their own at the same time: two four-hour sleeps took the clock on eight hours.
        closing, latest = self.simulation_time, self.simulation_time
        for character in characters_in_scene:
            if stage.off_stage(character):
                continue   # no closing business for someone who has gone (Ding walked out, then 'had a drink at Cafe')
            self.simulation_time = closing
            for label in stage.pending_business(character):
                fact = survival.do_business(self, character, label)
                if fact:
                    self.add_event(f'{character.name} -> {fact}')
                    character.add_perceptual_input(fact, percept=False, mode='internal')
                    told = re.sub(rf'^{re.escape(character.name)}\s+', '', stage.stage_line(character.name, '', fact).lstrip('. '))
                    self.message_queue.put({'name': character.name, 'text': told, 'line_kind': 'action'})
            latest = max(latest, self.simulation_time)
        self.simulation_time = latest
        true_outcomes = ''
        outcomes = await self.check_post_state_ambiguity(self.current_scene, scene=scene)
        if outcomes:
            true_outcomes = '\n'.join([outcome.get('outcome') for outcome in outcomes if outcome.get('test') == True])
        # The planned post_narrative is kept: overwriting it with the outcome labels ("Alliance
        # established") fed abstract labels to every character and to the next scene. The wrap-up
        # now says what physically happened, plus the assessed outcomes.
        self.current_scene['outcomes'] = outcomes
        # each event is 'HH:MM who did what -> visible result': keep the results, the latest few
        results = [e.split(' -> ', 1)[1] if ' -> ' in e else e.split(' ', 1)[-1] for e in self.scene_events]
        happened = ' '.join(r.split('. ')[0].rstrip('.') + '.' for r in results[-4:]) or 'little changed physically'
        happened = happened.rstrip('.')
        wrapup = f"Scene '{self.current_scene.get('scene_title', '')}' ends. What happened: {happened}."
        if true_outcomes:
            wrapup += f" Assessed outcome: {true_outcomes.replace(chr(10), '; ')}."
        self.current_scene['wrapup'] = wrapup
        self.act_scene_wrapups.append(wrapup)
        self.current_state += '\n' + wrapup
        # what happened is the characters' to remember; what the director judged it came to is not
        lived = wrapup.split(' Assessed outcome:')[0]
        for character in characters_in_scene:
            character.direction = None   # the notes were for this scene
            character.add_perceptual_input(lived, mode='internal')
            character.reason_over(lived)
            self.message_queue.put({'name':character.name, 'text':'character_update', 'data':await character.to_json_with_image()})
            self.message_queue.put({'name':character.name, 'text':f'character_detail', 'data':character.get_explorer_state()})
        shown = re.sub(r'\s*\([^)]*\)', '', true_outcomes.replace(chr(10), '; '))   # '(Actual State)'
        self.message_queue.put({'name':'', 'text': f" ---- {shown}\n" if shown else ' ----\n'})
        await asyncio.sleep(0.1)

        staleness = None
        if hasattr(self, 'staleness_detector'):
            try:
                from dataclasses import asdict
                staleness = asdict(self.staleness_detector.compute_staleness_metrics())
            except Exception:
                pass
        trace.event('scene_end', outcomes=outcomes, post_narrative=self.current_scene.get('post_narrative'),
                    wrapup=self.current_scene.get('wrapup'),
                    staleness=staleness)
        # plans write durations as numbers or as text ('20 minutes'); '>' on a string ended the narrative
        try:
            scene_duration = parse_duration(self.current_scene.get('duration', 0) or 0)
        except Exception:
            scene_duration = timedelta(0)
        # the planned duration is the least a scene takes; acts and travel have already moved the clock
        self.simulation_time = max(self.simulation_time, scene_start + scene_duration)
        await self.update(local_only=True)
        await asyncio.sleep(0.1)
        self.current_scene = None
        self._opening_done = True   # the play's opening description was for its first scene (record.seen_by)
        return true_outcomes


    async def run_narrative_act(self, act, act_number):
        """Run a narrative"""
        self._act_jumps = 0   # the play moves on at most MAX_JUMPS_IN_ACT times within an act
        self.establish_tension_points(act)
        self.current_act = act
        if self.screenplay.act:
            self.screenplay.act.plan = act
        scenes = act.get('scenes', [])
        if len(scenes) == 0:
            logger.error(f'No scenes in act: {act["act_title"]}')
            return
        print(f'Running act: {act["act_title"]}')
        self.act_events = []
        self._act_transcript_start = len(self.transcript)
        self.act_scene_wrapups = []
        # scenes are numbered in the order they are played, through the play (a plan's own
        # numbers gave act 3 scenes 12 to 14 and the closing act scenes 1 and 2)
        for scene in scenes:
            self._scenes_numbered = getattr(self, '_scenes_numbered', 0) + 1
            scene['scene_number'] = self._scenes_numbered
        trace.event('act_start', number=act_number, title=act.get('act_title'),
                    planned_pre=act.get('act_pre_state'), planned_post=act.get('act_post_state'),
                    scenes=len(scenes))
        self.message_queue.put({'name':'', 'text':f'\n----- ACT {act_number} ----- {act["act_title"]}'})
        await asyncio.sleep(0.1)
        scene_number = 0
        while scene_number < len(scenes):
            scene = scenes[scene_number]
            world_state_start_scene = self.current_state
            while self.step is False and  self.run is False:
                await asyncio.sleep(0.5)
            print(f'Running scene: {scene["scene_title"]}')
            true_outcomes = await self.run_scene(scene, act, opens_act=scene_number == 0)

            world_state_end_scene = self.current_state
            if (hasattr(self, 'staleness_detector') and act_number == 2 and scene_number == 1
                    and scene_number + 1 < len(scenes)): # allow one intervention in middle of act 2, if a scene follows
                # Run staleness analysis
                try:
                    changes = self.identify_state_changes(world_state_start_scene, world_state_end_scene)
                    staleness_analysis = await self.staleness_detector.analyze_staleness(changes, scenes[scene_number+1])
                    intervention = None
                    if staleness_analysis and self.staleness_detector.should_trigger_intervention(staleness_analysis):
                        intervention = await self.inject_staleness_intervention(staleness_analysis)
                    if intervention:
                        scene = await self.replan_scene(scenes[scene_number+1], scene_number+1, act, act_number, intervention)
                        scenes[scene_number+1] = scene
                except Exception as e:
                    traceback.print_exc()
                    print(f"Staleness detection error: {e}")
            scene_number += 1
            self.step = False
        # act_post_state stays as planned (it was replaced by the first True outcome label, which
        # then became the next act's starting point); the act's record is its scene wrap-ups
        if act.get('act_post_state'):
            outcomes = await self.check_post_state_ambiguity(act)
            act['act_outcomes'] = outcomes
            assessed = '; '.join(o.get('outcome') for o in outcomes if o.get('test') == True)
            act_summary = ' '.join(self.act_scene_wrapups[-4:]) or 'The act passed without notable events.'
            if assessed:
                act_summary += f' Assessed act outcome: {assessed}.'
            act['act_summary'] = act_summary
            self.current_state += f'\n{act_summary}\n'
            for character in cast(List[NarrativeCharacter], self.actors+self.extras):
                character.add_perceptual_input(act_summary, mode = 'internal')
        trace.event('act_end', number=act_number, post_state=act.get('act_post_state'))
        self.current_act = None
        #await self.update()
                
    async def introduce_characters(self, play_file, map_file=None):
        """called from the simulation server to create narratives for all characters"""

        """Create narratives for all characters,
            then share them with each other (selectively),
            then update them with the latest shared information"""
        self.narrative_level= False
        for character in cast(List[NarrativeCharacter], self.actors+self.extras):
            #self.message_queue.put({'name':character.name, 'text':f'---- introducing myself -----'})
            await character.introduce_myself(self.summarize_scenario())
        for round in range(1):
            for character in cast(List[NarrativeCharacter], self.actors+self.extras):
                await character.negotiate_central_narrative(round, self.summarize_scenario())
        await asyncio.sleep(0.4)

        #for character in cast(List[NarrativeCharacter], self.actors):
        #    self.message_queue.put({'name':character.name, 'text':f'---- creating narrative -----'})
        #    await character.write_narrative(self.summarize_scenario(), self.summarize_map(), self.central_narrative)
        #    await asyncio.sleep(0.1)

        self.message_queue.put({'name':self.name, 'text':f'----- {play_file} characters introduced -----'})
        await asyncio.sleep(0.1)

    async def create_character_narratives(self, play_file, map_file=None):
        """called from the simulation server to create narratives for all characters"""

        """Create narratives for all characters,
            then share them with each other (selectively),
            then update them with the latest shared information"""
        self.narrative_level= True
        for character in cast(List[NarrativeCharacter], self.actors+self.extras):
            #self.message_queue.put({'name':character.name, 'text':f'---- introducing myself -----'})
            await character.introduce_myself(self.summarize_scenario(), self.summarize_map())
        for round in range(1):
            for character in cast(List[NarrativeCharacter], self.actors):
                await character.negotiate_central_narrative(round, self.summarize_scenario(), self.summarize_map())
                self.message_queue.put({'name':character.name, 'text':f'character_detail', 'data':character.get_explorer_state()})
        await self.merge_central_narratives()
        self.message_queue.put({'name':'Prologue', 'text':f'{self.central_narrative}\n'})
        # Settled before any character plans its acts, so that the plans are written knowing
        # them: what this episode settles and leaves open, the clues that lie in the world, and
        # what each character has with it. (They were settled after the plans, at the head of
        # the first act: a plan resolved the mystery the frame then said must stay open, and
        # promised a find in a hut that held nothing.)
        self.frame_episode()
        self.place_clues()
        self.equip_cast()

        await asyncio.sleep(0.4)

        for character in cast(List[NarrativeCharacter], self.actors):
            await character.write_narrative(self.summarize_scenario(), self.summarize_map(), self.central_narrative)
            await asyncio.sleep(0.1)
        self.message_queue.put({'name':self.name, 'text':f'----- {play_file} character narratives created -----'})
        await asyncio.sleep(0.1)

    async def integrate_narratives(self, act_number, character_narrative_blocks, act_central_narrative, previous_act_post_state):
        """Integrate narratives into a single coherent narrative for a single act
        character_narrative_blocks is a list of lists, each containing a character and their planned next play act
        """
        # reformat character_narrative_blocks into a list of character plans for the prompt
        character_plans = []
        for character_block in character_narrative_blocks:
            character = character_block[0]
            character_narrative = self.reserialize_act_times(character_block[1])
            character_plans.append(f'{character.name}\n{character_narrative}\n')
        character_plans = '\n'.join(character_plans)
        kind = self.screenplay.kind(act_number)   # opening, middle, resolving or closing: what the act is for

        character_backgrounds = []
        for character in cast(List[NarrativeCharacter], self.actors):
            drives = 'Drives, most pressing first:\n' + drive_model.drive_lines(character)
            narrative = character.narrative.get_summary('medium')
            surroundings = character.look_percept
            memories = '\n'.join(memory.to_string() for memory in character.structured_memory.get_recent(8))
            emotionalState = EmotionalStance.for_character(character)        
            emotional_stance = emotionalState.to_definition()
            character_backgrounds.append(f'{character.name}\n{drives}\n{narrative}\n{surroundings}\n{memories}\n{emotional_stance}\n')
        character_backgrounds = '\n'.join(character_backgrounds)

        prompt = [UserMessage(content="""You are a skilled playwright and narrative integrator.
You are given a list of proposed acts, one from each of the characters in the play. 
These plans were created by each character independently, emphasizing their own role in the play.
Your job is to assimilate them into a single coherent act. This may require ignoring or merging scenes. Focus on those scenes that are most relevant to the dramatic context.

#Act Directives
{{$act_directives}}
##

#Act Central Narrative
{{$act_central_narrative}}
##

Most critically, this act is being written in the context of the actual situation after the previous act. Key elements to consider are:
#Key Elements to Consider
{{$previous_act_post_state}}
##

An act is a JSON object containing a sequence of scenes (think of a play).
Resolve any staging conflicts resulting from overlapping scenes with overlapping time signatures and overlapping characters (aka actors in the plans)
In integrating scenes within an act across characters, pay special attention to the pre- and post-states, as well as the characters's goals, to create a coherent scene. 
It may be necessary for the integrated narrative to have more scenes than any of the original plans, but keep the number of scenes to a minimum consistent with other directives. 
Do not attempt to resolve conflicts that result from conflicting character drives or goals. Rather, integrate acts and scenes in a way that provides dramatic tension and allows characters to resolve it among themselves.
The integrated plan should be a single JSON object with a sequence of acts, each with a sequence of scenes. 
The overall integrated plan should include as much of the original content as possible, ie, if one character's plan has an act or scene that is not in the others, it's narrative intent should be included in the integrated plan.
Be sure that that integrated act is consistent with the previous act's post-state.
Note carefully the proposed acts pre-conditions and locations, and resolve any inconsistencies among them or with the previous act's post-state.
{{$scene_rules}}

#Places
{{$places}}
##

Here are the original, independent, plans for each character:
{{$plans}}

In the performance it is currently {{$time}}. Use this information to generate a time-appropriate sequence of scenes for your integrated Act.
The following additional information about the state of the performance will be useful in integrating the acts.

{{$backgrounds}}

{{$plan_rule}}

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
- "act_pre_state": (string, description of the situation / goals / tensions before the act starts. Must be concrete and specific. e.g., 'I am pregnant', not 'there is a secret.')
- "act_post_state": (string, description of the situation / goals / tensions after the act ends)
- "tension_points": [
    {"characters": ["<Name>", ...], "issue": (string, concise description of the issue), "resolution_requirement": (string, "partial" / "complete")}
    ...
  ]
- "scenes"      (array) 

Each **scene** object must have:
{ "scene_number": int, // sequential within the play 
 "scene_title": string, // concise descriptor 
 {{$location_field}}
 "time": YYYY-MM-DDTHH:MM:SS, // the start time of the scene, in ISO 8601 format
 "duration": mm, // the duration of the scene, in minutes
 "characters": { "<Name>": { "goal": "<one-line playable goal>" }, … }, 
 "action_order": [ "<Name>", … ], // each name occurrence is a 'beat' in the scene lead by the named character. list only characters present in the scene 'characters' list.
 "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors. Must be concrete and specific. e.g., 'Developers are funding the project', not 'there is a secret.'", 
 "post_narrative": "Short prose (≤20 words): how things stand at the end and what tension lingers, shown in what the characters do. Nobody announces a feeling or a choice ('says aloud that he stays' is not an ending; his staying is)." 
 // OPTIONAL: 
 "task_budget": 3 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the action_order to avoid rambling or repetition. 
 }

Never break JSON validity.  Ensure the returned form is parseable by the json.loads function.
Keep the JSON human-readable (indent 2 spaces).

Return **only** the JSON.  No commentary, no code fences.
{{$rejected}}
End your response with </end>
""")]

        central_dramatic_question = self.central_narrative
        act_directives = ""
        if kind == 'opening':
            act_directives = f""" 
In performing this integration:
    1. Preserve character scene intentions where possible, but seek conciseness. This act should be short and to the point. Your target is 3 scenes maximum.
          This may require omitting or combining some scenes that have lesser importance to the act narrative.
    2. Observe scene time constraints. If it makes narrative sense to sequences scenes out of temporal order, do so and readjust scene times to maintain temporal coherence.
    2. Sequence scenes to introduce characters and create unresolved tension.
    3. Establish the central dramatic question clearly: {central_dramatic_question}
    4. Act post-state must specify: what the characters now know, what they've agreed to do together, and what specific tension remains unresolved.
    5. Final scene post-narrative must show characters making a concrete commitment or decision about their shared challenge.
    6. Ensure act post-state represents measurable progress from act pre-state, not just mood shifts.
 """
        elif kind == 'middle':
            act_directives = f""" 
In performing this integration:
    1. Each scene must advance the central dramatic question: {central_dramatic_question}. Your target is 4 scenes maximum.
    2. Midpoint scene should fundamentally and dramatically complicate the question (make it harder to answer or change its nature).
    3. Minimize lateral exploration - every scene should move closer to OR further from resolution.
    4. Preserve character scene intentions where possible. Combine overlapping scene intentions from different characters where possible.
    5. Avoid pointless repetition of scene intentions, but allow characters to develop their characters.
    6. Sequence scenes for continuously building tension, perhaps with minor temporary relief, E.G., create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)
    7. Ensure each scene raises stakes higher than the previous scene - avoid cycling back to earlier tension levels.
    9. Act post-state must show: what new obstacles emerged, what the characters now understand differently, and what desperate action they're forced to consider.
    10. Each scene post-narrative must demonstrate measurable escalation from the previous scene - not just "tension increases" but specific new problems or revelations.
"""
        elif kind == 'resolving':
            act_directives = f""" 
In performing this integration:
    1. Directly answer the central dramatic question: {central_dramatic_question}. Decide how the central dramatic question will be answered now, before generating the act, and stick to it.
    2. No scene should avoid engaging with the question's resolution you have chosen. Your target is 3 scenes maximum.
    3. Preserve character scene intentions where possible. Combine overlapping scene intentions from different characters where possible.
    4. Sequence scenes for maximum tension (alternate tension/relief beats)
    5. Create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)
    6. Add bridging scenes where character proposals create gaps, but only when essential.
    7. Act post-state must explicitly state: whether the central dramatic question was answered YES or NO, what specific outcome was achieved, and what the characters' final status is.
    8. Final scene post-narrative must show the concrete resolution to the primary tensions in the scene- not "they find peace" but "they escape the forest together" or "they remain trapped but united."
    9. No scene may end with ambiguous outcomes - each must show clear progress toward or away from resolving the central question.

"""
        else:   # the closing act
            act_directives = f""" 
In performing this integration:
    1. Show the immediate aftermath and consequences of how the last act left: {central_dramatic_question}
    2. Maximum of 2 scenes - focus on essential closure only, avoid extended exploration.
    3. Preserve character scene intentions where possible. Combine overlapping scene intentions from different characters where possible.
    4. First scene should show the immediate practical consequences of the resolution (what changes in their situation).
    5. Second scene (if needed) should show the emotional/relational aftermath (how characters have transformed).
    6. No new conflicts or dramatic questions - only reveal the implications of what was already resolved.
    7. Act post-state must specify: the characters' new equilibrium, what they've learned or become, and their final emotional state.
    8. Final scene post-narrative must provide definitive closure - show the "new normal" that results from their journey.
    9. Avoid ambiguity about outcomes - the coda confirms and completes the resolution..
"""

        bindings = {"time": self.simulation_time.isoformat(), 
                                 "plans": character_plans, 
                                 "backgrounds": character_backgrounds, 
                                 "act_central_narrative": act_central_narrative, 
                                 "central_dramatic_question": self.central_narrative,
                                 "act_directives": act_directives,
                                 "location_field": stage.LOCATION_FIELD, "scene_rules": stage.SCENE_RULES,
                                 "places": stage.gazetteer(self),
                                 "previous_act_post_state": previous_act_post_state,
                                 "plan_rule": survival.PLAN_RULE if survival.config(self) else '', "rejected": ''}
        # The act that is performed is held to the same form as a character's draft of one
        # (validate_narrative_act); it had been checked for its dates alone, and a scene with
        # no speaking order or no cast reached the scene loop. One more try with the reason.
        self.integration_fault = ''
        for attempt in range(2):
            response = self.llm.ask(bindings, prompt, max_tokens=4000, stops=['</end>'], tag='integrate_narratives')
            if not response:
                self.integration_fault = 'the playwright gave no answer'
                continue
            response = response.replace("```json", "").replace("```", "").strip()
            try:
                act = json.loads(response)
            except Exception as e:
                print(f'Error parsing integrated act: {e}')
                act = self.repair_json(response, e)
            if isinstance(act, dict) and isinstance(act.get('acts'), list) and act['acts']:
                act = act['acts'][0]   # 'a sequence of acts' was asked for in one place: take the act
            if isinstance(act, dict) and not isinstance(act.get('act_number'), int):
                act['act_number'] = act_number
            valid, result = self.validate_narrative_act(act, require_scenes=True) if isinstance(act, dict) else (False, 'not a JSON object')
            if valid:
                self.validate_dates_in_plan(result)
                logger.info(f'Integrated act: {self.reserialize_act_times(result)}')
                return result
            self.integration_fault = str(result)
            trace.event('act_refused', act=act_number, reason=self.integration_fault, attempt=attempt + 1)
            bindings['rejected'] = f"\nYour last answer could not be used: {self.integration_fault}. Return the whole act again, corrected.\n"
        return None

    def validate_dates_in_plan(self, narrative_plan):
        """Make scene times in a plan run forward from now: a scene starts no earlier than the
        current time, nor before the previous scene's planned end. (A scene time at or before the
        clock used to be pushed a whole day on, so every play opened a day after its start.)"""
        if isinstance(narrative_plan, dict):
            acts = narrative_plan.get('acts') or [narrative_plan]
        elif isinstance(narrative_plan, list):
            acts = narrative_plan
        else:
            return
        earliest = self.simulation_time
        for act in acts:
            for scene in act.get('scenes', []):
                if 'time' not in scene:
                    continue
                scene['time'] = max(self.validate_scene_time(scene), earliest)
                try:
                    earliest = scene['time'] + parse_duration(scene.get('duration', 0) or 0)
                except Exception:
                    earliest = scene['time']


    def frame_episode(self):
        """Settle, before the first act, what this playthrough is to settle and what it leaves
        open: an episode. The acts are planned one at a time and nothing said how the play should
        close, so it closed on whatever the last act happened to reach. And a play's largest
        question ('how did we come to be here?') is often one the world holds no answer to: no
        scene can pay it off, so it is named here as left open, for a later episode, instead of
        being promised and dropped. Returns the frame, {} when it could not be made."""
        prompt = [UserMessage(content="""You are planning one episode of an improvised play. An episode settles one thing and leaves one thing open.

<setting>
{{$setting}}
</setting>

<central_question>
{{$central}}
</central_question>

<characters>
{{$characters}}
</characters>

- What the episode settles is the central question, or as much of it as these characters can bring about themselves, here, by what they do to and with each other: something between them, not a single errand. It is shown in an act, not announced.
- Use only what the setting and the characters give you. Invent no object, sound, creature or event.
- """ + survival.PLAN_RULE + """
- What the characters do to settle it must be something this world lets them do: where places are listed below, the actions named at each are what can be done there (a thing can be handed over only if it can first be picked up or carried).
- What it leaves open is a larger question this episode cannot answer (how they came to be here, what this place is, what lies beyond it). It may be touched on; it is not answered, and nobody pretends it has been.
- The last image is what the audience is left looking at: one act or picture that shows what was settled without anyone saying it, with the open question still standing behind it.
{{$clue_rule}}

Respond with these lines only, one sentence each:
#settles what this episode settles
#open what it leaves open
#last the image or act it should end on
{{$clue_format}}
</end>
""")]
        places = stage.gazetteer(self) if survival.config(self) else ''
        clue_rule = ("""- Clues are the one exception to inventing nothing. Name two or three small, ordinary things that will really lie in this world, each at one of the places listed below, that touch the open question without answering it. Draw them from this setting and these people's lives, so that they could belong to no other play. A clue makes the question sharper; it never says who, how or why. Choose places the characters can reach in this episode.

<places>
""" + places + """
</places>""") if places else ''
        clue_format = "#clue <the thing, two to four plain words> | <exact name of a place listed> | <what finding it makes someone wonder>   (one line for each clue)" if places else ''
        reply = self.llm.ask({'setting': self.initial_state.strip(), 'central': str(self.central_narrative or ''),
                              'characters': '\n'.join(f"{c.name}: {'; '.join(d.text for d in (c.drives or [])[:3])}" for c in self.actors),
                              'clue_rule': clue_rule, 'clue_format': clue_format},
                             prompt, tag='frame_episode', temp=0.4, stops=['</end>'], max_tokens=340) or ''
        echo = r'^(what (this episode|it) (settles|leaves open)|the image or act it should end on)\s*:?\s*'   # the format line, said back
        frame = {k: re.sub(echo, '', (hash_utils.find(k, reply) or '').strip(), flags=re.I).strip() for k in ('settles', 'open', 'last')}
        frame = {k: (v[0].upper() + v[1:] if v else v) for k, v in frame.items()}
        self.episode = frame if frame['settles'] and frame['open'] else {}
        self.screenplay.clues = [{'thing': thing.strip(' ."\''), 'place': place.strip(), 'touches': touches.strip()}
                                 for thing, place, touches in re.findall(r'^\s*#clue\s+([^|\n]+)\|([^|\n]+)\|\s*(.+?)\s*$', reply, re.M)][:3]
        trace.event('episode', **self.episode, clues=self.screenplay.clues)
        return self.episode

    def place_clues(self):
        """Put the episode's clues in the world, each at the place the director named for it:
        hidden where this world has a way to search such a place, in plain sight where it has
        not. A clue whose place is not on the map is dropped from the screenplay, so that no
        plan is told of a thing that is not there."""
        if not survival.config(self) or not self.actors:
            self.screenplay.clues = []
            return []
        anchor = stage.here(self.actors[0])
        placed = []
        for clue in self.screenplay.clues:
            spot = stage.resolve(self, clue['place'], anchor)
            res = self.map.resource_registry.get(spot[1]) if spot else None
            if spot is None or res is None:
                trace.event('clue_dropped', **clue)
                continue
            clue['place'] = spot[1]
            clue['hidden'] = survival.searchable(self, res)
            if survival.add_thing(self, clue['thing'], at=spot[0], hidden=clue['hidden']):
                placed.append(clue)
        self.screenplay.clues = placed
        trace.event('clues_placed', clues=placed)
        return placed

    def equip_cast(self):
        """Before the first act, settle what each character has with it that the story turns on,
        and make it real. A world's characters began with empty hands unless the play file gave
        them something, so the widow whose whole part is her photographs reached into her handbag
        and 'came up empty', and then told the room she had photographs. Returns {name: [things]}."""
        cfg = survival.config(self)
        people = [c for c in self.actors if getattr(c, 'body', None) is not None]
        if not cfg or not people or getattr(self, '_cast_equipped', False):
            return {}
        self._cast_equipped = True
        prompt = [UserMessage(content="""A play is about to begin in a simulated world that keeps track of what each character carries. Settle what each character has with them at the start that the story depends on.

<setting>
{{$setting}}
</setting>

<central_question>
{{$central}}
</central_question>

<characters>
{{$characters}}
</characters>

Already carrying:
{{$carrying}}

Things this world's own actions produce, which are not to be handed out here: {{$items}}

- List only what a character's description, or the premise of the story, says or plainly implies they have with them when the play opens and that the story turns on: evidence, a letter, a keepsake, the tool of their trade.
- Most characters need nothing. Someone who begins with nothing (lost, stranded, robbed) gets nothing. At most two things each.
- No weapon, food, drink, medicine or money, and nothing they could only have by something happening in the play.
- What a person is wearing is not something they have with them: no clothes, shoes or anything else a description says they are dressed in.

Respond with one line for each thing, or the single line #none:
#has <character name> | <the thing, two or three plain words>
</end>
""")]
        reply = self.llm.ask({'setting': self.initial_state.strip(), 'central': str(self.central_narrative or ''),
                              'characters': '\n'.join(f"{c.name}: {' '.join(c.character.split())} Wants: {'; '.join(d.text for d in (c.drives or [])[:3])}" for c in people),
                              'carrying': '\n'.join(f"{c.name}: {', '.join(c.body.inventory) or 'nothing'}" for c in people),
                              'items': ', '.join(survival.item_kinds(self)) or 'none'},
                             prompt, tag='equip_cast', temp=0.2, stops=['</end>'], max_tokens=160) or ''
        given = {}
        own = [k.lower() for k in survival.item_kinds(self)]
        for name, thing in re.findall(r'^\s*#has\s+([^|\n]+?)\s*\|\s*(.+?)\s*$', reply, re.M):
            holder = next((c for c in people if c.name.lower() == name.strip().lower()), None)
            thing = thing.strip(' ."\'').lower()
            if holder is None or not thing or thing in own or len(given.get(holder.name, [])) >= 2:
                continue
            if survival.add_thing(self, thing, holder=holder):
                given.setdefault(holder.name, []).append(thing)
                holder.add_perceptual_input(f'You have with you: {thing}.', percept=False, mode='internal')
        trace.event('cast_equipped', given=given, reply=reply.strip())
        return given

    def episode_note(self, closing=False):
        """The episode's frame as a note for whoever plans an act; '' when there is none."""
        frame = getattr(self, 'episode', None) or {}
        if not frame:
            return ''
        if closing:
            return (f"This is the last act of the episode. It closes what the episode set out to settle: {frame['settles']} "
                    f"Show it settled in what the characters do; nobody says it. "
                    f"It leaves standing what the episode cannot answer: {frame['open']} Do not answer it, and do not have anyone sum it up. "
                    + (f"End on this, or something as quiet: {frame['last']}" if frame.get('last') else ''))
        return (f"This episode is to settle: {frame['settles']} Each act should bring that nearer or put it in doubt. "
                f"It leaves open: {frame['open']} That may be touched on, never answered in this episode."
                + (f"\n{self.screenplay.clue_note()}" if self.screenplay.clues else ''))

    async def run_integrated_narrative(self):
        """Play the screenplay, an act at a time. Every act, the closing one included, goes the
        same way: its focus is settled, each character drafts its own version of the act from
        where things now stand, the drafts are integrated into the act that is played, the act
        is played, and what it came to is recorded for the next.
        (The first act used to skip the characters' drafts, so its scenes were the integrator's
        alone; the closing act was a separate block, hard-coded as act 4, with no focus of its
        own.)"""
        def stop(reason):
            # these early exits used to be log-only: the UI just went quiet
            logger.error(reason)
            self.message_queue.put({'name': self.name, 'text': f'----- narrative stopped: {reason} -----'})

        players = cast(List[NarrativeCharacter], self.actors)
        count = self.screenplay.act_count = max((len((c.plan or {}).get('acts') or []) for c in players), default=0)
        if count == 0:
            stop('no character has any narrative acts')
            return
        post_state = ''
        try:
            for number in range(1, count + 2):
                record = self.screenplay.begin_act(number)
                closing = record.kind == 'closing'
                focus = await self.generate_act_central_narrative(number)
                if focus is None:
                    stop(f'could not settle what act {number} is about')
                    return
                if self.episode_note(closing=closing):   # every act is planned knowing what the episode is for
                    focus = f'{focus}\n\n{self.episode_note(closing=closing)}'
                drafts = []
                for character in players:
                    acts = (character.plan or {}).get('acts') or []
                    outline = acts[number - 1] if number <= len(acts) else {"act_number": number, "act_title": "Closing" if closing else f"Act {number}"}
                    drafts.append([character, character.replan_narrative_act(outline, focus, post_state, kind=record.kind)])
                if not drafts:
                    stop(f'no character has a plan for act {number}')
                    return
                act = await self.integrate_narratives(number, drafts, focus, post_state)
                if act is None:
                    stop(f'act {number} could not be written: {self.integration_fault}')
                    return
                act = await self.request_act_choice(act)   # the Director's chair may edit it
                await asyncio.sleep(0.1)
                if act is None:
                    stop(f'no act {number} to run')
                    return
                await self.run_narrative_act(act, number)
                post_state = await self.close_act(act)
            self.message_queue.put({'name': '', 'text': '\n----- end of episode -----'})
            trace.event('episode_end', **(self.episode or {}))
        except Exception as e:
            stop(f'error running integrated narrative: {type(e).__name__}: {e}')
            traceback.print_exc()

    async def check_post_state_ambiguity(self, act, scene=None):
        """What a scene or act came to, read from the record of it: [{'outcome', 'test'}].
        It used to be given the plan (all of it) and asked which planned outcome was so: an act
        was closed with its own planned ending word for word ('Wolf repelled; alliance forged'
        when no wolf had appeared), a scene with one of the plan's failure cases ('Separation
        occurs; one or both are killed') and with alternatives ('established dominance or
        secured food'). Now it is given what happened and what was said, and the planned ending
        only as the thing to compare: an outcome stands on a line of the record or it is false."""
        prompt = [UserMessage(content="""You keep the record of an improvised play. A {{$unit}} has just been played. Say what it came to, from the record below and from nothing else.

<planned_ending>
{{$post_state}}
</planned_ending>

<what_happened>
{{$events}}
</what_happened>

<what_was_said_and_shown>
{{$lines}}
</what_was_said_and_shown>

<bodies_now>
{{$bodies}}
</bodies_now>

Rules:
- Take the planned ending apart into its separate claims, and for each say whether the record shows it happened. A plan is not evidence: what was planned, intended, feared or only spoken of as a possibility did not happen.
- An animal, a person or a danger did not appear unless the record shows it appearing. Nobody is hurt, healed, fed, armed or dead unless what happened or the bodies say so.
- Then add what the record shows that matters and the plan did not foresee: a choice made, a thing handed over or refused, a parting, an injury, something admitted.
- A line written between dots (...like this...) is a private thought: it shows what someone thinks, not that anything happened.
- Each outcome is one plain statement of one thing, ten words or so. Never "X or Y". For each, quote a few words of the record that show it (fifteen words at most); with no such words, it did not happen.
- Six outcomes at most: the claims of the planned ending first, then only what changed how things stand. Not every deed is an outcome.

Answer in JSON only:
{"outcomes": [{"outcome": "<one plain statement>", "happened": true or false, "shown_by": "<words quoted from the record, or empty>"}]}
End your response with: </end>""")]
        if scene:
            post_state = scene.get('post_narrative', '')
            start = getattr(self, '_scene_transcript_start', 0)
        else:
            post_state = act.get('act_post_state', '')
            start = getattr(self, '_act_transcript_start', 0)
        cast = [c for c in self.actors + self.extras if getattr(c, 'body', None) is not None]
        reply = self.llm.ask({'unit': 'scene' if scene else 'act', 'post_state': post_state or 'none given',
                              'events': '\n'.join(self.scene_events if scene else self.act_events) or 'nothing recorded',
                              'lines': '\n'.join(str(l).strip() for l in self.transcript[start:] if str(l).strip())[-6000:] or 'nothing',
                              'bodies': '; '.join(f'{c.name}: {c.body.summary()}' for c in cast) or 'not tracked'},
                             prompt, tag='check_post_state_ambiguity', temp=0.1, stops=['</end>'], max_tokens=700)
        found = json_reply.parse(reply) if reply else None
        if not isinstance(found, dict):
            trace.event('check_unreadable', of='outcomes', reply=(reply or '')[:300])
            return []
        outcomes = []
        for entry in found.get('outcomes') or []:
            if isinstance(entry, dict) and str(entry.get('outcome') or '').strip():
                shown_by = str(entry.get('shown_by') or '').strip()
                # an outcome with nothing of the record behind it is not counted as having happened
                outcomes.append({'outcome': str(entry['outcome']).strip(), 'test': entry.get('happened') is True and bool(shown_by),
                                 'shown_by': shown_by[:200]})
        return outcomes

    async def check_decision_required(self, act, scene):
        """Whether the scene about to be played turns on a choice one of its characters makes:
        [] or [{'character', 'decision', 'choices'}], one at most.
        It used to be asked of nearly every scene (18 decisions in a play of 12 scenes) with
        no word on what the scene was for, and offered choices that walked out of it: a scene
        planned as 'survive the charge without abandoning Joe' got 'run away to save herself',
        and a closing act planned as 'ask nothing about the ribbon' got 'demand she reveal all
        hidden resources'. A choice now is a way of playing the character's planned goal, and
        the closing act, which holds no new conflict, is not asked."""
        if self.screenplay.act and self.screenplay.act.kind == 'closing':
            return []
        prompt = [UserMessage(content="""You are directing an improvised play. A scene is about to be played. Say whether it turns on a choice.

<act>
{{$act_focus}}
</act>

<scene>
Title: {{$title}}
Set-up: {{$setup}}
Where it is heading: {{$ending}}
What each character is after in it:
{{$goals}}
</scene>

- Most scenes turn on no choice: what happens in them follows from what each character is after. For those, answer with an empty list.
- A scene turns on a choice only when where it is heading depends on one character choosing, during the scene, between real alternatives: to tell or to withhold, to stay or to go, to accept or to refuse.
- One decision at most, for one character.
- Give two or three choices. Each is a way that character could go after what it is after in this scene, said in a few plain words as something it does or says. No choice gives up the scene, drops what the character is after, or abandons the other person; none is 'wait' or 'decide later'.

Answer in JSON only:
{"decisions": [{"character": "<name>", "decision": "<what has to be chosen>", "choices": ["<choice>", "<choice>"]}]}
End your response with: </end>""")]
        goals = '\n'.join(f"- {name}: {(entry or {}).get('goal', '')}" for name, entry in (scene.get('characters') or {}).items() if isinstance(entry, dict))
        reply = self.llm.ask({'act_focus': (self.act_frame or {}).get('focus') or self.act_central_narrative or 'not given',
                              'title': scene.get('scene_title', ''), 'setup': scene.get('pre_narrative', '') or 'none given',
                              'ending': scene.get('post_narrative', '') or 'not said', 'goals': goals or 'none given'},
                             prompt, tag='check_decision_required', temp=0.2, stops=['</end>'], max_tokens=220)
        found = json_reply.parse(reply) if reply else None
        decisions = []
        for entry in (found.get('decisions') or []) if isinstance(found, dict) else []:
            choices = [str(c).strip() for c in (entry.get('choices') or []) if str(c).strip()] if isinstance(entry, dict) else []
            if len(choices) >= 2 and str(entry.get('character') or '').strip() and str(entry.get('decision') or '').strip():
                decisions.append({'character': str(entry['character']).strip(), 'decision': str(entry['decision']).strip(), 'choices': choices[:3]})
        return decisions[:1]

    def exit_allowed(self, character, act):
        """Whether `character` may go out of the scene with this act. An exit that leaves one
        person or nobody on stage ends the scene, so it waits until the scene has been played:
        everyone in it has played one of their planned tasks, and half the planned tasks are
        played. ('Agreed, water first' marked as leaving ended a scene after one task of six.)
        A count, not a model's opinion of whether the scene feels over."""
        scene = self.current_scene
        plan = self.scene_integrated_task_plan or []
        if scene is None or not plan:
            return True
        party = act.leaves if isinstance(getattr(act, 'leaves', None), list) else []
        on_stage = [c for c in self.actors + self.extras + self.npcs
                    if getattr(c, 'current_scene', None) is scene and not stage.off_stage(c)]
        if len(on_stage) < 2 or len([c for c in on_stage if c is not character and c not in party]) > 1:
            return True   # the scene goes on without it
        played = plan[:self.scene_integrated_task_plan_index + 1]   # with the task now being played
        had_a_turn = {e['actor'] for e in played}
        allowed = len(played) * 2 >= len(plan) and all(c in had_a_turn for c in on_stage if any(e['actor'] is c for e in plan))
        if not allowed:
            trace.event('exit_held', character=character.name, played=len(played), planned=len(plan))
        return allowed

    def told_briefly(self, account):
        """An act's record (its scene wrap-ups) as two or three sentences for the audience."""
        prompt = [UserMessage(content="""Here is the record of what happened in the last act of a play:

{{$account}}

Tell it to the audience in two or three plain sentences, 60 words at most: what happened and where it leaves the characters. Only what the record says; no scene titles, no commentary.
End your response with </end>
""")]
        told = self.llm.ask({'account': account.strip()}, prompt, tag='coda_summary', temp=0.3, stops=['</end>'], max_tokens=120)
        return (told or account).strip()

    async def close_act(self, act):
        """An act is over: tell the audience what it came to, let each character think it over,
        and return what the next act starts from (the act's record, with what was planned).
        (This was run_coda, which also held a second, uncalled half that played a closing
        'Coda' task; the play's closing act is an act like the others.)"""
        planned = act.get('act_post_state') or ''
        summary = act.get('act_summary')
        if summary is None:   # an act with no planned post-state was not summed up by run_narrative_act
            outcomes = await self.check_post_state_ambiguity(act)
            assessed = '; '.join(o.get('outcome') for o in outcomes if o.get('test') == True)
            summary = ' '.join(self.act_scene_wrapups[-4:]) + (f' Assessed act outcome: {assessed}.' if assessed else '')
        post_state = summary.strip() + (f'\n(The act was planned to end with: {planned})' if planned else '')
        self.message_queue.put({'name':self.name, 'text':f'coda: {self.told_briefly(summary)}'})
        if self.screenplay.act:
            self.screenplay.act.summary, self.screenplay.act.post_state = summary.strip(), post_state
        for character in cast(List[NarrativeCharacter], self.actors):
            character.reason_over(post_state)
        return post_state


    async def merge_central_narratives(self):
        """Merge the central narratives of all characters into a single central narrative"""
        prompt = [UserMessage(content="""You are a skilled playwright and narrative integrator.
You are given a list of central narratives for a play, one from each of the characters in the play. 
These narratives were created by each character independently, negotiating with the others.
You are integrating multiple character-negotiated dramatic proposals into a single, cohesive central dramatic question that will drive the play.
The dramatic question usually revolves around how one or more characters will resolve a conflict, either internal (among conflicting drives or desires) or external (between or among characters).
Rarely, conflict can be achieved through compromise. More often, it must be resolved through making a choice between starkly defined alternatives and living with the resulting gains and losses. 
In integrating character-proposed narratives, you may have to choose among conflicting proposals about the central dramatic question. 
It is your job to decide which conflict or question is primary for the play.
It is NOT your job to decide now how that conflict will be resolved. That will emerge from the play.


## Character Proposals
{{$character_central_narratives}}

## Cast & Context
**Characters:** {{$character_names_and_brief_descriptions}}
**Setting:** {{$setting}}
**What each character wants, most pressing first (from the play as written):**
{{$drives}}
{{$left_out}}
## Integration Task
Create a unified central dramatic question that:
1. **Synthesizes character interests** - Incorporates core elements from multiple proposals. The question is larger than staying alive: it gives a place to what each character most wants (a mystery to solve, a person to win or to keep at a distance, something hidden). Survival is the pressure on the question, not the question.
2. **Assigns clear dramatic roles** - Each character has a specific function in the conflict
4. **Has binary resolution potential** - Can succeed or fail clearly
5. **Creates character interdependence** - No one can resolve it alone

## Output elements
1. Central Dramatic Question: [One clear concise sentence framing the core conflict]
2. Stakes: [What happens if this question isn't resolved - consequences that matter to all. Be terse, no more than 10 words.]
3. Character Roles: [for each character, what is their role in the conflict? Protagonist/Antagonist/Catalyst/Obstacle - with 4-6 word description]
4. Dramatic Tension Sources: [The main opposing forces. Be terse, no more than 8 words.]
5. Success Scenario: [Brief description of what "winning" looks like. Be terse, no more than 8 words.]
6. Failure Scenario: [Brief description of what "losing" looks like. Be terse, no more than 8 words.] 

## Output Format

*Write your response as one paragraph totalling no more than 100 words, with no introductory text, code fences or markdown formatting.

After the paragraph, repeat its elements on lines of their own, exactly in this form, one #role line for each character:
#question <the Central Dramatic Question>
#stakes <the Stakes>
#tension <the Dramatic Tension Sources>
#role <character name>: <that character's role>
#success <the Success Scenario>
#failure <the Failure Scenario>

What unified central dramatic question emerges from these proposals?         
                              """)]
        drives = '\n'.join(f"{c.name}: " + ' / '.join(d.text.strip() for d in c.drives) for c in self.actors if getattr(c, 'drives', None))
        bindings = {"character_central_narratives": '\n'.join([f'{character.name}: {character.current_proposal.to_string() if character.current_proposal else ""}' for character in cast(List[NarrativeCharacter], self.actors)]),
                    "character_names_and_brief_descriptions": '\n'.join([f'{character.name}: {character.character}' for character in cast(List[NarrativeCharacter], self.actors)]),
                    "setting": self.current_state, "drives": drives or 'not given', "left_out": ''}
        strip = lambda reply: re.sub(r'^\s*#(question|stakes|tension|success|failure|role)\b.*$', '', reply or '', flags=re.M).strip()
        response = self.llm.ask(bindings, prompt, max_tokens=480, stops=['</end>'], tag='merge_central_narratives')
        left_out = self.drives_left_out(strip(response), drives) if response and drives else []
        if left_out:
            # the play as written gave these people wants that the question has no place for
            # (lost: a mystery and a romance planned as berries, boar, wolf): written once more
            bindings['left_out'] = ("**A first draft of the question left these out; this one gives each a place:**\n"
                                    + '\n'.join(f"- {who}: {want}" for who, want in left_out) + '\n')
            again = self.llm.ask(bindings, prompt, max_tokens=480, stops=['</end>'], tag='merge_central_narratives')
            trace.event('plan_drives', left_out=[f'{who}: {want}' for who, want in left_out], rewritten=bool(again),
                        before=response[:600], after=(again or '')[:600])
            response = again or response
        if response:
            # the paragraph for the planning prompts and the audience; its elements, taken from
            # the tagged lines, for each actor's notes (Screenplay.story_for)
            self.screenplay.set_question(response)
            trace.event('central_question', **{k: v for k, v in self.screenplay.question.items() if k != 'text'})
        return self.central_narrative

    def drives_left_out(self, question, drives):
        """Which of the cast's wants (from the play file) the central question gives no place to:
        [(name, want)]. [] when all have one, or when the check cannot be read."""
        reply = self.llm.ask({'question': question, 'drives': drives}, [UserMessage(content="""A play's central dramatic question has been written. Below it is what each character wants, as the play's author wrote it.

<question>
{{$question}}
</question>

<wants>
{{$drives}}
</wants>

For each want, decide whether the question and its stakes, roles and tensions give it a place: could a scene that pursues this want be part of answering the question? A want for water, food, rest or safety from danger has a place in any play about survival. A want to solve a mystery, to find love or belonging, or for adventure has a place only if the question or its tensions speak of it.

Answer in JSON only, listing the wants that have NO place (an empty list if every want has one):
{"left_out": [{"name": "<character>", "want": "<the want, in a few of its own words>"}]}
End your response with: </end>""")], tag='plan_serves_drives', temp=0.1, stops=['</end>'], max_tokens=200)
        found = json_reply.parse(reply) if reply else None
        names = {c.name for c in self.actors}
        return [(e['name'], e['want']) for e in json_reply.items(found, 'left_out', fields=('name', 'want')) if e['name'] in names]

    async def request_act_choice(self, act_data):
        """Request an act choice from the UI - following request_goal_choice pattern"""
        request=False
        for actor in self.actors:
            if not actor.autonomy.act:
                request=True
        if not request:
            return act_data
        print(f"DEBUG: Context.request_act_choice called with act_data: {act_data.get('act_title', 'unknown')} (act #{act_data.get('act_number', 'unknown')})")
        
        # Send choice request to UI
        choice_request = {
            'text': 'act_choice',
            'choice_type': 'act',
            'act_data': act_data
        }
        
        # Drain any old responses from the queue
        while not self.choice_response.empty():
            _ = self.choice_response.get_nowait()
        
        # Send choice request to UI
        print(f"DEBUG: Context putting act_choice message on queue: {choice_request['text']}")
        self.message_queue.put(choice_request)
        await asyncio.sleep(0)
        
        # Wait for response with timeout
        waited = 0
        while waited < 600.0:
            await asyncio.sleep(0.1)
            waited += 0.1
            if not self.choice_response.empty():
                try:
                    response = self.choice_response.get_nowait()
                    if response and response.get('updated_act'):
                        new_act = response['updated_act']
                        valid, reason = self.validate_narrative_act(new_act, require_scenes=True)
                        if not valid:
                            print(f'Invalid act: {reason}')
                            return act_data
                        return new_act
                    elif response and response.get('selected_id') is not None:
                        # If UI just confirmed without changes
                        return act_data
                except Exception as e:
                    print(f'Context request_act_choice error: {e}')
                    break
        
        # If we get here, either timed out or had an error - return original
        return act_data

    def validate_narrative_json(self, json_data: Dict[str, Any], require_scenes=True) -> tuple[bool, str]:
        """
        Validates the narrative JSON structure and returns (is_valid, error_message)
        """
        # Check top-level structure
        if not isinstance(json_data, dict):
            return False, "Root must be a JSON object"
    
        if "title" not in json_data or not isinstance(json_data["title"], str):
            return False, "Missing or invalid 'title' field"
        
        if "acts" not in json_data or not isinstance(json_data["acts"], list):
            return False, "Missing or invalid 'acts' array"
        
        # Validate each act
        for n, act in enumerate(json_data["acts"]):
            if not isinstance(act, dict):
                return False, f"Act {n} must be a JSON object"
            else:
                valid, json_data["acts"][n] = self.validate_narrative_act(act, require_scenes=require_scenes)
                if not valid:
                    return False, f"Act {n} is invalid: {json_data['acts'][n]}"
                
        return True, "Valid narrative structure"
 
    def validate_narrative_act(self, act: Dict[str, Any], require_scenes=True) -> tuple[bool, str]:       # Validate each act
        if not isinstance(act, dict):
            return False, "Act must be a JSON object"
                
        # An act needs a number; the rest of its heading is filled in when missing (an act
        # written without tension points is still an act that can be played)
        if "act_number" not in act or not isinstance(act["act_number"], int):
            return False, "Missing or invalid 'act_number'"
        for field in ("act_title", "act_description", "act_pre_state", "act_post_state"):
            if not isinstance(act.get(field), str):
                act[field] = '' if field != "act_title" else f"Act {act['act_number']}"
        if not isinstance(act.get("act_goals"), dict):
            act["act_goals"] = {}
        act["tension_points"] = [t for t in (act.get("tension_points") if isinstance(act.get("tension_points"), list) else [])
                                 if isinstance(t, dict) and isinstance(t.get("characters"), list) and isinstance(t.get("issue"), str)]
        for tension_point in act["tension_points"]:
            if not isinstance(tension_point.get("resolution_requirement"), str):
                tension_point["resolution_requirement"] = "partial"
        if require_scenes and (not isinstance(act.get("scenes"), list) or not act["scenes"]):
            return False, "The act has no scenes"
        if "scenes" in act:
            # Validate each scene. A scene is refused only for what the scene loop cannot do
            # without (a title, its people and their goals). What the engine settles itself is
            # filled in: the clock sets a scene's time and the least it lasts, the cast walks to
            # "same" if no place is named, and the people speak in the order they are listed.
            # (An act was refused for a scene with no 'time', a field that is not used.)
            for n, scene in enumerate(act["scenes"]):
                if not isinstance(scene, dict):
                    return False, "Scene must be a JSON object"
                if not isinstance(scene.get("scene_title"), str) or not scene["scene_title"].strip():
                    return False, f"Scene {n + 1} has no 'scene_title'"
                if not isinstance(scene.get("characters"), dict) or not scene["characters"]:
                    return False, f"Scene '{scene['scene_title']}' has no 'characters'"
                if not isinstance(scene.get("scene_number"), int):
                    scene["scene_number"] = n + 1
                if not isinstance(scene.get("location"), str) or not scene["location"].strip():
                    scene["location"] = "same"
                if not isinstance(scene.get("duration"), int):
                    scene["duration"] = 15
                for field in ("pre_narrative", "post_narrative"):
                    if not isinstance(scene.get(field), str):
                        scene[field] = ''
                # One spelling for each of the play's people, here, once: the cast was looked up
                # three ways downstream (capitalised and resolved, by exact key, and without
                # regard to case), and a name the plan spelled its own way got an empty goal
                # from one of them and no tasks from another. A name that is nobody in the cast
                # is left as written: the scene brings that person on.
                cast_name = lambda name: next((c.name for c in self.actors + self.extras
                                               if c.name.lower() == str(name).strip().lower()), str(name).strip())
                scene["characters"] = {cast_name(name): entry for name, entry in scene["characters"].items()}
                if isinstance(scene.get("action_order"), list):
                    scene["action_order"] = [cast_name(name) for name in scene["action_order"]]
                names = [name for name in scene["characters"]]
                order = scene.get("action_order")
                if not isinstance(order, list) or not [name for name in order if name in names]:
                    scene["action_order"] = list(names)
                else:
                    scene["action_order"] = [name for name in order if name in names]
                scene["time"] = self.validate_scene_time(scene)   # a datetime: the plan's, or now

                # each person in the scene has a goal (a bare string is taken as the goal)
                for char_name, char_data in list(scene["characters"].items()):
                    if isinstance(char_data, str) and char_data.strip():
                        scene["characters"][char_name] = char_data = {"goal": char_data.strip()}
                    if not isinstance(char_data, dict) or not str(char_data.get("goal") or '').strip():
                        return False, f"Scene '{scene['scene_title']}' gives {char_name} no goal"
                budget = int(1.75*len(scene["action_order"])+1)
                if not isinstance(scene.get("task_budget"), int) or scene["task_budget"] > 2*len(scene["action_order"]):
                    scene["task_budget"] = budget
        
        return True, act
        
    def validate_scene_time(self, scene):
        """
        Validate and parse the scene time from ISO 8601 format.
        Returns a datetime object if valid, None if invalid or missing.
        """
        time_str = scene.get('time')
        if not time_str:
            return self.simulation_time
        if isinstance(time_str, datetime):  # already validated
            return time_str
        time_str = str(time_str).strip().replace('x', '0')
        time_str = time_str.replace('00T', '01T') # make sure day is not 00
        try:
            # Parse ISO 8601 format
            scene_time = datetime.fromisoformat(time_str)
            return scene_time
        except (ValueError, TypeError):
            # Handle invalid format or type
            return self.simulation_time

    async def request_scene_choice(self, scene_data):
        """Request a scene choice from the UI - following request_goal_choice pattern"""
        request=False
        for actor in self.actors:
            if not actor.autonomy.scene:
                request=True
        if not request:
            return scene_data
        # Send choice request to UI
        choice_request = {
            'text': 'scene_choice',
            'choice_type': 'scene',
            'scene_data': scene_data
        }
        
        # Drain any old responses from the queue
        while not self.choice_response.empty():
            _ = self.choice_response.get_nowait()
        
        # Send choice request to UI
        self.message_queue.put(choice_request)
        await asyncio.sleep(0.1)
        
        # Wait for response with timeout
        waited = 0
        while waited < 600.0:
            await asyncio.sleep(0.1)
            waited += 0.1
            if not self.choice_response.empty():
                response = self.choice_response.get_nowait()
                if response and response.get('updated_scene'):
                    new_scene = response['updated_scene']
                    try:
                        scene_time = self.validate_scene_time(new_scene)
                    except Exception as e:
                        print(f'Context request_scene_choice error: {e}')
                        return scene_data
                    if scene_time is None:
                        return scene_data
                    else: 
                        new_scene["time"] = scene_time # update the time field with the validated datetime object
                        return new_scene
                else:
                    # If UI just confirmed without changes
                    return scene_data
        
        # If we get here, either timed out or had an error - return original
        return scene_data

    async def generate_act_central_narrative(self, act_number ):
        if self.embedding_model is None:
            from utils.embedding_model import get_embedding_model
            self.embedding_model = get_embedding_model()
        """Generate a central narrative for the act"""
        prompt = [UserMessage(content="""You are creating a focused dramatic framework for Act {{$act_number}} based on the overall central narrative and character planning.

## Overall Central Narrative
{{$play_level_central_narrative}}
                              
## What has happened in the acts already played
{{$previous_scenes_summary}}

## Act {{$act_number}}

## Character Act Plans
{{$character_act_plans}}

## Actor narratives
{{$actor_narratives}}

## Current Situation
{{$current_situation}}

## Task
Generate an act-specific central narrative that:
1. **Focuses the overall question** - Narrows the play-level dramatic question to what THIS act must resolve
2. **Escalates from previous acts** - Raises stakes/tension from where the story currently stands  
3. **Creates scene-plannable conflict** - Gives characters specific dramatic targets for individual scenes
4. **Maintains character agency** - Respects the roles and drives established in character plans
5. **Forces binary resolution** - This act must clearly succeed or fail at its specific focus

## Act-Specific Guidelines
{{$act_specific_guidelines}}

## Output Format
A paragraph or two including the following elements:
1. Central Focus: [One terse (10-12 words) sentence stating what specific aspect of the overall question this act addresses]
2. Stakes: [What happens if this act's focus succeeds vs. fails - immediate consequences (8-10 words)]
3. Act-Specific Character Roles: [for each character]
    - {character_name}: [How their overall role manifests specifically in this act - (5-6 words each) who drives decisions, who creates obstacles, etc.]
4. Required Scene Types: [for each scene type]
    - {scene_type_name}: [tension builders, confrontations, forced choices, etc.] plus short description of scene focus (4-5 words)
5. Success Scenario: [What this act accomplishing its focus looks like, in terms of setting up the next act (5-6 words)]
6. Failure Scenario: [What this act failing at its focus looks like, in terms of setting up the next act (5-6 words)]

Your response should be consistent with the play-level central narrative and the previous acts.

Generate the act-specific framework that will guide scene-level planning.

After the framework, repeat three of its elements on lines of their own, exactly in this form, one #role line for each character:
#focus <the Central Focus>
#stakes <the Stakes>
#role <character name>: <that character's role in this act>
""")]
        # (each character's narrative summary was reset to empty here every act, erasing the
        # record of what had actually happened before the plans that use it)

        # by what the act is for, not by its number: a play may plan two acts or five
        act_guidance = {
            'opening': "ESTABLISH foundation - Introduce core conflict, form fragile alliances, set central stakes",
            'middle': "COMPLICATE dynamics - Deepen tension, raise external stakes, force difficult choices",
            'resolving': "RESOLVE or collapse - Final confrontation, decisive action, ultimate success or failure",
            'closing': "REFLECT aftermath - This is the closing act, after the question has been settled. Show consequences, "
                       "character transformation, new equilibrium. No new conflict; nothing here succeeds or fails. " + self.episode_note(closing=True),
        }
        character_act_plans = []
        for character in cast(List[NarrativeCharacter], self.actors):
            character.current_act = character.get_act_from_plan(act_number)
            character_act_plans.append(f'{character.name}: {json.dumps(character.current_act, indent=2, default=datetime_handler) if character.current_act else ""}')

        response = self.llm.ask({"play_level_central_narrative": self.central_narrative, 
                                 "act_number": act_number,
                                 "character_act_plans": '\n'.join(character_act_plans), 
                                 "previous_scenes_summary": self.screenplay.so_far() or 'None: this is the first act.',   # what happened, not what was planned
                                 "actor_narratives": '\n'.join([f'{character.name}: \n{character.narrative.get_summary()}\n' for character in cast(List[NarrativeCharacter], self.actors)]), 
                                 "act_specific_guidelines": act_guidance[self.screenplay.kind(act_number)],
                                 "current_situation": self.current_state}, 
                                 prompt, max_tokens=420, stops=['</end>'], tag='act_central_narrative')
        if response:
            # the tagged lines are what an actor is given (Character.act_direction): the act's focus
            # and stakes and its own role, not the others' roles or the scenes still to come
            find = lambda tag: (hash_utils.find(tag, response) or '').strip()
            act = self.screenplay.act
            act.focus = {'focus': find('focus'), 'stakes': find('stakes'),
                         'roles': {n.strip().strip('*'): r.strip() for n, r in re.findall(r'^\s*#role\s+([^:\n]+):\s*(.+)$', response, re.M)}}
            act.framework = re.sub(r'^\s*#(focus|stakes|role)\b.*$', '', response, flags=re.M).strip()
        return self.act_central_narrative or None
    
    def embed_task(self, task):
        """Embed a task"""

        try:
            embedding = self.embedding_model.encode(task.name+': '+task.description+'. because '+task.reason+' to achieve '+task.termination)
            self.scene_task_embeddings.append(embedding)
            return embedding
        except Exception as e:
            print(f"Warning: Could not embed task {task.name}: {e}")
            traceback.print_exc()
            raise e
    
    def cluster_tasks(self, task_embeddings):
        """Cluster tasks"""
        from sklearn.cluster import DBSCAN
        
        if not task_embeddings or len(task_embeddings) < 2:
            self.scene_task_clusters = []
            return self.scene_task_clusters
        
        clustering = DBSCAN(eps=0.20, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(task_embeddings)
        
        # Group by cluster labels
        clusters = []
        cluster_dict = {}
        for i, label in enumerate(labels):
            if label == -1:  # Outlier
                clusters.append([i])
            else:
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(i)
        
        clusters.extend(cluster_dict.values())
        self.scene_task_clusters = clusters
        return self.scene_task_clusters
    
    async def integrate_task_plans(self, scene):
        """Integrate the task plans of all characters in the scene"""
        actors_in_scene = []
        self.scene_task_embeddings = []
        task_id_to_embedding_index = {}
        total_input_task_count = 0
        total_actor_beats = len(scene['action_order'])


        actor_tasks = {}        # get actors and task_plans
        for actor_name in scene['characters']:
            if actor_name == 'Context':
                continue
            if actor_name not in scene['action_order']:
                continue
            actor = self.get_actor_by_name(actor_name)
            if actor.__class__.__name__ == 'NarrativeCharacter':
                actors_in_scene.append(actor)
                actor_tasks[actor.name] = {}
                if not actor.focus_goal: # generate goals if necessary
                    goals = await actor.generate_goals()
                    if goals:  # (called without its goals argument: TypeError ended the narrative)
                        await actor.request_goal_choice(goals)
                if actor.focus_goal and not actor.focus_goal.task_plan:
                    await actor.generate_task_plan(ntasks=(scene['action_order']).count(actor.name))
                actor_tasks[actor.name]['task_plan'] = actor.focus_goal.task_plan if actor.focus_goal and actor.focus_goal.task_plan else []
                total_input_task_count += len(actor_tasks[actor.name]['task_plan'])
                actor_tasks[actor.name]['next_task_index'] = 0
                for n, task in enumerate(actor_tasks[actor.name]['task_plan']):
                    #actor_tasks[actor]['task_plan'][task.id]['criticality'] = self.evaluate_task_criticality(task)
                    task_id_to_embedding_index[task.id] = len(self.scene_task_embeddings)
                    self.embed_task(task)

        prune_level = 0
        if total_input_task_count > 1.3*total_actor_beats:
            prune_level = 1
        if total_input_task_count > 1.5*total_actor_beats:
            prune_level = 2
        if total_input_task_count > 2*total_actor_beats:
            prune_level = 3
        scene_task_clusters = self.cluster_tasks(self.scene_task_embeddings)

        # Create mapping from task id to cluster index
        task_id_to_cluster = {}
        for cluster_idx, task_indices in enumerate(scene_task_clusters):
            for embedding_idx in task_indices:
                # Find task id that corresponds to this embedding index
                for task_id, emb_idx in task_id_to_embedding_index.items():
                    if emb_idx == embedding_idx:
                        task_id_to_cluster[task_id] = cluster_idx
                        break
        
        used_clusters = set()
        self.scene_integrated_task_plan = []
        # Each character's tasks are taken in its own order, interleaved by the scene's speaking
        # order. The list of tasks is the one gathered above: a character whose goal step failed
        # has none and is passed over (its focus_goal was dereferenced here, and raised).
        remaining = [a for a in actors_in_scene if actor_tasks[a.name]['task_plan']]
        while remaining:
            for name in scene['action_order']:
                actor: NarrativeCharacter = self.get_actor_by_name(name)
                if actor not in remaining:
                    continue
                plan = actor_tasks[actor.name]['task_plan']
                next_task_index = actor_tasks[actor.name]['next_task_index']
                current_task = plan[next_task_index]
                is_critical = next_task_index == len(plan) - 1   # a character's last task is never pruned

                # a task much like one already taken (its embedding in the same cluster) is
                # dropped when the scene has more tasks than beats
                cluster_idx = task_id_to_cluster.get(current_task.id, -1)
                if cluster_idx not in used_clusters or prune_level < 1 or is_critical:
                    self.scene_integrated_task_plan.append({'actor': actor, 'goal': actor.focus_goal, 'task': current_task})
                    used_clusters.add(cluster_idx)
                else:
                    trace.event('task_pruned', character=actor.name, task_id=current_task.id, task=current_task.name)

                actor_tasks[actor.name]['next_task_index'] += 1
                if actor_tasks[actor.name]['next_task_index'] >= len(plan):
                    remaining.remove(actor)

        return self.scene_integrated_task_plan
    
    def format_scene_integrated_task_plan(self):
        """Format the scene integrated task plan"""
        formatted_task_plan = []
        for scene_task in self.scene_integrated_task_plan:
            formatted_task_plan.append(f' - Task: {scene_task["actor"].name} {scene_task["goal"].name+': '+scene_task["goal"].description}\n\t {scene_task["task"].to_string()}')
        return '\n'.join(formatted_task_plan)
    

    def evaluate_commitment_significance(self, character:Character, other_character:Character, commitment_task:Task):
        prompt = [UserMessage(content="""Evaluate the significance of a tentative verbal commitment to the current dramatic context
    and determine if it is a significant commitment that should be added to the task plan.
The apparent commitment is inferred from a verbal statement to act now made by self, {{$name}}, to other, {{$target_name}}.
The apparent commitment is tentative, and may be redundant in the scene you are in.
The apparent commitment may be hypothetical, and may not be a commitment to act now.
The apparent commitment may be social chatter, e.g. simply a restatement of already established plans. These should be reported as NOISE.
The apparent commitment may simply be 'social noise' - a statement of intent or opinion, not a commitment to act now. Again, these should be reported as NOISE.
The apparent commitment may be relevant to the scene you are in or the dramatic context, and unique wrt tasks already in the task plan. These should be reported as RELEVANT.
The apparent commitment may be relevant to the scene you are in or the dramatic context, but redundant with tasks already in the task plan. These should be reported as REDUNDANT.
Alternatively, the apparent commitment may be unique and significant to the scene you are in or the central dramatic question. These should be reported as SIGNIFICANT.
The apparent commitment may be irrelevant to the scene you are in or the dramatic context. These should be reported as NOISE.

## Dramatic Context
<central_narrative>
{{$central_narrative}}
</central_narrative>

<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

<current_scene>
{{$current_scene}}
</current_scene>

##Scene Task Plan
<scene_integrated_task_plan>
{{$scene_integrated_task_plan}}
</scene_integrated_task_plan>

##Actual Scene History to date
<scene_history> 
{{$scene_history}}
</scene_history>

<transcript>    
{{$transcript}}
</transcript>
        
## Apparent Commitment
<commitment>
{{$commitment_task}}
</commitment>

Your task is to evaluate the significance of the apparent commitment.
Determine if it is 
    social NOISE that should be ignored.
    a REDUNDANT task that can be ignored. 
    a RELEVANT task that can be optionally included.
    a SIGNIFICANT commitment that should be added to the task plan.
    a CRUCIAL commitment, the task plan is incomplete relative to the dramatic context without it.
Respond with a single word: "NOISE", "REDUNDANT", "RELEVANT", "SIGNIFICANT", or "CRUCIAL".
Do not include any other text in your response.

End your response with </end>
"""
        )]
        significance = 'NOISE'
        response = self.llm.ask({"name": character.name, 
                               "target_name": other_character.name,
                               "commitment_task": commitment_task.to_string(),
                               "central_narrative": self.central_narrative,
                               "act_specific_narrative": self.act_central_narrative,
                               "current_scene": json.dumps(self.current_scene, indent=2, default=datetime_handler) if self.current_scene else '',
                               "scene_integrated_task_plan": self.format_scene_integrated_task_plan() if self.scene_integrated_task_plan else "",
                               "scene_history": '\n'.join([history for history in self.scene_history]) if self.scene_history else '',
                               "transcript": character.actor_models.get_actor_model(other_character.name, create_if_missing=True).dialog.get_transcript(8)},
                                prompt, max_tokens=100, stops=['</end>'], tag='evaluate_commitment_significance')
        return self.parse_significance(response)

    SIGNIFICANCE_LEVELS = ('NOISE', 'REDUNDANT', 'RELEVANT', 'SIGNIFICANT', 'CRUCIAL')

    @staticmethod
    def parse_significance(response):
        """First word of the reply as a significance level, else NOISE. (Substring tests read
        'insignificant' as SIGNIFICANT, 'irrelevant' as RELEVANT, any chatty 'crucial' as CRUCIAL.)"""
        words = re.findall(r'[A-Za-z]+', response or '')
        level = words[0].upper() if words else ''
        return level if level in Context.SIGNIFICANCE_LEVELS else 'NOISE'

    def add_event(self, text):
        """Record a concrete physical event (see recent_events)."""
        text = ' '.join(str(text).split())[:300]
        if text:
            line = f'{self.simulation_time.strftime("%H:%M")} {text}'
            self.recent_events.append(line)
            self.scene_events.append(line)
            self.act_events.append(line)
            trace.event('world_event', text=text)

    def recent_events_text(self, n=6):
        return '\n'.join(list(self.recent_events)[-n:]) or 'Nothing yet.'

    def place_commitment(self, character, task, significance, goal=None):
        """Put a commitment from a dialog into the running scene's task plan. Returns True if
        placed. A scene may grow, but not without end: the whole plan (tasks done and to come)
        is held to a multiple of the length it started with:
          CRUCIAL      up to 2x
          SIGNIFICANT  up to 1.5x
          RELEVANT     up to 1.25x, as an optional entry that run_scene skips if the scene is
                       over budget by the time it comes up
          REDUNDANT / NOISE  dropped
        (The cap used to count only the tasks still to run, so a scene that kept completing
        tasks kept accepting new ones: a 3-beat scene ran 18 tasks.)
        Several commitments from one dialog run in the order given (each was inserted at
        index+1, so they ran in reverse). Every drop is logged."""
        plan = self.scene_integrated_task_plan
        goal = goal or character.focus_goal
        why = None
        if plan is None or not self.current_scene:
            why = 'no running scene'
        elif goal is None:
            why = 'no goal to attach it to'
        elif significance in ('NOISE', 'REDUNDANT'):
            why = significance
        if why is None:
            budget = max(1, self.scene_task_budget or len(plan))
            limit = {'CRUCIAL': 2.0, 'SIGNIFICANT': 1.5, 'RELEVANT': 1.25}[significance] * budget
            if len(plan) >= limit:
                why = f'{significance} but the scene already has {len(plan)} tasks (limit {limit:g})'
        if why:
            logger.info(f'commitment dropped ({why}): {character.name} {task.name}')
            trace.event('commitment', character=character.name, task_id=task.id, task=task.name,
                        description=task.description, significance=significance, placed=False, reason=why)
            print(f'{character.name} commitment dropped ({why}): {task.name}')
            return False
        index = self.scene_integrated_task_plan_index
        pos = index + 1
        while pos < len(plan) and plan[pos].get('inserted_at') == index:
            pos += 1
        entry = {'actor': character, 'goal': goal, 'task': task, 'inserted_at': index}
        if significance == 'RELEVANT':
            entry['optional'] = True
        plan.insert(pos, entry)
        label = 'optional ' if entry.get('optional') else ''
        print(f'{character.name} new {label}committed task ({significance}): {task.name}')
        logger.info(f'commitment placed ({significance}{", optional" if label else ""}): {character.name} {task.name}')
        trace.event('commitment', character=character.name, task_id=task.id, task=task.name,
                    description=task.description, significance=significance, placed=True,
                    optional=bool(entry.get('optional')), position=pos)
        return True

    def repair_xml(self, response, error):
        """Repair XML if it is invalid"""
        
        # Try to extract XML from common wrapper patterns
        if not response.strip().startswith('<') and '<' in response:
            # Look for XML in code blocks or other wrapping
            import re
            xml_pattern = r'```xml\s*\n(.*?)\n```'
            xml_match = re.search(xml_pattern, response, re.DOTALL | re.IGNORECASE)
            if xml_match:
                response = xml_match.group(1).strip()
            else:
                # Look for first < to last >
                start = response.find('<')
                end = response.rfind('>')
                if start != -1 and end != -1 and end > start:
                    response = response[start:end+1]

        # Try to fix common XML issues
        # Remove newlines that break attribute parsing
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            # Keep line if it's whitespace or starts with <
            if line.strip() == '' or line.strip().startswith('<'):
                cleaned_lines.append(line)
            else:
                # If previous line doesn't end with >, append to it
                if cleaned_lines and not cleaned_lines[-1].strip().endswith('>'):
                    cleaned_lines[-1] += ' ' + line.strip()
                else:
                    cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        
        # Try parsing the cleaned XML
        try:
            import xml.etree.ElementTree as ET
            # Wrap in root element to handle multiple top-level elements
            wrapped_xml = f"<root>{response.strip()}</root>"
            root = ET.fromstring(wrapped_xml)
            return response  # Return original if parsing succeeds
        except ET.ParseError as e:
            error = e

        # OK, ask LLM to repair
        prompt = [UserMessage(content="""You are an XML repair tool.
Your task is to repair the following XML:

<xml>
{{$xml}}
</xml> 

The reported error is:
{{$error}}

Respond with the repaired XML string. Make sure the string is well-formed XML that can be parsed by xml.etree.ElementTree.fromstring. No commentary, no code fences. Just the raw XML.
""")]
        
        repaired = self.llm.ask({"xml": response, "error": error}, prompt, tag='repair_xml', temp=0.2, max_tokens=3500)
        
        # Clean up any remaining code fences or commentary
        repaired = repaired.replace("```xml", "").replace("```", "").strip()
        
        # Test the repair
        try:
            import xml.etree.ElementTree as ET
            wrapped_xml = f"<root>{repaired.strip()}</root>"
            ET.fromstring(wrapped_xml)
            return repaired
        except ET.ParseError as e:
            print(f'Error parsing repaired XML: {e}')
            return None
        
    

    async def replan_scene(self, scene, scene_number, act, act_number, intervention):
        """Rewrite the act with the latest shared information"""

        system_prompt = """You are a seasoned writer rewriting scene {{$scene_number}} in act {{$act_number}}.
This scene is about to be performed, an unexpected event has suddenly occurred, and you need to rewrite the scene to reflect this.

#The previous plan for this scene is:

{{$scene}}
##

#This scene was to occur in act {{$act_number}}:

{{$act}}
##

#This act was part of a performance focusing on the following overall dramatic question:

{{$central_narrative}}
##

#This new unexpected event has just occurred and should be central to the rewritten scene:

{{$intervention}}
##

The scene should be rewritten to reflect this event.

"""
        mission = """ 
#The actual scenes performed so far are:

{{$previous_scenes}}
##
"""
        suffix = """

Pay special attention to the goals you have already attempted and their completion status in replanning your scene. 
Do not repeat goals that have already been attempted and completed, unless you have a new reason to attempt them again or it is important for dramatic tension.

Note that in the current situation any assumptions or preconditions on which your original act and plan were based may no longer be valid given the event that has occurred.
However, the new scene should be consistent with the overall narrative arc of the performance, and represent an authentic response to the event that has occurred.

The following act-specific guidelines supplement the general guidelines above. 
These act-specific guidelines provide additional constraints in rewriting the scene about to be performed.
Again, the current act number is {{$act_number}} and the scene number is {{$scene_number}}:

** For act 1:
    1. This act should be short and to the point..
    2. Sequence scenes to introduce characters and create unresolved tension.
    3. Establish the central dramatic question clearly: {{$central_narrative}}
    4. Act post-state must specify: what the characters now know, what they've agreed together, and what specific tension remains unresolved.
    5. Final scene post-narrative must show characters making a concrete commitment or decision about their shared challenge.
    6. Ensure act post-state represents measurable progress from act pre-state, not just mood shifts.
 
** For act 2 (midpoint act):
    1. Each scene must advance the central dramatic question: {{$central_narrative}}
    2. Midpoint should fundamentally complicate the question (make it harder to answer or change its nature).
    3. Prevent lateral exploration - every scene should move closer to OR further from resolution..
    5. Avoid pointless repetition of scene intentions, but allow characters to develop their characters.
    6. Sequence scenes for continuously building tension, perhaps with minor temporary relief, E.G., create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)
    7. Ensure each scene raises stakes higher than the previous scene - avoid cycling back to earlier tension levels.
    8. Midpoint scene post-narrative must specify: what discovery/setback fundamentally changes the characters' approach to the central question.
    9. Act post-state must show: what new obstacles emerged, what the characters now understand differently, and what desperate action they're forced to consider.
    10. Each scene post-narrative must demonstrate measurable escalation from the previous scene - not just "tension increases" but specific new problems or revelations.

** For act 3 (final act):
    1. Directly answer the central dramatic question: {{$central_narrative}}
    2. No scene should avoid engaging with the question's resolution.
    3. Sequence scenes for maximum tension (eg, alternate elation/disappointment, tension/relief, etc. beats)
    4. create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)  
    5. Act post-state must explicitly state: whether the General dramatic question was answered YES or NO, what specific outcome was achieved, and what the characters' final status is.
    6. Final scene post-narrative must show the concrete resolution - not "they find peace" but "they escape the forest together" or "they remain trapped but united."
    7. No Scene may end with ambiguous outcomes - each must show clear progress toward or away from resolving the central question.

** For act 4 (coda):
    1. Show the immediate aftermath and consequences of Act 3's resolution of: {{$central_narrative}}
    2. Maximum two scenes - focus on essential closure only, avoid extended exploration.
    3. Preserve character scene intentions where possible. Combine overlapping scene intentions from different characters where possible.
    4. First scene should show the immediate practical consequences of the resolution (what changes in their situation).
    5. Second scene (if needed) should show the emotional/relational aftermath (how characters have transformed).
    6. No new conflicts or dramatic questions - only reveal the implications of what was already resolved.
    7. Act post-state must specify: the characters' new equilibrium, what they've learned or become, and their final emotional state.
    8. Final scene post-narrative must provide definitive closure - show the "new normal" that results from their journey.
    9. Avoid ambiguity about outcomes - the coda confirms and completes the resolution, not reopens questions.


The scene continues from where the previous one ended; move it only if the event demands it.
#Places
{{$places}}
##

Respond with the updated scene, using the following format:

```json
updated scene
```

Scene format is as follows:
{ "scene_number": int, // sequential within the play 
 "scene_title": string, // concise descriptor 
 {{$location_field}}
 "time": "2025-01-01T08:00:00", // the start time of the scene, in ISO 8601 format
 "characters": { "<Name>": { "goal": "<one-line playable goal>" }, … }, 
 "action_order": [ "<Name>", … ], // each name occurrence is a 'beat' in the scene lead by the named character. list only characters present in the scene 'characters' list.
 "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors.", 
 "post_narrative": "Short prose (≤20 words): how things stand at the end and what tension lingers, shown in what the characters do. Nobody announces a feeling or a choice ('says aloud that he stays' is not an ending; his staying is)." 
 // OPTIONAL: 
 "task_budget": 4 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the scene to avoid rambling or repetition, or to 1.67*len(characters) for scenes with complex goals or interactions.
 }


"""

        scene_number = scene.get('scene_number',0)
        response = await self.context_ask(system_prompt=system_prompt, prefix=mission, suffix=suffix,
                              addl_bindings={"name": self.name, "scene": json.dumps(scene, indent=1, default=datetime_handler), 
                                    "scene_number": scene_number,
                                    "act_number": act_number,
                                    "location_field": stage.LOCATION_FIELD,
                                    "places": stage.gazetteer(self),
                                    "act": json.dumps(act, indent=1, default=datetime_handler),
                                    "intervention": intervention,
                                    "act_central_narrative": self.act_central_narrative,
                                    "central_narrative": self.central_narrative,
                                    "previous_scenes": '\n'.join(filter(None, [self.screenplay.so_far(), ' '.join(self.act_scene_wrapups)]))},   # what happened, not what was planned
                              max_tokens=800, tag='NarrativeCharacter.replan_scene')
        try:
            updated_scene = None
            scene_id = scene.get('scene_number',0)
            if not response:
                return scene   # no rewrite: play the scene as planned (None here stopped the play)
            response = response.replace("```json", "").replace("```", "").strip()
            updated_scene = json.loads(response)
        except Exception as e:
            print(f'Error parsing updated scene: {e}')
            logger.error(f'Error parsing updated act: {e}')
            logger.error(traceback.format_exc())
            updated_scene = self.repair_json(response, e)
        if updated_scene:
            print(f'{self.name} updates scene {scene_id}')
        else:
            print(f'{self.name} failed to update scene {scene_id}')
            updated_scene = scene
        self.current_scene = updated_scene
        return updated_scene    