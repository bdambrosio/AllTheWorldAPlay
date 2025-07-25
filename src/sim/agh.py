from __future__ import annotations
import base64
import os, json, math, time, requests, sys

from sklearn import tree

from src.sim.SpeechStylizer import SpeechStylizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# Add parent directory to path to access existing simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
from enum import Enum
import json
import random
import string
import traceback
import time
from typing import List, Dict, Optional, TYPE_CHECKING, cast
from sim.cognitive import knownActor
from sim.cognitive import perceptualState
from sim.cognitive.driveSignal import Drive, DriveSignalManager, SignalCluster
from sim.memory.consolidation import MemoryConsolidator
from sim.memory.core import MemoryEntry, NarrativeSummary, StructuredMemory
from sim.memory.core import MemoryRetrieval
from src.sim.cognitive.EmotionalStance import EmotionalStance
from utils import llm_api
from utils.llm_api import generate_image
from utils.Messages import UserMessage, SystemMessage
import utils.xml_utils as xml
import sim.map as map
import utils.hash_utils as hash_utils
import utils.choice as choice   
from sim.cognitive.DialogManager import Dialog
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np # type: ignore
from sim.cognitive.perceptualState import PerceptualInput, PerceptualState, SensoryMode
from sim.cognitive.knownActor import KnownActor, KnownActorManager
from sim.cognitive.knownResources import KnownResource, KnownResourceManager
from sim.ResourceReferenceManager import ResourceReferenceManager
import re
import asyncio
from weakref import WeakValueDictionary
import logging
from sim.prompt import ask as default_ask
from sim.cognitivePrompt import CognitiveToolInterface
from src.sim.character_dataclasses import Mode, Stack, Act, Goal, Task, Autonomy, datetime_handler     
if TYPE_CHECKING:
    from sim.context import Context  # Only imported during type checking

logger = logging.getLogger('simulation_core')


def find_first_digit(s):
    for char in s:
        if char.isdigit():
            return char
    return None  # Return None if no digit is found


from datetime import timedelta

def parse_duration(duration_str: str) -> timedelta:
    """Convert duration string to timedelta
    Args:
        duration_str: Either minutes as int ("5") or with units ("2 minutes")
    Returns:
        timedelta object
    """
    try:
        # Try simple integer (minutes)
        return timedelta(minutes=int(duration_str))
    except ValueError:
        # Try "X units" format
        try:
            amount, unit = duration_str.strip().split()
            amount = int(amount)
            unit = unit.lower().rstrip('s')  # handle plural
            
            if unit == 'minute':
                return timedelta(minutes=amount)
            elif unit == 'hour':
                return timedelta(hours=amount)
            elif unit == 'day':
                return timedelta(days=amount)
            else:
                # Default to minutes if unit not recognized
                return timedelta(minutes=amount)
        except:
            # If all parsing fails, default to 1 minute
            return timedelta(minutes=1)


# Character base class
class Character:
    def __init__(self, name, character_description, reference_description='', init_x=50, init_y=50, server_name='local', mapAgent=True):
        print(f"Initializing Character {name}")  # Debug print
        self.name = name.strip().capitalize()
        self.character = character_description
        self.original_character = character_description
        self.llm = llm_api.LLM(server_name)
        if len(reference_description) > 0:
            self.reference_dscp = reference_description
        else:
            self.reference_dscp = self.llm.ask({}, [SystemMessage(content='generate a concise single sentence description for this character useful for reference resolution. Respond only with the description. End your response with: </end>'), 
                                                    UserMessage(content=f"""character name {self.name}\ncharacter description:\n{character_description}\n\nEnd your response with: </end>""")
        ], tag='reference_dscp', max_tokens=40, stops=["</end>"])
        if self.reference_dscp:
            self.reference_dscp = self.reference_dscp.strip()
        else:
            self.reference_dscp = ''
        self.context: Context = None

        self.show = ''  # to be displayed by main thread in UI public text widget
        self.reason = ''  # reason for action
        self.thought = ''  # thoughts - displayed in character thoughts window
        self.sense_input = ''
        self.widget = None
        
        # Initialize focus_task stack
        self.focus_task:Stack = Stack()
        self.action_history = []
        self.act_result = ''
        self.wakeup = True

        # Initialize narrative
        self.narrative = NarrativeSummary(
            recent_events="",
            ongoing_activities="",
            last_update=datetime.now(),  # Will be updated to simulation time when set_context is called
            active_drives=[]
        )

        # World integration attributes
        self.init_x = init_x # save these for later mapAgent initialization
        self.init_y = init_y
        if mapAgent:
            self.mapAgent = None  # Will be set later
            self.world = None
            self.my_map = [[{} for i in range(100)] for i in range(100)]
        self.perceptual_state = PerceptualState(self)
        self.last_sense_time = datetime.now()
        self.act = None
        self.previous_action_name = None
        self.look_percept = ''
        self.driveSignalManager = None
       # Initialize drives
        self.drives = [
            Drive( "immediate physiological needs: survival, water, food, clothing, shelter, rest."),
            Drive("safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events."),
            Drive("love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friendship, intimacy, trust, acceptance."),
            Drive("self-esteem, including recognition, respect, and achievement.")
        ]
            
        self.always_respond = True
        
        # Initialize memory systems
        self.cognitive_llm = CognitiveToolInterface(self)
        self.structured_memory:StructuredMemory = StructuredMemory(owner=self)
        self.memory_consolidator:MemoryConsolidator = MemoryConsolidator(self, self.llm, self.context)
        self.memory_retrieval:MemoryRetrieval = MemoryRetrieval()
        self.new_memory_cnt = 0
        self.next_task = None  # Add this line
        self.driveSignalManager:DriveSignalManager = DriveSignalManager(self, self.llm, ask=default_ask)
        self.driveSignalManager.set_context(self.context)
        self.focus_goal:Goal = None
        self.focus_goal_history:List[Goal] = []
        self.focus_task:Stack = Stack()
        self.last_task = None
        self.focus_action:Act = None
        self.goals:List[Goal] = []
        self.goal_history:List[Goal] = []
        #self.tasks = [] 
        self.intensions:List[Task] = []
        self.actions:List[Act] = []
        self.achievments:List[str] = [] # a list of short strings summarizing the character's achievments
        self.autonomy:Autonomy = Autonomy()
        self.step = 2 # start at step 2 to skip first update
        self.update_interval = 4
        self.emotionalStance:EmotionalStance = EmotionalStance()
        self.do_actions:str = '' #generated by NarrativeCharacter.generate_action_vocabulary()
        self.decisions:List[str] = [] # a list of decisions made by the character
        self.avoidance_cnt = 0 # a counter of how many times the character has avoided a decision
        #self.speech_stylizer = SpeechStylizer(self)

    def x(self):
        return self.mapAgent.x
    
    def y(self):
        return self.mapAgent.y

    def set_context(self, context: Context):
        self.context = context
        self.actor_models = KnownActorManager(self, context)
        self.resource_models = KnownResourceManager(self, context)
        self.resourceRefManager = ResourceReferenceManager(self, context, self.llm)
        self.narrative = NarrativeSummary(
            recent_events="",
            ongoing_activities="",
            last_update=self.context.simulation_time, 
            active_drives=[]
        )
        if self.driveSignalManager:
            self.driveSignalManager.context = context
        if self.memory_consolidator:
            self.memory_consolidator.context = context
        ranked_signalClusters = self.driveSignalManager.get_scored_clusters()
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order
        self.emotionalStance = EmotionalStance.from_signalClusters(focus_signalClusters, self)
        self.set_character_traits()
        self.speech_stylizer = SpeechStylizer(self) 
 

    def set_autonomy(self, autonomy_json):
        if 'act' in autonomy_json:
            self.autonomy.act = autonomy_json['act']
        if 'scene' in autonomy_json:
            self.autonomy.scene = autonomy_json['scene']
        if 'signal' in autonomy_json:
            self.autonomy.signal = autonomy_json['signal']
        if 'goal' in autonomy_json:
            self.autonomy.goal = autonomy_json['goal']
        if 'task' in autonomy_json:
            self.autonomy.task = autonomy_json['task']
        if 'action' in autonomy_json:
            self.autonomy.action = autonomy_json['action']

    def get_current_act_info(self):
        """Get current act information for display"""
        if hasattr(self, 'current_act') and self.current_act:
            # NarrativeCharacter has its own current_act
            return {
                'title': self.current_act.get('act_title', 'Unknown Act'),
                'description': self.current_act.get('act_description', ''),
                'number': self.current_act.get('act_number', 0),
                'pre': self.current_act.get('act_pre_state', ''),
                'post': self.current_act.get('act_post_state', '')
            }
        elif self.context and self.context.current_act:
            # Regular character uses context current_act
            return {
                'title': self.context.current_act.get('act_title', 'Unknown Act'),
                'description': self.context.current_act.get('act_description', ''),
                'number': self.context.current_act.get('act_number', 0),
                'pre': self.context.current_act.get('act_pre_state', ''),
                'post': self.context.current_act.get('act_post_state', '')
            }
        return {'title': 'No Act', 'description': '', 'number': 0}

    def get_current_scene_info(self):
        """Get current scene information for display"""
        if hasattr(self, 'current_scene') and self.current_scene:
            # NarrativeCharacter has its own current_scene
            return {
                'title': self.current_scene.get('scene_title', 'Unknown Scene'),
                'location': self.current_scene.get('location', ''),
                'number': self.current_scene.get('scene_number', 0)
            }
        elif self.context and self.context.current_scene:
            # Regular character uses context current_scene
            return {
                'title': self.context.current_scene.get('scene_title', 'Unknown Scene'),
                'location': self.context.current_scene.get('location', ''),
                'number': self.context.current_scene.get('scene_number', 0)
            }
        return {'title': 'No Scene', 'location': '', 'number': 0}

    def set_character_traits(self):
        prompt = [SystemMessage(content="""Given the following character description, extract the character's likely gender, age group, accent, and personality keywords as a hash-formatted block.

Character Name: {{$name}}
Description: {{$description}}

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Close the hash-formatted text with ##  on a separate line, as shown below.
be careful to insert line breaks only where shown, separating a value from the next tag:

#gender female / male
#age middle_aged / young / old
#accent american / british / australian / etc.
#personality expressive, confident / shy, introverted / etc.
##

End your response with:
</end>
""")]

        # 2. Call LLM and parse response
        llm_response = self.llm.ask({'name': self.name, 'description': self.character}, prompt, tag='voice_traits', stops=['</end>'], max_tokens=100)
        if llm_response is None:
            return None
        self.gender = hash_utils.find('gender', llm_response)
        self.age = hash_utils.find('age', llm_response)
        self.accent = hash_utils.find('accent', llm_response)
        self.personality = hash_utils.findList('personality', llm_response)


    def get_character_description(self):
        description = self.character
        if self.actor_models.get_known_actor_relationship(self.name):
            description += '\n\n' + self.actor_models.get_known_actor_relationship(self.name)
        if type(description) != str:
            description = ''
        return description
    
    def validate_and_create_goal(self, goal_hash, signalCluster=None):
        """Validate a goal hash and create a goal object
        
        Args:
            goal_hash: Hash-formatted goal definition
            signalCluster: SignalCluster this goal is for
        """
        name = hash_utils.find('goal', goal_hash)
        description = hash_utils.find('description', goal_hash)
        other_actor_name = hash_utils.find('otherActorName', goal_hash)
        signalCluster_id = hash_utils.find('signalCluster_id', goal_hash)
        preconditions = hash_utils.find('preconditions', goal_hash)
        termination = hash_utils.find('termination', goal_hash)

        signalCluster = None
        if signalCluster_id:
            signalCluster = SignalCluster.get_by_id(signalCluster_id.strip())
        other_actor = None
        if other_actor_name and other_actor_name.strip().lower() != 'none':
            other_actor,_ = self.actor_models.resolve_or_create_character(other_actor_name)

        if name and description and termination:
            goal = Goal(name=name, 
                        actors=[self, other_actor] if other_actor else [self],
                        description=description, 
                        preconditions=preconditions,
                        termination=termination.replace('##','').strip(), 
                        signalCluster=signalCluster, 
                        drives=signalCluster.drives if signalCluster else []
            )
            return goal
        else:
            print(f"Warning: Invalid goal generation response for {goal_hash}") 
            return None
        
    def characters_from_hash(self, hash_form):
        """Get character names from a hash string"""
        targets = []
        try:
            target_form = hash_utils.find('actors', hash_form)
            if not target_form:
                target_form = hash_utils.find('target', hash_form)
            if not target_form:
                target_form = hash_utils.find('targets', hash_form)
            if target_form:
                target_names = target_form.split(',') if target_form else []
                for name in target_names:
                    name = name.strip().capitalize()
                    names = name.split(' and ')
                    for n in names:
                        if n and n.strip() != '':
                            target,_ = self.actor_models.resolve_or_create_character(n)
                            if target:
                                targets.append(target)
        except Exception as e:
            targets = []
        return targets

    def validate_and_create_task(self, task_hash, goal=None):
        """Validate a task hash and create a task object
        
        Args:
            task_hash: Hash-formatted task definition
            goal: Goal this task is for
        """
        name = hash_utils.find('name', task_hash)
        if not name:
            name = hash_utils.find('task', task_hash)
        description = hash_utils.find('description', task_hash)
        reason = hash_utils.find('reason', task_hash)
        if not reason:
            reason = ' '
        termination = hash_utils.find('termination', task_hash)
        if not termination:
            termination = ''
        start_time = hash_utils.find('start_time', task_hash)
        duration = hash_utils.find('duration', task_hash)
        try:
            characters = self.characters_from_hash(task_hash)
        except Exception as e:
            print(f"Warning: invalid actors field in {task_hash}") 
            characters = []
        if self.mapAgent:
            default_x = self.mapAgent.x
            default_y = self.mapAgent.y
        else:
            default_x = 30
            default_y = 30
        for character in characters:
            if character and hasattr(character, 'mapAgent'):
                default_x = character.mapAgent.x
                default_y = character.mapAgent.y
            elif character:
                print(f"Warning: {character} is not a character")
        actors = [actor for actor in characters if actor is not None]
        if not self in actors:
            actors =  [self] + actors

        if name and description and reason and actors:
            task = Task(name, description=description, 
                        reason=reason, 
                        termination = termination.replace('##','').strip(), 
                        goal = goal, 
                        actors = actors, 
                        start_time = start_time, 
                        duration = duration)
            return task
        else:
            print(f"Warning: Invalid task generation response for {task_hash}") 
            return None
        
    def validate_and_create_act(self, hash_string, task):
        """Validate an XML act and create an Act object
        
        Args:
            hash_string: Hash-formatted act definition
            task: Task this act is for
        """
        mode = hash_utils.find('mode', hash_string)
        action = hash_utils.find('action', hash_string)
        duration = hash_utils.find('duration', hash_string)
        if isinstance(duration, str) and duration.strip().isdigit():
            duration = timedelta(minutes=int(duration))
        elif isinstance(duration, int):
            duration = timedelta(minutes=duration)

        name_stop_list = ['none', 'no', 'one', 'self', 'me', 'i', 'my', 'mine', 'ours', 'us', 'we', 'our', 'ourselves']
        target_names = hash_utils.findList('target', hash_string)
        targets = [self.actor_models.resolve_or_create_character(name.strip().capitalize())[0] for name in target_names if name.strip().lower() not in name_stop_list]
        actor_names = hash_utils.findList('actors', hash_string)
        actors = [self.actor_models.resolve_or_create_character(name.strip().capitalize())[0] for name in actor_names if name.strip().lower() not in name_stop_list]

        # Clean up mode string and validate against Mode enum
        if mode:
            mode = mode.strip().capitalize()
            try:
                mode_enum = Mode[mode]  # This validates against the enum
                mode = mode_enum.value  # Get standardized string value
            except KeyError:
                return None
        
        actors = [self]
        if targets and (not actors or len(actors) == 0):
            if self in targets:
                actors = targets
            else:
                actors = [self] + targets
        if mode and action:
            act = Act(
                mode=mode,
                action=action,
                actors=actors,
                reason=action,
                duration=duration,
                source=task,
                target=targets
            )
            return act
        else:
            print(f"Invalid actionable Hash: {hash_string}")
            return None
        
    def format_history(self, n=16):
        """Get memory context including both concrete and abstract memories"""
        # Get recent concrete memories
        recent_memories = self.structured_memory.get_recent(n)
        memory_text = []
        
        if recent_memories:
            concrete_text = '\n'.join(f"- {memory.to_string()}" for memory in recent_memories)
            memory_text.append("Recent Events:\n" + concrete_text)
        
        # Get current activity if any
        current = self.structured_memory.get_active_abstraction()
        if current:
            memory_text.append("Current Activity:\n" + current.summary)
        
        # Get recent completed activities
        recent_abstracts = self.structured_memory.get_recent_abstractions(4)
        if recent_abstracts:
            # Filter out current activity and format others
            completed = [mem for mem in recent_abstracts if not mem.is_active]
            if completed:
                abstract_text = '\n'.join(f"- {mem.summary}" for mem in completed)
                memory_text.append("Recent Activities:\n" + abstract_text)
        
        return '\n\n'.join(memory_text) if memory_text else ""


    def demote_drive(self, drive: Drive):
        """Demote a drive"""
        if drive in self.drives:
            self.drives.remove(drive)
        self.drives.append(drive)
        
    def promote_drive(self, drive: Drive):
        """Promote a drive"""   
        if drive in self.drives:
            self.drives.remove(drive)
        self.drives.insert(0, drive)    

    def _find_related_drives(self, message: str) -> List[Drive]:
        """Find drives related to a memory message"""
        related_drives = []
        
        # Create temporary memory entry to compare against drives
        temp_memory = MemoryEntry(
            text=message,
            timestamp=self.context.simulation_time,
            importance=0.5,
            confidence=1.0
        )
        temp_memory.embedding = self.memory_retrieval.embedding_model.encode(message)
        
        # Check each drive for relevance
        for drive in self.drives:
            similarity = self.memory_retrieval._compute_similarity(
                temp_memory.embedding,
                drive.embedding if drive.embedding is not None 
                else self.memory_retrieval.embedding_model.encode(drive.text)
            )
            if similarity >= 0.1:  # Use same threshold as retrieval
                related_drives.append(drive)
                
        return related_drives

    def update(self, now=False):
        """Move time forward, update state"""
        # Enhanced consolidation (semantic clustering, LLM summaries) if we have drives
        # Note: Basic abstraction now happens automatically on add_entry
        if hasattr(self, 'drives') and hasattr(self, 'memory_consolidator'):
            self.memory_consolidator.consolidate(self.structured_memory)

        # Add narrative update
        if hasattr(self, 'narrative'):
            self.memory_consolidator.update_cognitive_model(
                memory=self.structured_memory,
                narrative=self.narrative,
                knownActorManager=self.actor_models,
                current_time=self.context.simulation_time,
                character_desc=self.character,
                relationsOnly=False
            )
       # Reset new memory counter
        self.new_memory_cnt = 0
        self.step += 1 # skip next update
        if self.step < 5 and not now:
            return
        self.driveSignalManager.recluster() 
        self.step = 0
        system_prompt = """You are an actor with deep insight into human psychology and the character you are playing, {{$name}}, in an ongoing improvisational play.
Your task is to review the original and current descriptions of your character, and generate a new description of your character accurately reflecting your character's development.\
A primary task is to differentiate between temporary traits and decisions, and your character's fundamental personality and commitments.
Some traits and decisions may be temporary, and should be allowed to fade away, but your character's fundamental personality and commitments should be preserved. 
Carry forward your character's fundamental personality and commitments. Otherwise, allow this description to evolve slowly in keeping with your character's development. 
Preserve your character's fundamental decision-making capability - acknowledge doubt, but don't allow it to become dominant.
This description will inform the next scene and remainder of the play.
"""
        mission = """

#Original description
{{$original_description}}
##

#Current description
{{$current_description}}
##

"""
        suffix = """

#Original description
{{$original_description}}
##

#Current description
{{$current_description}}
##

Respond with a new description of your character, in the same overall style as the original description (ie, no new formatting, no new tags, no new structure), 
reflecting your character's development. The first sentence is used to identify the character and must contain two critical pieces of information: 
(1) the character's name, age, and gender - these should simply be copied from the original description; and (2) a description of the character's current appearance.
Limit your response to 125 words at most.
Do not include any reasoning, introductory, explanatory, or discursive text, or any markdown or other formatting. 

"""
        response = default_ask(self, system_prompt=system_prompt, prefix=mission, suffix=suffix, 
                                addl_bindings={"original_description":self.original_character, "current_description":self.character}, 
                                tag='Agh.update', max_tokens=300)
        if response:
            self.character = response
        return response








    def say_target(self, act_mode, text, source=None):
        """Determine the intended recipient of a message"""
        #if len(self.context.actors) == 2:
        #    for actor in self.context.actors:
        #        if actor.name != self.name:
        #            if actor.name.lower() in text.lower():
        #                return actor.name
        if 'dialog with ' in source.name:
            return source.name.strip().split('dialog with ')[1].strip()
        
        prompt = [UserMessage(content="""Determine the intended hearer of the message below spoken by you.
Background:
{{$character}}

You're recent history has been:

<history>
{{$history}}
</history>
        
Known other actors include:
        
<actors>
{{$actors}}
</actors>
        
The message is an {{$action_type}} and its content is:
        
<Message>
{{$message}}
</Message>

If the message is an internal thought, respond with the name of the actor the thought is about.
If the message is a spoken message, respond with the name of the intended recipient. For example 'Ask Mary about John' would respond with 'Mary'.
An intended recipient may be another known actor or a third party. Note that known actors are not always the intended recipient.
Respond using the following XML format:

  <name>intended recipient name</name>

End your response with:
</end>
""")]
        #print('say target')
        response = self.llm.ask({
            'character': self.get_character_description(),
            'history': self.narrative.get_summary('medium'),
            'actors': '\n'.join([actor.name for actor in self.context.actors if actor != self]+[npc.name for npc in self.context.npcs]),
            'action_type': 'internal thought' if act_mode == 'Think' else 'spoken message',
            "message": text
        },  prompt, tag='Agh.say_target', temp=0.2, stops=['</end>'], max_tokens=20)

        candidate = xml.find('<name>', response)
        if candidate is not None:
            target = self.context.resolve_reference(self, candidate.strip())
            if target:
                return target.name
        return None
    # World interaction methods

    def format_hash_percept(self, percept_hash):
        """Format a perceptual hash into a string"""
        percept = ''
        if len(percept_hash) > 1:
            print('surprise')
        percept_hash = percept_hash[0]
        if type(percept_hash) != str:
            print('surprise2')
        for form in percept_hash.split('\n'):
            form = form[1:].strip()+'; '
            percept += form
        percept = percept[:-2]
        return percept

    def process_consequences(self, act_arg, reason, consequences):
        """Determine if the consequences of an action are relevant to the character's goals"""
        prompt=[UserMessage(content="""{{$name}} does {{$act_arg}} and notices 
{{$consequences}}

Are any of these consequences relevant to the character's goals in performing this action as described in 
{{$act_arg}} 
or 
{{$reason}}?
Respond with a concise description of the relevant consequences, or an empty string if none are relevant.
Do not include any other introductory, discursive, formatting, or explanatory text in your response.
Respond only with the relevant consequences, or an empty string if none are relevant.

End response with:
</end>
""")]
        response = self.llm.ask({"name":self.name, "act_arg":act_arg, "reason":reason, "consequences":consequences}, 
                                prompt, tag='do_consequences', temp=0.2, stops=['</end>'], max_tokens=50)
        return response

    def look(self, act_mode=None, act_arg=None, reason=None):
        """Get visual information about surroundings"""
        if self.mapAgent is None:
            return ''  
        obs = self.mapAgent.look()
        view = {}
        for dir in ['Current','North', 'Northeast', 'East', 'Southeast', 
                   'South', 'Southwest', 'West', 'Northwest']:
            dir_obs = map.extract_direction_info(self.context.map, obs, dir)
            view[dir] = dir_obs
        self.my_map[self.mapAgent.x][self.mapAgent.y] = view

        view_text, resources, characters, paths, percept_summary = map.hash_direction_info(view, world=self.context.map)
        view_percept = self.add_perceptual_input(view_text, mode=SensoryMode.VISUAL)
        notices = self.process_consequences(act_arg, reason, view_text)
        if notices:
            self.add_perceptual_input(notices, mode='internal')
 
        visible_actors = []
        for dir in view.keys():
            if 'characters' in view[dir]:
                for character_name in view[dir]['characters']:
                    character, character_name= self.actor_models.resolve_character(character_name['name'])
                    if character:
                        visible_actors.append(character)  
        
        # update actor visibility.
        self.actor_models.set_all_actors_invisible()
        for actor in visible_actors:
            new_actor = False
            if not self.actor_models.known(actor.name):
                new_actor = True
            self.actor_models.get_actor_model(actor.name, create_if_missing=True).visible = True
            if new_actor:
                # first encounter with an actor is a big deal!
                perceptual_input = PerceptualInput(
                    mode=SensoryMode.VISUAL,
                    content=f'You see {actor.name}',
                    timestamp=self.context.simulation_time,
                    intensity=0.9,  # Medium-high for direct observation
                )       
                self.perceptual_state.add_input(perceptual_input)       

        items = self.perceptual_state.record_information_items_from_look(view_text, resources, characters)
        self.look_percept = percept_summary
        return percept_summary

    def format_look(self):
        """Format the agent's current map view"""
        obs = self.my_map[(self.mapAgent.x, self.mapAgent.y)]
        # Fix empty map check
        if obs is None or not obs or all(not v for v in obs.values()):
            return "You see nothing special."
        
        return f"""A look from the current orientation and in each of 8 compass points. 
terrain: environment type perceived.
slope:ground slope in the given direction
resources:a list of resource type detected and distance in the given direction from the current location
others: the other actors visible
water: the water resources visible
{json.dumps(obs, indent=2).replace('\'"{}', '').replace('\'"', '')}
"""

    # Utility methods
    def other(self):
        """Get the other actor in the context"""
        for actor in self.context.actors+self.context.npcs:
            if actor != self:
                return actor
        return None

    def synonym_check(self, term, candidate):
        """Check if two task names are synonymous - simple equality by default"""
        return term == candidate

    # Initialization and persistence
    def initialize(self, ui=None):
        """Called from worldsim once everything is set up"""
        self.ui = ui
        if hasattr(self, 'mapAgent'):
            self.look()
        """Initialize agent state"""
        # Set up initial drives
        if self.drives is None or len(self.drives) == 0:
            self.drives = [
            Drive("find food and water"),
            Drive("stay safe from threats"),
            Drive("explore the environment"),
            Drive("maintain social bonds")
        ]
        
        # Initialize narrative with drives
        self.narrative.active_drives = [d.text for d in self.drives]


    def greet(self):
        """Initial greeting behavior"""
        for actor in self.context.actors:
            if actor == self:
                return
            message = f"Hi, I'm {self.name}"
            actor.show += f'\n{self.name}: {message}'
            return
    
    def is_visible(self, actor):
        """Check if an actor is visible to the current actor"""
        return actor.mapAgent.is_visible(self)

    def see(self):
        """Add initial visual memories of other actors"""
        for actor in self.context.actors+self.context.npcs:
            if actor != self:
                self.add_perceptual_input(f'You see {actor.name}', mode='visual')

    def save(self, filepath):
        """Save character state"""
        allowed_types = (int, float, str, list, dict)
        filtered_data = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, allowed_types):
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    filtered_data[attr] = value
                elif isinstance(value, dict):
                    filtered_data[attr] = {k: v for k, v in value.items() if isinstance(v, allowed_types)}
                else:
                    filtered_data[attr] = value
        with open(filepath, 'w') as fp:
            json.dump(filtered_data, fp, indent=4)

    def load(self, filepath):
        """Load character state"""
        try:
            filename = self.name + '.json'
            with open(filepath / filename, 'r') as jf:
                data = json.load(jf)
                for attr, value in data.items():
                    setattr(self, attr, value)
        except Exception as e:
            print(f' error restoring {self.name}, {str(e)}')

    def set_drives(self, drive_texts: List[str]) -> None:
        """Initialize character drives with embeddings"""
        self.drives = [Drive(text) for text in drive_texts]

    def get_drive_texts(self) -> List[str]:
        """Get drive texts for backward compatibility"""
        return [drive.text for drive in self.drives]



    def set_llm(self, llm):
        self.llm = llm
        if self.memory_consolidator is not None:
            self.memory_consolidator.set_llm(llm)
        if self.driveSignalManager is not None:
            self.driveSignalManager.set_llm(llm)

    def save(self, filepath):
        allowed_types = (int, float, str, list, dict)  # Allowed types for serialization
        filtered_data = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, allowed_types):
                # Handle lists of strings specifically
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    filtered_data[attr] = value
                elif isinstance(value, dict):
                    filtered_data[attr] = {k: v for k, v in value.items() if isinstance(v, allowed_types)}
                else:
                    filtered_data[attr] = value
        with open(filepath, 'w') as fp:
            json.dump(filtered_data, fp, indent=4)

    def load(self, filepath):
        try:
            filename = self.name + '.json'
            with open(filepath / filename, 'r') as jf:
                data = json.load(jf)
                for attr, value in data.items():
                    setattr(self, attr, value)
        except Exception as e:
            print(f' error restoring {self.name}, {str(e)}')

    def generate_image_description(self):
        """Generate a description of the character for image generation
            character_description, current activity, emotional state, and environment"""

        try:
            context = ''
            i = 0
            candidates = self.context.current_state.split('.')
            while len(context) < 84 and i < len(candidates):
                context += candidates[i]+'. '
                i +=1
            context = context[:96]
            description = 'Portrait of a '+self.gender+': '+'. '.join(self.character.split('.')[:2])+'. '# assumes character description starts with 'You are <name>'
            
            prompt = [UserMessage(content="""Following is a description of a character in a play. 

<description>
{{$description}}
</description>
            
<recent_memories>
{{$recent_memories}}
</recent_memories>

Extract from this description two or three words that each describe an aspect of the character's emotional state.
Use common adjectives like happy, sad, frightened, worried, angry, curious, aroused, cold, hungry, tired, disoriented, etc.
The words should each describe a different aspect of the character's emotional state, and should be distinct from each other.

Respond using this format, without any additional text:

adjective, adjective, adjective
</end>
""")]

            concerns = self.focus_task.peek().to_string() if self.focus_task.peek() else ''
            state = description + '.\n '+concerns +'\n'+ context
            recent_memories = self.structured_memory.get_recent(8)
            recent_memories = '\n'.join(memory.to_string() for memory in recent_memories)
            #print("Char generate image description", end=' ')
            response = self.llm.ask({ "description": state, "recent_memories": recent_memories}, prompt, tag='image_description', temp=0.2, stops=['</end>'], max_tokens=10)
            if response:
                description = description[:192-min(len(context), 48)] + f'. {self.name} feels '+response.strip()+'. Background: '+context
            else:
                description = description[:192-min(len(context), 48)] + ', '+context

        except Exception as e:
            traceback.print_exc()
        return description

    def update_actor_image(self):
        try:
            description = self.generate_image_description()
            prompt = description
            #print(f' actor image prompt len {len(prompt)}')
            image_path = llm_api.generate_image(self.llm, prompt, size='192x192', filepath=self.name + '.png')
        except Exception as e:
            traceback.print_exc()
        return image_path

    def update_world_image(self):
        raise NotImplementedError("Derived classes must implement update_world_image")
    
    def add_perceptual_input(self, message: str, percept=True, mode=None):
        """Add a perceptual input to the agent's perceptual state"""
        content = message
        if mode is None or mode not in perceptualState.SensoryMode:
            prompt = [UserMessage(content="""Determine its sensory mode of the following message,
a terse description of the perceptual content,
and the emotionalintensity of the perceptual content.

sense mode may be:
auditory
visual
olfactory
tactile
movement
internal
unclassified

<message>
{{$message}}
</message>
Respond using this format:

mode
</end>

Respond only with a single choice of mode. Do not include any introductory, discursive, or explanatory text.
""")]
            #print("Perceptual input",end=' ')
            response = self.llm.ask({"message": message}, prompt, tag='perceptual_input', temp=0, stops=['</end>'], max_tokens=150)
            if response and type(response) == str:
                mode = response.strip().split()[0].strip()  # Split on any whitespace and take first word
            else:
                logger.error(f'Error parsing perceptual input: {response}')
                mode = 'unclassified'

        try:
            mode = perceptualState.SensoryMode(mode)
        except:
            # unknown mode, skip
            mode=perceptualState.SensoryMode.UNCLASSIFIED
        perceived_content = message
        intensity = 0.6 # assume low intensity
        perceptual_input = PerceptualInput(
            mode=mode,
            content=perceived_content if percept else message,
            timestamp=self.context.simulation_time if self.context else datetime.now()+timedelta(seconds=10),
            intensity=intensity
        )
        self.perceptual_state.add_input(perceptual_input)
        if self.context:
            input_time = self.context.simulation_time
        else:
            input_time = datetime.now()+timedelta(seconds=10)
        self.driveSignalManager.analyze_text(message, self.drives, input_time, mode=mode)
        return perceptual_input

    def add_to_history(self, message: str):
        """Add message to structured memory"""
        if message is None or message == '':
            return
        entry = MemoryEntry(
            text=message,
            importance=0.5,  # Default importance
            timestamp=self.context.simulation_time if self.context else datetime.now()+timedelta(seconds=10),
            confidence=1.0
        )
        self.structured_memory.add_entry(entry)
        self.new_memory_cnt += 1
        #self.add_perceptual_input(message)
       
    def _find_related_drives(self, message: str) -> List[Drive]:
        """Find drives related to a memory message"""
        related_drives = []
        
        # Create temporary memory entry to compare against drives
        temp_memory = MemoryEntry(
            text=message,
            timestamp=self.context.simulation_time,
            importance=0.5,
            confidence=1.0
        )
        temp_memory.embedding = self.memory_retrieval.embedding_model.encode(message)
        
        # Check each drive for relevance
        for drive in self.drives:
            similarity = self.memory_retrieval._compute_similarity(
                temp_memory.embedding,
                drive.embedding if drive.embedding is not None 
                else self.memory_retrieval.embedding_model.encode(drive.text)
            )
            if similarity >= 0.1:  # Use same threshold as retrieval
                related_drives.append(drive)
                
        return related_drives

    def format_history_for_UI(self, n=2):
        """Get n most recent memories"""
        recent_memories = self.structured_memory.get_recent(n)
        return ' \n '.join(memory.to_string() for memory in recent_memories)
    
    def map_goals(self, goal=None):
        """ map goals for llm input """
        header = """A goal is a terse description of a need or desire the character has.  
Each goal has an urgency and a trigger.  A goal is satisfied when the termination condition is met.
Here are this character's goals:
"""
        mapped = [header]
        if goal:
            mapped.append(goal.to_string())
        else:
            for goal in self.goals:
                mapped.append(goal.to_string())
        return '\n'.join(mapped)


    def make_task_name(self, reason):
        instruction=[UserMessage(content="""Generate a concise, 2-5 word task name from the motivation to act provided below.
Respond using this XML format:

<name>task-name</name>

<motivation>
{{$reason}}
</motivation>

Respond only with your task-name using the above XML
Do not include your reasoning in your response.
Do not include any introductory, discursive, or explanatory text.
End your response with:
</end>
""")]
        response = self.llm.ask({"reason":reason}, instruction, tag='task_name', temp=0.3, stops=['</end>'], max_tokens=12)
        return xml.find('<name>', response)
                    

    def repetitive(self, new_response, last_response, source):
        """Check if response is repetitive considering wider context"""
        # Get more historical context from structured memory
        recent_memories = self.structured_memory.get_recent(8)  # Increased window
        history = '\n'.join(mem.to_string() for mem in recent_memories)
        
        prompt = [UserMessage(content="""Given recent history and a new proposed action, 
determine if the new action is pointlessly repetitive and unrealistic. 

An action is considered repetitive ONLY if it:
1. Repeats the exact same verbal statement or physical action that was already performed
2. Adds no new information or progression to the interaction

Important distinctions:
- Converting internal thoughts into verbal communication is NOT repetitive
- Similar ideas expressed in different ways are NOT repetitive
- Questions or statements that naturally follow from previous context are NOT repetitive

Recent history:
<history>
{{$history}}
</history>

New action:
<action>
{{$response}}
</action>

Consider:
- Does this action repeat an already performed action word-for-word?
- Does this action naturally progress the interaction?
- Does this add something new to the scene or conversation?
- Is this a reasonable next step given the context?

Respond with only 'True' if repetitive or 'False' if the response adds something new.
End response with:
</end>""")]

        #print("Repetitive")
        result = self.llm.ask({
            'history': history,
            'response': new_response
        }, prompt, tag='repetitive', temp=0.2, stops=['</end>'], max_tokens=100)

        if 'true' in result.lower():
            return True
        else:
            return False

    def update_drives(self, goal: Goal):
        """Update drives based on goal satisfaction"""
        if hasattr(goal, 'drives'):
            for drive in cast(List[Drive], goal.drives):
                drive.satisfied_goals.append(goal)
                # don't update the primary drive
                if drive is not self.drives[0]:
                    if len(drive.attempted_goals) > 0:
                        if drive in self.drives: # only update if drive still exists, may have already been rewritten
                            drive.activation *= 0.9
                            update = drive.update_on_goal_completion(self, goal, 'goal completed')
                            if update:
                                try:
                                    self.drives.remove(drive)
                                    # remove from any signalClusters too!
                                except:
                                    pass
                                self.drives.append(update)

    async def clear_task_if_satisfied(self, task: Task, consequences='', world_updates=''):
        """Check if task is complete and update state"""
        termination_check = task.termination if task != None else None
        if termination_check is None or termination_check == '':
            return True

        # Test completion through cognitive processor's state system
        satisfied, progress = await self.test_termination(
            task,
            termination_check, 
            consequences,
            world_updates, 
            type='task'
        )

        weighted_acts = len(task.acts) + 2*len([act for act in task.acts if act.mode == Mode.Say])
        if satisfied or weighted_acts > random.randint(4,6): # basic timeout on task
            satisfied = True # force completion
            if task == self.focus_task.peek():
                self.focus_task.pop()
                self.focus_action = None
            if self.focus_goal and task in self.focus_goal.task_plan:
                self.focus_goal.task_plan.remove(task)
            if self.focus_goal:
                pass #self.focus_goal.tasks_completed.append(task)
            self.context.current_state += f"\nFollowing task has been satisfied. This may invalidate parts of the above:\n  {task.to_string()}"
            self.add_perceptual_input(f"Following task has been satisfied: {task.short_string()}", mode='internal')
            self.achievments.append(task.termination)
            #await self.context.update(local_only=True) # remove confusing obsolete data, task completion is a big event
 
        return satisfied

    async def clear_goal_if_satisfied(self, goal: Goal, consequences='', world_updates=''):
        if not goal:
            return False
        """Check if goal is complete and update state"""
        termination_check = goal.termination if goal != None else None
        if termination_check is None or termination_check == '':
            if goal.name == 'preconditions' or goal.name == 'postconditions':
                return True
            termination_check = goal.description+', at least partially satisfied'

        # Test completion through cognitive processor's state system
        satisfied, progress = await self.test_termination(
            goal,
            termination_check, 
            consequences,
            world_updates, 
            type='goal'
        )

        if satisfied:
            self.goal_history.append(goal)
            if goal.name != 'preconditions':
                # remove preconditions goal if it is satisfied
                for goal in self.goals.copy():
                    if goal.name == 'preconditions' and goal.description == self.focus_goal.preconditions:
                        self.goals.remove(goal)
            self.context.current_state += f"\nFollowing goal has been satisfied. This may invalidate parts of the above:\n{goal.to_string()}"
            #self.add_perceptual_input(f"Following goal has been satisfied. This may invalidate parts of the above:\n{goal.short_string()}", mode='internal')
            self.achievments.append(goal.termination)
            if goal in self.goals:
                self.goals.remove(goal)           
            self.update()

            await self.context.update(local_only=True) # remove confusing obsolete data, goal completion is a big event
            self.update_drives(goal)
            if self.focus_goal is goal:
                self.focus_goal_history.append(goal)
                self.focus_goal = None
        return satisfied


    def move_toward(self, target_string):
        """Move toward the target string"""
        if map.Direction.from_string(target_string.strip()):
            self.mapAgent.move(target_string.strip())
            return True
        resource = self.context.map.get_resource_by_id(target_string.strip())
        if resource:
            self.mapAgent.move_toward_location(resource['location'][0], resource['location'][1])
            return True
        else:
            actor, _ = self.actor_models.resolve_character(target_string)
            if actor:
                if type(actor.x) == int:
                    raise Exception(f'{self.name} move_toward: actor {actor.name} x is int, not method')
                self.mapAgent.move_toward_location(actor.x(), actor.y())
                return True
            else:
                return self.mapAgent.move(target_string)

    async def acts(self, act:Act, target: Character, act_mode: str, act_arg: str='', reason: str='', source: Task=None):
        """Execute an action and record results"""
        # Create action record with state before action
        try:
            mode = Mode(act_mode.capitalize())
        except ValueError:
            raise ValueError(f'Invalid action name: {act_mode}')
        self.act_result = ''
    
        # Store current state and reason
        self.reason = reason
        self.show = ''
        if act_mode is None or act_arg is None or len(act_mode) <= 0 or len(act_arg) <= 0:
            return


        # Update thought display
        self.thought +=  self.reason
        self.lastActResult = ''
            
        if act_mode == 'Move':
            try:
                act_arg = act_arg.strip()
                if act_arg.startswith('toward'):
                    act_arg = act_arg[len('toward'):]
                    location = None
                    if act_arg.startswith('s '): # just in case towards instead of toward
                        act_arg = act_arg[len('s '):]
                moved = self.move_toward(act_arg)
                if moved:
                    percept = self.look(act_mode, act_arg,reason)
                    self.show += ' moves to ' + act_arg + '.\n'#  and notices ' + percept
                    self.context.message_queue.put({'name':self.name, 'text':self.show})
                    self.context.transcript.append(f'{self.name}: {self.show}')
                    self.show = '' # has been added to message queue!
                    await asyncio.sleep(0)
                    self.show = '' # has been added to message queue!
                else: 
                    resource, canonical_name = self.resourceRefManager.resolve_reference_with_llm(act_arg)
                    if resource:
                        new_x = resource['location'][0]
                        new_y = resource['location'][1]
                        self.mapAgent.x = new_x
                        self.mapAgent.y = new_y
                        percept = self.look(act_mode, act_arg,reason)
                        self.show += f' moves to {resource["name"]}.\n  and notices ' + percept
                        self.context.message_queue.put({'name':self.name, 'text':self.show})
                        self.context.transcript.append(f'{self.name}: {self.show}')
                        self.show = '' # has been added to message queue!
                        await asyncio.sleep(0)
                    else:
                        act_mode = 'Do'
                        act_arg = 'move to ' + act_arg
            except Exception as e:
                traceback.print_exc()
                print(f'{self.name} move_toward failure: {e}')
 
        # Handle world interaction
        if act_mode == 'Do':
            # Get action consequences from world
            await asyncio.sleep(0)
            consequences, world_updates, character_updates = await self.context.do(self, act)
            if character_updates and len(character_updates) > 0:
                self.add_perceptual_input(f'{character_updates}', mode='internal')  
            notices = self.process_consequences(act_arg, reason, consequences)
            await asyncio.sleep(0)
            if notices and len(notices) > 0:
                self.add_perceptual_input(f'{notices}', mode='internal')           
                self.context.message_queue.put({'name':self.name, 'text':f'does {act_arg} {notices}'})
            else:
                self.context.message_queue.put({'name':self.name, 'text':f'does {act_arg}'})
            self.lastActResult = notices
            if source == None:
                source = self.focus_task.peek()
            task = source

            # Update displays

            if len(consequences.strip()) > 2:
                #self.show += 'Resulting in ' + consequences.strip()
                #self.context.message_queue.put({'name':self.name, 'text':self.show})
                self.context.transcript.append(f'{self.name}: {self.show}')
                await asyncio.sleep(0)
            self.show = '' # has been added to message queue!
            if len(world_updates) > 0:
                self.add_perceptual_input(f"{world_updates}", mode='visual')
            self.act_result = world_updates +'. '+character_updates
        
            # Update target's sensory input
            if target is not None:
                target.sense_input += '\n' + world_updates

        elif act_mode == 'Look':
            percept = self.look(act_mode, act_arg=act_arg,reason=reason)
            consequences = await self.context.look(self, act)
            if  consequences and len(consequences) > 0:
                self.add_perceptual_input(f'{consequences}', mode='internal')  
            self.lastActResult = consequences
            self.show += act_arg + '.\n ' + consequences
            self.context.message_queue.put({'name':self.name, 'text':self.show})
            await asyncio.sleep(0)

        elif act_mode == 'Think': # Say is handled below
            self.thought = act_arg
            self.show += f" \n...{self.thought}..."
            #self.add_perceptual_input(f"\nYou {act_mode}: {act_arg}", percept=False, mode='internal')
            style_block, elevenlabs_params = self.speech_stylizer.get_style_directive(self)
            self.context.message_queue.put({'name':self.name, 'text':f"...{act_arg}...", 'elevenlabs_params': json.dumps(elevenlabs_params)})
            self.context.transcript.append(f'{self.name}: ...{act_arg}...')
            await asyncio.sleep(0)

            if self.focus_task.peek() and not self.focus_task.peek().name.startswith('internal dialog with '+self.name): # no nested inner dialogs for now
                    # initiating an internal dialog
                dialog_task = Task(name = 'internal dialog with '+self.name, 
                                    description=self.name + ' thinks ' + act_arg, 
                                    reason=reason, 
                                    termination='natural end of internal dialog', 
                                    goal=None,
                                    start_time=self.context.simulation_time,
                                    duration=0,
                                    actors=[self])
                self.focus_task.push(dialog_task)
                self.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.activate()
            await self.think(act_arg, source)
            await asyncio.sleep(0)

        elif act_mode == 'Say':# must be a say
            self.show += f"{act_arg}'"
            style_block, elevenlabs_params = self.speech_stylizer.get_style_directive(target)
            act_arg = self.speech_stylizer.stylize(act_arg, target, style_block)
            #print(f"Queueing message for {self.name}: {act_arg}")  # Debug
            self.context.message_queue.put({'name':self.name, 'text':f"'{act_arg}'", 'elevenlabs_params': json.dumps(elevenlabs_params)})
            self.context.transcript.append(f'{self.name}: "{act_arg}"')
            await asyncio.sleep(0)
            content = re.sub(r'\.\.\..*?\.\.\.', '', act_arg)
            if not target and act.target and isinstance(act.target, list):
                target=act.target[0]
            elif target and isinstance(target, list):
                for candidate in target:
                    if candidate and isinstance(candidate, Character) and candidate != self:
                        target = candidate
                        break
                if target and isinstance(target, list) and len(target) > 0:
                    target = target[0]
            elif not target and act.target and isinstance(act.target, Character):
                target = act.target
            # open dialog
            if target:
                if not self.actor_models.get_actor_model(target.name, create_if_missing=True).dialog.active:
                    # start new dialog
                    dialog_task = Task('dialog with '+target.name, 
                                    description=self.name + ' talks with ' + target.name, 
                                    reason=act_arg+'; '+reason, 
                                    termination='natural end of dialog', 
                                    goal=None,
                                    start_time=self.context.simulation_time,
                                    duration=0,
                                    actors=[self, target])
                    self.focus_task.push(dialog_task)
                    self.actor_models.get_actor_model(target.name, create_if_missing=True).dialog.activate()
                    # new dialog, create a new dialog task, but only if we don't already have a dialog task, no nested dialogs for now
                    if target.focus_task.peek() and target.focus_task.peek().name.startswith('dialog with '+self.name):
                        target.focus_task.pop()
                    # create a new dialog task
                    target_ranked_signalClusters = target.driveSignalManager.get_scored_clusters()
                    target_focus_signalClusters = [rc[0] for rc in target_ranked_signalClusters[:3]] # first 3 in score order
                    target.emotionalStance = EmotionalStance.from_signalClusters(target_focus_signalClusters, target)
                    target_dialog_task = Task('dialog with '+self.name, 
                                    description='dialog with '+self.name, 
                                    reason=act_arg+'; '+reason, 
                                    termination='natural end of dialog', 
                                    goal=None,
                                    start_time=self.context.simulation_time,
                                    duration=2,
                                    actors=[target, self])
                    target.focus_task.push(target_dialog_task)
                    target.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.activate(source)

                # self is speaker, don't get confused about first arg to add_turn
                self.actor_models.get_actor_model(target.name).dialog.add_turn(self, content)
                target.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.add_turn(self, content)
                await target.hear(self, act_arg, source)

         # After action completes, update record with results
        # Notify other actors of action
        if act_mode != 'Say' and act_mode != 'Look' and act_mode != 'Think':  # everyone you do or move or look if they are visible
            for actor in self.context.actors+self.context.extras:
                if actor != self:
                    if actor != target:
                        actor_model = self.actor_models.get_actor_model(actor.name)
                        if actor_model != None and actor_model.visible:
                            percept = actor.add_perceptual_input(f"You see {self.name}: '{act_arg}'", percept=False, mode='visual')
                            actor.actor_models.get_actor_model(self.name, create_if_missing=True).infer_goal(percept)
        self.previous_action_mode = act_mode           

    def instantiate_narrative_goal(self, goal_statement, generate_conditions=True):
        """instantiate a goal from a narrative"""
        self.driveSignalManager.check_drive_signals()
        ranked_signalClusters = self.driveSignalManager.get_scored_clusters()
        #focus_signalClusters = choice.pick_weighted(ranked_signalClusters, weight=4.5, n=5) if len(ranked_signalClusters) > 0 else []
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order
        signal_memories = []
        for sc in focus_signalClusters:
            sc.emotional_stance = EmotionalStance.from_signalCluster(sc, self)
            perceptual_memories = self.perceptual_state.get_information_items(sc.text, threshold=0.01, max_results=3)
            signal_memories.extend(perceptual_memories)

        prompt = [UserMessage(content="""You are an actor playing the character {{$name}}.
Instantiate a goal object for the following directive. Depending on the situation and your character, you may choose to be direct, indirect, or even devious in your approach.

#Character
{{$character}}
##
                                                        
#Directive
{{$goal_statement}}
##

#Recent events
{{$recent_events}}
##
                              
#Recently achieved goals
{{$goal_history}}
##

Additional information about the your character to support developing your response
Drives are the character's basic motivations. Activation indicates how strongle the drive is active at present in the character's consciousness.

#Drives
{{$drives}}
##

Character's relationships with other actors

#Relationships
{{$relationships}}
##
                              
#Recent memories
{{$recent_memories}}
##


Following are a few signalClusters ranked by impact. These are issues or opportunities nagging at the periphery of the character's consciousness.
These clusters may overlap.
                              
#SignalClusters
{{$signalClusters}}
##

Following are memories relevant to the signalClusters above. Consider them in your goal generation:

#Signal memories
{{$signal_memories}}
##
Following are a few signalClusters ranked by impact. These are issues or opportunities nagging at the periphery of the character's consciousness.
These clusters may overlap.
                              
#SignalClusters
{{$signalClusters}}
##

Following are memories relevant to the signalClusters above. Consider them in your goal generation:

#Signal memories
{{$signal_memories}}
##

Following are the central dramatic question of the play, the act-specific narrative, and the scene your character is in.

#Central narrative
{{$central_narrative}}
##
                              
#Act specific narrative
{{$act_specific_narrative}}
##
                              
#Scene
{{$scene}}
##
                              
#Character's situation

#Situation
{{$situation}}
##

#Surroundings
{{$surroundings}}
##


Consider:
1 The goal directive is the play director's overall instruction to the character for this scene.
2. The character is about to perform a scene in an improvisational play, the players have agreed to the central narrative and act specific narrative given above. 
2. Any patterns or trends in the past goals, tasks, or actions and their outcomes should be considered in generating the specific goal for this character in this scene.
3. Identify any other actors involved in the goal, and their relationships to the character, insofar as the goal must be consistent with them. 
    Consistent, in this context, means the generated goal should advance the dramatic narrative, but does not limit the deception or surprise the goal may have for the other actors.
    Your goal will not be revealed to the other actors, but they may be able to infer it from your actions.


Respond with the instantiated goal consistent with the goal directive, in the following parts: 
    goal - a terse (5-8 words) name for the goal, 
    description - concise (8-14 words) further details of the goal, intended to guide task generation, 
    otherActorName - name of the other actor involved in this goal, or None if no other actor is involved, 
    signalCluster_id - the signalCluster id ('scn..') of the signalCluster that is most associated with this goal
    preconditions - a statement of conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any
    termination  - a condition (5-6 words) that would mark satisficing achievement of the goal.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Each goal should begin with a #goal tag, and should end with ## on a separate line as shown below:
be careful to insert line breaks only where shown, separating a value from the next tag:

#goal terse (5-8) words) name for this goal
#description concise (8-14) words) further details of this goal
#otherActorName name of the other actor involved in this goal, or None if no other actor is involved
#signalCluster_id id ('scn..') of the signalCluster that is the primary source for this goal  
#preconditions (3-4 words) statement of conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any
#termination terse (5-6 words) statement of condition that would mark achievement or partial achievement of this goal
##


Respond ONLY with the above hash-formatted text.
End response with:
</end>
""")]

        response = self.llm.ask({"name":self.name,
                                 "character": self.get_character_description(),
                                 "goal_statement":goal_statement,
                                 "central_narrative": self.context.central_narrative if self.context.central_narrative else '',
                                 "act_specific_narrative": self.context.act_central_narrative if self.context.act_central_narrative else '',
                                 "scene": json.dumps(self.context.current_scene, indent=2, default=datetime_handler) if self.context.current_scene else '',
                                 "signalClusters": "\n".join([sc.to_full_string() for sc in focus_signalClusters]),
                                 "drives": "\n".join([f'{d.id}: {d.text}; activation: {d.activation:.2f}' for d in self.drives]),
                                 "situation": self.context.current_state if self.context else "",
                                 "surroundings": self.look_percept,
                                 "character": self.get_character_description(),
                                 "recent_events": self.narrative.get_summary('medium'),
                                 "goal_history":'\n'.join([f'{g.name} - {g.description}\n\t{g.completion_statement}' for g in self.goal_history]) if self.goal_history else 'None to report',
                                 "relationships": self.actor_models.format_relationships(include_transcript=True),
                                 "recent_memories": "\n".join([m.to_string() for m in self.structured_memory.get_recent(16)]),
                                 "signal_memories": "\n".join([m.content for m in signal_memories]),
                                 #"drive_memories": "\n".join([m.text for m in self.memory_retrieval.get_by_drive(self.structured_memory, self.drives, threshold=0.1, max_results=5)])
                                 }, prompt, tag='instantiate_narrative_goal', temp=0.3, stops=['</end>'], max_tokens=240)
        goals = []
        forms = hash_utils.findall_forms(response)
        for goal_hash in forms:
            goal = self.validate_and_create_goal(goal_hash)
            if goal:
                if not generate_conditions:
                    goal.description = goal.name+': '+goal.description
                    goal.name = 'postconditions'
                    goal.preconditions = None
                    goal.termination = None
                print(f'{self.name} generated goal: {goal.to_string()}')
                goals.append(goal)
            else:
                print(f'Warning: Invalid goal generation response for {goal_hash}')

        if len(goals) > 0:
            self.focus_goal = goals[0]
            self.goals = goals
        if len(goals) > 1:
            print(f'{self.name} generated {len(goals)} goals for scene!')
        return goals


    async def generate_goals(self, previous_goal:Goal=None):
        """generate alternative goals to work on"""
        self.look()
        system_prompt = """You are an actor with deep insight into human psychology and behavior. You are acting in an improvisational play, playing the character {{$name}}.
You have just completed the following goal:
#previous goal
{{$previous_goal}}
##
Your task is to identify the highest priority goal alternatives the character should focus on next given the following information.

#goal related memories
{{$goal_related_memories}}
##

"""
        mission = """
Consider:
1. What is the central issue / opportunity / obligation demanding the character's attention?
2. Given the following available information about the character, the situation, and the surroundings, how can the character best advance the central dramatic question?
3. Identify any other actors involved in the goal, and their relationships to the character.
4. Try to identify a goal that might be the center of an overall story arc of a play or movie.
5. Goals must be distinct from one another.
6. Goals must be consistent with the character's drives and emotional stance.

"""
        suffix = """
Respond with up to three highest priority, most encompassing, next goal alternatives, in the following parts: 
    goal - a terse (5-8 words) name for the goal, 
    description - concise (8-14 words) further details of the goal, intended to guide task generation, 
    otherActorName - name of the other actor involved in this goal, or None if no other actor is involved, 
    signalCluster_id - the signalCluster id ('scn..') of the signalCluster that is most associated with this goal
    preconditions - a statement of situational conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any. 
        These should be necessary pre-conditions for the expected initial tasks of this goal.

Be sure to include a goal that responds to the character's primary drive in the context of the central narrative, the drive signals, and the information above.
Do NOT duplicate or significantly overlap the completed goal or generate a goal for objectives duplicating the completion statement of the completed goal.
Other goals must not duplicate or significantly overlap this goal or the completed goal. 
Nothing in this or other instructions limits your use of deception or surprise.

#primary drive
{{$primary_drive}}
##

#drive signals
{{$primary_drive_signals}}
##
                              
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Each goal should begin with a #goal tag, and should end with ## on a separate line as shown below:
be careful to insert line breaks only where shown, separating a value from the next tag:

#goal terse (5-8) words) name for this goal
#description concise (8-14) words) further details of this goal
#otherActorName name of the other actor involved in this goal, or None if no other actor is involved
#signalCluster_id id ('scn..') of the signalCluster that is the primary source for this goal  
#preconditions (3-4 words) statement of conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any
#termination terse (5-6 words) statement of condition that would mark achievement or partial achievement of this goal
##


Respond ONLY with the above hash-formatted text.
End response with:
</end>
"""


        goal_memories = []
        ranked_signalClusters = self.driveSignalManager.get_scored_clusters()
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:5]] # first 3 in score order
        for sc in focus_signalClusters:
            perceptual_memories = self.perceptual_state.get_information_items(sc.text, threshold=0.01, max_results=3)
            for pm in perceptual_memories:
                if pm not in goal_memories:
                    goal_memories.append(pm)
        for g in self.goals:
            memories = self.perceptual_state.get_information_items(g.short_string(), threshold=0.01, max_results=3)
            for gm in memories:
                if gm not in goal_memories:
                    goal_memories.append(gm)

        if self.context.scene_post_narrative:
            scene_narrative = f"\nThe narrative arc of the scene is from:  {self.context.scene_pre_narrative} to  {self.context.scene_post_narrative}\nThe task sequence should be consistent with this theme."
        else:
            scene_narrative = ''
        response = default_ask(self, 
                               system_prompt=system_prompt,
                               prefix=mission, 
                               suffix=suffix, 
                               addl_bindings={
                                   "primary_drive": f'{self.drives[0].id}: {self.drives[0].text}; activation: {self.drives[0].activation:.2f}',
                                   "primary_drive_signals": "\n".join([f'{sc.id}: {sc.text}' for sc in self.driveSignalManager.get_signals_for_drive(self.drives[0])]),
                                   "my_narrative": self.central_narrative if self.central_narrative else '',
                                   "previous_goal": previous_goal.to_string()+f'\n\n completion statement: {previous_goal.completion_statement}' if previous_goal else '',
                                   "goal_related_memories": "\n".join([m.content for m in goal_memories])},
                               tag = 'Character.generate_goals',
                               max_tokens=350, log=True)


        goals = []
        forms = hash_utils.findall_forms(response)
        for goal_hash in forms:
            goal = self.validate_and_create_goal(goal_hash)
            if goal:
                print(f'{self.name} generated goal: {goal.to_string()}')
                goals.append(goal)
            else:
                print(f'Warning: Invalid goal generation response for {goal_hash}')
        if len(goals) < 2:
            print(f'Warning: Only {len(goals)} goals generated for {self.name}')
        self.goals = goals
        self.driveSignalManager.clear_clusters(goals)
        print(f'{self.name} generated {len(goals)} goals')
        return goals

    async def generate_task_plan(self, goal:Goal=None, new_task:Task=None, replan=False, ntasks=None):
        if not self.focus_goal:
            raise ValueError(f'No focus goal for {self.name}')
        """generate task alternatives to achieve a focus goal"""
        if not goal:
            goal = self.focus_goal
        if not goal:
            return []
        if replan:
            if new_task and (goal.task_plan and len(goal.task_plan) == 0):
                goal.task_plan = [new_task]
                new_task.goal = goal
                return [new_task]
        
        suffix = """

Create at most {{$ntasks}} specific, actionable task(s), individually distinct and collectively exhaustive for achieving the focus goal.
The tasks must be designed to achieve the focus goal in light of the central narrative, act specific narrative, and scene given below.

##Dramatic Context
<central_narrative>
{{$central_narrative}}
</central_narrative>

<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

In addition, you have your own plans that may or may not be consistent with the central dramatic question. 
Potential conflict between your own plans and the central dramatic question is real, only you can decide. You may choose deception or surprise to advance your own plans.

<my_narrative>
{{$my_narrative}}
</my_narrative>

Scene you will be acting in:
<scene>
{{$scene}}
</scene>

In light of the narrative context above, your tasks should serve story momentum as well as character psychology.

Consider how your chosen task_plan:
- Advances the central narrative question toward resolution or deepens the mystery
- Escalates or defuses the current act's dramatic tension, for example by forcing a choice between incompatible drives or goals
- Creates new conflict, responds to existing conflict, or avoids confrontation, for example by creating a diversion or misdirection
- Reveals, conceals, or misdirects crucial information
- Forces decisions from others or demonstrates your own choices
- Affects your alliances and relationships in service of the story

Choose tasks that drive narrative forward, not just maintain status quo.

############
#While you are generating tasks, not specific actions in support of these tasks, you should be aware of the following thoughts the actor animating this character has had about physical action:

{{$do_actions}}

############

The tasks should be at a granularity such that they collectively cover all the steps necessary to achieve the focus goal.
Each task should be an important component of the overall narrative arc of the scene achieving the focus goal. Do not include tasks merely to fill time or number of tasks.
Each task should contain the potential for a narrative beat: learn something new, relationship shift, powerful emotion, change in power dynamics, etc.
Where appropriate, drawn from typical life scripts.
Also, the collective duration of the tasks should be less than any duration or completion time required for the focus goal.
                              
A task is a specific objective that can be achieved in the current situation and which is a major step in satisfying the focus goal.
The new task(s) should be distinct from one another, and each should advance the focus goal.
Use known actor names in the task description unless deliberately introducing a new actor. Characters known to this character include:
{{$known_actors}}

Start with an appropriate task in light of previous achievments and the dramatic context. 

{{$achievments}}


Make explicit reference to diverse known or observable resources where logically fitting, ensuring broader environmental engagement across tasks.

A task has a name, description, reason, list of actors, start time, duration, and a termination criterion as shown below.
Respond using the following hash-formatted text, where each task tag (field-name) is preceded by a # and followed by a single space, followed by its content.
Each task should begin with a #task tag, and should end with ## as shown below. Insert a single blank line between each task.
be careful to insert line breaks only where shown, separating a value from the next tag:

#name brief (4-6 words) task name
#description terse (6-8 words) statement of the action to be taken.
#reason (6-7 words) on why this action is important now
#actors the names of any other actors involved in this task, comma separated. if no other actors, use None
#start_time (2-3 words) expected start time of the action
#duration (2-3 words) expected duration of the action in minutes
#termination (5-7 words) condition test to validate goal completion, specific and objectively verifiable.
##


In refering to other actors. always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', etc.
Respond ONLY with the tasks in hash-formatted-text format and each ending with ## as shown above.
Order tasks in the assumed order of execution.

Remember, generate at most {{$ntasks}} tasks.

###Scene
{{$scene_narrative}}
"""
        mission = """You are an actor in an improvisational play, playing the character {{$name}}.
You are tasked with creating a sequence of at most {{$ntasks}} tasks to achieve the focus goal: 
###Focus Goal
{{$focus_goal}}
"""
        if goal and goal.task_plan and len(goal.task_plan) > 0:
            mission += """\n you already have a task_plan for this goal, which should be used as a basis for your new plan, but may need to be revised due to new information.

###Existing plan
{{$task_plan}}
"""
        if new_task:
            mission += """\nYou have been given this new task which should be integrated early into your plan. 
Do not simply prepend this task, but rather integrate it into your plan in a way that is consistent with the focus goal.
Do not introduce substantially new tasks, but rather integrate the new task into your existing plan.
Place tasks least important to the dramatic context early in the plan when possible.
            
###New Task
{{$new_task}}

"""

        ranked_signalClusters = self.driveSignalManager.get_scored_clusters()
        #focus_signalClusters = choice.pick_weighted(ranked_signalClusters, weight=4.5, n=5) if len(ranked_signalClusters) > 0 else []
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order
        goal_memories = []
        for sc in focus_signalClusters:
            perceptual_memories = self.perceptual_state.get_information_items(sc.text, threshold=0.01, max_results=3)
            for pm in perceptual_memories:
                if pm not in goal_memories:
                    goal_memories.append(pm)
        for g in self.goals:
            memories = self.perceptual_state.get_information_items(g.short_string(), threshold=0.01, max_results=3)
            for gm in memories:
                if gm not in goal_memories:
                    goal_memories.append(gm)

        if not goal:
            goal = self.focus_goal
        if not goal:
            raise ValueError(f'No focus goal for {self.name}')
        if not ntasks:
            ntasks = 2
            if goal.name == 'preconditions' or goal.name == 'postconditions':
                ntasks = 1
            elif self.__class__.__name__ == 'NarrativeCharacter' and self.current_scene:
                ntasks = self.context.compute_task_plan_limits(self,self.current_scene)
            elif self.__class__.__name__ == 'NarrativeCharacter':
                logger.info(f'{self.name} generate_task_plan: no current scene, using default ntasks: {ntasks}')
            elif self.context.current_scene:
                ntasks = self.context.compute_task_plan_limits(self, self.context.current_scene)
            if goal.task_plan and replan:
                ntasks = min(ntasks, max(1, len(goal.task_plan)))
        if self.context.scene_post_narrative:
            scene_narrative = f"\nThe narrative arc of the scene is from:  {self.context.scene_pre_narrative} to  {self.context.scene_post_narrative}\nThe task sequence should be consistent with this theme."
        else:
            scene_narrative = ''
        logger.info(f'{self.name} generate_task_plan: {goal.to_string()}')
        response = default_ask(self, 
                               prefix=mission, 
                               suffix=suffix, 
                               addl_bindings={"focus_goal":goal.to_string(),
                                "do_actions": self.do_actions,
                                "goal_memories": "\n".join([m.content for m in goal_memories]),
                                "ntasks": ntasks,
                                "known_actors": "\n".join([name for name in self.actor_models.names()]),
                                "achievments": '\n'.join(self.achievments[:5]),
                                "scene_narrative": scene_narrative,
                                "act_specific_narrative": self.context.act_central_narrative if self.context.act_central_narrative else '',
                                "central_narrative": self.context.central_narrative if self.context.central_narrative else '',
                                "my_narrative": self.central_narrative if self.central_narrative else '',
                                "scene": json.dumps(self.context.current_scene, indent=2, default=datetime_handler) if self.context.current_scene else '',
                                "task_plan": "\n".join([t.to_string() for t in goal.task_plan]) if goal.task_plan else '', 
                                "new_task": new_task.to_string() if new_task else ''},
                               tag = 'Character.generate_task_plan',
                               max_tokens=350, log=True)


        # add each new task, but first check for and delete any existing task with the same name
        task_plan = []
        start_index = -1
        forms = hash_utils.findall_forms(response)
        current_task_plan_length = len(goal.task_plan) if goal.task_plan else ntasks
        if current_task_plan_length < 2 and len(forms) > current_task_plan_length + 1:
            start_index = len(forms) - current_task_plan_length - 1 # skip early tasks if new plan is too long; preserve important closing tasks.
        elif current_task_plan_length >= 2 and len(forms) > current_task_plan_length:
            start_index = len(forms) - current_task_plan_length # skip early tasks if new plan is too long; preserve important closing tasks.
        for t, task_hash in enumerate(forms):
            if t < start_index:
                continue # skip early tasks if new plan is too long; preserve important closing tasks.
            print(f'\n{self.name} new task: {task_hash.replace('\n', '; ')}')
            if not self.focus_goal:
                print(f'{self.name} generate_plan: no focus goal, skipping task')
            task = self.validate_and_create_task(task_hash, goal)
            if task:
                task_plan.append(task)
        print(f'{self.name} generate_task_plan: {len(task_plan)} tasks found')
        goal.task_plan = task_plan # only use the first task for now, to force replanning in face of new information
        return task_plan

    def generate_completion_statement(self, object, termination_check, satisfied, progress, consequences='', updates=''):
        """Generate a statement about the completion of an goal or task"""
        prompt = [SystemMessage(content="""Generate a concise statement that accurately states the actual achievements with respect to the following goal or task."""),
                  UserMessage(content="""
Goal or task to complete:
{{$objective}}

Completion criterion:
{{$termination_check}}

Result of completion criterion assessment:
{{$satisfied}}
{{$progress}}

overall current personal situation:
{{$situation}}

overall current world state:
{{$world}}


world consequences of most recent recent act:
{{$consequences}}
</consequences>

observable world updates from most recent act
{{$updates}}
</updates>

<history>
{{$history}}
</history>

<events>
{{$events}}
</events>

<recent_memories>
{{$memories}}
</recent_memories>
                              
<relationships>
{{$relationships}}
</relationships>
                              
Respond ONLY with the concise (20-50 words) statement about the actual achievements with respect to the goal or task.
Include in a simple text paragraph:
 - The goal or task name
 - The termination criterion
 - The actual achievement, ie progress towards the termination criterion
 - What was achieved, ie a change in your information state, relationships, or world state
 - What remains unachieved, if the goal or task was not completely satisfied.
                              
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(8)
        memory_text = '\n'.join(memory.to_string() for memory in recent_memories)

        if consequences == '':
            consequences = self.context.last_consequences
        if updates == '':
            updates = self.context.last_updates
        response = self.llm.ask({"objective": object.to_string(),
                                "termination_check": termination_check.strip('##')+ ': wrt '+object.name+', '+object.description,
                                "satisfied": satisfied,
                                "progress": progress,
                                "situation": self.context.current_state,
                                "world": self.context.current_state,
                                "consequences": consequences,
                                "updates": updates,
                                "events": consequences + '\n' + updates,
                                "character": self.get_character_description(),
                                "history": self.format_history_for_UI(),
                                "relationships": self.narrative.get_summary('medium'),
                                "memories": memory_text
                                 }, prompt, tag='completion_statement', temp=0.5, stops=['</end>'], max_tokens=80)
        return response

    async def test_termination(self, object, termination_check, consequences, updates='', type=''):
        """Test if recent acts, events, or world update have satisfied termination"""
        prompt = [SystemMessage(content="""Test if recent acts, events, or world update have satisfied the CompletionCriterion is provided below. 
Reason step-by-step using the CompletionCriterion as a guide for this assessment.
Consider these factors in determining task completion:
- Sufficient progress towards goal for intended purpose
- Diminishing returns on continued effort
- Environmental or time constraints
- "Good enough" vs perfect completion

For concrete termination checks (e.g., 'sufficient food gathered'), the full completion criterion is the actual achievement of the termination check, not merely thought, conversation, or movement towards it.
A good way to test completion is to first generate a concise statement of actual achievement, and then test if the statement satisfies the termination check.
Do not output your reasoning or your achievment statement."""),
                  UserMessage(content="""
                              
<situation>
{{$situation}}
</situation>

<history>
{{$history}}
</history>

<events>
{{$events}}
</events>

<recent_memories>
{{$memories}}
</recent_memories>
                              
<relationships>
{{$relationships}}
</relationships>

<completion_criterion>
{{$termination_check}}
</completion_criterion>

Respond using this hash-formatted text:

#status complete/partial/insufficient
#progress 0-100
##

Respond ONLY with the above hash-formatted text.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]


        # Get recent memories
        recent_memories = self.structured_memory.get_recent(8)
        memory_text = '\n'.join(memory.to_string() for memory in recent_memories)

        if consequences == '':
            consequences = self.context.last_consequences
        if updates == '':
            updates = self.context.last_updates

        response = self.llm.ask({
            "termination_check": termination_check.strip('##')+ ': wrt '+object.name+', '+object.description,
            "situation": self.context.current_state,
            "memories": memory_text,  # Updated from 'memory'
            "events": consequences + '\n' + updates,
            "character": self.get_character_description(),
            "history": self.format_history_for_UI(),
            "relationships": self.narrative.get_summary('medium')
        }, prompt, tag='test_termination', temp=0.5, stops=['</end>'], max_tokens=20)

        satisfied = hash_utils.find('status', response)
        progress = hash_utils.find('progress', response)
        print(f'\n{self.name} testing {type} {object.name} termination: {termination_check}, ', end='')
        try:
            progress = int(progress.strip())
            object.progress = progress
        except:
            progress = 50
        if satisfied != None and satisfied.lower().strip() == 'complete':
            print(f'  **Satisfied!**')
            statement = self.generate_completion_statement(object, termination_check, satisfied, progress, consequences, updates)
            object.completion_statement = statement
            self.add_perceptual_input(statement, mode='internal')
            self.context.current_state += f"\n\nFollowing update may invalidate parts of above:\n{statement}"
            if self.context.current_state.count('Following update may invalidate') > 3:
                #await self.context.update(local_only=True)
                await asyncio.sleep(0)
            return True, 100
        elif satisfied != None and 'partial' in satisfied.lower():
            if type == 'task': threshold = 0.0
            else: threshold = 0.1 # goal threshold is higher
            if progress/100.0 > random.random() + threshold:
                print(f'  **Satisfied partially! {satisfied}, {progress}%**')
                statement = self.generate_completion_statement(object, termination_check, satisfied, progress, consequences, updates)
                object.completion_statement = statement
                self.add_perceptual_input(statement, mode='internal')
                self.context.current_state += f"\n\nFollowing update may invalidate parts of above:\n{statement}"
                if self.context.current_state.count('Following update may invalidate') > 3:
                    #await self.context.update(local_only=True)
                    await asyncio.sleep(0)
                return True, progress
        #elif satisfied != None and 'insufficient' in satisfied.lower():
        #    if progress/100.0 > random.random() + 0.5:
        #        print(f'  **Satisfied partially! {satisfied}, {progress}%**')
        #        self.add_perceptual_input(f"{object.name} is sufficiently complete: {termination_check}", mode='internal')
        #        return True, progress
            
        print(f'  **Not satisfied! {satisfied}, {progress}%**')
        return False, progress

    def refine_say_act(self, act: Act, task: Task):
        """Refine a say act to be more natural and concise"""
        if act.target:
            target = act.target[0]
            target_name = act.target[0].name
        elif task.actors:
            for actor in task.actors+self.context.npcs:
                if actor is not self:
                    target = actor
                    target_name = actor.name
                    break
        if not target or target_name is None:
            target_name = self.say_target(act.mode, act.action, task)
        if target_name is None:
            return act
        
        dialog = self.actor_models.get_actor_model(target_name, create_if_missing=True).dialog.get_transcript(20)
        if dialog is None or len(dialog) == 0:
            return act
        prompt = [UserMessage(content="""Revise the following proposed text to say given the previous dialog with {{$target}}.

<previous_dialog>
{{$dialog}}
</previous_dialog>

<proposed_text>
{{$act_arg}}
</proposed_text>  

<relationship>
{{$relationship}}
</relationship>

<character>
{{$character}}
</character>

Respond ONLY with the revised text to say.
Your revised text should be about the same length as the proposed text, and naturally extend the previous dialog.
Speak in the first person, using the character's voice.
Do not repeat the previous dialog.
Do not include any introductory, explanatory, formative, or discursive text.
End response with:
</end>
""")]

        relationship = self.actor_models.get_actor_model(target_name, create_if_missing=True).get_relationship()
        response = self.llm.ask({
                            "act_arg": act.action,
                            "dialog": dialog,
                            "target": target_name,
                            "relationship": relationship,
                            "character": self.get_character_description()
                    }, prompt, tag='refine_say_act', temp=0.6, stops=['</end>'])

        return act

    def generate_acts(self, task: Task, goal: Goal=None):
        in_dialog = self.check_if_in_dialog_subtask()
        modes = 'Think, Say, Look, Move, Do'
        if in_dialog:
            modes = 'Look, Move, Do'
        print(f'\n{self.name} generating acts for task: {task.to_string()}')
        mission = """generate a set of two alternative acts {{$modes}} for the next step of the following task:

<task>
{{$task_string}}
</task>

#this is derived from your goal:

<goal>
{{$goal_string}}
</goal>
##

#In the following narrative context, the act should be consistent with the character's drives and emotional stance.

<central_narrative>
{{$central_narrative}}
</central_narrative>
<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>
##

In light of the narrative context above, your acts should serve story momentum as well as character psychology.

Consider how your chosen action:
- Advances the central narrative question toward resolution or deepens the mystery
- Escalates or defuses the current act's dramatic tension  
- Creates new conflict, responds to existing conflict, or avoids confrontation
- Reveals, conceals, or misdirects crucial information
- Forces decisions from others or demonstrates your own choices
- Affects your alliances and relationships in service of the story

Choose actions that drive narrative forward, not just maintain status quo.
"""
        suffix = """

#recent percepts related to the current goals and tasks
{{$goal_memories}}
##

Respond with two alternative acts, including their Mode, action, target, and expected duration. 
The acts should vary in mode and action.

In choosing each act (see format below), you can choose from these Modes:
- Say - speak, to obtain or share information, to align, coordinate with, or manipulate others in service of your goals or drives; to reason jointly, or to establish or maintain a bond. 
    For example, if you want to build a shelter with Samantha, it might be effective to Say: 'Samantha, let's build a shelter.'
- Look - observe your surroundings, gaining information on features, actors, and resources at your current location and for the eight compass
    points North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- Move - move in any one of eight directions: North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
    Alternately, move towards a known resource or actor.
    Useful when you need to move towards a resource or actor.
- Do - perform an act (other than move) to achieve physical consequences in the world or demonstrate to others your commitment to a goal or drive (whether real or performative). 
    This is often appropriate when the task involves interacting physically with a resource or actor. See below for actor thoughts on physical actions.
- Think - mental reasoning about the current situation wrt your state and the task.
    Often useful when you need to plan or strategize, or when you need to understand your own motivations and emotions, but beware of overthinking. Thoughts are not revealed to others.

############
#Consider the following thoughts the actor animating this character has had about physical action ('Do' actions). 
#Remember, these are suggestive, neither limiting nor exhaustive, and should be used as a guide, not a constraint:

{{$do_actions}}
############

Review your character and current emotional stance when choosing Mode and action. 
Emotional tone and orientation can (and should) heavily influence the boldness, mode, phrasing and style of expression for an Act.

An act is:
- Is a specific thought, spoken text, physical movement or action.
- Includes only the actual thoughts, spoken words, physical movement, or action.
- Has a clear beginning and end point.
- Can be performed or acted out by a person.
- Can be easily visualized or imagined as a film clip.
- Makes sense as the next action given observed results of previous act . 
- Is consistent with any incomplete action commitments made in your last statements in RecentHistory.
- Does NOT repeat, literally or substantively, a previous act by you in RecentHistory, unless it is a continuation of the same action.
- Significantly advances the story or task at hand, and so may be deceptive or surprising to others.
- Is stated in the appropriate person (voice):
        If a thought (mode is 'Think') or speech (mode is 'Say'), is stated in the first person.
        If an act in the world (mode is 'Do'), is stated in the third person.
 
Prioritize actions that lead to meaningful progress in the narrative.
IMPORTANT: When evaluating a potential Act, the primary consideration is whether it can directly result in achievment of the task goal. 

Dialog guidance: If speaking (mode is Say), then:
- The specificAct must contain only the actual words to be spoken.
- Respond in the style of spoken dialog, not written text. Pay special attention to the character's emotional stance shown above in choosing tone and phrasing. 
    Use contractions and language appropriate to the character's personality, emotional tone and orientation. Speak in the first person. DO NOT repeat phrases used in recent dialog.
- If intended recipient is known  or has been spoken to before (e.g., in RecentHistory), 
    then pronoun reference is preferred to explicit naming, or can even be omitted. 
- In any case, when using pronouns, always match the pronoun gender (he, she, his, her, they, their,etc) to the sex of the referent, or use they/their if a group. 
- Avoid repeating phrases in RecentHistory derived from the task, for example: 'to help solve the mystery'.
- Avoid repeating stereotypical past dialog.
- Avoid repeating phrases or statements from past conversations:
{{$dialog_transcripts}}.

When describing an action:
- Reference previous action if this is a continuation
- Indicate progress toward goal (starting/continuing/nearly complete)
- Note changes in context or action details
- Describe progress toward goal
- Use the appropriate pronoun gender and case(he, she, his, her, etc.) for any referent, or they / their if the referent is a group.
    Use the following gender information:
    {{$actor_genders}}
Consider the previous act. E.G.:
- If the previous act was a Move, are you now at your destination? If not, do you want to keep moving towards it?
    If you are at your destination, what do you want to Do there? 
    Gather or use a resource? Talk to someone there? Do something else at your new location?
- If the previous act was a Look, what did you learn and how does it affect this act?
                              
Respond in hash-formatted text:

#mode one of {{$modes}}, corresponding to whether the act is a physical act, speech, or reasoning. Note that Move can take either a direction or a resource name.
#action thoughts, words to speak, direction to move, or physical action. For Move this can be a direction or a resource name. Be concise, limit your response to 16-20 words for mode Do or 30 words max for mode Say.
#target name(s) of the actor(s) you are thinking about, speaking to, looking for, moving towards, or acting on behalf of, comma separated, or omit if no target(s).
#duration expected duration of the action in minutes. Use a fraction of task duration according to the expected progress towards completion.
##

#NOTE#
mode can only be one of {{$modes}}.

===Examples===

Task:
Situation: increased security measures; State: fear of losing Annie
Actors: Annie, Madam

Response:
#mode Do
#action Call the building management to discuss increased security measures for Annie and the household.
#target building management
#actors Annie, Madam
#duration 10 minutes

----

Task:
Establish connection with Joe given RecentHistory element: "Who is this guy?"
Actors: Samantha

Response:
#mode Say
#action Hi, who are you?
#duration 1 minute
#target Joe
#actors Samantha
##

----

Task:
Find out where I am given Situation element: "This is very very strange. Where am I?"
Actors: Samantha

Response:
#mode Look
#action look around for landmarks or signs of civilization
#duration 1 minute
#actors Samantha
##

----

Task:
Find food.
Actors: Joe, Samantha

Response:
#mode Move
#action berries#2
#duration 15 minute
#target None
#actors Joe, Samantha
##

----

Task: dialog with Elijah: Maya talks with Elijah to maintain their bond despite her intention to leave.

Response:
#mode Say
#action Hey, Elijah, - you know I'd never leave this place, right?
#duration 1 minute
#target Elijah, Chrys
#actors Maya
##


===End Examples===

Use the following hash-formatted text format for each act.
Each act should be closed by a ## tag on a separate line.
be careful to insert line breaks only where shown, separating a value from the next tag:

#mode Think, Say, Do, Look, or Move
#action thoughts, words to say, direction to move, or physical action
#duration expected duration of the action in minutes
#target name of the actor you are thinking about, speaking to, looking for, moving towards, or acting on behalf of, if applicable. Otherwise omit.
#actors name(s) of the actor(s) of this act, comma separated.
##

Respond ONLY with the above hash-formatted text for each alternative act.
Your name is {{$name}}, phrase the statement of specific action in your voice.
If the mode is Say, the action should be the actual words to be spoken.
    e.g. 'Maya, how do you feel about the letter from the city gallery?' rather than a description like 'ask Maya about the letter from the city gallery and how it's making her feel'. 
Ensure you do not duplicate content of a previous specific act.

Again, the task to translate into alternative acts is:
<task>
{{$task_string}} 
</task>

{{$scene_post_narrative}}

Do not include any introductory, explanatory, or discursive text. Remember, respond with 2 or at most 3 alternative acts.
End your response with:
</end>
"""

        if not goal:
            goal = self.focus_goal
        if not goal and task and task.goal:
            goal = task.goal
        if not goal:
            raise ValueError(f'No goal for {self.name}')
        goal_memories = []
        actor_genders = ''
        for actor in self.context.actors+self.context.extras:
            actor_genders += f'\t{actor.name}: {actor.gender}\n'
        #generate emotional state for this task
        ranked_signalClusters = self.driveSignalManager.get_scored_clusters()
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order
        self.emotionalStance = EmotionalStance.from_signalClusters(focus_signalClusters, self)
        memories = self.perceptual_state.get_information_items(goal.short_string(), threshold=0.1, max_results=3)
        for t in goal.task_plan:
            memories = self.perceptual_state.get_information_items(t.short_string(), threshold=0.1, max_results=3)
            for tm in memories:
                if tm not in goal_memories:
                    goal_memories.append(tm)
        task = self.focus_task.peek()
        if self.context.scene_post_narrative:
            scene_post_narrative = f"\n\nThe dominant theme of the scene is: {self.context.scene_post_narrative}. The act alternatives should be consistent with this theme."
        else:
            scene_post_narrative = ''

        response = default_ask(self, prefix=mission, suffix=suffix, 
                               addl_bindings={"goal_string":goal.to_string(), "task":task, "task_string":task.to_fullstring(),
                                              "do_actions": self.do_actions,
                                              "goal_memories": "\n".join([m.content for m in goal_memories]),
                                              "scene_post_narrative": scene_post_narrative,
                                              "act_specific_narrative": self.context.act_central_narrative if self.context.act_central_narrative else '',
                                              "central_narrative": self.context.central_narrative if self.context.central_narrative else '',
                                              "scene": json.dumps(self.context.current_scene, indent=2, default=datetime_handler) if self.context.current_scene else '',
                                              "modes": modes,
                                              "name": self.name,
                                              "actor_genders": actor_genders,
                                              "dialog_transcripts": '\n'.join(self.actor_models.get_dialog_transcripts(20))}, 
                               tag = 'Character.generate_acts',
                               max_tokens=200, log=True)
        if response is not None:
            response = response.strip()
        # Rest of existing while loop...
        act_hashes = hash_utils.findall_forms(response)
        act_alternatives = []
        if len(act_hashes) == 0:
            print(f'No act found in response: {response}')
            self.actions = []
            return []
        if not task.goal:
            print(f'{self.name} generate_act_alternatives: task {task.name} has no goal!')
            if not self.focus_goal:
                print(f'{self.name} generate_act_alternatives: no focus goal either!')
            task.goal = self.focus_goal
        if len(act_hashes) < 2:
            print(f'{self.name} generate_act_alternatives: only {len(act_hashes)} acts found')
        for act_hash in act_hashes:
            try:
                act = self.validate_and_create_act(act_hash, task)
                if act:
                    act_alternatives.append(act)
            except Exception as e:
                print(f"Error parsing Hash, Invalid Act. (act_hash: {act_hash}) {e}")
        self.actions = act_alternatives
        return act_alternatives


    async def update_actions_wrt_say_think(self, source, act_mode, act_arg, reason, target=None):
        """Update actions based on speech or thought"""
        if not self.focus_goal or source.name.startswith('dialog'):
            # print(f' in dialog, no action updates')
            return
        if target is None:
            if source and source.target and hasattr(source.target, 'name'):
                target_name = source.target.name
            else:
                target_name = self.say_target(act_mode, act_arg, source)
        elif hasattr(target, 'name'):  # Check if target has name attribute instead of type
            target_name = target.name
        else:
            target_name = target
        # Skip action processing during active dialogs
        if target is not None and self.actor_models.get_actor_model(target_name, create_if_missing=True).dialog.active and source.name.startswith('dialog'):
            # print(f' in active dialog, no action updates')
            return
        
        # Rest of the existing function remains unchanged
        print(f'\n{self.name} Update actions from say or think\n {act_mode}, {act_arg};  reason: {reason}')
        
        if 'Viewer' in source.name:  # Still skip viewer
            print(f' source is Viewer, no action updates')
            return
        
        prompt=[UserMessage(content="""Your task is to analyze the following text, and extract a single new task expressing an intension for {{$name}} to act, if any.
Otherwise respond None.
Be careful to distinguish between hypotheticals and actual commitments. Respond only with actual commitments.

<text>
{{$text}}
</text>

Given the following background information:

<name>
{{$name}}
</name>

<focus_task>
{{$focus_task}}
</focus_task>

You have already formed the following intensions:
<intensions>
{{$intensions}}
</intensions>

<reason>
{{$reason}}
</reason>

Does it include an intension for 'I' to act, that is, a new task being committed to? 
An action can be physical or verbal.

Consider the current task and action reason in determining if there is a new task being committed to.

Do not include any intensions that are similar to those already formed.
Do not include any intensions that are similar to the focus task.

Respond with at most one task expressing an intension in the text.
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Each intension should be closed by a ## tag on a separate line.
be careful to insert line breaks only where shown, separating a value from the next tag:

#name brief (4-6 words) action name
#description terse (6-8 words) statement of the action to be taken
#reason (6-7 words) on why this action is important now
#duration (2-3 words) expected duration of the action in minutes
#termination (5-7 words) easily satisfied termination test to quickly end this task
#actors {{$name}}
##

In refering to other actors, always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', etc.
                            

===Examples===

Text:
'Good morning Annie. I'm heading to the office for the day. Call maintenance about the disposal noise please.'

<name>
Madam
</name>

Response:

#name Head to the office for the day.
#description Head to the office for the day.
#reason Need to go to work.
#termination Leave for the office
#actors Madam
##


Text:
'I really should reassure annie.'

<name>
Hank
</name>

Response:

#name Reassure Annie
#description Reassure Annie
#reason Need to reassure Annie
#termination Reassured Annie
#duration 10 minutes
#actors Hank
##


Text:
'Reflect on my thoughts and feelings to gain clarity and understanding, which will ultimately guide me towards finding my place in the world.'

Response:

#name Reflect on my thoughts and feelings
#description Reflect on my thoughts and feelings
#reason Gain clarity and understanding
#termination Gained clarity and understanding
#duration 10 minutes
#actors Annie
##

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the action analysis in hash-formatted text as shown above.
End your response with:
</end>""")]

        goal = self.focus_goal
        response = self.llm.ask({"text":f'{act_mode} {act_arg}',
                                 "focus_task":self.focus_task.peek(),
                                 "reason":reason,
                                 "intensions":'\n'.join(task.to_string() for task in self.intensions[-5:]),
                                 "name":self.name}, 
                                 prompt, tag='update_actions_wrt_say_think', temp=0.1, stops=['</end>'], max_tokens=150)
        intension_hashes = hash_utils.findall_forms(response)
        if len(intension_hashes) == 0:
            print(f'no new tasks in say or think')
            return
        for intension_hash in intension_hashes:
            intension = self.validate_and_create_task(intension_hash, goal)
            if intension and self.focus_goal:
                print(f'  New task from say or think: {intension_hash.replace('\n', '; ')}')
                self.focus_task.push(intension)
                self.step_task('')
                if self.focus_goal and self.focus_goal.task_plan and len(self.focus_goal.task_plan) > 0:
                    await self.generate_task_plan(self.focus_goal)

            elif intension:
                print('No focus goal, no new task from say or think')
        return
    
    async def update_individual_commitments_following_conversation(self, target:Character, transcript:str, joint_tasks=[]):
        """Update individual commitments after closing a dialog"""
        if not self.focus_goal or self.check_if_in_dialog_subtask():
            return []
        prompt=[UserMessage(content="""Your task is to analyze the following transcript of a dialog between {{$name}} and {{$target_name}},
your task is to analyze the following transcript of a conversation between {{$name}} and {{$target_name}},
and identify any new commitment to act individually, now,  made solely by you, {{$name}}, if any. Otherwise respond None.
Any commitment reported must be definite, immediate, non-hypothetical, and both significant to and not redundant in the scene you are in.
Be careful to distinguish between hypotheticals and actual commitments. Respond only with an actual commitment. 
Note that a commitment must come from a clear, firm, statement of intent to act, and not from a question, statement of opinion, statement of fact, or other conversational noise.
For example, the following transcript contains hypothetical commitments by Alex that should not be reported as new commitments to act:
                            
<hypotheticial commitments example>
Interviewer says: Alex, suppose you're thrown into a high-stakes project using unfamiliar tech,
     and the timeline's slashed in half—how would you keep things from unraveling without your usual backup?
Alex says: If that happened, I'd quickly get up to speed on the essentials and trim the project down. 
    I'd be clear about the risks from the start, tackle key features first, and adjust as needed, making sure everyone's on the same page about what's doable without backup.

</hypothetical commitments example>

Further, any new commitments should be consistent with the scene you will be acting in, as given below.
                            
<central_narrative>
{{$central_narrative}}
</central_narrative>
                            
<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

<scene>
{{$scene}}
</scene>

## the conversation up to this point has been:
# <transcript>
{{$transcript}}
</transcript>

Given the following background information:

<all_tasks> 
{{$all_tasks}}
</all_tasks>

<focus_task>
{{$focus_task}}
</focus_task>

<reason>
{{$reason}}
</reason>

<self>
{{$name}}
</self>

<other>
{{$target_name}}
</other>

<joint_tasks>
{{$joint_tasks}}
</joint_tasks>

Extract from this transcript a new commitment to act made by self, {{$name}}, to other, {{$target_name}} relevant to and not redundant in the scene you will are acting in.

Extract only commitments made by {{$name}} that are consistent with the entire transcript and remain unfulfilled at the end of the transcript.
Note that the joint_tasks listed above, are commitments made by both {{$name}} and {{$target_name}} to work together, and should not be reported as new commitments here.
                            
Does the transcript include an intension for {{$name}} to act alone, that is, a new task being committed to individually? 
An action can be physical or verbal.
Thought, e.g. 'reflect on my situation', should NOT be reported as an action.
Consider the all_tasks pending and current task and action reason in determining if a candidate task is in fact new.

Respond only with all intensions in the transcript, in the temporal order implied in the transcript.
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Each intension should be closed by a ## tag on a separate line.
be careful to insert line breaks only where shown, separating a value from the next tag:

#name brief (4 words max) task name
#description terse (6 words max) statement of the action to be taken
#actors {{$name}}
#start_time (2 words) expected start time of the action, typically elapsed time from now
#duration (2 words) expected duration of the action in minutes
#reason (4 words max) on why this action is important now
#termination (5 words max) easily satisfied termination test to quickly end this task
##

In refering to other actors, always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', etc.
                            

===Example===

<transcript>
Jean says: Hey Francoise, what needs doin' on the farm today?
Francoise says: Waterin' the south field's a priority, Jean. Crops are gettin' parched. We should check the wheat too, make sure it's not gettin' too ripe.
Jean says: Aye, let's get the waterin' done first, then we can take a gander at the wheat. I'll grab the buckets and meet you by the well.
Francoise says: I'll meet you by the well then, Jean. Don't forget to check the bucket handles, we don't want 'em breakin' on us.
Jean says: Aye, I'll give 'em a good check. See you by the well, Francoise. I'll bring the ropes too, just in case.
</transcript>

<self>
Francoise
</self>

<other>
Jean
</other>

Response:

#name bring ropes
#description bring ropes to meeting with Jean
#reason in case the well handle breaks
#termination Met Jean by the well
#start_time 0 minutes
#duration 10 minutes
#actors Francoise
#committed True
##


===End Example===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the action analysis in hash-formatted text as shown above.
End your response with:
</end>
""")]
        response = self.llm.ask({"transcript":transcript, 
                                 "all_tasks":'\n'.join([task.name for task in self.focus_task.stack]),
                                 "focus_task":self.focus_task.peek(),
                                 "joint_tasks":'\n'.join(hash_utils.find('name', task) for task in joint_tasks),
                                 "reason":self.focus_task.peek().reason if self.focus_task.peek() else '',
                                 "name":self.name, 
                                 "target_name":target.name,
                                 "central_narrative": self.context.central_narrative if self.context.central_narrative else '',
                                 "act_specific_narrative": self.context.act_central_narrative if self.context.act_central_narrative else '',
                                 "scene": json.dumps(self.context.current_scene, indent=2, default=datetime_handler) if self.context.current_scene else ''}, 
                                 prompt, tag='update_individual_commitments_following_conversation', temp=0.1, stops=['</end>'], max_tokens=180)
        source = Task('dialog with '+target.name, 
                      description='dialog with '+target.name, 
                      reason='dialog with '+target.name, 
                      termination='natural end of dialog', 
                      goal=self.focus_goal if self.focus_goal else None,
                      actors=[self, target],
                      start_time=self.context.simulation_time,
                      duration=0)    
        intension_hashes = hash_utils.findall_forms(response)
        if len(intension_hashes) == 0:
            print(f'no new tasks in conversation')
            return
        for intension_hash in intension_hashes:
            intension = self.validate_and_create_task(intension_hash)
            if intension:
                action_order_length = len(self.context.current_scene['action_order']) if self.context.current_scene else 10 # if not in a scene, skip the significance check
                if self.__class__.__name__ == 'NarrativeCharacter' and self.current_scene:
                    significance = self.context.evaluate_commitment_significance(self, target, intension)
                    if significance == 'NOISE':
                        continue
                    elif significance == 'RELEVANT' or significance == 'REDUNDANT':
                        intension.reason = intension.reason + ' (optional)'
                    elif (significance == 'SIGNIFICANT' and self.context.scene_integrated_task_plan and len(self.context.scene_integrated_task_plan) < 1.5*action_order_length) \
                        or (significance == 'CRUCIAL' and self.context.scene_integrated_task_plan and len(self.context.scene_integrated_task_plan) < 2*action_order_length):
                        print(f'\n{self.name} new individual committed task: {intension_hash.replace('\n', '; ')}')
                        if self.context.scene_integrated_task_plan:
                            insert_index = self.context.scene_integrated_task_plan_index+1
                            self.context.scene_integrated_task_plan.insert(insert_index, {'actor': self, 'goal': self.focus_goal, 'task': intension})
                elif self.focus_goal:
                    await self.generate_task_plan(self.focus_goal, new_task=intension, replan=True)
                else:
                    print('No focus goal, no new task from conversation')
        return
  
    async def update_joint_commitments_following_conversation(self, target:Character, transcript:str):
        if not self.focus_goal or self.check_if_in_dialog_subtask():
            return []
        """Update individual commitments after closing a dialog"""
        
        prompt=[UserMessage(content="""Your task is to analyze the following transcript of a conversation between {{$name}} and {{$target_name}},
     and identify any new commitment to act now made jointly by {{$name}} and {{$target_name}}, if any. Otherwise respond None.
Any commitment reported must be definite, immediate, non-hypothetical, and both significant to and not redundant in the scene you are in.
Note that a commitment must come from a clear, firm, statement of intent, and not from a question, statement of opinion, statement of fact.
Be careful to distinguish between hypotheticals and actual commitments. Respond only with actual commitments.
                            
<hypotheticial commitments example>
Interviewer says: Alex, suppose you're thrown into a high-stakes project using unfamiliar tech,
     and the timeline's slashed in half—how would you keep things from unraveling without your usual backup?
Alex says: If that happened, I'd quickly get up to speed on the essentials and trim the project down. 
    I'd be clear about the risks from the start, tackle key features first, and adjust as needed, making sure everyone's on the same page about what's doable without backup.
</hypothetical commitments example>

Further, any new commitments should be consistent with and not redundant in the scene you will be acting in, as given below.
                            
<central_narrative>
{{$central_narrative}}
</central_narrative>
                            
<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

<scene>
{{$scene}}
</scene>

## the conversation up to this point has been:
<transcript>
{{$transcript}}
</transcript>

Given the following background information:

<all_tasks> 
{{$all_tasks}}
</all_tasks>

<focus_task>
{{$focus_task}}
</focus_task>

<reason>
{{$reason}}
</reason>


Extract from this transcript a single most important new commitment to act jointly made by self, {{$name}} and other, {{$target_name}}, if any. Otherwise respond None.
Extract only an express commitment appear in the transcript and remaining unfulfilled at the end of the transcript.
If more than one joint action commitment is found, report the most concrete, immediate, prominent one in the transcript.
                            
An action can be physical or verbal.
Thought, e.g. 'reflect on our situation', should NOT be reported as a commitment to act.
Consider the all_tasks and current task and action reason in determining if there is a new task being committed to.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#name brief (4 words max) action name
#description terse (6 words max) statement of the action to be taken
#actors {{$name}}, {{$target_name}}
#reason (4 words max) on why this action is important now
#duration (2 words) expected duration of the action in minutes
#termination (5 words max) easily satisfied termination test to quickly end this task
##

In refering to other actors, always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', 'other', etc.
                            

===Example===

<transcript>
Jean says: Hey Francoise, what needs doin' on the farm today?
Francoise says: Waterin' the south field's a priority, Jean. Crops are gettin' parched. We should check the wheat too, make sure it's not gettin' too ripe.
Jean says: Aye, let's get the waterin' done first, then we can take a gander at the wheat. I'll grab the buckets and meet you by the well.
Francoise says: I'll meet you by the well then, Jean. Don't forget to check the bucket handles, we don't want 'em breakin' on us.
Jean says: Aye, I'll give 'em a good check. See you by the well, Francoise. I'll bring the ropes too, just in case.
</transcript>

<self>
Francoise
</self>

<other>
Jean
</other>

Response:

#name farm chores
#description Meet by the well  
#duration 10 minutes
#actors Jean, Francoise
#reason get water for the south field
#duration 30 minutes
#termination water bucket is full

===End Example===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the action analysis in hash-formatted text as shown above.
End your response with:
</end>
""")]
        response = self.llm.ask({"transcript":transcript, 
                                 "all_tasks":'\n'.join([task.name for task in self.focus_goal.task_plan]) if self.focus_goal else '',
                                 "focus_task":self.focus_task.peek().to_string() if self.focus_task.peek() else '',
                                 "reason":self.focus_task.peek().reason if self.focus_task.peek() else '',
                                 "name":self.name, 
                                 "target_name":target.name,
                                 "central_narrative": self.context.central_narrative if self.context.central_narrative else '',
                                 "act_specific_narrative": self.context.act_central_narrative if self.context.act_central_narrative else '',
                                 "scene": json.dumps(self.context.current_scene, indent=2, default=datetime_handler) if self.context.current_scene else ''}, 
                                 prompt, tag='update_joint_commitments_following_conversation', temp=0.1, stops=['</end>'], max_tokens=240)
        source = Task('dialog with '+target.name, 
                      description='dialog with '+target.name, 
                      reason='dialog with '+target.name, 
                      termination='natural end of dialog', 
                      start_time=self.context.simulation_time,
                      duration=0,
                      goal=None,
                      actors=[self, target])    
        intension_hashes = hash_utils.findall_forms(response)
        intensions = []
        if len(intension_hashes) == 0:
            print(f'no new joint intensions in turn')
            return []
        for intension_hash in intension_hashes:
            intension = self.validate_and_create_task(intension_hash)
            if intension:
                if self.__class__.__name__ == 'NarrativeCharacter' and self.current_scene:
                    action_order_length = len(self.context.current_scene['action_order']) if self.context.current_scene else 10 # if not in a scene, skip the significance check
                    significance = self.context.evaluate_commitment_significance(self, target, intension)
                    if significance == 'NOISE':
                        continue
                    elif significance == 'RELEVANT' or significance == 'REDUNDANT':
                        intension.reason = intension.reason + ' (optional)'
                    elif (significance == 'SIGNIFICANT' and self.context.scene_integrated_task_plan and len(self.context.scene_integrated_task_plan) < 1.5*action_order_length) \
                        or (significance == 'CRUCIAL' and self.context.scene_integrated_task_plan and len(self.context.scene_integrated_task_plan) < 2*action_order_length):
                        print(f'\n{self.name} new joint task: {intension_hash.replace('\n', '; ')}')
                        if self.context.scene_integrated_task_plan:
                            insert_index = self.context.scene_integrated_task_plan_index+1
                            self.context.scene_integrated_task_plan.insert(insert_index, {'actor': self, 'goal': self.focus_goal, 'task': intension})
                            intensions.append(intension)
                elif self.focus_goal:
                    await self.generate_task_plan(self.focus_goal, new_task=intension, replan=True)
                else:
                    print('No focus goal, no new task from conversation')
        return intensions

    def random_string(self, length=8):
        """Generate a random string of fixed length"""
        letters = string.ascii_lowercase
        return self.name+''.join(random.choices(letters, k=length))

    def tell(self, to_actor, message, source=None, respond=True):
        """Initiate or continue dialog with dialog context tracking"""
        if source.name.startswith('dialog'):
            
            if self.actor_models.get_actor_model(to_actor.name, create_if_missing=True).dialog.active is False:
                self.actor_models.get_actor_model(to_actor.name).dialog.activate()
                # Remove text between ellipses - thoughts don't count as dialog
                content = re.sub(r'\.\.\..*?\.\.\.', '', message)
                self.actor_models.get_actor_model(to_actor.name).dialog.add_turn(self, content)
        
        self.acts(Act(mode='Say', target=[to_actor], action=message, reason=message, duration=1, source=source), to_actor, 'tell', message, self.reason, source)
        

    def natural_dialog_end(self, from_actor):
        """ called from acts when a character says something to this character """
        #if self.actor_models.get_actor_model(from_actor.name).dialog.turn_count > 10):
        #    return True
        if from_actor and from_actor.name.lower() == 'viewer':
            return False
        prompt = [UserMessage(content="""Given the following dialog transcript, rate the naturalness of ending at this point.

#Transcript
{{$transcript}}
##
                              
For example, if the last entry in the transcript is a question that expects an answer (as opposed to merely musing), ending at this point is likely not expected.
On the other hand, if the last entry is an agreement to an earlier suggestion, this is a natural end.
Dialogs are short, and should be resolved quickly.
Respond only with a rating between 0 and 10, where
0 expects continuation of the dialog (i.e., termination at this point would be unnatural)
10 expects termination at this point (i.e., continuation is highly unexpected, unnatural, or repetitious).   
                                                  
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]   
        transcript = from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog() # this is dialog from the perspective of self.
        response = self.llm.ask({"transcript":transcript}, prompt, tag='natural_dialog_end', temp=0.1, stops=['</end>'], max_tokens=180)
        if response is None or type(response) != str:
            logger.error(f'Error parsing natural_dialog_end: {response}')
            return False
        try:
            rating = int(response.lower().replace('</end>','').strip())
        except ValueError:
            try:
                rating = int(''.join(filter(str.isdigit, response)))
                if rating < 0 or rating > 10:
                    rating = 7
            except ValueError:
                print(f'{self.name} natural_dialog_end: invalid rating: {response}')
                rating = 7
        # force end to run_on conversations
        end_point = rating > 7 or (random.randint(4, 10) < rating) or ((rating + len(transcript.split('\n'))) > random.randint(8,10))
        print(f'{self.name} natural_dialog_end: rating: {rating}, {end_point}')
        return end_point
    
    def reflect_on_dialog(self, dialog):
        """reflect on the thought dialog - duplicate, already have reason_over!"""
        system_prompt = """You are a rational agent reflecting about your situarion."""
        prefix = """You are reflecting on your situation given the following insights:

<dialog>
{{$dialog}}
</dialog>
"""
        suffix = """
Review all the information above and identify, in light of the focus provided by the dialog, up to 3 new insights or understandings that are significant to your situation and not present explicitly in the other above information.
An insight might be a new understanding of a relationship (e.g. I am falling in love with ...), a new understanding of a goal (e.g. I should achieve ...), or a new understanding of a situation (e.g. That door is the way out).
Respond only with a concise statement of no more than 20 words for each new insight.
Do not include and introductory, discursive, formatting, or explanatory text.
If you can derive nothing new of significance wrt thought, respond with 'None'.
"""
        response = default_ask(self, system_prompt=system_prompt, prefix=prefix, suffix=suffix, addl_bindings={"dialog":dialog}, tag='reflect_on_dialog', max_tokens=70)
        if response and response.strip().lower() != 'none':
            self.add_perceptual_input(f'Reflecting:\n {response}', percept=False, mode='internal')
        return
            
    async def think(self, message: str, source: Task=None, respond: bool=True):
        """ called from acts when a character says something to itself """
        # Initialize dialog manager if needed
        print(f'\n{self.name} thinks: {message}')
       
        # Remove text between ellipses - thoughts don't count as dialog
        message = re.sub(r'\.\.\..*?\.\.\.', '', message)
        if self.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.active is False:
            print(f'{self.name} has no active dialog, assertion error')
            return
        # otherwise, we have an active dialog in progress, decide whether to close it or continue it
        else:
            dialog_model = self.actor_models.get_actor_model(self.name, create_if_missing=True).dialog
            dialog_model.add_turn(self, message)
            if True: #self.natural_dialog_end(self): short-circuit for testing

                dialog = dialog_model.get_current_dialog()
                self.add_perceptual_input(f'Internal monologue:\n {dialog}', percept=False, mode='internal')
                dialog_as_list = dialog.split('\n') 
                self.actor_models.get_actor_model(self.name).short_update_relationship(dialog_as_list, use_all_texts=True)
                dialog_model.deactivate_dialog()
                self.reason_over(dialog) # if we think we should think about something, then do it!
                self.last_task = self.focus_task.peek()
                self.focus_task.pop()
                #self.driveSignalManager.recluster()
                return

    def reason_over(self, thought):
        """reason over the dialog"""
        system_prompt = """You are a rational agent reasoning over a thought, capable also of making intuitive leaps."""
        prefix ="""You are reasoning over the following thought:
<thought>
{{$thought}}
</thought>
"""
        suffix = """
Consider implications of known facts, relationships, your character, and your goals with respect to the thought.
After both inductive and deductive reasoning over all the above, generate a concise (15 words or less) statement that summarizes your newly surfaced understanding of the thought.
If that statement is a problem statement, formulate two very terse (3-4 words each) abstract solutions and add them to the statement.
Respond only with a concise statement of no more than 40 words total.
Do not include and introductory, discursive, formatting, or explanatory text.
If you can derive nothing new of significance wrt thought, respond with 'None'.

"""
        response = default_ask(self, system_prompt=system_prompt, prefix=prefix, suffix=suffix, addl_bindings={"thought":thought}, tag='reason_over', max_tokens=70)
        if response and response.strip().lower() != 'none':
            print(f'{self.name} reason_over: {response}')
            self.add_perceptual_input(f'Reasoning over thought:\n {response}', percept=False, mode='internal')
        return



    async def hear(self, from_actor: Character, message: str, source: Task=None, respond: bool=True):
        """ called from acts when a character says something to this character """
        # Initialize dialog manager if needed
        print(f'\n{self.name} hears from {from_actor.name}: {message}')
       
        # Special case for Owl-Doc interactions
        if self.name == 'Owl' and from_actor.name == 'Doc':
            text, response_source = self.generate_dialog_turn(from_actor, message, self.focus_task.peek())
            self.show = text
            return
        

        # Remove text between ellipses - thoughts don't count as dialog
        message = re.sub(r'\.\.\..*?\.\.\.', '', message)

        if (self.name == 'Viewer') or self.natural_dialog_end(from_actor): # test if the dialog has reached a natural end, or hearer is viewer.
            # close the dialog
            dialog = self.actor_models.get_actor_model(from_actor.name).dialog.get_current_dialog()
            self.add_perceptual_input(f'Conversation with {from_actor.name}:\n {dialog}', percept=False, mode='auditory')
            dialog_as_list = dialog.split('\n') 
            self.actor_models.get_actor_model(from_actor.name).short_update_relationship(dialog_as_list, use_all_texts=True)
    
            individual_tasks = []
            if self.name != 'Viewer' and from_actor.name != 'Viewer':
                joint_tasks = await self.update_joint_commitments_following_conversation(from_actor, dialog)
                await asyncio.sleep(0)
                if (not joint_tasks or len(joint_tasks) == 0):
                    individual_tasks = await self.update_individual_commitments_following_conversation(from_actor, dialog, joint_tasks)
                await asyncio.sleep(0)

            self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
            self.last_task = self.focus_task.peek()
            self.focus_task.pop()
            # it would probably be better to have the other actor deactivate the dialog itself
            dialog = from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog()
            from_actor.add_perceptual_input(f'Conversation with {self.name}:\n {dialog}', percept=False, mode='auditory')
            dialog_as_list = dialog.split('\n') 
            from_actor.actor_models.get_actor_model(self.name).short_update_relationship(dialog_as_list, use_all_texts=True)

            # only allow one individual task to be generated even if there are no joint tasks
            if from_actor.name != 'Viewer' and self.name != 'Viewer' and (not joint_tasks or len(joint_tasks) == 0) and (not individual_tasks or len(individual_tasks) == 0):
                await from_actor.update_individual_commitments_following_conversation(self, dialog, joint_tasks)
                await asyncio.sleep(0)

            from_actor.actor_models.get_actor_model(self.name).dialog.deactivate_dialog()
            from_actor.last_task = from_actor.focus_task.peek()
            from_actor.focus_task.pop()
            #self.driveSignalManager.recluster()
            #from_actor.driveSignalManager.recluster()
            return

        text, response_source = self.generate_dialog_turn(from_actor, message, self.focus_task.peek()) # Generate response using existing prompt-based method
        action = Act(mode='Say', action=text, actors=[self, from_actor], reason=text, duration=1, source=response_source, target=[from_actor])
        await self.act_on_action(action, response_source)
        if self.focus_action == action:
            self.focus_action = None
        await asyncio.sleep(0)

    def generate_dialog_turn(self, from_actor, message, source=None):
        #self.memory_consolidator.update_cognitive_model(
        #    memory=self.structured_memory,
        #    narrative=self.narrative,
        #    knownActorManager=self.actor_models,    
        #    current_time=self.context.simulation_time,
        #    character_desc=self.character,
        #    relationsOnly=True
        #)
            
        if not self.focus_task.peek():
            raise Exception(f'{self.name} has no focus task')
        duplicative_insert = ''
        system_prompt = """You are a seasoned writer writing dialog for a movie.
Keep the stakes personal and specific — pledge of a commitment, revelation of a secret, a deadline that can't be missed — so the audience feels the pulse of consequence.
Let conflict emerge through action, thought, spoken intention, and subtext, not narration.
Characters hold real agency; they pursue goals, make trade-offs, and can fail. Survival chores are background unless they expose or escalate the core mystery.
Use vivid but economical language, vary emotional tone, and avoid repeating imagery.
        """
        prompt_string = """Be faithful to your character as presented in the following.
Disagreement is not only allowed but expected when confidence is low or relationship is in question, fear is high, or the character perceives conflicting goals or a threat.

##Current dramatic context:
<central_narrative>
{{$central_narrative}}
</central_narrative>

<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

<scene>
{{$scene}}
</scene>

Given the following character description, emotional state, current situation, goals, memories, and recent history, """
        prompt_string += """generate a next thought in the internal dialog below.""" if self is from_actor else """generate a response to the statement below, based on your goals in the dialog."""
        prompt_string += """

<your_character>
{{$character}}
</your_character>

<emotional_state>
{{$emotionalState}}
</emotional_state>

<situation>
{{$situation}}
</situation>

<goals>
{{$goals}}
</goals>

<memories>
{{$memories}}
</memories>

<recent_history>
{{$history}}
</recent_history>

Your last action was:

{{$activity}}

"""
        prompt_string += """You think of yourself as:""" if self is from_actor else """You think of the speaker as:"""
        prompt_string += """
                              
<relationship>
{{$relationship}}
</relationship>
                              
"""

        prompt_string += """Your internal dialog up to this point has been:""" if self is from_actor else """Your dialog with the speaker up to this point has been:"""
        prompt_string += """
<dialog>
{{$dialog}}
</dialog>

"""

        prompt_string += """generate a next thought in the internal dialog below:""" if self is from_actor else """generate a response to the statement below in the dialog:"""
        prompt_string += """
<statement>
{{$statement}}
</statement>
"""

        prompt_string += "" if self is from_actor else """Your first step in composing your response is to decide if you agree with the speaker. 
Even if speaker is a close friend, you may disagree with them, and that is fine.
Use how you think of speaker as a key determinant in composing your response. 
Do you believe that speaker's drives and goals align with yours? Do you believe speaker's statement is sincere?
What is speaker's emotional state? Would they enjoy humor at this point, or should your response be cautious and measured?
"""
        prompt_string += """Use the following XML template in your response:

<response>response / next thought</response>
<reason>terse (4-6 words) reason for this response / thought</reason>

{{$duplicative_insert}}

Guidance: 
- Use the appropriate pronoun (he, she, him, her) according to declared gender of each character. {{$target}} gender is {{$target_gender}).
    Actor genders:
    {{$actor_genders}}
- The response can occasionally include occasional body language or facial expressions as well as speech
- Respond in a way that advances the dialog. E.g., express an opinion or propose a next step. Don't hesitate to disagree with the speaker if consistent with the character's personality and goals.
- Do not respond to a question with a question.
- If the intent is to agree, state agreement without repeating the statement.
- Above all, speak in your own voice. Do not echo the speech style of the statement. 
- Your emotional state should have maximum impact on the tone, phrasing, and content of the response.
- Respond in the style of natural spoken dialog. Use short sentences and casual language, but avoid repeating stereotypical phrases in the dialog to this point.
{{$scene_post_narrative}}
 
Respond only with the above XML
Do not include any additional text. 
End your response with:
</end>
"""

        prompt = [SystemMessage(content=system_prompt), UserMessage(content=prompt_string)]

        if self.focus_goal:
            mapped_goals = self.map_goals(self.focus_goal)
        else:
            mapped_goals = self.map_goals()
        activity = ''
        if self.focus_task.peek() != None and self.focus_task.peek().name.startswith('dialog'):
            activity = f'You are currently actively engaged in {self.focus_task.peek().name}'
        elif self.focus_task.peek() != None and self.focus_task.peek().name.startswith('internal dialog'):
            activity = f'You are currently actively engaged in an internal dialog'
        # Get recent memories
        recent_memories = self.structured_memory.get_recent(10)
        memory_text = '\n'.join(memory.to_string() for memory in recent_memories)
        
        #print("Hear",end=' ')
        duplicative_insert = ''
        trying = 0

        emotionalState = self.emotionalStance
        if self.context.scene_post_narrative:
            scene_post_narrative = f"- Most importantly, the dominant theme of the scene is:\n {self.context.scene_post_narrative}\n  The response must be consistent with this theme."
        else:
            scene_post_narrative = ''

        actor_genders = ''
        for actor in self.context.actors+self.context.extras:
            actor_genders += f'\t{actor.name}: {actor.gender}\n'

        answer_xml = self.llm.ask({
            'character': self.get_character_description(),
            'emotionalState': emotionalState.to_definition(),
            'statement': f'{from_actor.name} says {message}' if self is not from_actor else message,
            "situation": self.context.current_state,
            "name": self.name,
            "goals": mapped_goals,
            "memories": memory_text,  # Updated from 'memory'
            "activity": activity,
            'history': self.narrative.get_summary('medium'),
            'dialog': from_actor.actor_models.get_actor_model(self.name).dialog.get_transcript(max_turns=40),
            'target': 'Your' if self is from_actor else "The person you are speaking to's ",
            'target_gender': from_actor.gender,
            'your_gender': self.gender,
            'actor_genders': actor_genders,
            'relationship': self.actor_models.get_actor_model(from_actor.name).get_relationship(),
            'duplicative_insert': duplicative_insert,
            "scene_post_narrative": scene_post_narrative,
            "central_narrative": self.context.central_narrative if self.context.central_narrative else '',
            "act_specific_narrative": self.context.act_central_narrative if self.context.act_central_narrative else '',
            "scene": json.dumps(self.context.current_scene, indent=2, default=datetime_handler) if self.context.current_scene else ''
            }, prompt, tag='generate_dialog_turn', temp=0.8, stops=['</end>'], max_tokens=300)
        response = xml.find('<response>', answer_xml)
        if response is None:
            print(f'No response to hear')
            self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
            return '', source
 
        reason = xml.find('<reason>', answer_xml)
        if not source:
            response_source = Task('dialog with '+from_actor.name, 
                                   description='dialog with '+from_actor.name, 
                                   reason='dialog with '+from_actor.name, 
                                   termination='natural end of dialog', 
                                   goal=None,
                                   actors=[self, from_actor])
        else:
            response_source = source
            # self.show = response
            if from_actor.name == 'Viewer':
                self.context.message_queue.put({'name':self.name, 'text':f"'{response}'", 'chat_response': True})
                return response, response_source

        # Create action for response
        return response, response_source
    
    def format_tasks(self, tasks, labels):
        task_list = []
        for label, task in zip(labels, tasks):
            task_dscp = f'{label} - {hash_utils.find("name", task)} ({hash_utils.find("description", task)}), {hash_utils.find("reason", task)}.' 
            task_dscp += f'needs: {hash_utils.find("needs", task)}. committed: {hash_utils.find("committed", task)}'
            task_list.append(task_dscp)
        return '\n'.join(task_list)
    

    async def admissible_goals(self, goals):
        """test if any of the goals meet preconditions"""
        admissible_goals = []
        impossible_goals = []
        for goal in goals.copy():
            # test, goal may have been cleared by completion of another goal
            sat = await self.clear_goal_if_satisfied(goal)
            if not goal in self.goals:
                continue
            if goal.preconditions:
                prompt = [UserMessage(content="""Given the following goal and current situation, determine if the goal preconditions are weakly satisfied.
In this determination, assume a condition is satisfied unless there is evidence to the contrary.
If a precondition specifies an action, eg, 'wake up asap', then the precondition is satisfied if the current situation includes the action consequences, e.g. the actor is awake.
If the goal preconditions are weakly satisfied, respond with <admissible>True</admissible>, <impossible>False</impossible>.
If the goal preconditions are not weakly satisfied, but it is possible that they will be satisfied in the future, respond with <admissible>False</admissible>, <impossible>False</impossible>.
If the goal preconditions is impossible to satisfy(e.g., a time that has already passed), respond with <admissible>False</admissible>, <impossible>True</impossible>.

<goal>
{{$goal}}
</goal>

<surroundings>
{{$surroundings}}
</surroundings>

<recent_memories>
{{$recent_memories}}
</recent_memories>

<situation>
{{$situation}}
</situation>    

The following tasks and goals have been completed or achieved:

{{$achievments}}

<time>
{{$time}}
</time>

Respond in this XMLformat:
                                      
<admissible>True/False</admissible>
<impossible>False/True</impossible>

Only respond with the above XML
Do not include any additional text. 
End your response with:
</end>
""")]

                response = self.llm.ask({'goal': goal.to_string(), 
                                         'surroundings': self.look_percept,
                                         'achievments': '\n'.join(self.achievments[:5]),
                                         'recent_memories': '\n'.join([memory.to_string() for memory in self.structured_memory.get_recent(8)]), 
                                         'situation': self.context.current_state, 
                                         'time': self.context.simulation_time}, 
                                         prompt, tag='admissible_goals', temp=0.8, stops=['</end>'], max_tokens=30, log=True)
                if response:
                    admissible = xml.find('admissible', response)
                    impossible = xml.find('impossible', response)
                    if admissible.lower() == 'true' and impossible.lower() == 'false':
                        admissible_goals.append(goal)
                    elif admissible.lower() == 'false' and impossible.lower() == 'true':
                        logger.debug(f'{self.name} admissible_goals: goal {goal.name} is impossible')
                        impossible_goals.append(goal)
                        goal.preconditions = None # let it thru next time - maybe this should be llm to soften preconditions?
                        admissible_goals.append(goal)
                    else:
                        logger.debug(f'{self.name} admissible_goals: goal {goal.name} is not admissible')
            else:
                admissible_goals.append(goal)
        return admissible_goals, impossible_goals


    async def request_goal_choice(self, goals, narrative=False):
        """Request a goal choice from the UI"""
        if self.autonomy.goal:
            if narrative:
                self.focus_goal = choice.exp_weighted_choice(goals, 0.5)
                return self.focus_goal
            else:
                admissible_goals, impossible_goals = await self.admissible_goals(goals)
                if len(admissible_goals) == 0:
                    for goal in goals.copy():
                        if goal.preconditions:
                            subgoal = Goal(name='preconditions', actors=[self], description=goal.preconditions, preconditions=None, 
                                        termination=goal.preconditions, signalCluster=goal.signalCluster, drives=goal.drives)
                            self.goals.append(subgoal)
                            admissible_goals.append(subgoal)

            if len(admissible_goals) > 0:
                self.focus_goal = choice.exp_weighted_choice(admissible_goals, 0.5)
            else:
                self.focus_goal = None
            return self.focus_goal

        else:
            if len(self.goals) > 0: # debugging
                # Send choice request to UI
                choice_request = {
                    'text': 'goal_choice',
                    'character_name': self.name,
                    'options': [{
                    'id': i,
                    'name': goal.name,
                    'description': goal.description,
                    'termination': goal.termination,
                    'context': {
                        'signal_cluster': goal.signalCluster.to_string(),
                        'emotional_stance': {
                            'arousal': str(goal.signalCluster.emotional_stance.arousal.value),
                            'tone': str(goal.signalCluster.emotional_stance.tone.value),
                            'orientation': str(goal.signalCluster.emotional_stance.orientation.value)
                        }
                    }
                    } for i, goal in enumerate(goals)]
                }
                # Drain any old responses from the queue
                while not self.context.choice_response.empty():
                    _ = self.context.choice_response.get_nowait()
                
                # Send choice request to UI
                self.context.message_queue.put(choice_request)
                await asyncio.sleep(0)
            
                # Wait for response with timeout                                 "all_tasks":'\n'.join(task.name for task in self.focus_task.stack),

                waited = 0
                while waited < 600.0:
                    await asyncio.sleep(0.1)
                    waited += 0.1
                    if not self.context.choice_response.empty():
                        try:
                            response = self.context.choice_response.get_nowait()
                            if response and response.get('custom_data'):
                                print(f'{self.name} request_goal_choice: custom data {response["custom_data"]}')
                                custom_data = response['custom_data']
                                if custom_data.get('name') and custom_data.get('description') and custom_data.get('actors') and custom_data.get('termination'):
                                    actors = [self.actor_models.resolve_character(actor_name)[0] for actor_name in custom_data['actors']]
                                    actors = [actor for actor in actors if actor] # strip actors that could not be found
                                    self.driveSignalManager.recluster()
                                    self.focus_goal = Goal(name=custom_data['name'], description=custom_data['description'], 
                                                           actors=actors, termination=custom_data['termination'], signalCluster=None, 
                                                           drives=[], preconditions='')
                                return self.focus_goal
                            elif response and response.get('selected_id') is not None:
                                self.focus_goal = goals[response['selected_id']]
                                return self.focus_goal
                        except Exception as e:
                            print(f'{self.name} request_goal_choice error: {e}')
                            break
            
                # If we get here, either timed out or had an error
                self.focus_goal = choice.exp_weighted_choice(goals, 0.9)
                return self.focus_goal
            else:
                self.focus_goal = None
                return None

    async def request_task_choice(self, tasks):
        """Request an act choice from the UI"""
        if self.autonomy.task:
            if self.focus_goal and self.focus_goal.task_plan and len(self.focus_goal.task_plan) > 0:
                self.focus_task.push(self.focus_goal.task_plan[0])
                self.focus_goal.task_plan.pop(0)
                return self.focus_task.peek()
            elif len(tasks) > 0:
                self.focus_task.push(choice.exp_weighted_choice(tasks, 0.67))
                return self.focus_task.peek()
            else:
                return None
        else:
            choice_request = {
                    'text': 'task_choice',
                    'character_name': self.name,
                    'options': [{
                        'id': i,
                        'name': task.name,
                        'description': task.description,
                        'reason': task.reason,
                        'termination': task.termination,
                        'context': {
                            'signal_cluster': task.goal.signalCluster.to_string() if task.goal and task.goal.signalCluster else '',
                            'emotional_stance': {
                                'arousal': str(task.goal.signalCluster.emotional_stance.arousal.value) if task.goal and task.goal.signalCluster else '',
                                'tone': str(task.goal.signalCluster.emotional_stance.tone.value) if task.goal and task.goal.signalCluster else '',
                                'orientation': str(task.goal.signalCluster.emotional_stance.orientation.value) if task.goal and task.goal.signalCluster else ''
                            }
                        }
                    } for i, task in enumerate(tasks)]
                }
            # Drain any old responses from the queue
            while not self.context.choice_response.empty():
                 _ = self.context.choice_response.get_nowait()
                
            # Send choice request to UI
            self.context.message_queue.put(choice_request)
            
            # Wait for response with timeout
            waited = 0
            while waited < 600.0:
                await asyncio.sleep(0.1)
                waited += 0.1
                if not self.context.choice_response.empty():
                    try:
                        response = self.context.choice_response.get_nowait()
                        if response and response.get('custom_data'):
                            print(f'{self.name} request_task_choice: custom data {response["custom_data"]}')
                            custom_data = response['custom_data']
                            if custom_data.get('name') and custom_data.get('description') and custom_data.get('actors') and custom_data.get('reason') and custom_data.get('termination'):
                                actors = [self.actor_models.resolve_character(actor_name)[0] for actor_name in custom_data['actors']]
                                actors = [actor for actor in actors if actor] # strip actors that could not be found
                                task = Task(name=custom_data['name'], 
                                             description=custom_data['description'],
                                             reason=custom_data['reason'],
                                             termination=custom_data['termination'],
                                             goal=self.focus_goal,
                                             actors=actors,
                                             start_time=self.context.simulation_time,
                                             duration=1)
                                self.focus_task.push(task)
                                return task
                        elif response and response.get('selected_id') is not None:
                            focus_task = tasks[response['selected_id']]
                            if self.focus_task.peek() != focus_task:
                                self.focus_task.push(focus_task)
                            return self.focus_task.peek()
                    except Exception as e:
                        print(f'{self.name} request_task_choice error: {e}')
                        break
            
            # If we get here, either timed out or had an error
            self.focus_task.push(choice.exp_weighted_choice(tasks, 0.67))
            return self.focus_task.peek()


    def ActWeight(self, count, n, act):
        """Weight for act choice using exponential decay"""
        base = 0.75  # Controls how quickly weights decay
        raw = pow(base, n)  # Exponential decay
        if act.mode == 'Think':
            return raw * 0.3
        if act.mode == 'Say' and self.check_if_in_dialog_subtask():
            return raw * 0.4
        return raw

    def get_action_target(self, action):
        if action.target and isinstance(action.target, list) and len(action.target) > 0:
            if action.target[0] and isinstance(action.target[0], Character):
                return action.target[0].name
            elif action.target[0] and isinstance(action.target[0], str):
                return action.target[0]
        elif action.target and isinstance(action.target, Character):
            return action.target.name
        else:
            return ''

    async def request_action_choice(self, actions):
        """Request an act choice from the UI"""
        if self.autonomy.action:
            #if len(acts) > 0 and acts[0].mode != 'Think':
            #    self.focus_action = acts[0]
            #    return self.focus_action
            self.focus_action = choice.pick_weighted(list(zip(actions, [self.ActWeight(len(actions), n, action) for n, action in enumerate(actions)])))[0]
            if not self.focus_action and len(actions) > 0:
                self.focus_action = actions[0]
            elif not self.focus_action:
                print(f'{self.name} request_action_choice: no act selected')
                return
            return self.focus_action
        
        if len(actions) < 2:
                print(f'{self.name} request_action_choice: not enough acts')
        for action in actions:
            if not action.source or not action.source.goal or not action.source.goal.signalCluster:
                print(f'{self.name} request_action_choice: act {action.mode} {action.action} has no source or goal or signal cluster')
        if len(actions) > 0: # debugging
                # Send choice request to UI
            choice_request = {
                'text': 'action_choice',
                'character_name': self.name,
                'options': [{
                    'id': i,
                    'mode': action.mode,
                    'action': action.action,
                    'reason': action.reason,
                    'duration': action.duration,
                    'target': self.get_action_target(action),
                    } for i, action in enumerate(actions)]
            }
            # Drain any old responses from the queue
            while not self.context.choice_response.empty():
                _ = self.context.choice_response.get_nowait()

            # Send choice request to UI
            self.context.message_queue.put(choice_request)
            
            # Wait for response with timeout
            waited = 0
            while waited < 600.0:
                await asyncio.sleep(0.1)
                waited += 0.1
                if not self.context.choice_response.empty():
                    try:
                        response = self.context.choice_response.get_nowait()
                        if response and response.get('selected_id') is not None:
                            if response.get('selected_id') == 'custom' and response.get('custom_data'):
                                # Handle custom act
                                custom_data = response['custom_data']
                                mode = custom_data['mode']
                                if mode:
                                    mode = mode.strip().capitalize()
                                    try:
                                        mode_enum = Mode[mode]  # This validates against the enum
                                        mode = mode_enum.value  # Get standardized string value
                                    except KeyError:
                                        self.focus_action = actions[0]
                                        return self.focus_action

                                actors = [self.actor_models.resolve_character(actor_name)[0] for actor_name in custom_data['actors']]
                                actors = [actor for actor in actors if actor] # strip actors that could not be found
                                target = self.actor_models.resolve_character(custom_data['target'])[0]
                                custom_act = Act(
                                        mode=mode,
                                        action=custom_data['action'],
                                        actors=actors,
                                        reason=custom_data.get('reason', ''),
                                        duration=custom_data.get('duration', 1),
                                        source=self.focus_task.peek(),
                                        target=[target]
                                    )   
                                print(f'{self.name} request_action_choice: custom act {custom_act.mode} {custom_act.action}')
                                self.focus_action = custom_act
                                return self.focus_action
                            self.focus_action = actions[response['selected_id']]
                            return self.focus_action    
                    except Exception as e:
                        print(f'{self.name} request_action_choice error: {e}')
                        break
            
            # If we get here, either timed out or had an error
            self.focus_action = choice.pick_weighted(list(zip(actions, [self.ActWeight(len(actions), n, action) for n, action in enumerate(actions)])))[0]
            return self.focus_action


    def check_if_in_dialog_subtask(self):
        if self.focus_task.peek() and not self.focus_task.peek().name.startswith('dialog with '):
            # current task is not a dialog
            for task in self.focus_task.stack:
                if task.name.startswith('dialog with '):
                    # but there IS a dialog on the stack
                    return True
        return False

    async def cognitive_cycle(self, sense_data='', narrative=False, ui_queue=None):
        """Perform a complete cognitive cycle"""
        print(f'{self.name} cognitive_cycle')
        logger.info(f'{self.name} cognitive_cycle')
        if not narrative:
            self.context.message_queue.put({'name':'\n\n'+self.name, 'text':f'-----cognitive cycle----- {self.context.simulation_time.isoformat()}'})    
        await asyncio.sleep(0)
        self.thought = ''
        if self.focus_goal: 
            pass #self.reason_over(self.focus_goal.name+'. '+self.focus_goal.description)

        self.memory_consolidator.update_cognitive_model(self.structured_memory, 
                                                  self.narrative, 
                                                  self.actor_models,
                                                  self.context.simulation_time, 
                                                  self.get_character_description().strip(),
                                                  relationsOnly=self.step % self.update_interval != 0 )
        
        # Push Act and Scene information if autonomy is enabled for display
        if hasattr(self, 'autonomy'):
            if self.autonomy.act or self.autonomy.scene:
                # Push current act and scene info to UI for display
                self.context.message_queue.put({
                    'name': self.name, 
                    'text': 'character_update', 
                    'data': self.to_json()
                })
                await asyncio.sleep(0.1)

        if not narrative:
            if self.focus_goal:
                goal = self.focus_goal
                satisfied = await self.clear_goal_if_satisfied(self.focus_goal)
                if satisfied:
                    self.focus_goal_history.append(goal)
                    await self.generate_goals(previous_goal=goal)
                    await self.request_goal_choice(self.goals)

            if not self.focus_goal and (not self.goals or len(self.goals) == 0):
                await self.generate_goals()
                await self.request_goal_choice(self.goals)
                for goal in self.goals.copy():
                    if goal is self.focus_goal:
                        pass
                    else:
                        self.goals.remove(goal) # once we've selected a goal, remove all other goals -why?
        
            if not self.focus_goal and self.goals and len(self.goals) > 0:
                await self.request_goal_choice(self.goals)
                if not self.focus_goal: # if we don't have a goal
                    return

            if self.focus_goal and (not self.focus_goal.task_plan or (len(self.focus_goal.task_plan) == 0)):
                await self.generate_task_plan(self.focus_goal)

        else: #narrative  
            if not self.focus_goal:
                return

        logger.debug(f'{self.name} cognitive_cycle: focus goal {self.focus_goal.name} task plan {self.focus_goal.task_plan}')

        await self.request_task_choice(self.focus_goal.task_plan)
        if not self.focus_task.peek(): # prev task completed, proceed to next task in plan
            print(f'{self.name} narrative cognitive_cycle: no focus task')
            if self.focus_goal:
                self.focus_goal_history.append(self.focus_goal)
                self.focus_goal = None
            return
            
        # input - a task on top of self.focus_task, the next task to run for self.focus_goal
        task_to_run = self.focus_task.peek()
        await self.step_tasks()

        delay = 0.0
        #delay = await self.context.choose_delay()
        self.context.message_queue.put({'name':self.name, 'text':f'character_update', 'data':self.to_json()})
        self.context.message_queue.put({'name':self.name, 'text':f'character_detail', 'data':self.get_explorer_state()})
        try:
            old_time = self.context.simulation_time
            self.context.simulation_time += timedelta(hours=delay)
            if old_time.day!= self.context.simulation_time.day:
                pass # need to figure out out to force UI to no sho old image
            if delay > 15.0:
                self.context.update(changes_only=True)
                self.context.message_queue.put({'name':self.name, 'text':f'world_update', 'data':self.context.to_json()})
                await asyncio.sleep(0)
        except Exception as e:
            print(f'{self.name} cognitive_cycle error: {e}')

        if not narrative:
            goal = self.focus_goal
            satisfied = await self.clear_goal_if_satisfied(self.focus_goal)
            if satisfied:
                await self.generate_goals(previous_goal=goal)
                await self.request_goal_choice(self.goals)
        return

       
    async def step_tasks(self, n: int=2):
        #runs through up to n tasks in a task_plan
        # like step_task, the invariant is that the task stack is empty on return, and the first task in focus_goal task_plan is the next task to run, if it exists
        task_count = 0
        logger.info(f'{self.name} step_tasks: {self.focus_task.peek()}')
        while task_count < n and self.focus_task.peek():
    
            task = self.focus_task.peek()
            task_to_run = task
            outcome = await self.step_task(task)
            task_count += 1

            if task == self.focus_task.peek():
                print(f'{self.name} step_tasks: task {task.name} still on focus stack!')
                self.focus_task.pop()
            # outcome is False if task is not completed. if true proceed to next task
            if not outcome or not self.focus_goal:
                return False
            
            if self.focus_goal and not self.focus_goal.task_plan:
                if self.focus_goal.name == 'preconditions':
                    # ok, we ran preconditions goal, get rid of it and clear preconditions clause from source goal, make it focus_goal.
                    base_goal = None
                    for goal2 in self.goals:
                        if goal2.preconditions == self.focus_goal.termination:
                            goal2.preconditions = None # remove preconditions, we've already satisfied them
                            base_goal = goal2
                    if self.focus_goal in self.goals:
                        self.goals.remove(self.focus_goal)
                    if base_goal:
                        self.focus_goal = base_goal
                        return False

                elif self.focus_goal.name == 'postconditions':
                    # ok, we ran postconditions goal, get rid of it
                    satisfied = await self.clear_goal_if_satisfied(self.focus_goal) # do all the end of goal stuff for source goal
                    if not satisfied:
                        logger.debug(f'{self.name} step_tasks: assertion error: postconditions goal {self.focus_goal.name} not satisfied')
                    if self.focus_goal and self.focus_goal in self.goals:
                        self.goals.remove(self.focus_goal)
                        self.focus_goal_history.append(self.focus_goal)
                        self.focus_goal = None
                        return True
                elif self.focus_goal and self.focus_goal.termination and self.focus_goal.name != 'preconditions' and self.focus_goal.name != 'postconditions':
                    # termination not yet satisfied, and this is a main goal (has termination), subgoal on termination clause
                    old_goal = self.focus_goal
                    satisfied = await self.clear_goal_if_satisfied(self.focus_goal)
                    if not satisfied: # goal not satisfied and no remaining tasks
                        if not self.context.narrative: # extend task plan
                            await self.generate_task_plan(self.focus_goal)
                            await self.request_task_choice(self.focus_goal.task_plan)
                    elif old_goal in self.goals: #satisfied.
                        self.goals.remove(old_goal)
                        if old_goal == self.focus_goal:
                            self.focus_goal = None
                    return True
                else:
                    logger.debug(f'{self.name} step_tasks: assertion error: focus goal {self.focus_goal.name} not handled')
                    return False
            if task_count < n-1: # if we're not at the last task, push the next task onto the stack
                    self.focus_task.push(self.focus_goal.task_plan[0])
                    self.focus_goal.task_plan.pop(0)

            if self.focus_task.peek() == task_to_run:
                self.focus_task.pop()
                print(f'{self.name} step_tasks: task {task_to_run.name} still on focus stack, invariant violated')
            #if self.focus_goal:
            #    self.clear_goal_if_satisfied(self.focus_goal)
        return 
            
        
    async def step_task(self, sense_data='', ui_queue=None):
        """Perform a single task from the current goal task_plan. 
           Task is popped from goal plan and focus_task if completed.
           Returns False if task is not completed.
           However, current code always pretends task is completed. (outcome is always True, task is popped)"""
        # if I have an active task, keep on with it.
        if not self.focus_task.peek():
            raise Exception(f'No focus task')

        print(f'\n{self.name} decides, focus task {self.focus_task.peek().name}')
        logger.info(f'{self.name} step_task: focus task {self.focus_task.peek().name}')
        task: Task = self.focus_task.peek()
        # iterate over task until it is no longer the focus task. 
        # This is to allow for multiple acts on the same task, clear_task_if_satisfied will stop the loop with poisson timeout or completion
        act_count=0
        subtask_count = 0
        while act_count < 2 and self.focus_task.peek():
            #self.look(task.name+': '+task.description)
            action_alternatives: List[Act] = self.generate_acts(task)
            await self.request_action_choice(action_alternatives) # sets self.focus_action
            self.context.message_queue.put({'name':self.name, 'text':f'character_update', 'data':self.to_json()})
            print(f'selected {self.name} {self.focus_action.mode} {self.focus_action.action} for task {task.name}')
            await asyncio.sleep(0)

            await self.act_on_action(self.focus_action, self.focus_task.peek())
            
            # Record turn for staleness detection
            if hasattr(self.context, 'staleness_detector'):
                # Also record turns for other affected actors
                affected_actors = set([self])
                if hasattr(self.focus_action, 'actors') and self.focus_action.actors:
                    affected_actors.update(self.focus_action.actors)
                if hasattr(self.focus_action, 'target') and self.focus_action.target:
                    if isinstance(self.focus_action.target, (list, tuple)):
                        affected_actors.update(self.focus_action.target)
                    else:
                        affected_actors.add(self.focus_action.target)
                
                for actor in affected_actors:
                    if hasattr(actor, 'name'):  # Ensure it's a valid character
                        self.context.staleness_detector.record_turn(actor)
            act_count += 1
 
               
            # Did we push opportunistic subtasks? This should be obv
            while task in self.focus_task.stack and self.focus_task.peek() != task and subtask_count < 1: # only allow 1 subtasks
                # just recursively call step_task here?
                subtask = self.focus_task.peek()
                #self.look(task.name+': '+task.description)
                action_alternatives: List[Act] = self.generate_acts(task)
                await self.request_action_choice(action_alternatives) # sets self.focus_action
                self.context.message_queue.put({'name':self.name, 'text':f'character_update', 'data':self.to_json()})
                print(f'selected {self.name} {self.focus_action.mode} {self.focus_action.action} for task {task.name}')
                await asyncio.sleep(0)
                await self.act_on_action(self.focus_action, task)
                
                # Record turn for staleness detection
                if hasattr(self.context, 'staleness_detector'):
                    affected_actors = set([self])
                    if hasattr(self.focus_action, 'actors') and self.focus_action.actors:
                        affected_actors.update(self.focus_action.actors)
                    if hasattr(self.focus_action, 'target') and self.focus_action.target:
                        if isinstance(self.focus_action.target, (list, tuple)):
                            affected_actors.update(self.focus_action.target)
                        else:
                            affected_actors.add(self.focus_action.target)
                    
                    for actor in affected_actors:
                        if actor != self and hasattr(actor, 'name'):  # Ensure it's a valid character
                            self.context.staleness_detector.record_turn(actor)
                    
                subtask_count += 1
                while task in self.focus_task.stack and subtask in self.focus_task.stack: 
                    # safeguard against popping the wrong task, but subtasks aren't allowed to spawn further subtasks
                    self.focus_task.pop()
            
            while task in self.focus_task.stack and self.focus_task.peek() != task: 
                # clear any remaining subtasks
                self.focus_task.pop()

            if self.focus_task.peek() is task: #are we on main task?
                await self.clear_task_if_satisfied(task) # will pop focus_task if task is done
                await asyncio.sleep(0)
                if self.focus_task.peek() != task: # task completed
                    return True
            else:
                print(f'{self.name} step_task: task {task.name} not found in focus_task.stack')
            return True
            
        if self.focus_task.peek() is task:
            return False
            #self.focus_task.pop() # didn't get done by turn limit pretend task is done            return True # probably need to replan around step failure, but we're not going to do that here
        else: # clear assigned task and all intensions created by it, out of turns for this task
            while task in self.focus_task.stack:
                self.focus_task.pop()
        return True # probably need to replan around step failure, maybe we pushed an intension and exhausted our turns, who knows?


    async def act_on_action(self, action: Act, task: Task):
        if task and task.goal:
            for drive in task.goal.drives:
                if task.goal not in drive.attempted_goals:
                    drive.attempted_goals.append(task.goal)
        self.act = action
        self.last_act = action
        if task:
            task.acts.append(action)
        act_mode = action.mode
        if act_mode is not None:
            self.act_mode = act_mode.strip()
        act_arg = action.action
        print(f'\n{self.name} act_on_action: {act_mode} {act_arg}')
        logger.info(f'{self.name} act_on_action: {act_mode} {act_arg}')
        self.reason = action.reason
        source = action.source
        #print(f'{self.name} choose {action}')
        target = None
        target_name = None
        #responses, at least, explicitly name target of speech.
        if action.target and (action.target[0] != self or (self.focus_task.peek() and self.focus_task.peek().name.startswith('dialog with '+self.name))):
            if action.target[0] and isinstance(action.target[0], Character):
                target_name = action.target[0].name
                target = action.target[0]
        elif self.focus_task.peek() and self.focus_task.peek().name.startswith('internal dialog'):
            target_name = self.name
            target = self
        elif act_mode == 'Say':
            if target_name != None and target is None:
                target = self.context.resolve_character(target_name)[0]
        #self.context.message_queue.put({'name':self.name, 'text':f'character_update'})
        #await asyncio.sleep(0.1)
        await self.acts(action, target, act_mode, act_arg, self.reason, source)
        logger.debug(f'{self.name} act_on_action: {act_mode} {act_arg} {self.reason} {source} {target}')
        if hasattr(action, 'duration'):
            act_duration = action.duration
        elif action.mode == 'Think': 
            act_duration = timedelta(seconds=15)
        elif action.mode == 'Say': 
            act_duration = timedelta(minutes=2)
        else:
            act_duration = timedelta(minutes=10)
        self.context.simulation_time += act_duration

        return

    def format_thought_for_UI (self):
        #<action> <mode>{mode}</mode> <act>{action}</act> <reason>{reason}</reason> <source>{source}</source></action>'
        action = ''
        if self.act:
            action = f'{self.act.mode}: {self.act.action}'
        reason = ''
        if self.focus_task.peek() and type(self.focus_task.peek()) == Task:
            reason = f'{self.focus_task.peek().reason}'
        elif self.last_task:
            reason = f'{self.last_task.reason}'
        return f'{self.thought.strip()}'
   
    def get_focus_goal_string(self):
        if self.focus_goal:
            return self.focus_goal.short_string()
        elif self.focus_goal_history and len(self.focus_goal_history) > 0:
            return self.focus_goal_history[-1].short_string()
        else:
            return ''
        
    def format_tasks_for_UI(self):
        """Format tasks for UI display as array of strings"""
        tasks = []
        if self.focus_goal:
            tasks.append(f"Goal: {self.get_focus_goal_string()}")
        if self.focus_task.peek():
            tasks.append(f"Task: {self.focus_task.peek().short_string()}")
        elif self.last_task:
            tasks.append(f"Last Task: {self.last_task.short_string()}")
        return tasks
    
    #def format_history(self):
    #    return '\n'.join([xml.find('<text>', memory) for memory in self.history])

    def to_json(self, gen_image=False):
        """Return JSON-serializable representation"""
        image_data = None
        if gen_image:
            description = self.generate_image_description()
            if description:
                image_path = generate_image(self.llm, description, filepath=self.name+'.png')
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()

        data = {
            'name': self.name,
            'show': self.show.strip(),  # Current visible state
            'thoughts': self.format_thought_for_UI(),  # Current thoughts
            'tasks': self.format_tasks_for_UI(),
            'description': self.get_character_description().strip(),  # For image generation
            'history': self.format_history_for_UI().strip(), # Recent history, limited to last 5 entries
            'narrative': {
                'recent_events': self.narrative.recent_events,
                'ongoing_activities': self.narrative.ongoing_activities,
                'relationships': self.actor_models.get_known_relationships(),
            },
            'signals': self.focus_goal.name if self.focus_goal else '',
            'current_act': self.get_current_act_info(),
            'current_scene': self.get_current_scene_info()
        }
        if image_data:
            data['image'] = image_data
        return data

    def find_say_result(self) -> str:
        """Find if recent say action influenced other actors"""
        # Get recent percepts in reverse chronological order (most recent first)
        recent_percepts = self.perceptual_state.get_current_percepts()
        
        # Find the most recent say by self and collect all percepts after it
        say_and_after = []
        found_say = False
        # Use reversed() to iterate from oldest to most recent
        for percept in reversed(recent_percepts):
            if not found_say:
                # Look specifically for self saying something
                if (percept.content is not None and percept.content.lower().strip().startswith('you say:')):  # Matches format from perceptual input
                    found_say = True
                    say_and_after = []
            if found_say:
                say_and_after.append(percept.content)
                
        if not say_and_after:
            return ""
            
        # Call LLM to analyze influence (stub for now)
        #print("Find Say Result",end=' ')
        prompt = [UserMessage(content=f"""
Analyze if these percepts show influence from the say action:
 - {'\n - '.join(say_and_after)}

Respond with an extremely terse statement of the effect of the say action.
Do not include any other text.
end your response with:
</end>
""")]
        
        result = self.llm.ask({}, prompt, tag='find_say_result', max_tokens=100, stops=['</end>'])
        return result or ""

    def get_explorer_state(self):
        """Return detailed state for explorer UI"""
        focus_task = self.focus_task.peek()
        last_act = self.action_history[-1] if self.action_history and len(self.action_history) > 0 else None

        return {
            'character': self.character,
            'currentTask': focus_task.to_string() if focus_task else 'idle',
            'actions': [act.to_string() for act in (focus_task.acts if focus_task else [])],
            'lastAction': {
                'mode': last_act.mode if last_act else '',
                'action': last_act.action if last_act else '',
                'result': last_act.result if last_act else '',
                'reason': last_act.reason if last_act else ''
            },
            'decisions': self.decisions or [],  # ADD THIS LINE
            'drives': [{'text': drive.text or '', 'activation': drive.activation} for drive in (self.drives or [])],
            'emotional_state': [
                {
                    'text': percept.content or '',
                    'mode': percept.mode.name if percept.mode else 'unknown',
                    'time': percept.timestamp.isoformat() if percept.timestamp else self.context.simulation_time.isoformat()
                }
                for percept in (self.perceptual_state.get_current_percepts(chronological=True) or [])
            ],
            'memories': [
                {
                    'text': memory.to_string() or '',
                    'timestamp': memory.timestamp.isoformat() if memory.timestamp else self.context.simulation_time.isoformat()
                } 
                for memory in (self.structured_memory.get_recent(8) or [])
            ],
            'narrative': {
                'recent_events': self.narrative.recent_events if self.narrative else '',
                'ongoing_activities': self.narrative.ongoing_activities if self.narrative else ''
            },
            'social': {
                'known_actors': [
                    {
                        'name': model.name or '',
                        'relationship': model.relationship or '',
                        'dialog': {
                            'active': bool(model.dialog.active) if model.dialog else False,
                            'transcript': model.dialog.get_transcript(10) or '' if model.dialog else ''
                        } if model.dialog else None
                    }
                    for model in (self.actor_models.known_actors.values() if self.actor_models else [])
                ]
            },
            'cognitive': {
                'goals': [
                    {
                        'name': goal.name,
                        'description': goal.description,
                        'preconditions': goal.preconditions,
                        'termination': goal.termination,
                        'progress': goal.progress,
                        'drives': [d.text for d in goal.drives]  # Send as array
                    }
                    for goal in self.goals
                ],
                'tasks': [  # Add empty list as default
                    {
                        'name': p.name,
                        'description': p.description,
                        'reason': p.reason,
                        'actors': [a.name for a in p.actors],
                    } for p in (self.focus_task.stack or [])
                ],
                'intensions': [intension.name for intension in self.intensions]  # List[str]
            },
            'signals': [
                {
                    'text': cluster_pair[0].text,
                    'drives': [d.text for d in cluster_pair[0].drives],  # Changed from single drive to list
                    'type': cluster_pair[0].type,
                    'signals': [s.text for s in cluster_pair[0].signals],
                    'score': cluster_pair[1],
                    'last_seen': cluster_pair[0].get_latest_timestamp().isoformat()
                }
                for cluster_pair in self.driveSignalManager.get_scored_clusters()
            ]
        }

