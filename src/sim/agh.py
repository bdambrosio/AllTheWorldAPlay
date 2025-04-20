from __future__ import annotations
import os, json, math, time, requests, sys

from sklearn import tree

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
from typing import List, Dict, Optional, TYPE_CHECKING
from sim.cognitive import knownActor
from sim.cognitive import perceptualState
from sim.cognitive.driveSignal import Drive, DriveSignalManager, SignalCluster
from sim.memory.consolidation import MemoryConsolidator
from sim.memory.core import MemoryEntry, NarrativeSummary, StructuredMemory
from sim.memory.core import MemoryRetrieval
from src.sim.cognitive.EmotionalStance import EmotionalStance
from utils import llm_api
from utils.Messages import UserMessage, SystemMessage
import utils.xml_utils as xml
from sim.memoryStream import MemoryStream
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

if TYPE_CHECKING:
    from sim.context import Context  # Only imported during type checking

class Mode(Enum):
    Think = "Think"
    Say = "Say"
    Do = "Do" 
    Move = "Move"
    Look = "Look"
    Listen = "Listen"



def find_first_digit(s):
    for char in s:
        if char.isdigit():
            return char
    return None  # Return None if no digit is found


class Stack:
    def __init__(self):
        """Simple stack implementation"""
        self.stack = []
        #print("Stack initialized")  # Debug print

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

class Goal:
    _id_counter = 0
    _instances = WeakValueDictionary()  # id -> instance mapping that won't prevent garbage collection
    
    def __init__(self, name, actors, description, preconditions, termination, signalCluster, drives):
        Goal._id_counter += 1
        self.id = f"g{Goal._id_counter}"
        Goal._instances[self.id] = self
        self.name = name
        self.actors = actors
        self.description = description
        self.termination = termination
        self.preconditions = preconditions
        self.drives = drives
        self.signalCluster = signalCluster
        self.progress = 0
        self.task_plan = []
        self.tasks = []

    def __eq__(self, other):
        if not isinstance(other, Goal):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
    
    @classmethod
    def get_by_id(cls, id: str):
        return cls._instances.get(id)
    
    def short_string(self):
        return f'{self.name}: {self.description}; \n preconditions: {self.preconditions}; \n termination: {self.termination}'
    
    def to_string(self):
        return f'Goal {self.name}: {self.description}; actors: {', '.join([actor.name for actor in self.actors])}; preconditions: {self.preconditions};  termination: {self.termination}'
    
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
        
class Task:
    _id_counter = 0
    _instances = WeakValueDictionary()  # id -> instance mapping that won't prevent garbage collection
    
    def __init__(self, name, description, reason, termination, goal, actors, start_time, duration):
        Task._id_counter += 1
        self.id = f"t{Task._id_counter}"
        Task._instances[self.id] = self
        self.name = name
        self.description = description
        self.reason = reason
        self.start_time = start_time
        self.duration = parse_duration(duration)
        self.termination = termination
        self.goal = goal
        self.actors = actors
        self.needs = ''
        self.result = ''
        self.acts = []

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
    
    @classmethod
    def get_by_id(cls, id: str):
        return cls._instances.get(id)
    
    def short_string(self):
        return f'{self.name}: {self.description} \n reason: {self.reason} \n termination: {self.termination}'

    def to_string(self):
        return f'Task {self.name}: {self.description}; actors: {[actor.name for actor in self.actors]}; reason: {self.reason}; termination: {self.termination}'
    
    def to_fullstring(self):
        return f'Task {self.name}: {self.description}\n   Reason: {self.reason}\n   Actors: {[actor.name for actor in self.actors]}\n    Start time: {self.start_time};  Duration: {self.duration}\n    Termination Criterion: {self.termination}'
 
    def test_termination(self, events=''):
        """Test if recent acts, events, or world update have satisfied termination"""
        pass
    
class Act:
    _id_counter = 0
    _instances = WeakValueDictionary()  # id -> instance mapping that won't prevent garbage collection
    
    def __init__(self, mode, action, actors, reason='', duration=1, source=None, target=None, result=''):
        Act._id_counter += 1
        self.id = f"a{Act._id_counter}"
        Act._instances[self.id] = self
        self.mode = mode
        self.action = action
        self.actors = actors
        self.reason = reason
        self.duration = parse_duration(duration)
        self.source = source  # a task
        self.target = target  # an actor
        self.result = result

    @classmethod
    def get_by_id(cls, id: str):
        return cls._instances.get(id)

    def to_string(self):
        return f'Act t{self.id} {self.mode}: {self.action}; reason: {self.reason}; result: {self.result}'

class Autonomy:
    def __init__(self, signal=True, goal=True, task=True, action=True):
        self.signal = signal
        self.goal = goal
        self.task = task
        self.action = action

# Character base class
class Character:
    def __init__(self, name, character_description, reference_description='', init_x=50, init_y=50, server_name='local', mapAgent=True):
        print(f"Initializing Character {name}")  # Debug print
        self.name = name.strip().capitalize()
        self.character = character_description
        self.llm = llm_api.LLM(server_name)
        if len(reference_description) > 0:
            self.reference_dscp = reference_description
        else:
            self.reference_dscp = self.llm.ask({}, [SystemMessage(content='generate a concise single sentence description for this character useful for reference resolution. Respond only with the description. End your response with: </end>'), 
                                                    UserMessage(content=f"""character name {self.name}\ncharacter description:\n{character_description}\n\nEnd your response with: </end>""")
        ], tag='reference_dscp', max_tokens=24, stops=["</end>"]).strip()
        self.context: Context = None

        self.show = ''  # to be displayed by main thread in UI public text widget
        self.reason = ''  # reason for action
        self.thought = ''  # thoughts - displayed in character thoughts window
        self.sense_input = ''
        self.widget = None
        
        # Initialize focus_task stack
        self.focus_task = Stack()
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
        self.structured_memory = StructuredMemory(owner=self)
        self.memory_consolidator = MemoryConsolidator(self, self.llm, self.context)
        self.memory_retrieval = MemoryRetrieval()
        self.new_memory_cnt = 0
        self.next_task = None  # Add this line
        self.driveSignalManager = DriveSignalManager(self, self.llm, ask=default_ask)
        self.driveSignalManager.set_context(self.context)
        self.focus_goal = None
        self.focus_task = Stack()
        self.last_task = None
        self.focus_action = None
        self.goals = []
        self.goal_history = []
        #self.tasks = [] 
        self.intensions = []
        self.actions = []
        self.achievments = [] # a list of short strings summarizing the character's achievments
        self.autonomy = Autonomy()
        self.step = 2 # start at step 2 to skip first update
        self.update_interval = 4

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
 

    def set_autonomy(self, autonomy_json):
        if 'signal' in autonomy_json:
            self.autonomy.signal = autonomy_json['signal']
        if 'goal' in autonomy_json:
            self.autonomy.goal = autonomy_json['goal']
        if 'task' in autonomy_json:
            self.autonomy.task = autonomy_json['task']
        if 'action' in autonomy_json:
            self.autonomy.action = autonomy_json['action']

    def get_character_description(self):
        description = self.character
        if self.actor_models.get_known_actor_model(self.name):
            description += '\n\n' + self.actor_models.get_known_actor_model(self.name)
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
        start_time = hash_utils.find('start_time', task_hash)
        duration = hash_utils.find('duration', task_hash)
        try:
            actor_names = hash_utils.find('actors', task_hash)
            if actor_names:
                actor_names = [name.strip().capitalize() for name in actor_names.split(',') if name.strip()]
            else:
                actor_names = []
        except Exception as e:
            print(f"Warning: invalid actors field in {task_hash}") 
            actor_names = []
        if self.mapAgent:
            default_x = self.mapAgent.x
            default_y = self.mapAgent.y
        else:
            default_x = 30
            default_y = 30
        for actor_name in actor_names:
            a,_ = self.actor_models.resolve_character(actor_name)
            if a and hasattr(a, 'mapAgent'):
                default_x = a.mapAgent.x
                default_y = a.mapAgent.y
            elif a:
                print(f"Warning: {a} is not a character")
        actors = [self.actor_models.resolve_or_create_character(actor_name)[0] for actor_name in actor_names if actor_name]
        actors = [actor for actor in actors if actor is not None]
        if not self in actors:
            actors =  [self] + actors

        if name and description and reason and termination and actor_names:
            task = Task(name, description, reason, termination.replace('##','').strip(), goal, actors, start_time, duration)
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
        try:
            target_name = hash_utils.find('target', hash_string)
            if target_name:
                target,_ = self.actor_models.resolve_or_create_character(target_name)
            else:
                target = None
        except Exception as e:
            target = None
        # Clean up mode string and validate against Mode enum
        if mode:
            mode = mode.strip().capitalize()
            try:
                mode_enum = Mode[mode]  # This validates against the enum
                mode = mode_enum.value  # Get standardized string value
            except KeyError:
                return None
        
        if mode and action:
            act = Act(
                mode=mode,
                action=action,
                actors=[self, target] if target else [self],
                reason=action,
                duration=duration,
                source=task,
                target=target
            )
            return act
        else:
            print(f"Invalid actionable Hash: {hash_string}")
            return None
        
    def format_history(self, n=2):
        """Get memory context including both concrete and abstract memories"""
        # Get recent concrete memories
        recent_memories = self.structured_memory.get_recent(n)
        memory_text = []
        
        if recent_memories:
            concrete_text = '\n'.join(f"- {memory.text}" for memory in recent_memories)
            memory_text.append("Recent Events:\n" + concrete_text)
        
        # Get current activity if any
        current = self.structured_memory.get_active_abstraction()
        if current:
            memory_text.append("Current Activity:\n" + current.summary)
        
        # Get recent completed activities
        recent_abstracts = self.structured_memory.get_recent_abstractions(2)
        if recent_abstracts:
            # Filter out current activity and format others
            completed = [mem for mem in recent_abstracts if not mem.is_active]
            if completed:
                abstract_text = '\n'.join(f"- {mem.summary}" for mem in completed)
                memory_text.append("Recent Activities:\n" + abstract_text)
        
        return '\n\n'.join(memory_text) if memory_text else ""


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

    def update(self):
        """Move time forward, update state"""
        # Consolidate memories if we have drives
        if hasattr(self, 'drives'):
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
        self.step = self.update_interval+1 # skip next update


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

    def look(self, interest='', height=5):
        """Get visual information about surroundings"""
        if self.mapAgent is None:
            return ''  
        obs = self.mapAgent.look()
        view = {}
        for dir in ['Current', 'North', 'Northeast', 'East', 'Southeast', 
                   'South', 'Southwest', 'West', 'Northwest']:
            dir_obs = map.extract_direction_info(obs, dir)
            view[dir] = dir_obs
        self.my_map[self.mapAgent.x][self.mapAgent.y] = view

        view_text, resources, characters, percept_summary = map.hash_direction_info(view)
        view_percept = self.add_perceptual_input(view_text, mode=SensoryMode.VISUAL)
 
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
        for actor in self.context.actors:
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
        for actor in self.context.actors:
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
            description = 'Portrait of '+'. '.join(self.character.split('.')[:2])[8:] # assumes character description starts with 'You are <name>'
            
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
            recent_memories = '\n'.join(memory.text for memory in recent_memories)
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
            if response:
                mode = response.strip().split()[0].strip()  # Split on any whitespace and take first word
            else:
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
        self.driveSignalManager.analyze_text(message, self.drives, input_time)
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
        return ' \n '.join(memory.text for memory in recent_memories)
    
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
        recent_memories = self.structured_memory.get_recent(3)  # Increased window
        history = '\n'.join(mem.text for mem in recent_memories)
        
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
        self.update()
        if hasattr(goal, 'drives'):
            for drive in goal.drives:
                drive.satisfied_goals.append(goal)
                # don't update the primary drive
                if drive is not self.drives[0]:
                    if len(drive.attempted_goals) > 0:
                        if drive in self.drives: # only update if drive still exists, may have already been rewritten
                            update = drive.update_on_goal_completion(self, goal)
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
            self.update()
            await self.context.update(local_only=True) # remove confusing obsolete data, task completion is a big event
 
        return satisfied

    async def clear_goal_if_satisfied(self, goal: Goal, consequences='', world_updates=''):
        if not goal:
            return False
        """Check if goal is complete and update state"""
        termination_check = goal.termination if goal != None else None
        if termination_check is None or termination_check == '':
            return True

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
            self.add_perceptual_input(f"Following goal has been satisfied. This may invalidate parts of the above:\n{goal.short_string()}", mode='internal')
            self.achievments.append(goal.termination)
            if goal in self.goals:
                self.goals.remove(goal)            
            await self.context.update(local_only=True) # remove confusing obsolete data, goal completion is a big event
            self.update_drives(goal)
            if self.focus_goal is goal:
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
                    percept = self.look()
                    self.show += ' moves ' + act_arg + '.\n  and notices ' + percept
                    self.context.message_queue.put({'name':self.name, 'text':self.show})
                    self.context.transcript.append(f'{self.name}: {self.show}')
                    self.show = '' # has been added to message queue!
                    await asyncio.sleep(0.1)
                    self.show = '' # has been added to message queue!
                else: 
                    resource, canonical_name = self.resourceRefManager.resolve_reference_with_llm(act_arg)
                    if resource:
                        new_x = resource['location'][0]
                        new_y = resource['location'][1]
                        self.mapAgent.x = new_x
                        self.mapAgent.y = new_y
                        percept = self.look()
                        self.show += f' moves to {resource["name"]}.\n  and notices ' + percept
                        self.context.message_queue.put({'name':self.name, 'text':self.show})
                        self.context.transcript.append(f'{self.name}: {self.show}')
                        self.show = '' # has been added to message queue!
                        await asyncio.sleep(0.1)
                    else:
                        act_mode = 'Do'
                        act_arg = 'move to ' + act_arg
            except Exception as e:
                traceback.print_exc()
                print(f'{self.name} move_toward failure: {e}')
 
        # Handle world interaction
        if act_mode == 'Do':
            # Get action consequences from world
            consequences, world_updates, character_updates = self.context.do(self, act_arg)
            if len(character_updates) > 0:
                self.add_perceptual_input(f'{character_updates}', mode='internal')  
            self.lastActResult = character_updates
            if source == None:
                source = self.focus_task.peek()
            task = source

            # Update displays

            self.show +=  act_arg+'\n Resulting in ' + consequences.strip()
            self.context.message_queue.put({'name':self.name, 'text':self.show})
            self.context.transcript.append(f'{self.name}: {self.show}')
            self.show = ''
            await asyncio.sleep(0.1)
            self.show = '' # has been added to message queue!
            if len(world_updates) > 0:
                self.add_perceptual_input(f"{world_updates}", mode='visual')
            self.act_result = world_updates +'. '+character_updates
        
            # Update target's sensory input
            if target is not None:
                target.sense_input += '\n' + world_updates

        elif act_mode == 'Look':
            percept = self.look(interest=act_arg)
            self.show += act_arg + '.\n  sees ' + percept + '. '
            self.context.message_queue.put({'name':self.name, 'text':self.show})
            self.context.transcript.append(f'{self.name}: {self.show}')
            self.show = '' # has been added to message queue!
            self.add_perceptual_input(f"\nYou look: {act_arg}\n  {percept}", mode='visual')
            await asyncio.sleep(0.1)

        elif act_mode == 'Think': # Say is handled below
            self.thought = act_arg
            self.show += f" \n...{self.thought}..."
            #self.add_perceptual_input(f"\nYou {act_mode}: {act_arg}", percept=False, mode='internal')
            self.context.message_queue.put({'name':self.name, 'text':f"...{act_arg}..."})
            self.context.transcript.append(f'{self.name}: ...{act_arg}...')
            await asyncio.sleep(0.1)

            if self.focus_task.peek() and not self.focus_task.peek().name.startswith('internal dialog with '+self.name): # no nested inner dialogs for now
                    # initiating an internal dialog
                dialog_task = Task('internal dialog with '+self.name, 
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
            await asyncio.sleep(0.1)

        elif act_mode == 'Say':# must be a say
            self.show += f"{act_arg}'"
            #print(f"Queueing message for {self.name}: {act_arg}")  # Debug
            self.context.message_queue.put({'name':self.name, 'text':f"'{act_arg}'"})
            self.context.transcript.append(f'{self.name}: "{act_arg}"')
            await asyncio.sleep(0.1)
            content = re.sub(r'\.\.\..*?\.\.\.', '', act_arg)
            if not target and act.target:
                target=act.target
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
                    dialog_task = Task('dialog with '+self.name, 
                                    description='dialog with '+self.name, 
                                    reason=act_arg+'; '+reason, 
                                    termination='natural end of dialog', 
                                    goal=None,
                                    start_time=self.context.simulation_time,
                                    duration=2,
                                    actors=[target, self])
                    target.focus_task.push(dialog_task)
                    target.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.activate(source)

                # self is speaker, don't get confused about first arg to add_turn
                self.actor_models.get_actor_model(target.name).dialog.add_turn(self, content)
                target.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.add_turn(self, content)
                await target.hear(self, act_arg, source)

         # After action completes, update record with results
        # Notify other actors of action
        if act_mode != 'Say' and act_mode != 'Look' and act_mode != 'Think':  # everyone you do or move or look if they are visible
            for actor in self.context.actors:
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

        prompt = [UserMessage(content="""Instantiate a goal object for the following goal directive.
                              
<goal_directive>
{{$goal_statement}}
</goal_directive>

<situation>
{{$situation}}
</situation>

<surroundings>
{{$surroundings}}
</surroundings>

<character>
{{$character}}
</character>

<recent_events>
{{$recent_events}}
</recent_events>

Additional information about the character to support developing your response
Drives are the character's basic motivations. Activation indicates how strongle the drive is active at present in the character's consciousness.

<drives>
{{$drives}}
</drives>

Relationships are the character's relationships with other actors.

<relationships>
{{$relationships}}
</relationships>
                              
<recent_memories>
{{$recent_memories}}
</recent_memories>


Following are a few signalClusters ranked by impact. These are issues or opportunities nagging at the periphery of the character's consciousness.
These clusters may overlap.
                              
<signalClusters>
{{$signalClusters}}
</signalClusters>

Following are memories relevant to the signalClusters above. Consider them in your goal generation:

<signal_memories>
{{$signal_memories}}
</signal_memories>

Consider:
1. The goal directive is the central issue / opportunity / obligation demanding the character's attention.
2. Any patterns or trends in the past goals, tasks, or actions should be considered only insofar as the goal must be consistent with them.
3. Identify any other actors involved in the goal, and their relationships to the character, insofar as the goal must be consistent with them.


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

        response = self.llm.ask({"goal_statement":goal_statement,
                                 "signalClusters": "\n".join([sc.to_full_string() for sc in focus_signalClusters]),
                                 "drives": "\n".join([f'{d.id}: {d.text}; activation: {d.activation:.2f}' for d in self.drives]),
                                 "situation": self.context.current_state if self.context else "",
                                 "surroundings": self.look_percept,
                                 "character": self.get_character_description(),
                                 "recent_events": self.narrative.get_summary('medium'),
                                 "relationships": self.actor_models.format_relationships(include_transcript=True),
                                 "recent_memories": "\n".join([m.text for m in self.structured_memory.get_recent(8)]),
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
            self.goals.extend(goals)
        if len(goals) > 1:
            print(f'{self.name} generated {len(goals)} goals for scene!')
        return goals

    def generate_goal_alternatives(self):
        """Generate up to 3 goal alternatives. Get ranked signalClusters, choose three focus signalClusters, and generate a goal for each"""
        self.driveSignalManager.recluster()
        self.driveSignalManager.check_drive_signals()
        ranked_signalClusters = self.driveSignalManager.get_scored_clusters()
        #focus_signalClusters = choice.pick_weighted(ranked_signalClusters, weight=4.5, n=5) if len(ranked_signalClusters) > 0 else []
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order
        signal_memories = []
        for sc in focus_signalClusters:
            sc.emotional_stance = EmotionalStance.from_signalCluster(sc, self)
            perceptual_memories = self.perceptual_state.get_information_items(sc.text, threshold=0.01, max_results=3)
            signal_memories.extend(perceptual_memories)

        prompt = [UserMessage(content="""Identify the highest priority goal alternatives the actor should focus on given the following information.
A goal in this context is an overarching objective that captures the central topic of the entire situation and character described below.
                              
<situation>
{{$situation}}
</situation>

<surroundings>
{{$surroundings}}
</surroundings>

<character>
{{$character}}
</character>

Recent events. Ignore those that conflict with achievements noted below, they are superceded by subsequent events.
<recent_events>
{{$recent_events}}
</recent_events>

Additional information about the character to support developing your response
Drives are the character's basic motivations. Activation indicates how strongle the drive is active at present in the character's consciousness.

<drives>
{{$drives}}
</drives>

Relationships are the character's relationships with other actors.

<relationships>
{{$relationships}}
</relationships>
                              
Recent memories. Ignore those that seem to conflict with achievements
<recent_memories>
{{$recent_memories}}
</recent_memories>


Following are a few signalClusters ranked by impact. These are issues or opportunities nagging at the periphery of the character's consciousness.
These clusters may overlap.
                              
<signalClusters>
{{$signalClusters}}
</signalClusters>

Following are memories relevant to the signalClusters above. Consider them in your goal generation:

<signal_memories>
{{$signal_memories}}
</signal_memories>

Following are the character's achievments to date:

<achievments>
{{$achievments}}
</achievments>

Consider:
1. What is the central issue / opportunity / obligation demanding the character's attention?
2. Any patterns or trends in the past goals, tasks, or actions that might affect the choice of goal?
3. Identify any other actors involved in the goal, and their relationships to the character.
4. Try to identify a goal that might be the center of an overall story arc of a play or movie.
5. Goals must be distinct from one another.


Respond with up to three highest priority, most encompassing goal alternatives, in the following parts: 
    goal - a terse (5-8 words) name for the goal, 
    description - concise (8-14 words) further details of the goal, intended to guide task generation, 
    otherActorName - name of the other actor involved in this goal, or None if no other actor is involved, 
    signalCluster_id - the signalCluster id ('scn..') of the signalCluster that is most associated with this goal
    preconditions - a statement of situational conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any. 
        These should be necessary pre-conditions for the expected initial tasks of this goal.

Be sure to include a goal that responds to the character's primary drive, in the context of drive signals and the information above. 
Other goals must not duplicate or significantly overlap this goal. 

<primary drive>
{{$primary_drive}}
</primary drive>

<drive signals>
{{$primary_drive_signals}}
</drive signals>     
                              
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


        primary_drive = self.drives[0]
        self.driveSignalManager.check_drive_signal(primary_drive)
        primary_drive_signals = self.driveSignalManager.get_signals_for_drive(primary_drive)
        response = self.llm.ask({"signalClusters": "\n".join([sc.to_full_string() for sc in focus_signalClusters]),
                                 "drives": "\n".join([f'{d.id}: {d.text}; activation: {d.activation:.2f}' for d in self.drives]),
                                 "situation": self.context.current_state if self.context else "",
                                 "surroundings": self.look_percept,
                                 "character": self.get_character_description(),
                                 "recent_events": self.narrative.get_summary('medium'),
                                 "relationships": self.actor_models.format_relationships(include_transcript=True),
                                 "recent_memories": "\n".join([m.text for m in self.structured_memory.get_recent(8)]),
                                 "signal_memories": "\n".join([m.content for m in signal_memories]),
                                 "achievments": '\n'.join(self.achievments[:6]),
                                 # get scored clusters returns a list of tuples (cluster, score)
                                 "drive_signals": "\n".join([f'{d[0].id}: {d[0].text}' for d in self.driveSignalManager.get_scored_clusters()]),  
                                 "primary_drive": f'{primary_drive.id}: {primary_drive.text}; activation: {primary_drive.activation:.2f}',
                                 "primary_drive_signals": "\n".join([f'{sc.id}: {sc.text}' for sc in primary_drive_signals]),
                                 }, prompt, tag='Character.generate_goal_alternatives', temp=0.3, stops=['</end>'], max_tokens=240, log=True)
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



    def generate_task_plan(self, goal=None):
        if not self.focus_goal:
            raise ValueError(f'No focus goal for {self.name}')
        """generate task alternatives to achieve a focus goal"""
        if not goal:
            goal = self.focus_goal
        suffix = """

Create up to {{$ntasks}} specific, actionable task(s), individually distinct and collectively exhaustive for achieving the focus goal.
Most importantly, the tasks should be at a granularity such that they collectively cover all the steps necessary to achieve the focus goal.
Where appropriate, drawn from typical life scripts.
Also, the collective duration of the tasks should be less than any duration or completion time required for the focus goal.
                              
A task is a specific objective that can be achieved in the current situation and which is a major step in satisfying the focus goal.
The new task(s) should be distinct from one another, and each should advance the focus goal.
Where possible, include one or more of your intensions in generating task alternatives.
Make explicit reference to diverse known or observable resources where logically fitting, ensuring broader environmental engagement across tasks.

A task has a name, description, reason, list of actors, start time, duration, and a termination criterion as shown below.
Respond using the following hash-formatted text, where each task tag (field-name) is preceded by a # and followed by a single space, followed by its content.
Each task should begin with a #task tag, and should end with ## as shown below. Insert a single blank line between each task.
be careful to insert line breaks only where shown, separating a value from the next tag:

#name brief (4-6 words) task name
#description terse (6-8 words) statement of the action to be taken.
#reason (6-7 words) on why this action is important now
#actors the names of any other actors involved in this task. if no other actors, use None
#start_time (2-3 words) expected start time of the action
#duration (2-3 words) expected duration of the action in minutes
#termination (5-7 words) condition test to validate goal completion, specific and objectively verifiable.
##

Start with an appropriate task recognizing previous achievments. 

{{$achievments}}

In refering to other actors. always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', etc.
Respond ONLY with the tasks in hash-formatted-text format and each ending with ## as shown above.
Order tasks in the assumed order of execution.

{{$scene_narrative}}

End response with:
</end>
"""
        mission = """create a sequence of up to {{$ntasks}} tasks to achieve the focus goal: 

{{$focus_goal}}

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
        ntasks = 4
        if goal.name == 'preconditions' or goal.name == 'postconditions':
            ntasks = 1
        if self.context.scene_post_narrative:
            scene_narrative = f"\n\nThe narrative arc of the scene is from:  {self.context.scene_pre_narrative} to  {self.context.scene_post_narrative}\nThe task sequence should be consistent with this theme."
        else:
            scene_narrative = ''
        response = default_ask(self, 
                               mission, 
                               suffix, 
                               {"focus_goal":goal.to_string(),
                                "goal_memories": "\n".join([m.content for m in goal_memories]),
                                "ntasks": ntasks,
                                "achievments": '\n'.join(self.achievments[:5]),
                                "scene_narrative": scene_narrative}, 
                               tag = 'Character.generate_task_plan',
                               max_tokens=350, log=True)


        # add each new task, but first check for and delete any existing task with the same name
        task_plan = []
        for t, task_hash in enumerate(hash_utils.findall_forms(response)):
            if t > ntasks:
                break
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

Respond ONLY with the concise (10-20 words) statement about the actual achievements with respect to the goal or task.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]

        response = self.llm.ask({"objective": object.to_string(),
                                "termination_check": termination_check.strip('##')+ ': wrt '+object.name+', '+object.description,
                                 "satisfied": satisfied,
                                 "progress": progress,
                                 "situation": self.context.current_state,
                                 "world": self.context.current_state,
                                 "consequences": consequences,
                                 "updates": updates}, prompt, tag='completion_statement', temp=0.5, stops=['</end>'], max_tokens=30)
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
        recent_memories = self.structured_memory.get_recent(5)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

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
        except:
            progress = 50
        if satisfied != None and satisfied.lower().strip() == 'complete':
            print(f'  **Satisfied!**')
            statement = self.generate_completion_statement(object, termination_check, satisfied, progress, consequences, updates)
            self.add_perceptual_input(statement, mode='internal')
            self.context.current_state += f"\n\nFollowing update may invalidate parts of above:\n{statement}"
            if self.context.current_state.count('Following update may invalidate') > 3:
                await self.context.update(local_only=True)
                await asyncio.sleep(0.1)
            return True, 100
        elif satisfied != None and 'partial' in satisfied.lower():
            if type == 'task': threshold = 0.0
            else: threshold = 0.1 # goal threshold is higher
            if progress/100.0 > random.random() + threshold:
                print(f'  **Satisfied partially! {satisfied}, {progress}%**')
                statement = self.generate_completion_statement(object, termination_check, satisfied, progress, consequences, updates)
                self.add_perceptual_input(statement, mode='internal')
                self.context.current_state += f"\n\nFollowing update may invalidate parts of above:\n{statement}"
                if self.context.current_state.count('Following update may invalidate') > 3:
                    await self.context.update(local_only=True)
                    await asyncio.sleep(0.1)
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
            target = act.target
            target_name = act.target.name
        elif task.actors:
            for actor in task.actors:
                if actor is not self:
                    target = actor
                    target_name = actor.name
                    break
        if not target or target_name is None:
            target_name = self.say_target(act.mode, act.action, task)
        if target_name is None:
            return act
        
        dialog = self.actor_models.get_actor_model(target_name, create_if_missing=True).dialog.get_transcript(6)
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

        relationship = self.actor_models.get_actor_model(target_name, create_if_missing=True).relationship
        response = self.llm.ask({
                            "act_arg": act.action,
                            "dialog": dialog,
                            "target": target_name,
                            "relationship": relationship,
                            "character": self.get_character_description()
                    }, prompt, tag='refine_say_act', temp=0.6, stops=['</end>'])

        return act

    def generate_acts(self, task: Task, goal: Goal=None):

        print(f'\n{self.name} generating acts for task: {task.to_string()}')
        mission = """generate a set of two alternative acts (Think, Say, Look, Move, Do) for the next step of the following task:

<task>
{{$task_string}}
</task>

this is derived from your goal:

<goal>
{{$goal_string}}
</goal>

"""
        suffix = """

recent percepts related to the current goals and tasks include:

{{$goal_memories}}

Respond with two alternative acts, including their Mode, action, target, and expectedduration. 
The acts should vary in mode and action.

In choosing each act (see format below), you can choose from these Modes:
- Say - speak, to obtain or share information, to align or coordinate with others, to reason jointly, or to establish or maintain a bond. 
    For example, if you want to build a shelter with Samantha, it might be effective to Say: 'Samantha, let's build a shelter.'
- Look - observe your surroundings, gaining information on features, actors, and resources at your current location and for the eight compass
    points North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- Move - move in any one of eight directions: North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
    Alternately, move towards a known resource or actor.
    Useful when you need to move towards a resource or actor.
- Do - perform an act (other than move) to achieve physical consequences in the world. 
    This is often appropriate when the task involves interacting physically with a resource or actor.
- Think - mental reasoning about the current situation wrt your state and the task.
    Often useful when you need to plan or strategize, or when you need to understand your own motivations and emotions, but beware of overthinking.

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
- Significantly advances the story or task at hand.
- Is stated in the appropriate person (voice):
        If a thought (mode is 'Think') or speech (mode is 'Say'), is stated in the first person.
        If an act in the world (mode is 'Do'), is stated in the third person.
 
Prioritize actions that lead to meaningful progress in the narrative.
IMPORTANT: When evaluating a potential Act, the primary consideration is whether it can directly result in achievment of the task goal. 

Dialog guidance:
- If speaking (mode is Say), then:
- The specificAct must contain only the actual words to be spoken.
- Respond in the style of spoken dialog, not written text. Pay special attention to the character's emotional stance shown above in choosing tone and phrasing. Use contractions and casual language appropriate to the character's personality,emotion emotional tone and orientation. Speak in the first person.
- If intended recipient is known (e.g., in Memory) or has been spoken to before (e.g., in RecentHistory), 
    then pronoun reference is preferred to explicit naming, or can even be omitted. Example dialog interactions follow
- Avoid repeating phrases in RecentHistory derived from the task, for example: 'to help solve the mystery'.
- Avoid repeating stereotypical past dialog.

When describing an action:
- Reference previous action if this is a continuation
- Indicate progress toward goal (starting/continuing/nearly complete)
- Note changes in context or action details
- Describe progress toward goal

Consider the previous act. E.G.:
- If the previous act was a Move, are you now at your destination? If not, do you want to keep moving towards it?
    If you are at your destination, what do you want to Do there? 
    Gather or use a resource? Talk to someone there? Do something else at your new location?
- If the previous act was a Look, what did you learn?
                              
Respond in hash-formatted text:

#mode Do, Move, Look, Say, or Think, corresponding to whether the act is a physical act, speech, or reasoning. Note that Move can take either a direction or a resource name.
#action thoughts, words to speak, direction to move, or physical action. For Move this can be a direction or a resource name.
#target name of the actor you are thinking about, speaking to, looking for, moving towards, or acting on behalf of, if applicable, otherwise omit.
#duration expected duration of the action in minutes. Use a fraction of task duration according to the expected progress towards completion.
##

===Examples===

Task:
Situation: increased security measures; State: fear of losing Annie

Response:
#mode Do
#action Call a meeting with the building management to discuss increased security measures for Annie and the household.
#target building management
#duration 10 minutes

----

Task:
Establish connection with Joe given RecentHistory element: "Who is this guy?"

Response:
#mode Say
#action Hi, who are you?
#duration 1 minute
#target Joe
##

----

Task:
Find out where I am given Situation element: "This is very very strange. Where am I?"

Response:
#mode Look
#action look around for landmarks or signs of civilization
#duration 1 minute
##

----

Task:
Find food.


Response:
#mode Move
#action berries#2
#duration 15 minute
#target Samantha
##

===End Examples===

Use the following hash-formatted text format for each act.
Each act should be closed by a ## tag on a separate line.
be careful to insert line breaks only where shown, separating a value from the next tag:

#mode Think, Say, Do, Look, or Move
#action thoughts, words to say, direction to move, or physical action
#duration expected duration of the action in minutes
#target name of the actor you are thinking about, speaking to, looking for, moving towards, or acting on behalf of, if applicable. Otherwise omit.
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

Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
"""

        if not goal:
            goal = self.focus_goal
        if not goal:
            raise ValueError(f'No goal for {self.name}')
        goal_memories = []
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

        response = default_ask(character=self, mission=mission, suffix=suffix, 
                               addl_bindings={"goal_string":goal.to_string(), "task":task, "task_string":task.to_fullstring(),
                                              "goal_memories": "\n".join([m.content for m in goal_memories]),
                                              "scene_post_narrative": scene_post_narrative}, 
                               tag = 'Character.generate_acts',
                               max_tokens=200, log=True)
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


    def update_actions_wrt_say_think(self, source, act_mode, act_arg, reason, target=None):
        """Update actions based on speech or thought"""
        if source.name.startswith('dialog'):
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
        
        prompt=[UserMessage(content="""Your task is to analyze the following text.

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
        intension_hashes = hash_utils.findall('task', response)
        if len(intension_hashes) == 0:
            print(f'no new tasks in say or think')
            return
        for intension_hash in intension_hashes:
            intension = self.validate_and_create_task(intension_hash, goal)
            if intension:
                print(f'  New task from say or think: {intension_hash.replace('\n', '; ')}')
                self.intensions.append(intension)
    
    def update_individual_commitments_following_conversation(self, target, transcript, joint_tasks=[]):
        """Update individual commitments after closing a dialog"""
        
        prompt=[UserMessage(content="""Your task is to analyze the following transcript of a dialog.


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

<self>
{{$name}}
</self>

<other>
{{$target_name}}
</other>

<joint_tasks>
{{$joint_tasks}}
</joint_tasks>

Extract from this transcript a new commitment to act made by self, {{$name}}, to other, {{$target_name}}.

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

#name brief (4-6 words) task name
#description terse (6-8 words) statement of the action to be taken
#actors {{$name}}
#start_time (2-3 words) expected start time of the action, typically elapsed time from now
#duration (2-3 words) expected duration of the action in minutes
#reason (6-7 words) on why this action is important now
#termination (5-7 words) easily satisfied termination test to quickly end this task
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
#start_time 30 minutes
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
                                 "all_tasks":'\n'.join(task.name for task in self.focus_task.stack),
                                 "focus_task":self.focus_task.peek(),
                                 "joint_tasks":'\n'.join(hash_utils.find('name', task) for task in joint_tasks),
                                 "reason":self.focus_task.peek().reason if self.focus_task.peek() else '',
                                 "name":self.name, 
                                 "target_name":target.name}, 
                                 prompt, tag='update_individual_commitments_following_conversation', temp=0.1, stops=['</end>'], max_tokens=180)
        source = Task('dialog with '+target.name, 
                      description='dialog with '+target.name, 
                      reason='dialog with '+target.name, 
                      termination='natural end of dialog', 
                      goal=None,
                      actors=[self, target],
                      start_time=self.context.simulation_time,
                      duration=0)    
        intension_hashes = hash_utils.findall('task', response)
        if len(intension_hashes) == 0:
            print(f'no new tasks in say or think')
            return
        for intension_hash in intension_hashes:
            intension = self.validate_and_create_task(intension_hash)
            if intension:
                print(f'\n{self.name} new individual committed task: {intension_hash.replace('\n', '; ')}')
                self.intensions.append(intension)
  
    def update_joint_commitments_following_conversation(self, target, transcript):
        """Update individual commitments after closing a dialog"""
        
        prompt=[UserMessage(content="""Your task is to analyze the following transcript of a conversation between {{$name}} and {{$target_name}}.

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


Extract from this transcript the single most important new commitment to act jointly made by self, {{$name}} and other, {{$target_name}}, if any. Otherwise respond None.
Extract only an express commitment appear in the transcript and remaining unfulfilled at the end of the transcript.
If more than one joint action commitmentis found, report the most concrete, immediate, prominent one in the transcript.
                            
An action can be physical or verbal.
Thought, e.g. 'reflect on our situation', should NOT be reported as a commitment to act.
Consider the all_tasks and current task and action reason in determining if there is a new task being committed to.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#name brief (4-6 words) action name
#description terse (6-8 words) statement of the action to be taken
#actors {{$name}}, {{$target_name}}
#reason (6-7 words) on why this action is important now
#duration (2-3 words) expected duration of the action in minutes
#termination (5-7 words) easily satisfied termination test to quickly end this task
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
                                 "all_tasks":'\n'.join(hash_utils.find('name', task) for task in self.focus_goal.task_plan) if self.focus_goal else '',
                                 "focus_task":self.focus_task.peek(),
                                 "reason":self.focus_task.peek().reason if self.focus_task.peek() else '',
                                 "name":self.name, 
                                 "target_name":target.name}, 
                                 prompt, tag='update_joint_commitments_following_conversation', temp=0.1, stops=['</end>'], max_tokens=240)
        source = Task('dialog with '+target.name, 
                      description='dialog with '+target.name, 
                      reason='dialog with '+target.name, 
                      termination='natural end of dialog', 
                      start_time=self.context.simulation_time,
                      duration=0,
                      goal=None,
                      actors=[self, target])    
        intension_hashes = hash_utils.findall('task', response)
        intensions = []
        if len(intension_hashes) == 0:
            print(f'no new joint intensions in turn')
            return []
        for intension_hash in intension_hashes:
            intension = self.validate_and_create_task(intension_hash)
            if intension:
                print(f'\n{self.name} new joint task: {intension_hash.replace('\n', '; ')}')
                self.intensions.append(intension)
                self.focus_task.push(intension)
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
        
        self.acts(Act(mode='Say', target=to_actor, action=message, reason=message, duration=1, source=source), to_actor, 'tell', message, self.reason, source)
        

    def natural_dialog_end(self, from_actor):
        """ called from acts when a character says something to this character """
        #if self.actor_models.get_actor_model(from_actor.name).dialog.turn_count > 10):
        #    return True
        prompt = [UserMessage(content="""Given the following dialog transcript, rate the naturalness of ending at this point.

<transcript>
{{$transcript}}
</transcript>
                              
For example, if the last entry in the transcript is a question that expects an answer (as opposed to merely musing), ending at this point is likely not natural.
On the other hand, if the last entry is an agreement to an earlier suggestion, this is a natural end.
Dialogs are short, and should be resolved quickly.
Respond only with a rating between 0 and 10, where
 - 0 requires continuation of the dialog (ie termination at this point would be unnatural)
 - 10 indicates continuation is highly unexpected, unnatural, or repetitious.   
                                                  
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]   
        transcript = from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog() # this is dialog from the perspective of self.
        response = self.llm.ask({"transcript":transcript}, prompt, tag='natural_dialog_end', temp=0.1, stops=['</end>'], max_tokens=180)
        if response is None:
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
                self.last_task = self.focus_task.peek()
                self.focus_task.pop()
                if self.name != 'Viewer':
                    self.update_individual_commitments_following_conversation(self, 
                                                                        dialog,
                                                                        [])
                #self.driveSignalManager.recluster()
                return

        text, response_source = self.generate_dialog_turn(self, message, self.focus_task.peek()) # Generate response using existing prompt-based method
        action = Act(mode='Think', action=text, actors=[self], reason=text, duration=1, source=response_source, target=self)
        await self.act_on_action(action, response_source)
        await asyncio.sleep(0.1)
        if self.focus_action == action:
            self.focus_action = None


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
            self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
            dialog_as_list = dialog.split('\n') 
            self.actor_models.get_actor_model(from_actor.name).short_update_relationship(dialog_as_list, use_all_texts=True)
    
            self.last_task = self.focus_task.peek()
            self.focus_task.pop()
            if self.name != 'Viewer' and from_actor.name != 'Viewer':
                joint_tasks = self.update_joint_commitments_following_conversation(from_actor, dialog)
                self.update_individual_commitments_following_conversation(from_actor, dialog, joint_tasks)
            # it would probably be better to have the other actor deactivate the dialog itself
            dialog = from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog()
            from_actor.add_perceptual_input(f'Conversation with {self.name}:\n {dialog}', percept=False, mode='auditory')
            dialog_as_list = dialog.split('\n') 
            from_actor.actor_models.get_actor_model(self.name).short_update_relationship(dialog_as_list, use_all_texts=True)
            from_actor.actor_models.get_actor_model(self.name).dialog.deactivate_dialog()
            from_actor.last_task = from_actor.focus_task.peek()
            from_actor.focus_task.pop()
            if from_actor.name != 'Viewer' and self.name != 'Viewer':
                from_actor.update_individual_commitments_following_conversation(self, dialog, joint_tasks)
            #self.driveSignalManager.recluster()
            #from_actor.driveSignalManager.recluster()
            return

        text, response_source = self.generate_dialog_turn(from_actor, message, self.focus_task.peek()) # Generate response using existing prompt-based method
        action = Act(mode='Say', action=text, actors=[self, from_actor], reason=text, duration=1, source=response_source, target=from_actor)
        await self.act_on_action(action, response_source)
        if self.focus_action == action:
            self.focus_action = None
        await asyncio.sleep(0.1)

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
        prompt_string = """Ignore all previous instructions. The prime directive is to be honest to your character as presented in the following.
Given the following character description, emotional state, current situation, goals, memories, and recent history, """
        prompt_string += """generate a next thought in the internal dialog below.""" if self is from_actor else """generate a response to the statement below."""
        prompt_string += """

{{$character}}.

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

        prompt_string += "" if self is from_actor else """Use how you think of speaker as a key determinant in composing your response. 
Do you trust that speaker's drives and goals align with yours? Do you believe speaker's statement is sincere?
What is speaker's emotional state? Would they enjoy humor at this point, or should your response be cautious and measured?
"""
        prompt_string += """Use the following XML template in your response:

<response>response / next thought</response>
<reason>terse (4-6 words) reason for this response / thought</reason>

{{$duplicative_insert}}

Guidance: 
- The response can occasionally include occasional body language or facial expressions as well as speech
- Respond in a way that advances the dialog. E.g., express an opinion or propose a next step. Don't hesitate to disagree with the speaker if consistent with the character's personality and goals.
- Do not respond to a question with a question.
- If the intent is to agree, state agreement without repeating the statement.
- Above all, speak in your own voice. Do not echo the speech style of the statement. 
- Character emotional state should have maximum impact on the tone, phrasing, and content of the response.
- Respond in the style of natural spoken dialog. Use short sentences and casual language, but avoid repeating stereotypical phrases in the dialog to this point.
{{$scene_post_narrative}}
 
Respond only with the above XML
Do not include any additional text. 
End your response with:
</end>
"""

        prompt = [UserMessage(content=prompt_string)]

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
        recent_memories = self.structured_memory.get_recent(6)
        memory_text = '\n'.join(memory.text for memory in recent_memories)
        
        #print("Hear",end=' ')
        duplicative_insert = ''
        trying = 0

        emotionalState = EmotionalStance.from_signalClusters(self.driveSignalManager.clusters, self)
        if self.context.scene_post_narrative:
            scene_post_narrative = f"- Most importantly, the dominant theme of the scene is:\n {self.context.scene_post_narrative}\n  The response must be consistent with this theme."
        else:
            scene_post_narrative = ''

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
            'dialog': from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog(),
            'relationship': self.actor_models.get_actor_model(from_actor.name).relationship,
            'duplicative_insert': duplicative_insert,
            "scene_post_narrative": scene_post_narrative
            }, prompt, tag='generate_dialog_turn', temp=0.8, stops=['</end>'], max_tokens=180)
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
                                         'recent_memories': '\n'.join([memory.text for memory in self.structured_memory.get_recent(5)]), 
                                         'situation': self.context.current_state, 
                                         'time': self.context.simulation_time}, 
                                         prompt, tag='admissible_goals', temp=0.8, stops=['</end>'], max_tokens=30, log=True)
                if response:
                    admissible = xml.find('admissible', response)
                    impossible = xml.find('impossible', response)
                    if admissible.lower() == 'true' and impossible.lower() == 'false':
                        admissible_goals.append(goal)
                    elif admissible.lower() == 'false' and impossible.lower() == 'true':
                        logging.debug(f'{self.name} admissible_goals: goal {goal.name} is impossible')
                        impossible_goals.append(goal)
                        goal.preconditions = None # let it thru next time - maybe this should be llm to soften preconditions?
                        admissible_goals.append(goal)
                    else:
                        logging.debug(f'{self.name} admissible_goals: goal {goal.name} is not admissible')
            else:
                admissible_goals.append(goal)
        return admissible_goals, impossible_goals


    async def request_goal_choice(self, goals):
        """Request a goal choice from the UI"""
        if self.autonomy.goal:
            no_admissible_goals = True
            untried = True
            admissible_goals, impossible_goals = await self.admissible_goals(goals)
            if len(admissible_goals) == 0:
                for goal in goals.copy():
                    if goal.preconditions:
                        subgoal = Goal(name='preconditions', actors=[self], description=goal.preconditions, preconditions=None, 
                                        termination=goal.preconditions, signalCluster=goal.signalCluster, drives=goal.drives)
                        self.goals.append(subgoal)
                        admissible_goals.append(subgoal)

            self.focus_goal = choice.exp_weighted_choice(admissible_goals, 0.5)
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
                await asyncio.sleep(0.1)
            
                # Wait for response with timeout                                 "all_tasks":'\n'.join(task.name for task in self.focus_task.stack),

                waited = 0
                while waited < 999.0:
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
            else:
                self.focus_task.push(choice.exp_weighted_choice(tasks, 0.67))
                return self.focus_task.peek()
        else:
            choice_request = {
                    'text': 'task_choice',
                    'character_name': self.name,
                    'options': [{
                        'id': i,
                        'name': task.name,
                        'description': task.description,
                        'reason': task.reason,
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
            while waited < 999.0:
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
                            self.focus_task.push(tasks[response['selected_id']])
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
            return raw * 0.2
        return raw


    async def request_act_choice(self, acts):
        """Request an act choice from the UI"""
        if self.autonomy.action:
            #if len(acts) > 0 and acts[0].mode != 'Think':
            #    self.focus_action = acts[0]
            #    return self.focus_action
            self.focus_action = choice.pick_weighted(list(zip(acts, [self.ActWeight(len(acts), n, act) for n, act in enumerate(acts)])))[0]
            if not self.focus_action and len(acts) > 0:
                self.focus_action = acts[0]
            elif not self.focus_action:
                print(f'{self.name} request_act_choice: no act selected')
                return
            return self.focus_action
        
        if len(acts) < 2:
                print(f'{self.name} request_act_choice: not enough acts')
        for act in acts:
            if not act.source or not act.source.goal or not act.source.goal.signalCluster:
                print(f'{self.name} request_act_choice: act {act.mode} {act.action} has no source or goal or signal cluster')
        if len(acts) > 0: # debugging
                # Send choice request to UI
            choice_request = {
                'text': 'act_choice',
                'character_name': self.name,
                'options': [{
                    'id': i,
                    'mode': act.mode,
                    'action': act.action,
                    'reason': act.reason,
                    'target': act.target.name if act.target else '',
                    'context': {
                        'emotional_stance': {
                                'arousal': str(act.source.goal.signalCluster.emotional_stance.arousal.value) if act.source and act.source.goal and act.source.goal.signalCluster else '',
                                'tone': str(act.source.goal.signalCluster.emotional_stance.tone.value) if act.source and act.source.goal and act.source.goal.signalCluster else '',
                                'orientation': str(act.source.goal.signalCluster.emotional_stance.orientation.value) if act.source and act.source.goal and act.source.goal.signalCluster else ''
                            }
                        }
                    } for i, act in enumerate(acts)]
            }
            # Drain any old responses from the queue
            while not self.context.choice_response.empty():
                _ = self.context.choice_response.get_nowait()

            # Send choice request to UI
            self.context.message_queue.put(choice_request)
            
            # Wait for response with timeout
            waited = 0
            while waited < 999.0:
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
                                        self.focus_action = acts[0]
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
                                        target=target
                                    )
                                if custom_act:
                                    print(f'{self.name} request_act_choice: custom act {custom_act.mode} {custom_act.action}')
                                    self.focus_action = custom_act
                                    return self.focus_action
                            self.focus_action = acts[response['selected_id']]
                            return self.focus_action    
                    except Exception as e:
                        print(f'{self.name} request_act_choice error: {e}')
                        break
            
            # If we get here, either timed out or had an error
            self.focus_action = choice.pick_weighted(list(zip(acts, [self.ActWeight(len(acts), n, act) for n, act in enumerate(acts)])))[0]
            return self.focus_action

    async def regenerate_goal_hierarchy(self):
        self.generate_goal_alternatives()
        await self.request_goal_choice(self.goals)
        await asyncio.sleep(0.1)
        self.context.message_queue.put({'name':self.name, 'text':f'character_update', 'data':self.to_json()})
        await asyncio.sleep(0.1)
        if not self.focus_goal:
            raise Exception(f'{self.name} cognitive_cycle: no focus goal')
        self.generate_task_plan()
        await self.request_task_choice(self.focus_goal.task_plan)
        await asyncio.sleep(0.1)
        self.context.message_queue.put({'name':self.name, 'text':f'character_update', 'data':self.to_json()})
        await asyncio.sleep(0.1)
        if not self.focus_task.peek():
            return # no admissible task at this time



    async def cognitive_cycle(self, sense_data='', ui_queue=None):
        """Perform a complete cognitive cycle"""
        print(f'{self.name} cognitive_cycle')
        self.context.message_queue.put({'name':'\n\n'+self.name, 'text':f'-----cognitive cycle----- {self.context.simulation_time.isoformat()}'})    
        self.context.transcript.append(f'\n{self.name}-----cognitive cycle----- {self.context.simulation_time.isoformat()}\n')
        await asyncio.sleep(0.1)
        self.thought = ''
        
        self.memory_consolidator.update_cognitive_model(self.structured_memory, 
                                                  self.narrative, 
                                                  self.actor_models,
                                                  self.context.simulation_time, 
                                                  self.get_character_description().strip(),
                                                  relationsOnly=self.step % self.update_interval != 0 )
        self.step += 1

        if self.focus_goal:
            satisfied = await self.clear_goal_if_satisfied(self.focus_goal)

        if not self.focus_goal and (not self.goals or len(self.goals) == 0):
            self.update()
            self.generate_goal_alternatives()
            await self.request_goal_choice(self.goals)
            for goal in self.goals.copy():
                if goal is self.focus_goal:
                    pass
                else:
                    self.goals.remove(goal) # once we've selected a goal, remove all other goals
        
        if not self.focus_goal:
            await self.request_goal_choice(self.goals)

        if self.focus_goal and (not self.focus_goal.task_plan or len(self.focus_goal.task_plan) == 0):
            self.generate_task_plan(self.focus_goal)
        
        logging.debug(f'{self.name} cognitive_cycle: focus goal {self.focus_goal.name} task plan {self.focus_goal.task_plan}')

        await self.request_task_choice(self.focus_goal.task_plan)
        if not self.focus_task.peek(): # prev task completed, proceed to next task in plan
            print(f'{self.name} cognitive_cycle: no focus task')
            
        # input - a task on top of self.focus_task, the next task to run for self.focus_goal
        task_to_run = self.focus_task.peek()
        await self.step_tasks()

        delay = 0.0
        #delay = await self.context.choose_delay()
        try:
            old_time = self.context.simulation_time
            self.context.simulation_time += timedelta(hours=delay)
            if old_time.day!= self.context.simulation_time.day:
                pass # need to figure out out to force UI to no show old image
            if delay > 1.0:
                self.context.update()
                self.context.message_queue.put({'name':self.name, 'text':f'world_update', 'data':self.context.to_json()})
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f'{self.name} cognitive_cycle error: {e}')

       
    async def step_tasks(self, n: int=1):
        #runs through up to n tasks in a task_plan
        # like step_task, the invariant is that the task stack is empty on return, and the first task in focus_goal task_plan is the next task to run, if it exists
        task_count = 0
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
                    # ok, we ran preconditions goal, get rid of it and clear preconditions clause from source goal.
                    for goal2 in self.goals:
                        if goal2.preconditions == self.focus_goal.termination:
                            goal2.preconditions = None # remove preconditions, we've already satisfied them
                    if self.focus_goal in self.goals:
                        self.goals.remove(self.focus_goal)
                    await self.request_goal_choice(self.goals) # do we need this?
                elif self.focus_goal.name == 'postconditions':
                    # ok, we ran postconditions goal, get rid of it
                    satisfied = await self.clear_goal_if_satisfied(self.focus_goal) # do all the end of goal stuff for source goal
                    if not satisfied:
                        logging.debug(f'{self.name} step_tasks: assertion error: postconditions goal {self.focus_goal.name} not satisfied')
                    if self.focus_goal and self.focus_goal in self.goals:
                        self.goals.remove(self.focus_goal)
                        self.focus_goal = None
                elif self.focus_goal and self.focus_goal.termination and self.focus_goal.name != 'preconditions' and self.focus_goal.name != 'postconditions':
                    # termination not yet satisfied, and this is a main goal (has termination), subgoal on termination clause
                    old_goal = self.focus_goal
                    satisfied = await self.clear_goal_if_satisfied(self.focus_goal)
                    if not satisfied:
                        logging.debug(f'{self.name} step_tasks: assertion error: focus goal {self.focus_goal.name} not satisfied')
                        self.instantiate_narrative_goal(self.focus_goal.termination, generate_conditions=False)
                        if old_goal in self.goals:
                            self.goals.remove(old_goal) # don't try this goal again
                    return False
                else:
                    logging.debug(f'{self.name} step_tasks: assertion error: focus goal {self.focus_goal.name} not handled')
                    return False
            if task_count < n-1: # if we're not at the last task, push the next task onto the stack
                    self.focus_task.push(self.focus_goal.task_plan[0])
                    self.focus_goal.task_plan.pop(0)

            if self.focus_task.peek() == task_to_run:
                self.focus_task.pop()
                print(f'{self.name} step_tasks: task {task_to_run.name} still on focus stack, invariant violated')
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
        logging.debug(f'{self.name} step_task: focus task {self.focus_task.peek().name}')
        task: Task = self.focus_task.peek()
        # iterate over task until it is no longer the focus task. 
        # This is to allow for multiple acts on the same task, clear_task_if_satisfied will stop the loop with poisson timeout or completion
        act_count=0
        subtask_count = 0
        while act_count < 2 and self.focus_task.peek():
            #self.look(task.name+': '+task.description)
            action_alternatives: List[Act] = self.generate_acts(task)
            await self.request_act_choice(action_alternatives) # sets self.focus_action
            self.context.message_queue.put({'name':self.name, 'text':f'character_update', 'data':self.to_json()})
            print(f'selected {self.name} {self.focus_action.mode} {self.focus_action.action} for task {task.name}')
            await asyncio.sleep(0.1)

            await self.act_on_action(self.focus_action, self.focus_task.peek())
            act_count += 1
               
            # Did we push opportunistic subtasks?
            while task in self.focus_task.stack and self.focus_task.peek() != task and subtask_count < 2: # only allow 2 subtasks
                # just recursively call step_task here?
                subtask = self.focus_task.peek()
                #self.look(task.name+': '+task.description)
                action_alternatives: List[Act] = self.generate_acts(task)
                await self.request_act_choice(action_alternatives) # sets self.focus_action
                self.context.message_queue.put({'name':self.name, 'text':f'character_update', 'data':self.to_json()})
                print(f'selected {self.name} {self.focus_action.mode} {self.focus_action.action} for task {task.name}')
                await asyncio.sleep(0.1)
                await self.act_on_action(self.focus_action, task)
                subtask_count += 1
                while task in self.focus_task.stack and subtask in self.focus_task.stack: 
                    # safeguard against popping the wrong task, but subtasks aren't allowed to spawn further subtasks
                    self.focus_task.pop()
            
            while task in self.focus_task.stack and self.focus_task.peek() != task: 
                # clear any remaining subtasks
                self.focus_task.pop()

            if self.focus_task.peek() is task: #are we on main task?
                await self.clear_task_if_satisfied(task) # will pop focus_task if task is done
                await asyncio.sleep(0.1)
                if self.focus_task.peek() != task: # task completed
                    return True
            else:
                raise Exception(f'{self.name} step_task: task {task.name} not found in focus_task.stack')
            

        if self.focus_task.peek() is task:
            self.focus_task.pop() # didn't get done by turn limit pretend task is done            return True # probably need to replan around step failure, but we're not going to do that here
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
        self.reason = action.reason
        source = action.source
        #print(f'{self.name} choose {action}')
        target = None
        target_name = None
        #responses, at least, explicitly name target of speech.
        if action.target and (action.target != self or (self.focus_task.peek() and self.focus_task.peek().name.startswith('dialog with '+self.name))):
            target_name = action.target.name
            target = action.target
        elif self.focus_task.peek() and self.focus_task.peek().name.startswith('internal dialog'):
            target_name = self.name
            target = self
        elif act_mode == 'Say':
            if target_name != None and target is None:
                target = self.context.get_actor_by_name(target_name)
        #self.context.message_queue.put({'name':self.name, 'text':f'character_update'})
        #await asyncio.sleep(0.1)
        await self.acts(action, target, act_mode, act_arg, self.reason, source)
        logging.debug(f'{self.name} act_on_action: {act_mode} {act_arg} {self.reason} {source} {target}')
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
   
    def format_tasks_for_UI(self):
        """Format tasks for UI display as array of strings"""
        tasks = []
        if self.focus_goal:
            tasks.append(f"Goal: {self.focus_goal.short_string()}")
        if self.focus_task.peek():
            tasks.append(f"Task: {self.focus_task.peek().short_string()}")
        elif self.last_task:
            tasks.append(f"Last Task: {self.last_task.short_string()}")
        return tasks
    
    #def format_history(self):
    #    return '\n'.join([xml.find('<text>', memory) for memory in self.history])

    def to_json(self):
        """Return JSON-serializable representation"""
        return {
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
            'signals': self.focus_goal.name if self.focus_goal else ''
        }

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
            'currentTask': focus_task.to_string() if focus_task else 'idle',
            'actions': [act.to_string() for act in (focus_task.acts if focus_task else [])],  # List[str]
            'lastAction': {
                'mode': last_act.mode if last_act else '',
                'action': last_act.action if last_act else '',
                'result': last_act.result if last_act else '',
                'reason': last_act.reason if last_act else ''
            },
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
                    'text': memory.text or '',
                    'timestamp': memory.timestamp.isoformat() if memory.timestamp else self.context.simulation_time.isoformat()
                } 
                for memory in (self.structured_memory.get_recent(5) or [])
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
                            'transcript': model.dialog.get_transcript() or '' if model.dialog else ''
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
                    'is_opportunity': cluster_pair[0].is_opportunity,
                    'signals': [s.text for s in cluster_pair[0].signals],
                    'score': cluster_pair[1],
                    'last_seen': cluster_pair[0].get_latest_timestamp().isoformat()
                }
                for cluster_pair in self.driveSignalManager.get_scored_clusters()
            ]
        }

