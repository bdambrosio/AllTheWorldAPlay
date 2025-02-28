from __future__ import annotations
from datetime import datetime, timedelta
from enum import Enum
import json
import random
import string
import traceback
import time
from typing import List, Dict, Optional
from sim.cognitive import knownActor
from sim.cognitive import perceptualState
from sim.cognitive.driveSignal import Drive, DriveSignalManager, SignalCluster
from sim.memory.consolidation import MemoryConsolidator
from sim.memory.core import MemoryEntry, NarrativeSummary, StructuredMemory, Drive
from sim.memory.core import MemoryRetrieval
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
import re
import asyncio

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
    def __init__(self, name, actors, description, termination, signalCluster, drives):
        self.name = name
        self.actors = actors
        self.description = description
        self.termination = termination
        self.drives = drives
        self.signalCluster = signalCluster
        self.progress = 0
        self.tasks = []

    def short_string(self):
        return f'{self.name}: {self.description}; \n termination: {self.termination}'
    
    def to_string(self):
        return f'Goal {self.name}: {self.description}; actors: {', '.join([actor.name for actor in self.actors])}; signals: {self.signalCluster.text}; termination: {self.termination}'
    
    def test_termination(self, events=''):
        """Test if recent acts, events, or world update have satisfied termination"""
        pass

class Task:
    def __init__(self, name, description, reason, termination, goal, actors):
        self.name = name
        self.description = description
        self.reason = reason
        self.termination = termination
        self.actors = actors
        self.goal = goal
        self.acts = []
        self.progress = 0

    def short_string(self):
        return f'{self.name}: {self.description} \n reason: {self.reason} \n termination: {self.termination}'

    def to_string(self):
        return f'Task {self.name}: {self.description}; actors: {[actor.name for actor in self.actors]}; reason: {self.reason}; termination: {self.termination}'
    
    def test_termination(self, events=''):
        """Test if recent acts, events, or world update have satisfied termination"""
        pass
    
class Act:
    def __init__(self, name, mode, action, actors, reason, source, target=None, result=''):
        self.name = name
        self.mode = mode
        self.action = action
        self.actors = actors
        self.reason = reason
        self.source = source # a task
        self.target = target # an actor
        self.result = result

    def to_string(self):
        return f'Act {self.name}: {self.action}; reason: {self.reason}; result: {self.result}'

# Character base class
class Character:
    def __init__(self, name, character_description, server_name='local', mapAgent=True):
        print(f"Initializing Character {name}")  # Debug print
        self.name = name
        self.character = character_description
        self.llm = llm_api.LLM(server_name)
        self.context = None

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
            last_update=datetime.now(),  # Will be updated to simulation time
            active_drives=[]
        )

        # World integration attributes
        if mapAgent:
            self.mapAgent = None  # Will be set later
            self.world = None
            self.my_map = [[{} for i in range(100)] for i in range(100)]
            self.x = 50
            self.y = 50
        self.perceptual_state = PerceptualState(self)
        self.last_sense_time = datetime.now()
        self.sense_threshold = timedelta(hours=4)
        self.act = None
        self.previous_action_name = None
        self.look_percept = ''
        self.driveSignalManager = None
       # Initialize drives
        self.drives = [
            Drive( "immediate physiological needs: survival, water, food, clothing, shelter, rest."),
            Drive("safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events."),
            Drive("assurance of short-term future physiological needs (e.g. adequate water and food supplies, shelter maintenance)."),
            Drive("love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friendship, intimacy, trust, acceptance.")
        ]
            
        self.always_respond = True
        
        # Initialize memory systems
        self.structured_memory = StructuredMemory(owner=self)
        self.memory_consolidator = MemoryConsolidator(self.llm)
        self.memory_retrieval = MemoryRetrieval()
        self.new_memory_cnt = 0
        self.next_task = None  # Add this line
        self.driveSignalManager = DriveSignalManager(self.llm)
        self.driveSignalManager.set_context(self.context)
        self.focus_goal = None
        self.focus_task = Stack()
        self.goals = []
        self.tasks = [] 
        self.intensions = []

    def set_context(self, context):
        self.context = context
        self.actor_models = KnownActorManager(self, context)
        if self.driveSignalManager:
            self.driveSignalManager.context = context
 
    def validate_and_create_task(self, task_hash, goal=None):
        """Validate a task hash and create a task object
        
        Args:
            task_hash: Hash-formatted task definition
            goal: Goal this task is for
        """
        name = hash_utils.find('name', task_hash)
        description = hash_utils.find('description', task_hash)
        reason = hash_utils.find('reason', task_hash)
        termination = hash_utils.find('termination', task_hash)
        try:
            actor_names = hash_utils.find('actors', task_hash)
            if actor_names:
                actor_names = actor_names.strip().split()
            else:
                actor_names = []
        except Exception as e:
            print(f"Warning: invalid actors field in {task_hash}") 
            actor_names = []
        actors = [self.context.resolve_reference(actor_name) for actor_name in actor_names if actor_name]
        actors = [actor for actor in actors if actor is not None]
        if not self in actors:
            actors =  [self] + actors

        if name and description and reason and termination and actor_names:
            task = Task(name, description, reason, termination.replace('##','').strip(), goal, actors)
            return task
        else:
            print(f"Warning: Invalid task generation response for {task_hash}") 
            return None
        
    def validate_and_create_act(self, xml_string, task):
        """Validate an XML actionable and create an Act object
        
        Args:
            xml_string: XML-formatted actionable definition
            task: Task this act is for
        """
        mode = xml.find('<mode>', xml_string)
        action = xml.find('<specificAct>', xml_string)
        try:
            target_name = xml.find('<target>', xml_string)
            if target_name:
                target = self.context.resolve_reference(target_name, create_if_missing=True)
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
                raise ValueError(f'Invalid mode {mode} in actionable. Must be one of: {", ".join(m.value for m in Mode)}')
        
        if mode and action:
            act = Act(
                name=mode,
                mode=mode, 
                action=action,
                actors=[self, target] if target else [self],
                reason=task.reason,
                source=task,
                target=target
            )
            return act
        else:
            raise ValueError(f"Invalid actionable XML: {xml_string}")

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

    def forward(self, step):
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

        # Update state and tasks
        self.focus_goal = None
        self.goals = []
        self.focus_task = Stack()
        self.intensions = []
        # Reset new memory counter
        self.new_memory_cnt = 0


    def say_target(self, act_name, text, source=None):
        """Determine the intended recipient of a message"""
        if len(self.context.actors) == 2:
            for actor in self.context.actors:
                if actor.name != self.name:
                    if actor.name.lower() in text.lower():
                        return actor.name
        elif 'dialog with' in source.name:
            return source.strip().split('dialog with ')[1].strip()
        
        prompt = [UserMessage(content="""Determine the intended hearer of the following message spoken by you.
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
If the message is a spoken message, respond with the name of the intended recipient. An intended recipient may be another known actor or 
Respond using the following XML format:

<target>
  <name>intended recipient name</name>
</target>

End your response with:
</end>
""")]
        #print('say target')
        response = self.llm.ask({
            'character': self.character,
            'history': self.narrative.get_summary('medium'),
            'actors': '\n'.join([actor.name for actor in self.context.actors if actor != self]),
            'action_type': 'internal thought' if act_name == 'Think' else 'spoken message',
            "message": text
        }, prompt, temp=0.2, stops=['</end>'], max_tokens=180)

        candidate = xml.find('<name>', response)
        if candidate is not None:
            target = self.context.resolve_reference(candidate)
            if target:
                return target.name
        return None
    # World interaction methods
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
        self.my_map[self.x][self.y] = view

        text_view = ""
        visible_actors = []
        for dir in view.keys():
            try:
                text_view += f"{dir}:"
                if 'visibility' in view[dir]:
                    text_view += f" visibility {view[dir]['visibility']}, "
                if 'terrain' in view[dir]:
                    text_view += f"terrain {view[dir]['terrain']}, "
                if'slope' in view[dir]:
                    text_view += f"slope {view[dir]['slope']}, "
                if 'resources' in view[dir]:
                    text_view += f"resources {view[dir]['resources']}, "
                if 'agents' in view[dir]:
                    text_view += f"others {view[dir]['agents']}, "
                    visible_actors.extend(view[dir]['agents'])
                if 'water' in view[dir]:
                    text_view += f"water {view[dir]['water']}"
            except Exception as e:
                pass
            text_view += "\n"
        
        # update actor visibility.
        self.actor_models.set_all_actors_invisible()
        for actor in visible_actors:
            self.actor_models.get_actor_model(actor['name'], create_if_missing=True).visible = True
         # Create visual perceptual input
        prompt = [UserMessage(content="""Your current state is :
                              
<interests>
{{$interests}}
</interests>

<state>
{{$state}}
</state>

And your current focus task is:

<focus_task>
{{$focus_task}}
</focus_task>

You see the following:

<view>
{{$view}}
</view>

Provide a concise description of what you notice, highlighting the most important features given your current interests, state and tasks. 
Respond using the following XML format:

<perception>a concise (30 words or less) description of perceptual content</perception>

End your response with:
<end/>
""")]
        #print("Look",end=' ')
        response = self.llm.ask({"view": text_view, 
                                 "interests": interest,
                                 "state": self.narrative.get_summary(),
                                 "focus_task": self.focus_task.peek().to_string() if self.focus_task.peek() else ''}, 
                                prompt, temp=0.2, stops=['<end/>'], max_tokens=100)
        percept = xml.find('<perception>', response)
        perceptual_input = PerceptualInput(
            mode=SensoryMode.VISUAL,
            content=percept,
            timestamp=self.context.simulation_time,
            intensity=0.7,  # Medium-high for direct observation
        )       
        self.perceptual_state.add_input(perceptual_input)       
        self.look_percept = percept
        return percept

    def format_look(self):
        """Format the agent's current map view"""
        obs = self.my_map[self.x][self.y]
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
            actor.tell(self, message)
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
<end/>
""")]

            concerns = self.focus_task.peek().to_string() if self.focus_task.peek() else ''
            state = description + '.\n '+concerns +'\n'+ context
            recent_memories = self.structured_memory.get_recent(8)
            recent_memories = '\n'.join(memory.text for memory in recent_memories)
            #print("Char generate image description", end=' ')
            response = self.llm.ask({ "description": state, "recent_memories": recent_memories}, prompt, temp=0.2, stops=['<end/>'], max_tokens=10)
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
<end/>

Respond only with a single choice of mode. Do not include any introductory, discursive, or explanatory text.
""")]
            #print("Perceptual input",end=' ')
            response = self.llm.ask({"message": message}, prompt, temp=0, stops=['<end/>'], max_tokens=150)
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
            timestamp=datetime.now(),
            intensity=intensity
        )
        self.perceptual_state.add_input(perceptual_input)
        if self.context:
            input_time = self.context.simulation_time
        else:
            input_time = datetime.now()
        self.driveSignalManager.analyze_text(message, self.drives, input_time)
        return perceptual_input

    def add_to_history(self, message: str):
        """Add message to structured memory"""
        if message is None or message == '':
            return
        entry = MemoryEntry(
            text=message,
            importance=0.5,  # Default importance
            timestamp=datetime.now(),
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
    
    def map_goals(self):
        """ map goals for llm input """
        header = """A goal is a terse description of a need or desire the character has.  
Each goal has an urgency and a trigger.  A goal is satisfied when the termination condition is met.
Here are this character's goals:
"""
        mapped = [header]
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
<end/>
""")]
        response = self.llm.ask({"reason":reason}, instruction, temp=0.3, stops=['<end/>'], max_tokens=12)
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
<end/>""")]

        #print("Repetitive")
        result = self.llm.ask({
            'history': history,
            'response': new_response
        }, prompt, temp=0.2, stops=['<end/>'], max_tokens=100)

        if 'true' in result.lower():
            return True
        else:
            return False

    def clear_task_if_satisfied(self, task, consequences='', world_updates=''):
        """Check if task is complete and update state"""
        termination_check = task.termination if task != None else None
        if termination_check is None or termination_check == '':
            return True

        # Test completion through cognitive processor's state system
        satisfied, progress = self.test_termination(
            task,
            termination_check, 
            consequences,
            world_updates, 
            type='task'
        )

        if satisfied or type(object) is Task and len(object.acts) > random.randint(3,5): # basic timeout on task
            if task == self.focus_task.peek():
                self.focus_task.pop()

        return satisfied

    def clear_goal_if_satisfied(self, goal, consequences='', world_updates=''):
        """Check if goal is complete and update state"""
        termination_check = goal.termination if goal != None else None
        if termination_check is None or termination_check == '':
            return True

        # Test completion through cognitive processor's state system
        satisfied, progress = self.test_termination(
            goal,
            termination_check, 
            consequences,
            world_updates, 
            type='goal'
        )

        if satisfied:
            if goal == self.focus_goal:
                self.focus_goal = None

        return satisfied


    async def acts(self, act:Act, target: Character, act_name: str, act_arg: str='', reason: str='', source: Task=None):
        """Execute an action and record results"""
        # Create action record with state before action
        try:
            mode = Mode(act_name.capitalize())
        except ValueError:
            raise ValueError(f'Invalid action name: {act_name}')
        self.act_result = ''
    
        # Store current state and reason
        self.reason = reason
        self.show = ''
        if act_name is None or act_arg is None or len(act_name) <= 0 or len(act_arg) <= 0:
            return


        # Update thought display
        if act_name == 'Think':
            self.thought = act_arg
            self.show += f" \n...{self.thought}..."
            self.context.message_queue.put({'name':self.name, 'text':f"{self.show}"})  
            await asyncio.sleep(0.1)
        else:
            self.thought +=  self.reason

        # Handle world interaction
        if act_name == 'Do':
            # Get action consequences from world
            consequences, world_updates = self.context.do(self, act_arg)
        
            if source == None:
                source = self.focus_task.peek()
            task = source

            # Update displays

            self.show +=  act_arg+'\n Resulting in ' + consequences.strip()
            self.context.message_queue.put({'name':self.name, 'text':self.show})
            self.show = ''
            await asyncio.sleep(0.1)
            self.show = '' # has been added to message queue!
            self.add_perceptual_input(f"You observe {world_updates}", mode='visual')
            self.act_result = world_updates
        
            # Update target's sensory input
            if target is not None:
                target.sense_input += '\n' + world_updates
            
        elif act_name == 'Move':
            moved = self.mapAgent.move(act_arg)
            if moved:
                dx, dy = self.mapAgent.get_direction_offset(act_arg)
                self.x = self.x + dx
                self.y = self.y + dy
                percept = self.look(interest=act_arg)
                self.show += ' moves ' + act_arg + '.\n  and notices ' + percept
                self.context.message_queue.put({'name':self.name, 'text':self.show})
                self.show = '' # has been added to message queue!
                await asyncio.sleep(0.1)
                self.show = '' # has been added to message queue!
                self.add_perceptual_input(f"\nYou {act_name}: {act_arg}\n  {percept}", mode='visual')
        elif act_name == 'Look':
            percept = self.look(interest=act_arg)
            self.show += act_arg + '.\n  sees ' + percept + '. '
            self.context.message_queue.put({'name':self.name, 'text':self.show})
            self.show = '' # has been added to message queue!
            self.add_perceptual_input(f"\nYou look: {act_arg}\n  {percept}", mode='visual')
            await asyncio.sleep(0.1)

        if act_name == 'Think': # Say is handled below
            # Update actions based on thought
            self.update_actions_wrt_say_think(source, act_name, act_arg, reason, target) 
            self.add_perceptual_input(f"\nYou {act_name}: {act_arg}", percept=False, mode='internal')
            await asyncio.sleep(0.1)
        # After action completes, update record with results
        # Notify other actors of action
        if act_name != 'Say' and act_name != 'Look' and act_name != 'Think':  # everyone you do or move or look if they are visible
            for actor in self.context.actors:
                if actor != self:
                    if actor != target:
                        actor_model = self.actor_models.get_actor_model(actor.name)
                        if actor_model != None and actor_model.visible:
                            percept = actor.add_perceptual_input(f"You see {self.name}: '{act_arg}'", percept=False, mode='visual')
                            actor.actor_models.get_actor_model(self.name, create_if_missing=True).infer_goal(percept)
        elif act_name == 'Say':# must be a say
            self.show += f"{act_arg}'"
            #print(f"Queueing message for {self.name}: {act_arg}")  # Debug
            self.context.message_queue.put({'name':self.name, 'text':f"'{act_arg}'"})
            await asyncio.sleep(0.1)
            content = re.sub(r'\.\.\..*?\.\.\.', '', act_arg)
            if target: 
                # NPCs are initialized with a dialog task
                if self.focus_task.peek() and not self.focus_task.peek().name.startswith('dialog'): # no nested dialogs for now
                    # initiating a dialog
                    dialog_task = Task('dialog with '+target.name, 
                                        description=self.name + ' says ' + act_arg, 
                                        reason=reason, 
                                        termination='natural end of dialog', 
                                        goal=None,
                                        actors=[self, target])
                    self.focus_task.push(dialog_task)
                self.actor_models.get_actor_model(target.name, create_if_missing=True).dialog.activate()
                self.actor_models.get_actor_model(target.name, create_if_missing=True).dialog.add_turn(self, content)
                await target.hear(self, act_arg, source)
            elif act.target:
                # target is an NPC-like object
                target_name = act.target.strip()
                target = self.context.get_npc_by_name(target_name, create_if_missing=True)
                if self.focus_task.peek() and not self.focus_task.peek().name.startswith('dialog'): # no nested dialogs for now
                    # initiating a dialog
                    dialog_task = Task('dialog with '+target.name, 
                                        description=self.name + ' says ' + act_arg, 
                                        reason=reason, 
                                        termination='natural end of dialog', 
                                        goal=None,
                                        actors=[self, target])
                    self.focus_task.push(dialog_task)
                self.actor_models.get_actor_model(target.name, create_if_missing=True).dialog.activate()
                self.actor_models.get_actor_model(target.name, create_if_missing=True).dialog.add_turn(self, content)

                self.add_perceptual_input(f"You say: {act_arg}", mode='internal')

        self.previous_action_name = act_name           

    def generate_goal_alternatives(self):
        """Generate up to 3 goal alternatives. Get ranked signalClusters, choose three focus signalClusters, and generate a goal for each"""
        ranked_signalClusters = self.driveSignalManager.get_scored_clusters()
        focus_signalClusters = choice.pick_weighted(ranked_signalClusters, weight=4.5, n=3) if len(ranked_signalClusters) > 0 else []
        goals = []
        scores = []
        for signalCluster in focus_signalClusters:
            goal = self.generate_goal(signalCluster)
            goals.append(goal)
            scores.append(signalCluster.score)
        self.goals = goals
        
        """Choose the best goals to focus on based on the goal alternatives ranked by signalCluster ranking and choose three stochastically"""

        options = [(goal, score) for goal, score in zip(goals,scores)]
        self.focus_goal = choice.pick_weighted(options, weight=3, n=1) if len(options) > 0 else None
        self.focus_goal = self.focus_goal[0] if self.focus_goal else None
        if not self.focus_goal:
            raise ValueError(f'No focus goal for {self.name}')
        return self.focus_goal

    def generate_goal(self, signalCluster):
        """Generate a goal for a signalCluster"""
        drives = signalCluster.drives
        
        prompt = [UserMessage(content="""Given a signalCluster, generate a goal to achieve the opportunity or ameliorate the issue.

<signalCluster>
{{$signalCluster}}
</signalCluster>

<drives>
{{$drives}}
</drives>

<situation>
{{$situation}}
</situation>

<character>
{{$character}}
</character>

<relationships>
{{$relationships}}
</relationships>
                              
<recent_memories>
{{$recent_memories}}
</recent_memories>

<drive_related_memories>
{{$drive_memories}}
</drive_related_memories>


Consider this signalCluster and and your drives, memories, situation, and character, paying special attention to the signalCluster's drives in drive_memories.
Consider:
1. What is the central issue / opportunity this signalCluster indicates wrt your drives in the context of your character, situation, etc?
2. Recent events that affect the drive.
3. Any patterns or trends in the past goals, tasks, or actions related to this signalCluster.

Respond with a goal, in four parts: 
    name - a terse (3-4 words) name for the goal, 
    description - a concise (5-8 words) description of the goal, intended to guide task generation, 
    other_actor_name - name of the other actor involved in this goal, or None if no other actor is involved, 
    and termination  - a condition (5-6 words) that would mark achievement or partial achievement of the goal.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#goal 
#name terse (3-4 words) name for this goal
#description concise (5-8 words) description of this goal
#otherActorName name of the other actor involved in this goal, or None if no other actor is involved
#termination terse (5-6 words) statement of condition that would mark achievement or partial achievement of this goal
##

Respond ONLY with the above hash-formatted text.
End response with:
<end/>
""")]

        print(f'\n{self.name} generating goal for signalCluster: {signalCluster.to_string()}')
            
        # Get relevant memories
        drive_memories = set()
        for drive in signalCluster.drives:
            drive_memories.update([m.text for m in self.memory_retrieval.get_by_drive(
                memory=self.structured_memory,
                drive=drive,
                threshold=0.1,
                max_results=5
            )])
            
            # Get recent memories regardless of drive
        recent_memories = self.structured_memory.get_recent(5)
            
        # Format memories for LLM
        drive_memories_text = "\n".join(drive_memories)
            
        recent_memories_text = "\n".join([mem.text for mem in recent_memories])
            
        # Generate state assessment
        response = self.llm.ask({
            "signalCluster": signalCluster.to_string(),
            "drives": "\n".join([d.text for d in signalCluster.drives]),
            "recent_memories": recent_memories_text,
            "drive_memories": drive_memories_text,
            "relationships": self.actor_models.format_relationships(include_transcript=True),
            "situation": self.context.current_state if self.context else "",
            "character": self.character
        }, prompt, temp=0.3, stops=['<end/>'])     
            
        # Parse response
        try:
            name = hash_utils.find('name', response)
            description = hash_utils.find('description', response)
            other_actor_name = hash_utils.find('otherActorName', response)
            termination = hash_utils.find('termination', response)
                
            if other_actor_name:
                other_actor = self.context.resolve_reference(other_actor_name, create_if_missing=True)
                if other_actor is None:
                    actors = [self]
                else:
                    actors = [self, other_actor]
            else:
                actors = [self]
            if name and description and termination:
                goal = Goal(name=name, 
                            actors=[self],
                            description=description, 
                            termination=termination.replace('##','').strip(), 
                            signalCluster=signalCluster, 
                            drives=signalCluster.drives)
                print(f'   generated goal: '+goal.to_string())
                return goal
            else:
                print(f"Warning: Invalid goal generation response for {signalCluster.to_string()}")
                return None
        except Exception as e:
            print(f"Error parsing goal generation response: {e}")
            traceback.print_exc()
                
        return None

    def generate_task_alternatives(self):
        if not self.focus_goal:
            raise ValueError(f'No focus goal for {self.name}')
        """generate task alternatives to achieve a focus goal"""
        prompt = [UserMessage(content="""Given your character, drives, current goal, intensions, and recent memories, create up to {{$n_new_tasks}} task alternatives.

<character>
{{$character}}
</character>

<situation>
{{$situation}}
</situation>

<goals>
{{$goals}}
</goals>

While you would like to achieve all goals, you have chosen to focus on:
                              
<focus_goal>
{{$focus_goal}}
</focus_goal>

In thinking about how to act to achieve this, bear in mind any intensions you have expressed in previous thought or conversation:
                              
<intensions>
{{$intensions}}
</intensions>

Also bear in mind your recent memories:
                              
<recent_memories>
{{$memories}}
</recent_memories>

The relationships you have with other actors:
                              
<relationships>
{{$relationships}}
</relationships>

And the recent events in the world:
                              
<recent_events>
{{$recent_events}}
</recent_events>


Create up to {{$n_new_tasks}} specific, actionable tasks.
                              
The new tasks should be distinct from one another, and cover both the focus goal.
Where possible, use one or more of your intensions in generating task alternatives.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#task
#name brief (4-6 words) action name
#description terse (6-8 words) statement of the action to be taken
#reason (6-7 words) on why this action is important now
#actors the names of any other actors involved in this task. if no other actors, use None
#termination (5-7 words) condition test which, if met, would satisfy the goal of this action
##

In refering to other actors. always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', etc.
Respond ONLY with three tasks in hash-formatted-text format and each ending with ## as shown above.
Order tasks from highest to lowest priority.
End response with:
<end/>
""")]

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(8)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        #print("Update Tasks")
        response = self.llm.ask({
            'character': self.character.replace('\n', ' '),
            'situation': self.context.current_state.replace('\n', ' '),
            'goals': '\n'.join([goal.to_string() for goal in self.goals]),
            'focus_goal': self.focus_goal.to_string(),
            'memories': memory_text,    
            'intensions': '\n'.join([intension.to_string() for intension in self.intensions]),
            'name': self.name,
            'recent_events': self.narrative.get_summary('medium'),
            'relationships': self.actor_models.format_relationships(include_transcript=True),
            'n_new_tasks': 3
        }, prompt, temp=0.7, stops=['<end/>'])

        # add each new task, but first check for and delete any existing task with the same name
        task_alternatives = []
        for task_hash in hash_utils.findall_forms(response):
            print(f'\n{self.name} new task: {task_hash.replace('\n', '; ')}')
            task = self.validate_and_create_task(task_hash, self.focus_goal)
            if task:
                task_alternatives.append(task)
        focus_task = choice.pick_weighted([(task, 1) for task in task_alternatives], weight=3, n=1)
        focus_task = focus_task[0] if focus_task else None
        if not focus_task:
            raise ValueError(f'No task alternatives for {self.name}')
        self.focus_task = Stack()
        self.focus_task.push(focus_task)
        return focus_task

    def choose_task_to_activate(self):
        """Choose the best task to activate based on the task alternatives"""
        pass

    def test_termination(self, object, termination_check, consequences, updates='', type=''):
        """Test if recent acts, events, or world update have satisfied termination"""
        prompt = [UserMessage(content="""Test if recent acts, events, or world update have satisfied the CompletionCriterion is provided below. 
Reason step-by-step using the CompletionCriterion as a guide for this assessment.
Consider these factors in determining task completion:
- Sufficient progress towards goal for intended purpose
- Diminishing returns on continued effort
- Environmental or time constraints
- "Good enough" vs perfect completion
                    
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
#progress>0-100
##

Respond ONLY with the above hash-formatted text.
Do not include any introductory, explanatory, or discursive text.
End your response with:
<end/>
""")]


        # Get recent memories
        recent_memories = self.structured_memory.get_recent(5)
        memory_text = '\n'.join(memory.text for memory in recent_memories)


        response = self.llm.ask({
            "termination_check": termination_check.strip('##')+ ': wrt '+object.name+', '+object.description,
            "situation": self.context.current_state,
            "memories": memory_text,  # Updated from 'memory'
            "events": consequences + '\n' + updates,
            "character": self.character,
            "history": self.format_history_for_UI(),
            "relationships": self.narrative.get_summary('medium')
        }, prompt, temp=0.5, stops=['<end/>'], max_tokens=120)

        satisfied = hash_utils.find('status', response)
        progress = hash_utils.find('progress', response)
        print(f'\n{self.name} testing {type} {object.name} termination: {termination_check}, ', end='')
        try:
            progress = int(progress.strip())
        except:
            progress = 50
        if satisfied != None and satisfied.lower().strip() == 'complete':
            print(f'  **Satisfied!**')
            self.add_perceptual_input(f" {object.name} is complete: {termination_check}", mode='internal')
            return True, 100
        elif satisfied != None and 'partial' in satisfied.lower():
            if progress/100.0 > 0.67 * random.random():
                print(f'  **Satisfied partially! {satisfied}, {progress}%**')
                self.add_perceptual_input(f" {object.name} is pretty much complete: {termination_check}", mode='internal')
                return True, progress
        elif satisfied != None and 'insufficient' in satisfied.lower():
            if progress/100.0 > random.random():
                print(f'  **Satisfied partially! {satisfied}, {progress}%**')
                self.add_perceptual_input(f"{object.name} is sufficiently complete: {termination_check}", mode='internal')
                return True, progress
            
        print(f'  **Not satisfied! {satisfied}, {progress}%**')
        return False, progress

    def refine_say_act(self, act, task):
        """Refine a say act to be more natural and concise"""
        target_name = self.say_target(act.name, act.action, task)
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
<end/>
""")]

        relationship = self.actor_models.get_actor_model(target_name, create_if_missing=True).relationship
        response = self.llm.ask({
            "act_name": act.name,
            "act_arg": act.action,
            "dialog": dialog,
            "target": target_name,
            "relationship": relationship,
            "character": self.character
        }, prompt, temp=0.6, stops=['<end/>'])

        return act

    def actualize_task(self, task):

        print(f'\n{self.name} actualizing task: {task.to_string()}')
        prompt = [UserMessage(content="""You are {{$character}}.
Your task is to generate an Actionable (a 'Think', 'Say', 'Look', Move', or 'Do') for the next step of the following task.

<task>
{{$task}}
</task>

Your current situation is:

<situation>
{{$situation}}
</situation>

Your current goals are:

<goals>
{{$goals}}
</goals>
                              
And in particular, you are focusing on:

<focus_goal>
{{$focus_goal}}
</focus_goal>

Your recent memories include:

<recent_memories>
{{$memories}}
</recent_memories>

Recent history includes:
<history>
{{$history}}
</history>

The previous specific act for this task, if any, was:

<previousSpecificAct>
{{$lastAct}}
</previousSpecificAct>

And the observed result of that was:
<observed_result>
{{$lastActResult}}.
</observed_result>

Respond with an Actionable, including its Mode and SpecificAct. 

In choosing an Actionable (see format below), you can choose from these Modes:
- Think - reason about the current situation wrt your state and the task.
- Say - speak, to motivate others to act, to align or coordinate with them, to reason jointly, or to establish or maintain a bond. 
    Say is especially appropriate when there is an actor you are unsure of, you are feeling insecure or worried, or need help.
    For example, if you want to build a shelter with Samantha, it might be effective to Say: 'Samantha, let's build a shelter.'
- Look - observe your surroundings, gaining information on features, actors, and resources at your current location and for the eight compass
    points North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- Move - move in any one of eight directions: North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- Do - perform an act (other than move) with physical consequences in the world. 
    This is often appropriate when the task involves interacting with a resource or actor, particularly when the actor is not in {{$known_actor_names}}.

Review your character for Mode preference. (e.g., 'xxx is thoughtful' implies higher percentage of 'Think' Actionables.) 

A SpecificAct is one which:
- Is a specific thought, spoken text, physical movement or action.
- Includes only the actual thoughts, spoken words, physical movement, or action.
- Has a clear beginning and end point.
- Can be performed or acted out by a person.
- Can be easily visualized or imagined as a film clip.
- Makes sense as the next action given observed results of previous act . 
- Is consistent with any incomplete action commitments made in your last statements in RecentHistory.
- Does NOT repeat, literally or substantively, the previous specific act or other acts by you in RecentHistory.
- Significantly advances the story or task at hand.
- Is stated in the appropriate person (voice):
        If a thought (mode is 'Think') or speech (mode is 'Say'), is stated in the first person.
        If an act in the world (mode is 'Do'), is stated in the third person.
 
Prioritize actions that lead to meaningful progress in the narrative.

Dialog guidance:
- If speaking (mode is Say), then:
- The specificAct must contain only the actual words to be spoken.
- Respond in the style of natural spoken dialog, not written text. Use short sentences, contractions, and casual language. Speak in the first person.
- If intended recipient is known (e.g., in Memory) or has been spoken to before (e.g., in RecentHistory), 
    then pronoun reference is preferred to explicit naming, or can even be omitted. Example dialog interactions follow
- Avoid repeating phrases in RecentHistory derived from the task, for example: 'to help solve the mystery'.

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
                              
Respond in XML:
<actionable>
  <mode>Think, Say, Look, Move, or Do, corresponding to whether the act is a reasoning, speech, or physical act</mode>
  <specificAct>thoughts, words to speak, direction to move, or physical action</specificAct>
  <target>name of the actor you are thinking about, speaking to, looking for, moving towards, or acting on behalf of, if applicable, otherwise omit.</target>
</actionable>

===Examples===

Task:
Situation: increased security measures; State: fear of losing Annie

Response:
<actionable>
  <mode>Do</mode>
  <specificAct>Call a meeting with the building management to discuss increased security measures for Annie and the household.</specificAct>
  <target>building management</target>
</actionable>

----

Task:
Establish connection with Joe given RecentHistory element: "Who is this guy?"

Response:
<actionable>
  <mode>Say</mode>
  <specificAct>Hi, who are you?</specificAct>
  <target>Joe</target>
</actionable>

----

Task:
Find out where I am given Situation element: "This is very very strange. Where am I?"

Response:
<actionable>
  <mode>Look</mode>
  <specificAct>look around for landmarks or signs of civilization</specificAct>
  <target>Samantha</target>
</actionable>

----

Task:
Find food.


Response:
<actionable>
  <mode>Move</mode>
  <specificAct>SouthWest</specificAct>
  <target>Samantha</target>
</actionable>

===End Examples===

Use the XML format for the actionable:

<actionable> 
  <mode>Think, Say, Do, Look, or Move</mode>
  <specificAct>thoughts, words to say, direction to move, or physical action</specificAct> 
  <target>name of the actor you are thinking about, speaking to, looking for, moving towards, or acting on behalf of, if applicable. Otherwise omit.</target>
</actionable>

Respond ONLY with the above XML.
Your name is {{$name}}, phrase the statement of specific action in your voice.
Ensure you do not duplicate content of a previous specific act.
{{$duplicative}}

Again, the task to translate into an Actionable is:
<task>
{{$task}} 
</task>

Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]
        #print(f'{self.name} act_result: {self.act_result}')
        act = None
        tries = 0
        mapped_goals = self.map_goals()
        duplicative_insert = ''
        temp = 0.6

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(10)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        #print("Actualize Task",end=' ')
        while act is None and tries < 2:
            response = self.llm.ask({
                'character': self.character.replace('\n', ' '),
                'memories': memory_text,  # Updated from 'memory'
                'duplicative': duplicative_insert,
                'history': self.narrative.get_summary('medium'),
                'name': self.name,
                "situation": self.context.current_state.replace('\n', ' ') + '\n\n' + self.look_percept + '\n',
                "goals": mapped_goals,
                "focus_goal": self.focus_goal.to_string() if self.focus_goal else '',
                "task": task.to_string(),
                "reason": task.reason,
                "lastAct": task.acts[-1].name if task.acts and len(task.acts) > 0 else '',
                "lastActResult": task.acts[-1].result if task.acts and len(task.acts) > 0 else '',
                "known_actor_names": ', '.join(actor.name for actor in task.actors)
            }, prompt, temp=temp, top_p=1.0, stops=['</end>'], max_tokens=180)

            # Rest of existing while loop...
            try:
                act = self.validate_and_create_act(response, task)
            except Exception as e:
                raise Exception(f"Error parsing XML, Invalid Act: {e}")
            # test for dup act
            if act.mode == 'Say':
                dup = self.repetitive(act.mode+': '+act.action, task.acts[-1] if task.acts and len(task.acts) > 0 else '', self.format_history_for_UI(6))
                if dup:
                    #print(f' Duplicate test failed\n    previous act: {last_act}\n    candidate act: {act}\n')
                    duplicative_insert = f"""\n****\nResponse:\n{act.mode+': '+act.action}\n is repetitive. Try something new\n****"""
                    if tries == 0:
                        act = None  # force redo
                        temp += .3
                    else:
                        pass

            elif act.mode == 'Do' or act.mode == 'Move':
                dup = self.repetitive(act.mode+': '+act.action, task.acts[-1] if task.acts and len(task.acts) > 0 else '', self.format_history_for_UI(4))
                if dup:
                    # print(f' Duplicate test failed\n    previous act: {task.acts[-1] if task.acts and len(task.acts) > 0 else ""}\n    candidate act: {act}\n')
                    #print(f'*****Response: {mode+': '+act} is repetitive of an earlier statement.****')
                    if tries < 1:
                        #act = None  # force redo
                        temp += .3
                    else:
                        pass
            elif act.mode == self.previous_action_name:
                dup = self.repetitive(act.mode+': '+act.action, task.acts[-1] if task.acts and len(task.acts) > 0 else '', self.format_history_for_UI(4))
                if dup:
                    # print(f' Duplicate test failed\n    previous act: {task.acts[-1] if task.acts and len(task.acts) > 0 else ''}\n    candidate act: {act}\n')
                    #print(f'\n*****Repetitive act test failed*****\n  {act}\n')
                    duplicative_insert = f"""\n****\nResponse:\n{act.mode+': '+act.action}\n is repetitive. Try something new\n****\n"""
                if tries < 1:
                    act = None  # force redo
                    temp += .3
                else:
                    pass
            tries += 1

        
        if act.mode is not None and act.mode == 'Say':
            act = self.refine_say_act(act, task)
        
        if act is not None:
            print(f'actualized task: {act.to_string()}')
            return act
        else:
            print(f'No action constructed, presumably duplicate')
            return None

    def validate_task(self, task):
        """Validate a task"""
        if task is None:
            return False
        name = hash_utils.find('name', task)
        if name is None or len(name) < 3:
            return False
        description = hash_utils.find('description', task)
        if description is None or len(description) < 3:
            return False
        reason = hash_utils.find('reason', task)
        if reason is None or len(reason) < 3:
            return False
        termination = hash_utils.find('termination', task)
        if termination is None or len(termination) < 3:
            return False
        actors = hash_utils.find('actors', task)
        if actors is None :
            return False
        actors = actors.split(',')
        if len(actors) == 0:
            return False
        for actor in actors:
            if self.context.get_actor_by_name(actor.strip()) is None:
                # not an actor in the context, try to resolve reference?
                pass
        committed = hash_utils.find('committed', task)
        #committed is optional, but if present, must be true
        if not (committed is None or committed=='' or committed.lower().strip() == 'true' or committed.lower().strip() == 'false'):
            return False
        return True

    def update_actions_wrt_say_think(self, source, act_name, act_arg, reason, target=None):
        """Update actions based on speech or thought"""
        if source.name.startswith('dialog'):
            # print(f' in dialog, no action updates')
            return
        if target is None:
            target_name = self.say_target(act_name, act_arg, source)
        elif hasattr(target, 'name'):  # Check if target has name attribute instead of type
            target_name = target.name
        else:
            target_name = target
        # Skip action processing during active dialogs
        if target is not None and self.actor_models.get_actor_model(target_name, create_if_missing=True).dialog.active and source.name.startswith('dialog'):
            # print(f' in active dialog, no action updates')
            return
        
        # Rest of the existing function remains unchanged
        print(f'\n{self.name} Update actions from say or think\n {act_name}, {act_arg};  reason: {reason}')
        
        if 'viewer' in source.name:  # Still skip viewer
            print(f' source is viewer, no action updates')
            return
        
        prompt=[UserMessage(content="""Your task is to analyze the following text.

<text>
{{$text}}
</text>

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
Thought, e.g. 'reflect on my situation', should NOT be reported as an intension to act.
Consider the current task and action reason in determining if there is a new task being committed to.

Do not include any intensions that are similar to those already formed.
Do not include any intensions that are similar to the focus task.

Respond with at most one task expressing an intension in the text.
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#task
#name brief (4-6 words) action name
#description terse (6-8 words) statement of the action to be taken
#reason (6-7 words) on why this action is important now
#termination (5-7 words) condition test which, if met, would satisfy the goal of this action
#actors {{$name}}
#committed True
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
#task
#name Head to the office for the day.
#description Head to the office for the day.
#reason Need to go to work.
#termination Leave for the office
#actors {{$name}}
#committed True
##

Text:
'I really should reassure annie.'

<name>
Hank
</name>

Response:
#task
#name Reassure Annie
#description Reassure Annie
#reason Need to reassure Annie
#termination Reassured Annie
#actors {{$name}}
#committed True
##

Text:
'Good morning Annie. Call maintenance about the disposal noise please.'

<name>
Annie
</name>

Response:
#task
#name Call maintenance about the disposal noise
#description Call maintenance about the disposal noise

Text:
'Reflect on my thoughts and feelings to gain clarity and understanding, which will ultimately guide me towards finding my place in the world.'

Response:
#task
#name Reflect on my thoughts and feelings
#description Reflect on my thoughts and feelings
#reason Gain clarity and understanding
#termination Gained clarity and understanding
#actors Annie
#committed True
##

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the action analysis in hash-formatted text as shown above.
End your response with:
</end>""")]
        response = self.llm.ask({"text":f'{act_name} {act_arg}',
                                 "focus_task":self.focus_task.peek(),
                                 "reason":reason,
                                 "intensions":'\n'.join(task.to_string() for task in self.intensions[-5:]),
                                 "name":self.name}, 
                                 prompt, temp=0.1, stops=['</end>'], max_tokens=150)
        intension_hashes = hash_utils.findall('task', response)
        if len(intension_hashes) == 0:
            print(f'no new tasks in say or think')
            return
        for intension_hash in intension_hashes:
            intension = self.validate_and_create_task(intension_hash)
            if intension:
                print(f'  New task from say or think: {intension_hash.replace('\n', '; ')}')
                self.intensions.append(intension)
    
    def update_individual_commitments_following_conversation(self, target, transcript, joint_tasks=None):
        """Update individual commitments after closing a dialog"""
        
        prompt=[UserMessage(content="""Your task is to analyze the following transcript of a dialog.


<transcript>
{{$transcript}}
</transcript>

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

Extract from this transcript up to three new commitments to act made by self, {{$name}}, to other, {{$target_name}}.

Extract only commitments made by self that are consistent with the entire transcript and remain unfulfilled at the end of the transcript.
Note that the joint_tasks, as listed above, are commitments made by both self and other to work together, and should not be reported as new commitments here.
                            
Does the transcript include an intension for 'I' to act alone, that is, a new task being committed to individually? 
An action can be physical or verbal.
Thought, e.g. 'reflect on my situation', should NOT be reported as an action.
Consider the all_tasks pendingand current task and action reason in determining if a candidate task is in fact new.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#task
#name brief (4-6 words) action name
#description terse (6-8 words) statement of the action to be taken
#actors {{$name}}
#reason (6-7 words) on why this action is important now
#termination (5-7 words) condition test which, if met, would satisfy the goal of this action
#committed True
##

In refering to other actors, always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', etc.
                            

===Examples===

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
#task
#name bring ropes
#description bring ropes to meeting with Jean
#reason in case the well handle breaks
#termination Met Jean by the well
#actors Francoise
#committed True
##


===End Examples===

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
                                 prompt, temp=0.1, stops=['</end>'], max_tokens=180)
        source = Task('dialog with '+target.name, 
                      description='dialog with '+target.name, 
                      reason='dialog with '+target.name, 
                      termination='natural end of dialog', 
                      goal=None,
                      actors=[self, target])    
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

<all_tasks> 
{{$all_tasks}}
</all_tasks>

<focus_task>
{{$focus_task}}
</focus_task>

<reason>
{{$reason}}
</reason>

<transcript>
{{$transcript}}
</transcript>


Extract from this transcript the single most important new commitment to act jointly made by self, {{$name}} and other, {{$target_name}}, if any. Otherwise respond None.
Extract only new joint actions that are consistent with the entire transcript and remain unfulfilled at the end of the transcript.
If more than one joint action is found, and they are similar, combine them into a single commitment.
If more than one joint action is found, and they are different, and there are dependencies among them, use the 'needs' tag to indicate the dependencies.
                            
An action can be physical or verbal.
Thought, e.g. 'reflect on our situation', should NOT be reported as a commitment to act.
Consider the all_tasks and current task and action reason in determining if there is a new task being committed to.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#task
#name brief (4-6 words) action name
#description terse (6-8 words) statement of the action to be taken
#actors {{$name}}, {{$target_name}}
#reason (6-7 words) on why this action is important now
#termination (5-7 words) condition test which, if met, would satisfy the goal of this action
#committed True
##

In refering to other actors, always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', 'other', etc.
                            

===Examples===

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
#task
#name Meet by the well
#description Meet by the well  
#actors Jean, Francoise
#reason get water for the south field
#termination water bucket is full
##
#task
#name Water south field
#description Water the south field
#needs Meet by the well
#actors Jean, Francoise
#reason crops are getting parched
#termination south field is watered
##
#task
#name Check the wheat
#description Check the wheat
#actors Jean, Francoise
#reason To ensure it's not getting too ripe
#termination wheat condition is checked

##

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the action analysis in hash-formatted text as shown above.
End your response with:
</end>
""")]
        response = self.llm.ask({"transcript":transcript, 
                                 "all_tasks":'\n'.join(hash_utils.find('name', task) for task in self.tasks),
                                 "focus_task":self.focus_task.peek(),
                                 "reason":self.focus_task.peek().reason if self.focus_task.peek() else '',
                                 "name":self.name, 
                                 "target_name":target.name}, 
                                 prompt, temp=0.1, stops=['</end>'], max_tokens=240)
        source = Task('dialog with '+target.name, 
                      description='dialog with '+target.name, 
                      reason='dialog with '+target.name, 
                      termination='natural end of dialog', 
                      goal=None,
                      actors=[self, target])    
        intension_hashes = hash_utils.findall('task', response)
        if len(intension_hashes) == 0:
            print(f'no new joint intensions in turn')
            return
        for intension_hash in intension_hashes:
            intension = self.validate_and_create_task(intension_hash)
            if intension:
                print(f'\n{self.name} new joint task: {intension_hash.replace('\n', '; ')}')
                self.intensions.append(intension)
        return intension_hashes

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
        
        self.acts(Act(name='tell', mode='Say', target=to_actor, action=message, reason=message, source=source), to_actor, 'tell', message, self.reason, source)
        

    def natural_dialog_end(self, from_actor):
        """ called from acts when a character says something to this character """
        #if self.actor_models.get_actor_model(from_actor.name).dialog.turn_count > 10):
        #    return True
        prompt = [UserMessage(content="""Given the following dialog transcript, rate the naturalness of ending at this point.

<transcript>
{{$transcript}}
</transcript>
                              
For example, if the last entry in the transcript is a question that expects an answer (as opposed to merely musing), ending at this point is likely not natural.
On the other hand, if the last entry is an agreement to act this may be a natural end.
Respond only with a rating between 0 and 10, where
 - 0 requires continuation of the dialog (ie termination at this point would be unnatural)
 - 10 indicates continuation is unexpected, unnatural, or repetitious.   
                                                  
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]   
        transcript = self.actor_models.get_actor_model(from_actor.name).dialog.get_current_dialog()
        response = self.llm.ask({"transcript":transcript}, prompt, temp=0.1, stops=['</end>'], max_tokens=180)
        if response is None:
            return False
        try:
            rating = int(response.lower().strip())
        except ValueError:
            print(f'{self.name} natural_dialog_end: invalid rating: {response}')
            rating = 7
        # force end to run_on conversations
        end_point = rating > 8 or random.randint(4, 11) < rating or rating + len(transcript.split('\n')) > random.randint(10,14)
        print(f'{self.name} natural_dialog_end: rating: {rating}, {end_point}')
        return end_point
            
    async def hear(self, from_actor: Character, message: str, source: Task=None, respond: bool=True):
        """ called from acts when a character says something to this character """
        # Initialize dialog manager if needed
        print(f'\n{self.name} hears from {from_actor.name}: {message}')
       
        # Special case for Owl-Doc interactions
        if self.name == 'Owl' and from_actor.name == 'Doc':
            # doc is asking a question or assigning a task
            new_task_name = self.random_string()
            new_task = f"""#task
#name {new_task_name}
#description {message}
#termination_check Responded
#actors {self.name}
#committed True
#reason engaging with Doc: completing his assignments.
"""
            self.focus_task.push(new_task_name)
            return

        # Remove text between ellipses - thoughts don't count as dialog
        message = re.sub(r'\.\.\..*?\.\.\.', '', message)
        if self.actor_models.get_actor_model(from_actor.name, create_if_missing=True).dialog.active is False:
            # new dialog, create a new dialog task, but only if we don't already have a dialog task, no nested dialogs for now
            if self.focus_task.peek() and self.focus_task.peek().name.startswith('dialog'):
                print(f'{self.name} already has a dialog task, assertion error')
                self.focus_task.pop()
                raise Exception(f'{self.name} already has a dialog task, assertion error')
            # we don't have a dialog task, so we activate a new one
            self.actor_models.get_actor_model(from_actor.name).dialog.activate(source)
            # create a new dialog task
            dialog_task = Task('dialog with '+from_actor.name, 
                                description='dialog with '+from_actor.name, 
                                reason=from_actor.name+' says '+message, 
                                termination='natural end of dialog', 
                                goal=None,
                                actors=[self, from_actor])
            self.focus_task.push(dialog_task)
            self.actor_models.get_actor_model(from_actor.name, create_if_missing=True).dialog.add_turn(from_actor, message)

        # otherwise, we have an active dialog in progress, decide whether to close it or continue it
        else:
            self.actor_models.get_actor_model(from_actor.name, create_if_missing=True).dialog.add_turn(from_actor, message)
            if self.natural_dialog_end(from_actor) or (self.name == 'viewer'): # test if the dialog has reached a natural end, or hearer is viewer.

                dialog = self.actor_models.get_actor_model(from_actor.name).dialog.get_current_dialog()
                self.add_perceptual_input(f'Conversation with {from_actor.name}:\n {dialog}', percept=False, mode='auditory')
                self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
                self.focus_task.pop()
                if not self.focus_task.peek():
                    # no previous focus task!
                    print(f'{self.name} has no previous focus task!')
                if self.name != 'viewer' and from_actor.name != 'viewer':
                    joint_tasks = self.update_joint_commitments_following_conversation(from_actor, 
                                                                                    from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog())
                    self.update_individual_commitments_following_conversation(from_actor, 
                                                                        self.actor_models.get_actor_model(from_actor.name).dialog.get_current_dialog(),
                                                                        joint_tasks)
                # it would probably be better to have the other actor deactivate the dialog itself
                dialog = from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog()
                from_actor.add_perceptual_input(f'Conversation with {self.name}:\n {dialog}', percept=False, mode='auditory')
                from_actor.actor_models.get_actor_model(self.name).dialog.deactivate_dialog()
                if from_actor.focus_task.peek() and from_actor.focus_task.peek().name.startswith('dialog'):
                    from_actor.focus_task.pop()
                if not self.focus_task.peek():
                    # no previous focus task!
                    print(f'{from_actor.name} has no previous focus task!')
                if from_actor.name != 'viewer' and self.name != 'viewer':
                    joint_tasks = from_actor.update_joint_commitments_following_conversation(self, 
                                                                              from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog())
                    from_actor.update_individual_commitments_following_conversation(self, 
                                                                              from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog(),
                                                                              joint_tasks)
                self.driveSignalManager.recluster()
                from_actor.driveSignalManager.recluster()
                return

        text, response_source = self.generate_dialog_turn(from_actor, message, self.focus_task.peek()) # Generate response using existing prompt-based method
        action = Act(name='Say', mode='Say', action=text, actors=[self, from_actor], reason=text, source=response_source, target=from_actor)
        await self.act_on_action(action, response_source)

    def generate_dialog_turn(self, from_actor, message, source=None):
        self.memory_consolidator.update_cognitive_model(
            memory=self.structured_memory,
            narrative=self.narrative,
            knownActorManager=self.actor_models,    
            current_time=self.context.simulation_time,
            character_desc=self.character,
            relationsOnly=True
        )
            
        if not self.focus_task.peek():
            raise Exception(f'{self.name} has no focus task')
        duplicative_insert = ''
        prompt = [UserMessage(content="""Given the following character description, current situation, goals, memories, and recent history, 
generate a response to the statement below.

{{$character}}.

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

Your relationship with the speaker is:
                              
<relationship>
{{$relationship}}
</relationship>
                              
Your dialog with the speaker up to this point has been:
                              
<dialog>
{{$dialog}}
</dialog>
                              
your commitments at present are (possibly as a result of this dialog) are:

<commitments>
{{$commitments}}
</commitments>


Given all the above, generate a response to the statement below:
                              
<statement>
{{$statement}}
</statement>
                              
Use the following XML template in your response:
                              
<response>response to this statement</response>
<reason>terse (4-6 words) reason for this answer</reason>

{{$duplicative_insert}}

Reminders: 
- The response can include body language or facial expressions as well as speech
- Respond in a way that advances the dialog. E.g., express an opinion or propose a next step.
- If the intent is to agree, state agreement without repeating the statement.
- Speak in your own voice. Do not echo the speech style of the Input. 
- Respond in the style of natural spoken dialog. Use short sentences and casual language.
 
Respond only with the above XML
Do not include any additional text. 
End your response with:
</end>
""")]

        mapped_goals = self.map_goals()
        activity = ''
        if self.focus_task.peek() != None and self.focus_task.peek().name.startswith('dialog'):
            activity = f'You are currently actively engaged in {self.focus_task.peek().name}'

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(6)
        memory_text = '\n'.join(memory.text for memory in recent_memories)
        
        #print("Hear",end=' ')
        duplicative_insert = ''
        trying = 0

        answer_xml = self.llm.ask({
            'character': self.character,
            'statement': f'{from_actor.name} says {message}',
            "situation": self.context.current_state,
            "name": self.name,
                "goals": mapped_goals,
            "memories": memory_text,  # Updated from 'memory'
            "activity": activity,
            'history': self.narrative.get_summary('medium'),
                'dialog': self.actor_models.get_actor_model(from_actor.name).dialog.get_current_dialog(),
                'relationship': self.actor_models.get_actor_model(from_actor.name).relationship,
                'commitments': '\n'.join([str(hash_utils.find('name', task)) 
                                          + ' - ' + str(hash_utils.find('description', task)) 
                                          + ' - ' + str(hash_utils.find('reason', task)) 
                                          for task in self.tasks if hash_utils.find('committed', task) == 'True']),
                'duplicative_insert': duplicative_insert
            }, prompt, temp=0.8, stops=['</end>'], max_tokens=180)
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
            if from_actor.name == 'viewer':
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
    
    
    def choose(self, sense_data, task_choices):
        if len(task_choices) == 1:
            return 0
        prompt = [UserMessage(content=self.character.replace('\n', ' ') + """The task is to order the execution of a set of task options, listed under <tasks> below.
Your current situation is:

<situation>
{{$situation}}
</situation>

Your goals are:

<goals>
{{$goals}}
</goals>

Your recent memories include:

<recent_memories>
{{$memories}}
</recent_memories>

Especially relevant to your task options, you remember:
                              
{{$tasks_memories}}

A summary of recent events is:
<recent_history>
{{$history}}
</recent_history>

Your current tasks include:
<tasks>
{{$tasks}}
</tasks>

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
    and respond with the following XML format:

<task><label>label of chosen task</label><order>execution order (an int, 1-{{$num_options}})</order></task>
<task>...</task>

Review to ensure the assigned execution order is consistent with the task option dependencies, urgency, and importance.
Respond only with the above XML, instantiated with the selected task label from the Task list. 
Do not include any introductory, explanatory, or discursive text, 
End your response with:
</end>
"""
                              )]

        mapped_goals = self.map_goals()
        labels = ['Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega']
        random.shuffle(labels)
        
        # Get recent memories from structured memory
        recent_memories = self.structured_memory.get_recent(8)  # Get 5 most recent memories
        memory_text = '\n'.join(memory.text for memory in recent_memories)
        
        # Change from dict to set for collecting memory texts
        tasks_memories = set()  # This is correct
        for task in task_choices:
            memories = self.memory_retrieval.get_by_text(
                memory=self.structured_memory,
                search_text=hash_utils.find('name', task)+': '+hash_utils.find('description', task) + ' ' + hash_utils.find('reason', task),
                threshold=0.1,
                max_results=5
            )
            # More explicit memory text collection
            for memory in memories:
                tasks_memories.add(memory.text)  # Add each memory text individually
                
        tasks_memories = '\n'.join(tasks_memories)  # Join at the end
        #print("Choose",end=' ')
        response = self.llm.ask({
                'input': sense_data + self.sense_input, 
                'history': self.narrative.get_summary('medium'),
                "memories": memory_text,
                "situation": self.context.current_state,
            "goals": mapped_goals, 
                "drives": '\n'.join(drive.text for drive in self.drives),
            "tasks": '\n'.join([str(hash_utils.find('name', task)) for task in self.tasks]),
            "tasks_memories": tasks_memories,
            "tasks": self.format_tasks(task_choices, labels[:len(task_choices)]),
            "num_options": len(task_choices)
        }, prompt, temp=0.0, stops=['</end>'], max_tokens=150)
        # print(f'sense\n{response}\n')
        index = -1
        ordering = xml.findall('<task>', response)
        pairs = [(xml.find('<label>', item).strip(), xml.find('<order>', item).strip()) for item in ordering]
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        label_choice = choice.exp_weighted_choice(sorted_pairs, 0.75)
        task_to_execute = task_choices[labels.index(label_choice[0])]
        print(f'  Chosen label: {label_choice} task: {task_to_execute.replace('\n', '; ')}')
        return task_to_execute

    async def cognitive_cycle(self, sense_data='', ui_queue=None):
        """Perform a complete cognitive cycle"""
        self.context.message_queue.put({'name':' \n'+self.name, 'text':f'cognitive cycle'})
        await asyncio.sleep(0.1)
        self.thought = ''
        self.memory_consolidator.update_cognitive_model(self.structured_memory, 
                                                  self.narrative, 
                                                  self.actor_models,
                                                  self.context.simulation_time, 
                                                  self.character.strip(),
                                                  relationsOnly=True )


        # clear intension tasks that are satisfied - a temporary hack to check if others actions satisfied a task.

        if self.focus_goal:
            focus_goal_satisfied = self.clear_goal_if_satisfied(self.focus_goal)
            if focus_goal_satisfied:
                self.driveSignalManager.recluster()
                self.generate_goal_alternatives()
                self.generate_task_alternatives()

        if self.focus_task.peek():
            self.clear_task_if_satisfied(self.focus_task.peek())
            
        if not self.focus_task.peek():
            if not self.focus_goal:
                self.generate_goal_alternatives()
            self.generate_task_alternatives()

        await self.step_task()

       
    async def step_task(self, sense_data='', ui_queue=None):
  
        # if I have an active task, keep on with it.
        if not self.focus_task.peek():
            raise Exception(f'No focus task')

        print(f'\n{self.name} decides, focus task {self.focus_task.peek().name}')
        task = self.focus_task.peek()
        # iterate over task until it is no longer the focus task. 
        # This is to allow for multiple acts on the same task, clear_task_if_satisfied will stop the loop with poisson timeout or completion
        while self.focus_task.peek() is task:
            action = self.actualize_task(task)
            if action:
                await self.act_on_action(action, task)
                #this will affect selected act and determine consequences
                if self.context:
                    self.context.simulation_time += timedelta(minutes=15)
                if self.focus_task.peek() is task:
                    self.clear_task_if_satisfied(task)
            else:
                print(f'No action for task {task.name}')
                return
                return

    async def act_on_action(self, action, task):
        self.act = action
        if task:
            task.acts.append(action)
        act_name = action.name
        if act_name is not None:
            self.act_name = act_name.strip()
        act_arg = action.action
        print(f'\n{self.name} act_on_action: {act_name} {act_arg}')
        self.reason = action.reason
        source = action.source
        #print(f'{self.name} choose {action}')
        target = None
            #responses, at least, explicitly name target of speech.
        if action.target and action.target != self:
            target_name = action.target.name
            target = action.target
        else:
            target_name = self.say_target(act_name, act_arg, source)
            if target_name != None:
                target = self.context.get_actor_by_name(target_name)
        #self.context.message_queue.put({'name':self.name, 'text':f'character_update'})
        #await asyncio.sleep(0.1)
        await self.acts(action, target, act_name, act_arg, self.reason, source)

    def format_thought_for_UI (self):
        #<action> <mode>{mode}</mode> <act>{action}</act> <reason>{reason}</reason> <source>{source}</source></action>'
        action = ''
        if self.act:
            action = f'{self.act.name}: {self.act.action}'
        reason = ''
        if self.focus_task.peek() and type(self.focus_task.peek()) == Task:
            reason = f'{self.focus_task.peek().reason}'
        return f'{self.thought.strip()}'
   
    def format_tasks_for_UI(self):
        """Format tasks for UI display as array of strings"""
        tasks = []
        if self.focus_goal:
            tasks.append(f"Goal: {self.focus_goal.short_string()}")
        if self.focus_task.peek():
            tasks.append(f"Task: {self.focus_task.peek().short_string()}")
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
            'description': self.character.strip(),  # For image generation
            'history': self.format_history_for_UI().strip(), # Recent history, limited to last 5 entries
            'narrative': {
                'recent_events': self.narrative.recent_events,
                'ongoing_activities': self.narrative.ongoing_activities,
                'relationships': self.actor_models.get_known_relationships(),
            },
            'signals': self.driveSignalManager.get_scored_clusters()[0][0].text
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
        
        result = self.llm.ask({}, prompt, max_tokens=100, stops=['</end>'])
        return result or ""

    def get_explorer_state(self):
        """Return detailed state for explorer UI"""
        focus_task = self.focus_task.peek()
        last_act = self.action_history[-1] if self.action_history and len(self.action_history) > 0 else None

        return {
            'currentTask': focus_task.to_string() if focus_task else 'idle',
            'actions': [act.to_string() for act in (focus_task.acts if focus_task else [])],  # List[str]
            'lastAction': {
                'name': last_act.name if last_act else '',
                'result': last_act.result if last_act else '',
                'reason': last_act.reason if last_act else ''
            },
            'drives': [{'text': drive.text or ''} for drive in (self.drives or [])],
            'emotional_state': [
                {
                    'text': percept.content or '',
                    'mode': percept.mode.name if percept.mode else 'unknown',
                    'time': percept.timestamp.isoformat() if percept.timestamp else datetime.now().isoformat()
                }
                for percept in (self.perceptual_state.get_current_percepts(chronological=True) or [])
            ],
            'memories': [
                {
                    'text': memory.text or '',
                    'timestamp': memory.timestamp.isoformat() if memory.timestamp else datetime.now().isoformat()
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

