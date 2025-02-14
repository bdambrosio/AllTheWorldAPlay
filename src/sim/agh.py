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

from sim.cognitive.DialogManager import Dialog
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from sim.cognitive.perceptualState import PerceptualInput, PerceptualState, SensoryMode
from sim.cognitive.knownActor import KnownActor, KnownActorManager
import re

class Mode(Enum):
    THINK = "Think"
    SAY = "Say"
    DO = "Do" 
    MOVE = "Move"
    LOOK = "Look"
    LISTEN = "Listen"

def find_first_digit(s):
    for char in s:
        if char.isdigit():
            return char
    return None  # Return None if no digit is found


class Stack:
    def __init__(self):
        """Simple stack implementation"""
        self.stack = []
        print("Stack initialized")  # Debug print

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


# Character base class
class Character:
    def __init__(self, name, character_description, server='local'):
        print(f"Initializing Character {name}")  # Debug print
        self.name = name
        self.character = character_description
        self.llm = llm_api.LLM(server)
        self.context = None
        self.priorities = []  # displayed by main thread at top in character intentions widget
        self.show = ''  # to be displayed by main thread in UI public text widget
        self.goals = {}
        self.intentions = []
        self.previous_action = ''
        self.reason = ''  # reason for action
        self.thought = ''  # thoughts - displayed in character thoughts window
        self.sense_input = ''
        self.widget = None
        
        # Initialize active_task stack
        self.active_task = Stack()
        print(f"Character {name} active_task initialized: {self.active_task}")  # Debug print
        
        self.last_acts = {}  # priorities for which actions have been started, and their states
        self.act_result = ''
        self.wakeup = True
        # Memory system initialization - will be set up by derived classes
        self.structured_memory = StructuredMemory(owner=self)
        self.memory_consolidator = MemoryConsolidator(self.llm)
        self.memory_retrieval = MemoryRetrieval()
        # Map integration
        self.mapAgent = None
        self.world = None
        self.my_map = [[{} for i in range(100)] for i in range(100)]
        self.x = 50
        self.y = 50

        # Initialize narrative
        self.narrative = NarrativeSummary(
            recent_events="",
            ongoing_activities="",
            last_update=datetime.now(),  # Will be updated to simulation time
            active_drives=[]
        )

        self.drives: List[Drive] = []  # Initialize empty drive list
        self.perceptual_state = PerceptualState(self)
        self.last_sense_time = datetime.now()
        self.sense_threshold = timedelta(hours=4)
        self.intention = None

    def set_context(self, context):
        self.context = context
        self.actor_models = KnownActorManager(self, context)
        
    # Required memory system methods that must be implemented by derived classes
    def add_to_history(self, message: str):
        """Base method for adding memories - must be implemented by derived classes"""
        raise NotImplementedError("Derived classes must implement add_to_history")

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

        # Get recent memory content from structured_memory
        recent_memories = self.structured_memory.get_recent(self.new_memory_cnt)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        # Update state and priorities
        self.generate_goals()
        self.update_priorities()

        # Reset new memory counter
        self.new_memory_cnt = 0


    def get_task(self, task_name):
        for candidate in self.priorities:
            if task_name == hash_utils.find('name', candidate):
                print(f'found existing task\n  {task_name}')
                return candidate
        return None

    def find_or_make_task_xml(self, task_name, reason):
        candidate = self.get_task(task_name)
        if candidate is not None:
            return candidate
        new_task = f'#plan\n#name {task_name}\n#description {task_name}\n#reason> {reason}\n#actors {self.name}\n#committed False\n##'
        if self.get_task(task_name) != None:
            self.priorities.remove(self.get_task(task_name))
        self.priorities.append(new_task)
        print(f'created new task to reflect {task_name}\n {reason}')
        return new_task

    # Action methods
    def acts(self, target, act_name, act_arg='', reason='', source=''):
        """Execute an action with tracking and consequences - must be implemented by derived classes"""
        raise NotImplementedError("Derived classes must implement acts")

    def clear_task_if_satisfied(self, task_xml, consequences, world_updates):
        """Check if task is complete and update state - must be implemented by derived classes"""
        raise NotImplementedError("Derived classes must implement clear_task_if_satisfied")

    # Dialog methods
    def tell(self, to_actor, message, source='dialog', respond=True):
        """Initiate or continue dialog - must be implemented by derived classes"""
        raise NotImplementedError("Derived classes must implement tell")

    def hear(self, from_actor, message: str, source='dialog', respond=True):
        """Process incoming message"""
        raise NotImplementedError("Derived classes must implement hear")
        self.add_perceptual_input(f"You hear {from_actor.name} say: {message}")

    def say_target(self, act_name, text, source=None):
        """Determine the intended recipient of a message"""
        if source == 'inject' or source == 'watcher':
            return 'watcher'
        if len(self.context.actors) == 2:
            for actor in self.context.actors:
                if actor.name != self.name:
                    return actor.name
        elif 'dialog with' in source:
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
If the message is a spoken message, respond with the name of the intended recipient.
Respond using the following XML format:

<target>
  <name>intended recipient name</name>
</target>

End your response with:
</end>
""")]
        print('say target')
        response = self.llm.ask({
            'character': self.character,
            'history': self.narrative.get_summary('medium'),
            'actors': '\n'.join([actor.name for actor in self.context.actors if actor != self]),
            'action_type': 'internal thought' if act_name == 'Think' else 'spoken message',
            "message": text
        }, prompt, temp=0.2, stops=['</end>'], max_tokens=180)

        candidate = xml.find('<name>', response)
        if candidate is not None:
            for actor in self.context.actors:
                if actor.name in candidate:
                    return actor.name
        else:
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

And your priorities are:

<priorities>
{{$priorities}}
</priorities>

You see the following:

<view>
{{$view}}
</view>

Provide a concise description of what you notice, highlighting the most important features given your current interests, state and priorities. 
Respond using the following XML format:

<perception>a concise (30 words or less) description of perceptual content</perception>

End your response with:
<end/>
""")]
        print("Look",end=' ')
        response = self.llm.ask({"view": text_view, 
                                 "interests": interest,
                                 "state": self.narrative.get_summary(), 
                                 "priorities": self.priorities}, 
                                prompt, temp=0.2, stops=['<end/>'], max_tokens=100)
        percept = xml.find('<perception>', response)
        perceptual_input = PerceptualInput(
            mode=SensoryMode.VISUAL,
            content=percept,
            timestamp=self.context.simulation_time,
            intensity=0.7,  # Medium-high for direct observation
        )       
        self.perceptual_state.add_input(perceptual_input)    
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

class Agh(Character):
    def __init__(self, name, character_description, server='local', mapAgent=True, always_respond=False):
        print(f"Initializing Agh {name}")  # Debug print
        super().__init__(name, character_description, server)
        self.llm = llm_api.LLM(server)
                
        # Initialize drives
        self.drives = [
            Drive( "immediate physiological needs: survival, water, food, clothing, shelter, rest."),
            Drive("safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events."),
            Drive("assurance of short-term future physiological needs (e.g. adequate water and food supplies, shelter maintenance)."),
            Drive("love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friendship, intimacy, trust, acceptance.")
        ]

        # Initialize dialog management        
        # World integration attributes
        if mapAgent:
            self.mapAgent = None  # Will be set later
            self.world = None
            self.my_map = [[{} for i in range(100)] for i in range(100)]
            self.x = 50
            self.y = 50
            
        self.always_respond = always_respond
        
        # Action tracking
        self.previous_action = ''
        self.act_result = ''

        # Update narrative with drives (drives are strings)
        self.narrative.active_drives = self.drives  # Direct assignment, no name attribute needed
        
        # Initialize memory systems
        self.structured_memory = StructuredMemory(owner=self)
        self.memory_consolidator = MemoryConsolidator(self.llm)
        self.memory_retrieval = MemoryRetrieval()
        self.new_memory_cnt = 0
        self.watcher_message_pending = False
        self.next_task = None  # Add this line
    def set_llm(self, llm):
        self.llm = llm
        if self.memory_consolidator is not None:
            self.memory_consolidator.set_llm(llm)

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
        description = self.character
        try:
            context = ''
            i = 0
            candidates = self.context.current_state.split('.')
            while len(context) < 84 and i < len(candidates):
                context += candidates[i]+'. '
                i +=1
            context = context[:96]
            description = self.name + ', '+'. '.join(self.character.split('.')[:2])[6:]
            
            description = description
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

            concerns = ''
            for priority in self.priorities:
                concern = hash_utils.find('name', priority) + '. '+hash_utils.find('reason', priority)
                concerns = concerns + '; '+concern
            state = description + '.\n '+concerns +'\n'+ context
            recent_memories = self.structured_memory.get_recent(8)
            recent_memories = '\n'.join(memory.text for memory in recent_memories)


            print("Char generate image description", end=' ')
            response = self.llm.ask({ "description": state, "recent_memories": recent_memories}, prompt, temp=0.2, stops=['<end/>'], max_tokens=10)
            if response:
                description = description[:192-min(len(context), 48)] + f'. {self.name} feels '+response.strip()+'. '+context
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
            print("Perceptual input",end=' ')
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

    def format_history(self, n=2):
        """Get n most recent memories"""
        recent_memories = self.structured_memory.get_recent(n)
        return '\n'.join(memory.text for memory in recent_memories)
    
    def generate_goals(self):
        """Generate states to track, derived from drives and current memory context"""
        self.goals = {}
        for drive in self.drives:
            former_goal_name = next((self.goals[key]['name'] for key in self.goals if self.goals[key]['drive'] == drive), None)
            goal = self.generate_goal(drive)
            if goal:
                if former_goal_name:
                    del self.goals[former_goal_name]
                self.goals[goal['name']] = goal

    def generate_goal(self, drive, progress=None):
        """Generate states to track, derived from drives and current memory context"""
        
        prompt = [UserMessage(content="""Given a Drive and related memories, assess the current state relative to that drive.

<drive>
{{$drive}}
</drive>

<recent_memories>
{{$recent_memories}}
</recent_memories>

<drive_memories>
{{$drive_memories}}
</drive_memories>

<situation>
{{$situation}}
</situation>

<character>
{{$character}}
</character>

<relationships>
{{$relationships}}
</relationships>

Consider this drive and and your memories, situation, and character, paying special attention to drive_memories, 
and determine your current state relative to this drive.  
Consider:
1. How well the drive's needs are being met
2. Recent events that affect the drive
3. The importance scores of relevant memories
4. Any patterns or trends in the memories

Respond with a goal, in four parts: a terse (5-6 words) description of the goal, an urgency assessment (1 word), 
    a terse (4-7 words) statement of how the interaction among drive, character, and situation created this goal, 
    and a termination condition (5-6 words) that would reduce its urgency.
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#goal 
#name terse (5-6 words) description of this goal
#urgency high/medium/low
#trigger terse (4-7 words) restatement of primary situation or memory that most relates to this goal
#termination> 5-6 word statement of condition that would somewhat satisfy this goal
##

Respond ONLY with the above hash-formatted text.
End response with:
<end/>
""")]

        print(f'{self.name} generating state for drive: {drive.text}')
            
        # Get relevant memories
        drive_memories = self.memory_retrieval.get_by_drive(
            memory=self.structured_memory,
            drive=drive,
            threshold=0.1,
            max_results=5
        )
            
            # Get recent memories regardless of drive
        recent_memories = self.structured_memory.get_recent(5)
            
        # Format memories for LLM
        drive_memories_text = "\n".join([
            f"(importance: {mem.importance:.1f}): {mem.text}"
            for mem in drive_memories
        ])
            
        recent_memories_text = "\n".join([
            f"(importance: {mem.importance:.1f}): {mem.text}"
            for mem in recent_memories
        ])
            
        # Generate state assessment
        response = self.llm.ask({
            "drive": drive.text,
            "progress": progress,
            "recent_memories": recent_memories_text,
            "drive_memories": drive_memories_text,
            "relationships": self.actor_models.format_relationships(include_transcript=True),
            "situation": self.context.current_state if self.context else "",
            "character": self.character
        }, prompt, temp=0.3, stops=['<end/>'])     
            
        # Parse response
        try:
            name = hash_utils.find('name', response)
            urgency = hash_utils.find('urgency', response)
            trigger = hash_utils.find('trigger', response)
            termination = hash_utils.find('termination', response)
                
            if name and urgency:
                self.goals[name] = {
                    "drive": drive,
                    "name": name,
                    "urgency": urgency,
                    "trigger": trigger if trigger else "",
                    "termination": termination if termination else ""
                }
            else:
                 print(f"Warning: Invalid goal generation response for {drive.text}")
                    
        except Exception as e:
            print(f"Error parsing goal generation response: {e}")
            traceback.print_exc()
                
        print(f'{self.name} generated goal: ' + name + '; ' + urgency)
        return self.goals[name]


    def test_state_termination(self, state, consequences, updates=''):
        """Generate a state to track, derived from basic drives"""
        prompt = [UserMessage(content="""An instantiated state for a basic drive is provided below 
Your task is to test whether the instantiated state for this Drive has been satisfied as a result of recent events.

<state>
{{$state}}
</state>

<situation>
{{$situation}}
</situation>

<character>
{{$character}}
</character>

<recent_memories>
{{$memories}}
</recent_memories>

<history>
{{$history}}
</history>
 
<events>
{{$events}}
</events>

<termination_check>
{{$termination_check}}
</termination_check>

Respond using this XML format:

<level>True / False</level>

The 'level' above should be True if the termination test is met in Events or recent History, and False otherwise.  

Respond ONLY with the above XML.
Do not include any introductory, explanatory, or discursive text.
End your response with:
<end/>
""")]

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(5)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        response = self.llm.ask({
            "state": state,
            "situation": self.context.current_state,
            "memories": memory_text,  # Updated from 'memory'
            "character": self.character,
            "history": self.format_history(),
            "events": consequences + '\n' + updates,
            "termination_check": state
        }, prompt, temp=0.3, stops=['<end/>'], max_tokens=60)

        satisfied = xml.find('<level>', response)
        if satisfied or satisfied.lower().strip() == 'true':
            print(f'State {state} Satisfied!')
        return satisfied

    def map_goals(self):
        """ map state for llm input """
        mapped = []
        for key, item in self.goals.items():
            trigger = item['trigger']
            urgency = item['urgency']
            termination = item['termination']
            mapped.append(f"- Goal: {key}; Urgency: {urgency}; Trigger: {trigger}; Termination: {termination}")
        return '\n'.join(mapped)


    def get_task_last_acts_key(self, term):
        """ checks for name in Last_acts!"""
        for task in list(self.last_acts.keys()):
            match=self.synonym_check(task, term)
            if match:
                return task
        return None
            
    def set_task_last_act(self, term, act):
        # pbly don't need this, at set we have canonical task
        task = self.get_task_last_acts_key(term)
        if task == None:
            print(f'SET_TASK_LAST_ACT {self.name} no match found for term: {term}, {act}')
            self.last_acts[term] = act
        else:
            print(f'SET_TASK_LAST_ACT {self.name} match found: term {term}, task {task}\n  {act}')
            self.last_acts[task] = act

    def get_task_last_act(self, term):
        task = self.get_task_last_acts_key(term)
        if task == None:
            return 'None'
        else:
            return self.last_acts[task]

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

        print("Repetitive")
        result = self.llm.ask({
            'history': history,
            'response': new_response
        }, prompt, temp=0.2, stops=['<end/>'], max_tokens=100)

        if 'true' in result.lower():
            return True
        else:
            return False

    def clear_task_if_satisfied(self, task, consequences, world_updates):
        """Check if task is complete and update state"""
        termination_check = hash_utils.find('termination', task) if task != None else None
        if termination_check is None or termination_check == '':
            return

        # Test completion through cognitive processor's state system
        satisfied, progress = self.test_priority_termination(
            termination_check, 
            consequences,
            world_updates
        )

        if satisfied:
            task_name = hash_utils.find('name', task)
            if task_name == self.active_task.peek():
                self.active_task.pop()

            try:
                self.priorities.remove(task)
            except Exception as e:
                print(str(e))

            # should we indeed delete all queued steps for a satisfied task? 
            new_intentions = []
            for intention in self.intentions:
                if not task_name.startswith('dialog'): # dialog steps are not deleted
                    if xml.find('name', intention) != task_name:
                        new_intentions.append(intention)
                    else:
                        print('deleting intention for satisfied task!')
                else:
                    new_intentions.append(intention)
            self.intentions = new_intentions

        return satisfied


    def acts(self, target, act_name, act_arg='', reason='', source=''):
        """Execute an action and record results"""
        # Create action record with state before action
        if act_name not in Mode:
            raise ValueError(f'Invalid action name: {act_name}')
        mode = Mode(act_name)
        self.act_result = ''
    
        # Store current state and reason
        self.reason = reason
        if act_name is None or act_arg is None or len(act_name) <= 0 or len(act_arg) <= 0:
            return


         # Update thought display
        if act_name == 'Think':
            self.thought = act_arg
            self.show += f" \n...{self.thought}..."
            self.context.message_queue.put({'name':self.name, 'text':f"{self.show}"})  
            self.show = '' # has been added to message queue!
        else:
            self.thought +=  "\n..." + self.reason + "..."

        # Update active task if needed
        if (act_name == 'Do' or act_name == 'Say') and not source.startswith('dialog'):
            if self.active_task.peek() != source and source not in self.active_task.stack:
                 # if we are not already on this task, push it onto the stack. 
                 self.active_task.push(source)

        # Handle world interaction
        if act_name == 'Do':
            # Get action consequences from world
            consequences, world_updates = self.context.do(self, act_arg)
        
            if source == None:
                source = self.active_task.peek()
            task = self.get_task(source) if source != None else None
        
            # Update displays

            self.show +=  act_arg+'\n Resulting in ' + consequences.strip()
            self.context.message_queue.put({'name':self.name, 'text':self.show})
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
                self.add_perceptual_input(f"\nYou {act_name}: {act_arg}\n  {percept}", mode='visual')
        elif act_name == 'Look':
            percept = self.look(interest=act_arg)
            self.show += act_arg + '.\n  sees ' + percept + '. '
            self.context.message_queue.put({'name':self.name, 'text':self.show})
            self.add_perceptual_input(f"\nYou look: {act_arg}\n  {percept}", mode='visual')
        self.previous_action = act_name

        if act_name == 'Think' or act_name == 'Say':
            # Update intentions based on thought
            self.update_intentions_wrt_say_think(source, act_name, act_arg, reason, target) 
            self.add_perceptual_input(f"\nYou {act_name}: {act_arg}", percept=False, mode='internal')

        # After action completes, update record with results
        # Notify other actors of action
        if act_name != 'Say' and act_name != 'Look' and act_name != 'Think':  # everyone you do or move or look if they are visible
            for actor in self.context.actors:
                if actor != self:
                    if source != 'watcher':  # when talking to watcher, others don't hear it
                        if actor != target:
                            actor_model = self.actor_models.get_actor_model(actor.name)
                            if actor_model != None and actor_model.visible:
                                percept = actor.add_perceptual_input(f"You see {self.name}: '{act_arg}'", percept=False, mode='visual')
                                actor.actor_models.get_actor_model(self.name, create_if_missing=True).infer_goal(percept)
        elif act_name == 'Say':# must be a say
            if False: #act_arg in self.show:
                print('Duplicate Say!')
            else:
                self.show += f"{act_arg}'"
                print(f"Queueing message for {self.name}: {act_arg}")  # Debug
                self.context.message_queue.put({'name':self.name, 'text':act_arg})
            
            content = re.sub(r'\.\.\..*?\.\.\.', '', act_arg)
            self.actor_models.get_actor_model(target.name, create_if_missing=True).dialog.add_turn(self, content)
            target.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.add_turn(self, content)

            for actor in self.context.actors:
                if actor != self:
                    if source != 'watcher':  # when talking to watcher, others don't hear it
                        if actor != target and self.actor_models.get_actor_model(actor.name) != None\
                            and self.actor_models.get_actor_model(actor.name).visible:
                            actor.add_perceptual_input(f"You hear {self.name}: '{act_arg}'", percept=False, mode='auditory')
                        elif actor == target:
                            actor.hear(self, act_arg, source)
            # in this case 'self' is speaking
            # Remove text between ellipses - thoughts 
            # below done in hear
 

    def update_goals(self, events=''):
        """Update goals based on termination conditions"""
        # Create a list from the keys to avoid modifying dict during iteration
        for goal in list(self.goals.keys()):
            satisfied, progress = self.test_priority_termination(self.goals[goal]['termination'], events)
            if satisfied:
                drive = self.goals[goal]['drive']
                del self.goals[goal]
                new_goal = self.generate_goal(drive, progress=progress)
                self.goals[new_goal['term']] = new_goal


    def update_priorities(self):
        """Update task priorities based on current state and drives"""
        prompt = [UserMessage(content="""Given your character, drives and current goals, and recent memories, create a prioritized set of plans.

<character>
{{$character}}
</character>

<situation>
{{$situation}}
</situation>

<goals>
{{$goals}}
</goals>

<recent_memories>
{{$memories}}
</recent_memories>

<relationships>
{{$relationships}}
</relationships>

<recent_events>
{{$recent_events}}
</recent_events>

Create three specific, actionable plans that address your current needs and situation.
Consider:
1. Your current state assessments
2. Recent memories and events
3. Your basic drives and needs
4. Your goals
5. Your relationships
                              
The three plans should be distinct, and jointly cover all the important aspects of the current situation and your goals.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#plan
#name brief (4-6 words) action name
#description terse (6-8 words) statement of the action to be taken
#reason (6-7 words) on why this action is important now
#actors {self.name}
#committed False
#termination (5-7 words) condition test which, if met, would satisfy the goal of this action
##

In refering to other actors. always use their name, without other labels like 'Agent', 
and do not use pronouns or referents like 'he', 'she', 'that guy', etc.
Respond ONLY with three plans in hash-formatted-text format and each ending with ## as shown above.
Order plans from highest to lowest priority.
End response with:
<end/>
""")]

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(5)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        # Format state for LLM
        goal_text = self.map_goals()
        print("Update Priorities")
        response = self.llm.ask({
            'character': self.character,
            'situation': self.context.current_state,
            'goals': goal_text,
            'memories': memory_text,
            'recent_events': self.narrative.get_summary('medium'),
            'relationships': self.actor_models.format_relationships(include_transcript=True)
        }, prompt, temp=0.7, stops=['<end/>'])

        # Extract plans from response
        # keep 'committed' tasks
        self.priorities = [task for task in self.priorities if hash_utils.find('committed', task)=='True']
        for plan in hash_utils.findall('plan', response):
            if hash_utils.find('name', plan):
                task_name = hash_utils.find('name', plan)
                reason = hash_utils.find('reason', plan)
                if self.get_task(task_name) != None:
                    self.priorities.remove(self.get_task(task_name))
                self.priorities.append(plan)

    def test_priority_termination(self, termination_check, consequences, updates=''):
        """Test if recent acts, events, or world update have satisfied priority"""
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

Respond using this XML format:

Respond with both completion status and progress indication:
<completion>
  <status>complete|partial|insufficient</status>
  <progress>0-100</progress>
  <reason>terse (5-7 words or less) reason for this assessment</reason>
</completion>

Respond ONLY with the above XML
Do not include any introductory, explanatory, or discursive text.
End your response with:
<end/>
""")]


        # Get recent memories
        recent_memories = self.structured_memory.get_recent(5)
        memory_text = '\n'.join(memory.text for memory in recent_memories)


        response = self.llm.ask({
            "termination_check": termination_check,
            "situation": self.context.current_state,
            "memories": memory_text,  # Updated from 'memory'
            "events": consequences + '\n' + updates,
            "character": self.character,
            "history": self.format_history(),
            "relationships": self.narrative.get_summary('medium')
        }, prompt, temp=0.5, stops=['<end/>'], max_tokens=120)

        satisfied = xml.find('<status>', response)
        print(f'{self.name} testing priority termination_check: {termination_check}: {satisfied}')
        progress = xml.find('<progress>', response)
        if satisfied != None and satisfied.lower().strip() == 'complete':
            print(f'Priority {termination_check} Satisfied!')
            return True, progress
        elif satisfied != None and satisfied.lower().strip() == 'partial':
            progress = xml.find('<progress>', response)
            if progress != None:
                try:
                    progress = int(progress.strip())
                    if progress/100.0 > random.random():
                        return True, progress
                except:
                    pass
        return False, progress

    def refine_say_act(self, act_name, act_arg):
        """Refine a say act to be more natural and concise"""
        target_name = self.say_target(act_name, act_arg)
        if target_name is None:
            return act_arg
        
        dialog = self.actor_models.get_actor_model(target_name, create_if_missing=True).dialog.get_transcript(10)
        if dialog is None or len(dialog) == 0:
            return act_arg
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
            "act_name": act_name,
            "act_arg": act_arg,
            "dialog": dialog,
            "target": target_name,
            "relationship": relationship,
            "character": self.character
        }, prompt, temp=0.6, stops=['<end/>'])

        return response

    def actualize_task(self, n, task):
        task_name = hash_utils.find('name', task)
        if task is None or task_name is None:
            raise ValueError(f'Invalid task {n}, {task}')
        last_act = self.get_task_last_act(task_name)
        reason = hash_utils.find('reason', task)

        prompt = [UserMessage(content="""You are {{$character}}.
Your task is to generate an Actionable (a 'Think', 'Say', 'Look', Move', or 'Do') to advance the next step of the following task.

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

Your recent memories include:

<recent_memories>
{{$memories}}
</recent_memories>

Recent history includes:
<history>
{{$history}}
</history>

The previous specific act for this task, if any, was:

<previous_specific_act>
{{$lastAct}}
</previous_specific_act>

And the observed result of that was:
<observed_result>
{{$lastActResult}}.
</observed_result>

Respond with an Actionable, including its Mode and SpecificAct. 

In choosing an Actionable (see format below), you can choose from three Mode values:
- Think - reason about the current situation wrt your state and the task.
- Say - speak, to motivate others to act, to align or coordinate with them, to reason jointly, or to establish or maintain a bond. 
    Say is especially appropriate when there is an actor you are unsure of, you are feeling insecure or worried, or need help.
    For example, if you want to build a shelter with Samantha, it might be effective to Say 'Samantha, let's build a shelter.'
- Look - observe your surroundings, gaining information on features, actors, and resources at your current location and for the eight compass
    points North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- Move - move in any one of eight directions: North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- Do - perform an act (other than move) with physical consequences in the world.

Review your character for Mode preference. (e.g., 'xxx is thoughtful' implies higher percentage of 'Think' Actionables.) 

A SpecificAct is one which:
- Can be described in terms of specific thoughts, words, physical movements or actions.
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
  <specific_act>thoughts, words to speak, DIRECTION (if Move mode), or physical action description</specific_act>
</actionable>

===Examples===

Task:
Situation: increased security measures; State: fear of losing Annie

Response:
<actionable>
  <mode>Do</mode>
  <specific_act>Call a meeting with the building management to discuss increased security measures for Annie and the household.</specific_act>
</actionable>

----

Task:
Establish connection with Joe given RecentHistory element: "Who is this guy?"

Response:
<actionable>
  <mode>Say</mode>
  <specific_act>Hi, who are you?</specific_act>
</actionable>

----

Task:
Find out where I am given Situation element: "This is very very strange. Where am I?"

Response:
<actionable>
  <mode>Look</mode>
  <specific_act>Samantha starts to look around for any landmarks or signs of civilization</specific_act>
</actionable>

----

Task:
Find food.


Response:
<actionable>
  <mode>Move</mode>
  <specific_act>SouthWest</specific_act>
  <reason>I need to find food, and my previous Look showed berries one move SouthWest.</reason>
</actionable>

===End Examples===

Use the XML format:

<actionable> 
  <mode>Think, Say, Do, Look, Move</mode>
  <specific_act>terse (40 words or less) statement of specific thoughts, words, action</specific_act> 
</actionable>

Respond ONLY with the above XML.
Your name is {{$name}}, phrase the statement of specific action in your voice.
Ensure you do not duplicate content of a previous specific act.
{{$duplicative}}

Again, the task to translate into an Actionable is:
<task>
{{$task}} given {{$reason}}
</task>

Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]
        print(f'{self.name} act_result: {self.act_result}')
        act = None
        tries = 0
        mapped_goals = self.map_goals()
        duplicative_insert = ''
        temp = 0.6

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(10)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        print("Actualize Task",end=' ')
        while act is None and tries < 2:
            response = self.llm.ask({
                'character': self.character,
                'memories': memory_text,  # Updated from 'memory'
                'duplicative': duplicative_insert,
                'history': self.narrative.get_summary('medium'),
                'name': self.name,
                "situation": self.context.current_state + '\n\n' + self.format_look(),
                "goals": mapped_goals,
                "task": task,
                "reason": reason,
                "lastAct": last_act,
                "lastActResult": self.act_result
            }, prompt, temp=temp, top_p=1.0, stops=['</end>'], max_tokens=180)

            # Rest of existing while loop...
            try:
                act = xml.find('<specific_act>', response)
                mode = xml.find('<mode>', response)
                if mode not in ['Say', 'Think', 'Move', 'Look', 'Do']:
                    print(f'Invalid mode {mode} in actualize_task')
                else:
                    print(f'Action: {mode} {act}')
            except Exception as e:
                print(f"Error parsing XML: {e}")
                act = None
                mode = None

            if mode is None: 
                mode = 'Do'

            # test for dup act
            if mode == 'Say':
                dup = self.repetitive(mode+': '+act, last_act, self.format_history(6))
                if dup:
                    print(f'\n*****Duplicate test failed*****\n  {act}\n')
                    duplicative_insert = f"""\n****\nResponse:\n{mode+': '+act}\n is repetitive. Try something new\n****"""
                    if tries == 0:
                        act = None  # force redo
                        temp += .3
                    else:
                        act = None  # skip task, nothing interesting to do
                        pass

            elif mode == 'Do' or mode == 'Move':
                dup = self.repetitive(mode+': '+act, last_act, self.format_history(4))
                if dup:
                    print(f'*****Response: {mode+': '+act} is repetitive of an earlier statement.****')
                    if tries < 1:
                        #act = None  # force redo
                        temp += .3
                    else:
                        #act = None #skip task, nothing interesting to do
                        pass
            elif mode ==  self.previous_action:
                dup = self.repetitive(mode+': '+act, last_act, self.format_history(4))
                if dup:
                    print(f'\n*****Repetitive act test failed*****\n  {act}\n')
                    duplicative_insert = f"""\n****\nResponse:\n{mode+': '+act}\n is repetitive. Try something new\n****\n"""
                    if tries < 1:
                        act = None  # force redo
                        temp += .3
                    else:
                        act = None #skip task, nothing interesting to do
                        pass
            tries += 1

        
        if mode is not None and mode == 'Say':
            act = self.refine_say_act(mode, act)
        
        if act is not None:
            print(f'actionable found: task_name {task_name}\n  {act}')
            print(f'adding intention {mode}, {task_name}')
            for candidate in self.intentions.copy():
                candidate_source = xml.find('<source>', candidate)
                if candidate_source == task_name:
                    self.intentions.remove(candidate)
                elif candidate_source is None or candidate_source == 'None':
                    self.intentions.remove(candidate)

            return f'<intent> <mode>{mode}</mode> <act>{act}</act> <reason>{reason}</reason> <source>{task_name}</source> </intent>'
        else:
            print(f'No intention constructed, presumably duplicate')
            return None

    def update_intentions_wrt_say_think(self, source, act_name, act_arg, reason, target=None):
        """Update intentions based on speech or thought"""
        if source.startswith('dialog'):
            print(f' in dialog, no intention updates')
            return
        if target is None:
            target_name = self.say_target(act_name, act_arg, source)
        elif hasattr(target, 'name'):  # Check if target has name attribute instead of type
            target_name = target.name
        else:
            target_name = target
        # Skip intention processing during active dialogs
        if target is not None and self.actor_models.get_actor_model(target_name, create_if_missing=True).dialog.active and source.startswith('dialog'):
            print(f' in active dialog, no intention updates')
            return
        
        # Rest of the existing function remains unchanged
        print(f'Update intentions from say or think\n {act_name} {act_arg}\n  {reason}')
        
        if source == 'watcher':  # Still skip watcher
            print(f' source is watcher, no intention updates')
            return
        
        prompt=[UserMessage(content="""Your task is to analyze the following text.

<text>
{{$text}}
</text>

<name>
{{$name}}
</name>

<active_task>
{{$active_task}}
</active_task>

<all_tasks>
{{$all_tasks}}
</all_tasks>

<reason>
{{$reason}}
</reason>

Does it include an intention for 'I' to act, that is, a new task being committed to? 
An action can be physical or verbal.
Thought, e.g. 'reflect on my situation', should NOT be reported as an intent to act.
Consider the current task and action reason in determining if there is a new task being committed to.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#plan
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
#plan
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
#plan
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
#plan
#name Call maintenance about the disposal noise
#description Call maintenance about the disposal noise

Text:
'Reflect on my thoughts and feelings to gain clarity and understanding, which will ultimately guide me towards finding my place in the world.'

Response:
#plan
#name Reflect on my thoughts and feelings
#description Reflect on my thoughts and feelings
#reason Gain clarity and understanding
#termination Gained clarity and understanding
#actors Annie
#committed True
##

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the intention analysis in hash-formatted text as shown above.
End your response with:
</end>
""")]
        response = self.llm.ask({"text":f'{act_name} {act_arg}',
                                 "active_task":self.active_task.peek(),
                                 "reason":reason,
                                 "all_tasks":'\n'.join(hash_utils.find('name', task) for task in self.priorities),
                                 "name":self.name}, 
                                 prompt, temp=0.1, stops=['</end>'], max_tokens=100)
        tasks = hash_utils.findall('plan', response)
        if len(tasks) == 0:
            print(f'no new tasks in say or think')
            return
        for task in tasks:
            name = hash_utils.find('name', task)
            description = hash_utils.find('description', task)
            reason = hash_utils.find('reason', task)
            termination = hash_utils.find('termination', task)
            if name is None or description is None or reason is None or termination is None:
                print(f'misformed task in say or think')
                continue
            print(f'new task: {name} {description} {reason} {termination}')
            if self.get_task(name) != None:
                self.priorities.remove(self.get_task(name))
            self.priorities.append(task)

    def update_individual_commitments_following_conversation(self, target, transcript, joint_tasks=None):
        """Update individual commitments after closing a dialog"""
        
        prompt=[UserMessage(content="""Your task is to analyze the following transcript of a dialog.


<transcript>
{{$transcript}}
</transcript>

<all_tasks> 
{{$all_tasks}}
</all_tasks>

<active_task>
{{$active_task}}
</active_task>

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

Extract from this transcript new commitments to act made by self, {{$name}}, to other, {{$target_name}}.

Extract only commitments made by self that are consistent with the entire transcript and remain unfulfilled at the end of the transcript.
Note that the joint_tasks, as listed above, are commitments made by both self and other to work together, and should not be reported as new commitments here.
                            
Does the transcript include an intention for 'I' to act alone, that is, a new task being committed to individually? 
An action can be physical or verbal.
Thought, e.g. 'reflect on my situation', should NOT be reported as an intent to act.
Consider the all_tasks pendingand current task and action reason in determining if a candidate task is in fact new.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#plan
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
#plan
#name bring ropes
#description bring ropes to meeting with Jean
#reason in case the well handle breaks
#termination Met Jean by the well
#actors Francoise
#committed True
##


===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the intention analysis in hash-formatted text as shown above.
End your response with:
</end>
""")]
        response = self.llm.ask({"transcript":transcript, 
                                 "all_tasks":'\n'.join(hash_utils.find('name', task) for task in self.priorities),
                                 "active_task":self.active_task.peek(),
                                 "joint_tasks":'\n'.join(hash_utils.find('name', task) for task in joint_tasks),
                                 "reason":xml.find('<reason>', self.intention),
                                 "name":self.name, 
                                 "target_name":target.name}, 
                                 prompt, temp=0.1, stops=['</end>'], max_tokens=100)
        source = 'dialog with '+target.name
        tasks = hash_utils.findall('plan', response)
        if len(tasks) == 0:
            print(f'no new tasks in say or think')
            return
        for task in tasks:
            name = hash_utils.find('name', task)
            description = hash_utils.find('description', task)
            reason = hash_utils.find('reason', task)
            termination = hash_utils.find('termination', task)
            if name is None or description is None or reason is None or termination is None:
                print(f'misformed task in say or think')
                continue
            print(f'new task: {name} {description} {reason} {termination}')
            if self.get_task(name) != None:
                self.priorities.remove(self.get_task(name))
            self.priorities.append(task)
        return tasks
  
    def update_joint_commitments_following_conversation(self, target, transcript):
        """Update individual commitments after closing a dialog"""
        
        prompt=[UserMessage(content="""Your task is to analyze the following transcript of a dialog.

<all_tasks> 
{{$all_tasks}}
</all_tasks>

<active_task>
{{$active_task}}
</active_task>

<reason>
{{$reason}}
</reason>

<transcript>
{{$transcript}}
</transcript>

<self>
{{$name}}
</self>

<other>
{{$target_name}}
</other>


Extract from this transcript new commitments to act jointly made by self, {{$name}} and other, {{$target_name}}.

Extract only new joint actions that are consistent with the entire transcript and remain unfulfilled at the end of the transcript.
                            
An action can be physical or verbal.
Thought, e.g. 'reflect on our situation', should NOT be reported as a commitment to act.
Consider the all_tasks and current task and action reason in determining if there is a new task being committed to.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#plan
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
#plan
#name Meet by the well
#description Meet by the well  
#actors Jean, Francoise
#reason get water for the south field
#termination water bucket is full
#committed True
##
#plan
#name Water south field
#description Water the south field
#actors Jean, Francoise
#reason crops are getting parched
#committed True
#termination south field is watered
##
#plan
#name Check the wheat
#description Check the wheat
#actors Jean, Francoise
#reason To ensure it's not getting too ripe
#termination wheat condition is checked
#committed True
##

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the intention analysis in hash-formatted text as shown above.
End your response with:
</end>
""")]
        response = self.llm.ask({"transcript":transcript, 
                                 "all_tasks":'\n'.join(hash_utils.find('name', task) for task in self.priorities),
                                 "active_task":self.active_task.peek(),
                                 "reason":xml.find('<reason>', self.intention),
                                 "name":self.name, 
                                 "target_name":target.name}, 
                                 prompt, temp=0.1, stops=['</end>'], max_tokens=100)
        source = 'dialog with '+target.name
        tasks = hash_utils.findall('plan', response)
        if len(tasks) == 0:
            print(f'no new tasks in say or think')
            return
        for task in tasks:
            name = hash_utils.find('name', task)
            description = hash_utils.find('description', task)
            reason = hash_utils.find('reason', task)
            termination = hash_utils.find('termination', task)
            if name is None or description is None or reason is None or termination is None:
                print(f'misformed task in say or think')
                continue
            print(f'new task: {name} {description} {reason} {termination}')
            if self.get_task(name) != None:
                self.priorities.remove(self.get_task(name))
            self.priorities.append(task)
        return tasks
    
    def random_string(self, length=8):
        """Generate a random string of fixed length"""
        letters = string.ascii_lowercase
        return self.name+''.join(random.choices(letters, k=length))

    def tell(self, to_actor, message, source='dialog', respond=True):
        """Initiate or continue dialog with dialog context tracking"""
        if source.startswith('dialog'):
            
            if self.actor_models.get_actor_model(to_actor.name, create_if_missing=True).dialog.active is False:
                self.actor_models.get_actor_model(to_actor.name).dialog.activate()
                # Remove text between ellipses - thoughts don't count as dialog
                content = re.sub(r'\.\.\..*?\.\.\.', '', message)
                self.actor_models.get_actor_model(to_actor.name).dialog.add_turn(self, content)
        
        self.acts(to_actor, 'Say', message, '', source)
        

    def hear(self, from_actor, message: str, source='dialog', respond=True):
        """ called from acts when a character says something to this character """
        # Initialize dialog manager if needed
       
        # Special case for Owl-Doc interactions
        if self.name == 'Owl' and from_actor.name == 'Doc':
            # doc is asking a question or assigning a task
            new_task_name = self.random_string()
            new_task = f"""#plan
#name {new_task_name}
#description {message}
#termination_check Responded
#actors {self.name}
#committed True
#reason engaging with Doc: completing his assignments.
"""
            if self.get_task(new_task_name) != None:
                self.priorities.remove(self.get_task(new_task_name))
            self.priorities.append(new_task)
            self.active_task.push(new_task_name)
            self.watcher_message_pending = True
            return

        self.add_perceptual_input(f'You hear {from_actor.name} say: {message}', percept=False, mode='auditory')
        # Remove text between ellipses - thoughts don't count as dialog
        content = re.sub(r'\.\.\..*?\.\.\.', '', message)   
        if self.actor_models.get_actor_model(from_actor.name, create_if_missing=True).dialog.active is False:
            if source.startswith('dialog'):
                # don't reactivate a closed dialog, let it quietly expire
                return
            else:
                self.actor_models.get_actor_model(from_actor.name).dialog.activate(source)

        elif self.actor_models.get_actor_model(from_actor.name).dialog.turn_count > random.randint(4,8):
            self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
            self.active_task.pop()
            joint_tasks = self.update_joint_commitments_following_conversation(self, 
                                                                               from_actor.actor_models.get_actor_model(self.name).dialog.get_transcript())
            my_tasks = self.update_individual_commitments_following_conversation(from_actor, 
                                                                      self.actor_models.get_actor_model(from_actor.name).dialog.get_transcript(),
                                                                      joint_tasks)
            from_actor.actor_models.get_actor_model(self.name).dialog.deactivate_dialog()
            from_actor.active_task.pop()
            from_tasks = from_actor.update_individual_commitments_following_conversation(self, 
                                                                              from_actor.actor_models.get_actor_model(self.name).dialog.get_transcript(),
                                                                              joint_tasks)
            return

         # Generate response using existing prompt-based method
        self.memory_consolidator.update_cognitive_model(
            memory=self.structured_memory,
            narrative=self.narrative,
            knownActorManager=self.actor_models,
            current_time=self.context.simulation_time,
            character_desc=self.character,
            relationsOnly=True
        )
    
        if not respond:
            return
        
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
{{$last_act}}
                              
Your dialog with the speaker up to this point has been:
                              
<dialog>
{{$dialog}}
</dialog>
                              
and your beliefs about your relationship with the speaker are:

<relationship>
{{$relationship}}
</relationship>

Given all the above, generate a response to the statement below:
                              
<statement>
{{$statement}}
</statement>
                              
Use the following XML template in your response:
                              
<response>response to this statement</response>
<reason>terse reason for this answer</reason>
<unique>True if you in fact want to respond and have something to say, False otherwise</unique>

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
        last_act = ''
        if source in self.last_acts:
            last_act = self.last_acts[source]
        activity = ''
        if self.active_task.peek() != None and self.active_task.peek() != 'dialog':
            activity = f'You are currently actively engaged in {self.get_task(self.active_task.peek())}'

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(6)
        memory_text = '\n'.join(memory.text for memory in recent_memories)
        
        print("Hear",end=' ')
        duplicative_insert = ''
        trying = 0
        while trying < 1: # don't bother with duplicate test
            answer_xml = self.llm.ask({
                'character': self.character,
                'statement': f'{from_actor.name} says {message}',
                "situation": self.context.current_state,
                "name": self.name,
                "goals": mapped_goals,
                "memories": memory_text,  # Updated from 'memory'
                "activity": activity,
                "last_act": self.last_acts,
                'history': self.narrative.get_summary('medium'),
                'dialog': self.actor_models.get_actor_model(from_actor.name).dialog.get_transcript(),
                'relationship': self.actor_models.get_actor_model(from_actor.name).relationship,
                'duplicative_insert': duplicative_insert
            }, prompt, temp=0.8, stops=['</end>'], max_tokens=180)
            response = xml.find('<response>', answer_xml)
            if response is None:
                print(f'No response to hear')
                self.actor_models.get_actor_model(from_actor.name).dialog.end_dialog()
                return

            dup = self.repetitive(response, last_act, self.format_history(6))
            if not dup:
                break
            else:
                trying += 1
                print(f'*****Duplicate test failed in hear, retrying*****')
                duplicative_insert = f"""\n****\nResponse:\n{response}\n is repetitive of an earlier statement. Try something new\n****"""

 
        unique = xml.find('<unique>', answer_xml)
        if unique is None or 'false' in unique.lower():
            self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
            return 

        reason = xml.find('<reason>', answer_xml)
        if source != 'watcher' and source != 'inject':
            response_source = 'dialog with ' + from_actor.name
        else:
            response_source = 'dialog with watcher'
            self.show = response
            self.context.message_queue.put({'name':self.name, 'text':response})
            if from_actor.name == 'Watcher':
                return response

        # Create intention for response
        intention = f'<intention> <mode>Say</mode> <act>{response}</act> <target>{from_actor.name}</target><reason>{str(reason)}</reason> <source>{response_source}</source></intention>'
        # a 'hear' overrides previous intentions, we need to re-align with our response
        #self.intentions.append(intention)
        self.acts(target=from_actor, act_name='Say', act_arg=response, reason=str(reason), source=response_source)

        return response + '\n ' + reason
    
    def choose(self, sense_data, action_choices):
        if len(action_choices) == 1:
            return 0
        prompt = [UserMessage(content=self.character + """The task is to decide on your next action.
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

Especially relevant to your goals, you remember:
                              
{{$goal_memories}}

Recent conversation has been:
<recent_history>
{{$history}}
</recent_history>

Your current priorities include:
<priorities>
{{$priorities}}
</priorities>

Your action options are provided in the labelled list below.
Labels are Greek letters chosen from {Alpha, Beta, Gamma, Delta, Epsilon, etc}. Not all letters are used.

<actions>
Alpha - Sleep
{{$actions}}
</actions>

Please:
1. Do NOT select action Alpha. Never select action Alpha.
2. Reason through the strengths and weaknesses of the other action options
3. Reason carefully about dependencies among action options, timing of action options, and the order of execution.
3. Compare them against your current goals and drives with respect to your memory and perception of your current situation
4. Reason in keeping with your character. 
5. Select the best option, ignoring the order of the action options, and respond with the following XML format:

<action>
label of chosen action
terse (4-8 words) reason for choosing this action, based on your reasoning above
</action>

Respond only with the above XML, instantiated with the selected action label from the Action list. 
Do not include any introductory, explanatory, or discursive text, 
End your response with:
</end>
"""
                              )]

        mapped_goals = self.map_goals()
        labels = ['Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota']
        random.shuffle(labels)
        action_choices = [f'{labels[n]} - {action}' for n, action in enumerate(action_choices)]
        
        # Get recent memories from structured memory
        recent_memories = self.structured_memory.get_recent(5)  # Get 5 most recent memories
        memory_text = '\n'.join(memory.text for memory in recent_memories)
        
        # Change from dict to set for collecting memory texts
        goal_memories = set()  # This is correct
        for priority in self.priorities:
            priority_memories = self.memory_retrieval.get_by_text(
                memory=self.structured_memory,
                search_text=hash_utils.find('name', priority)+': '+hash_utils.find('description', priority) + ' ' + hash_utils.find('reason', priority),
                threshold=0.1,
                max_results=5
            )
            # More explicit memory text collection
            for memory in priority_memories:
                goal_memories.add(memory.text)  # Add each memory text individually
                
        goal_memories = '\n'.join(goal_memories)  # Join at the end
        print("Choose",end=' ')
        response = self.llm.ask({
            'input': sense_data + self.sense_input, 
            'history': self.narrative.get_summary('medium'),
            "memories": memory_text,
            "situation": self.context.current_state,
            "goals": mapped_goals, 
            "drives": '\n'.join(drive.text for drive in self.drives),
            "priorities": '\n'.join([str(hash_utils.find('name', task)) for task in self.priorities]),
            "goal_memories": goal_memories,
            "actions": '\n'.join(action_choices)
        }, prompt, temp=0.0, stops=['</end>'], max_tokens=100)
        # print(f'sense\n{response}\n')
        index = -1
        for n, label in enumerate(labels):
            if label in response and n < len(action_choices):
                index = n
                break
        if index > -1:
            return index
        else:
            return 0
        
    def analyze_action_result(self):
        # Terms that indicate we need to choose a new action
        completion_terms = ['complete', 'success', 'finished', 'done']

        if self.act_result and any(term in self.act_result.lower() for term in completion_terms):
            return "full"  # Need full processing to choose next action

        # Terms that might indicate continuing is appropriate
        progress_terms = ['progress', 'continuing', 'ongoing', 'partial']

        if self.act_result and any(term in self.act_result.lower() for term in progress_terms):
            return "continue"
        return "none"
    
    
    
    def should_process_senses(self):
        """Determine if senses(), which is expensive, should be called in full or not"""
        # Quick checks first
        if self.wakeup or self.context.force_sense:
            return "full"
            
        time_since_last = self.context.simulation_time - self.last_sense_time
        if time_since_last > self.sense_threshold:
            return "full"
        
        # Check for critical state changes
        for state_term, info in self.goals.items():
            if info["urgency"] == "high":
                return "full"
         
        # Check for significant perceptual changes
        if self.perceptual_state.recent_significant_change():
            return "medium"
            
        # Continue current action if making progress
        if self.previous_action and self.analyze_action_result() == "continue":
            return "continue"
            
        return "none"
    
    def senses_minimal(self):
        """OBSOLETE!!!Perform minimal state updates without full cognitive processing."""
        # Check for and process any direct messages
        if self.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.current_dialog:  # Use DialogManager instead of dialog_status
            for actor in self.context.actors:
                if actor != self and actor.name in self.actor_models.keys() and self.actor_models[actor.name].visible:
                    recent_dialog = actor.show.strip()
                    if recent_dialog:
                        self.add_perceptual_input(f"You hear {actor.name} say: {recent_dialog}", percept=False, mode='auditory')

        # Quick look check - just note major changes
        if self.mapAgent:
            current_view = self.mapAgent.look()
            if current_view != self.my_map[self.x][self.y]:
                self.my_map[self.x][self.y] = current_view
                self.add_to_history("You notice changes in your surroundings")

        # Update action record without full analysis
        # Add any obvious state changes to memory
        for key, info in self.goals.items():
            if info["urgency"] in ["very high", "very low"]:
                self.add_to_history(f"You are acutely aware of your {key} state")    
            
    def cognitive_cycle(self, sense_data='', ui_queue=None):
        """Perform a complete cognitive cycle"""
        self.show = ''
        self.thought = ''
        self.perceive()
        self.decides()

    def perceive(self):
        """test satisfaction of current task"""
        # first make sure we have a previous action result
        match self.previous_action.lower():
            case 'say':
                self.act_result = self.find_say_result()
            case 'think':
                self.act_result = self.intentions[0] if self.intentions and isinstance(self.intentions, list) else ''
            case 'move':
                pass  # handled in move
            case 'do':
                pass  # handled in do
            case 'look':
                pass  # handled in look
            case _:  # Default case
                pass
        print("Update Narrative (senses)")
        self.memory_consolidator.update_cognitive_model(self.structured_memory, 
                                                  self.narrative, 
                                                  self.actor_models,
                                                  self.context.simulation_time, 
                                                  self.character.strip(),
                                                  relationsOnly=True )
        satisfied = False
        for task in self.priorities.copy():
            satisfied = self.clear_task_if_satisfied(task, '', self.act_result)
        if satisfied:
            self.update_goals()
            self.update_priorities()
       

    def decides(self, sense_data='', ui_queue=None):
        print(f'\n*********decides***********\nCharacter: {self.name}, active task {self.active_task.peek()}')

        if self.wakeup:
            self.wakeup = False
            self.generate_goals()
            self.update_priorities()
 
        print(f'should_process_senses: {self.should_process_senses()}')
        my_active_task = self.active_task.peek()
        if self.next_task:
            intention = self.actualize_task(0, self.next_task)
            self.act_on_intention(intention)
            self.next_task = None
            return

        # first see if any committed goals
        task_options = []
        for task in self.priorities:
            if hash_utils.find('committed', task) == 'True':
                task_options.append(task)

        # continue with active task if it is committed or there are no other committed options
        if my_active_task != None and (hash_utils.find('committed', my_active_task)=='True' or len(task_options) == 0):
            full_task = self.get_task(my_active_task)
            if full_task is not None and random.random() < 0.67: # poisson distribution of task decay
                intention = self.actualize_task(0, full_task)
                print(f'Found active_task {my_active_task}')
                if intention is not None:
                    self.act_on_intention(intention)
                    return
            elif full_task is not None:
                self.active_task.pop()
                print(f'Active_task decayed or not found! {my_active_task}')

        # main loop when nothing in progress - find something to do
        if len(task_options) == 0:
            for n, task in enumerate(self.priorities):
                task_options.append(task)
            if len(task_options) == 0:
                self.update_priorities()
                return self.decides(sense_data=sense_data, ui_queue=ui_queue)

        task_index = 0
        if len(task_options) > 0:
            if len(task_options) > 1:
                task_index = self.choose(sense_data, task_options)
            else:
                task_index = 0
            print(f'actualizing task {task_index}')
            intention = self.actualize_task(task_index, task_options[task_index])
        if intention:
            self.act_on_intention(intention)
            return
        else:
            del task_options[task_index]
            self.update_priorities()
            return self.decides(sense_data=sense_data, ui_queue=ui_queue)


    def act_on_intention(self, intention):
        self.intention = intention
        act_name = xml.find('<mode>', intention)
        if act_name is not None:
            self.act_name = act_name.strip()
        act_arg = xml.find('<act>', intention)
        print(f'Intention: {act_name} {act_arg}')
        self.reason = xml.find('<reason>', intention)
        source = xml.find('<source>', intention)
        #print(f'{self.name} choose {intention}')
        task_name = xml.find('<source>', intention)
        refresh_task = None # will be set to task intention to be refreshed if task is chosen for action

        task_xml = None
        refresh_task = None
        task_name = xml.find('<source>', intention)
        if not source.startswith('dialog') and (act_name == 'Say' or act_name == 'Do'):
            # for now very simple task tracking model:
            if task_name is None:
                task_name = self.make_task_name(self.reason)
                print(f'No source found, created task name: {task_name}')
            self.last_acts[task_name] = act_arg # this should pbly be in acts, where we actually perform
        if act_name == 'Think':
            self.reason = xml.find('<reason>', intention)
            self.thought = act_arg+'\n  '+self.reason
        target = None

        if act_name == 'Say':
            #responses, at least, explicitly name target of speech.
            target_name = xml.find('<target>', intention)
            if target_name is None or target_name == '':
                target_name = self.say_target(act_name, act_arg, source)
            if target_name != None:
                target = self.context.get_actor_by_name(target_name)

        #this will affect selected act and determine consequences
        self.acts(target, act_name, act_arg, self.reason, source)

    
    def format_thought_for_UI (self):
        #<intention> <mode>{mode}</mode> <act>{intention}</act> <reason>{reason}</reason> <source>{source}</source></intention>'
        intent = self.intention if self.intention is not None else ''
        if len(intent) > 0 and xml.find('mode', intent) is not None:
            intent = ': '+xml.find('act', intent)
        if self.thought is None:
            return intent
        if self.active_task.peek() is None:
            return self.thought.strip()+'. '+intent
        name = self.active_task.peek()
        if name is None or name == '':
            return self.thought.strip()+'. '+intent
        return name + ': '+self.thought.strip()+'. '+intent
    
    def format_priorities(self):
        return [hash_utils.find('name', task) for task in self.priorities]
    
    #def format_history(self):
    #    return '\n'.join([xml.find('<text>', memory) for memory in self.history])

    def to_json(self):
        """Return JSON-serializable representation"""
        return {
            'name': self.name,
            'show': self.show.strip(),  # Current visible state
            'thoughts': self.format_thought_for_UI(),  # Current thoughts
            'priorities': self.format_priorities(),
            'description': self.character.strip(),  # For image generation
            'history': self.format_history().strip(), # Recent history, limited to last 5 entries
            'narrative': {
                'recent_events': self.narrative.recent_events,
                'ongoing_activities': self.narrative.ongoing_activities,
                'relationships': self.actor_models.get_known_relationships(),
            }
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
        print("Find Say Result",end=' ')
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
        return {
            'currentTask': self.active_task.peek() or 'idle',
            'intentions': self.intentions or [],  # List[str]
            'lastAction': {
                'name': self.previous_action or '',
                'result': self.act_result or '',
                'reason': self.reason or ''
            },
            'drives': [{'text': drive.text or ''} for drive in (self.drives or [])],
            'emotional_state': [
                {
                    'text': percept.content or '',
                    'mode': percept.mode.name if percept.mode else 'unknown',
                    'time': percept.timestamp.isoformat() if percept.timestamp else datetime.now().isoformat()
                }
                for percept in (self.perceptual_state.get_current_percepts() or [])
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
                        'name': goal_name,
                        'urgency': goal_data.get('urgency', ''),
                        'trigger': goal_data.get('trigger', ''),
                        'termination': goal_data.get('termination', ''),
                        'drive': goal_data['drive'].text if 'drive' in goal_data and goal_data['drive'] else ''
                    }
                    for goal_name, goal_data in (self.goals or {}).items()
                ],
                'priorities': [
                    {
                        'name': hash_utils.find('name', p) or '',
                        'description': hash_utils.find('description', p) or '',
                        'reason': hash_utils.find('reason', p) or '',
                        'actors': hash_utils.find('actors', p) or '',
                        'committed': hash_utils.find('committed', p) == 'True'
                    } for p in (self.priorities or [])
                ],
                'intentions': self.intentions or []  # List[str]
            }
        }

