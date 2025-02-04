from datetime import datetime, timedelta
import json
import random
import string
import traceback
import time
from typing import List, Dict, Optional
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
from sim.ActionRecord import (
    ActionMemory, 
    ActionRecord,
    StateSnapshot,
    Mode,
    create_action_record
)

from sim.DialogManagement import DialogManager
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from sim.perceptualState import PerceptualInput, PerceptualState, SensoryMode


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
        self.priorities = ['']  # displayed by main thread at top in character intentions widget
        self.show = ''  # to be displayed by main thread in UI public text widget
        self.state = {}
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
            key_relationships={},
            active_drives=[]
        )

        self.drives: List[Drive] = []  # Initialize empty drive list
        self.perceptual_state = PerceptualState(self)
        self.last_sense_time = datetime.now()
        self.sense_threshold = timedelta(hours=4)

        self.dialog_manager = DialogManager(self)  # Initialize at creation instead of on-demand
        

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
            self.memory_consolidator.update_narrative(
                memory=self.structured_memory,
                narrative=self.narrative,
                current_time=self.context.simulation_time,
                character_desc=self.character,
                relationsOnly=False
            )

        # Get recent memory content from structured_memory
        recent_memories = self.structured_memory.get_recent(self.new_memory_cnt)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        # Update state and priorities
        self.generate_state()
        self.update_priorities()

        # Reset new memory counter
        self.new_memory_cnt = 0

    # Task management methods
    def get_task_last_acts_key(self, term):
        """Checks for name in Last_acts"""
        for task in list(self.last_acts.keys()):
            match = self.synonym_check(task, term)
            if match:
                return task
        return None

    def set_task_last_act(self, term, act):
        task = self.get_task_last_acts_key(term)
        if task is None:
            print(f'SET_TASK_LAST_ACT {self.name} no match found for term: {term}, {act}')
            self.last_acts[term] = act
        else:
            print(f'SET_TASK_LAST_ACT {self.name} match found: term {term}, task {task}\n  {act}')
            self.last_acts[task] = act

    def get_task_last_act(self, term):
        task = self.get_task_last_acts_key(term)
        if task is None:
            return 'None'
        else:
            return self.last_acts[task]

    def get_task_xml(self, task_name):
        for candidate in self.priorities:
            if task_name == xml.find('<name>', candidate):
                print(f'found existing task\n  {task_name}')
                return candidate
        return None

    def find_or_make_task_xml(self, task_name, reason):
        candidate = self.get_task_xml(task_name)
        if candidate is not None:
            return candidate
        new_task = f'<plan><name>{task_name}</name><reason>{reason}</reason></plan>'
        self.priorities.append(new_task)
        print(f'created new task to reflect {task_name}\n {reason}\n  {new_task}')
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

    def say_target(self, text):
        """Determine the intended recipient of a message"""
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
        
The message is:
        
<Message>
{{$message}}
</Message>

Respond using the following XML format:

<target>
  <name>intended recipient name</name>
</target>

End your response with:
</end>
""")]
        response = self.llm.ask({
            'character': self.character,
            'history': self.narrative.get_summary('medium'),
            'actors': '\n'.join([actor.name for actor in self.context.actors]),
            "message": text
        }, prompt, temp=0.2, stops=['</end>'], max_tokens=180)

        return xml.find('<name>', response)

    # World interaction methods
    def look(self, height=5):
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
                    text_view += f"agents {view[dir]['agents']}, "
                if 'water' in view[dir]:
                    text_view += f"water {view[dir]['water']}"
            except Exception as e:
                pass
            text_view += "\n"
         # Create visual perceptual input
        prompt = [UserMessage(content="""Your current state is :
                              
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

Provide a concise description of what you notice, highlighting the most important features given your current state and priorities. 
Respond using the following XML format:

<perception>a concise (30 words or less) description of perceptual content</perception>

End your response with:
<end/>
""")]
        response = self.llm.ask({"view": text_view, "state": self.show, "priorities": self.priorities}, prompt, temp=0.2, stops=['<end/>'], max_tokens=100)
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
agents: the other actors visible
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

    def see(self):
        """Add initial visual memories of other actors"""
        for actor in self.context.actors:
            if actor != self:
                self.add_perceptual_input(f'You see {actor.name}')

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
        self.dialog_manager = DialogManager(self)
        self.dialog_status = 'Waiting'
        self.dialog_length = 0
        
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
        self.action_memory = ActionMemory()

        # Update narrative with drives (drives are strings)
        self.narrative.active_drives = self.drives  # Direct assignment, no name attribute needed
        
        # Initialize memory systems
        self.structured_memory = StructuredMemory(owner=self)
        self.memory_consolidator = MemoryConsolidator(self.llm)
        self.memory_retrieval = MemoryRetrieval()
        self.new_memory_cnt = 0
        self.watcher_message_pending = False

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
            
            description = description +', '+ self.show.replace(self.name, '').strip()
            prompt = [UserMessage(content="""Following is a description of a character in a play. 

<description>
{{$description}}
</description>
            
Extract from this description two or three words that describe the character's emotional state.
Use common adjectives like happy, sad, frightened, worriedangry, curious, aroused, cold, hungry, tired, disoriented, etc.
The words should each describe a different aspect of the character's emotional state, and should be distinct from each other.

Respond using this XML format:

<emotion>
  <state>adjective</state>
</emotion>

End your response with:
<end/>
""")]
            concerns = ''
            for priority in self.priorities:
                concern = xml.find('<name>', priority) + '. '+xml.find('<reason>', priority)
                concerns = concerns + '; '+concern
            state = description + '.\n '+concerns +'\n'+ context
            response = self.llm.ask({ "description": state}, prompt, temp=0.2, stops=['<end/>'], max_tokens=100)
            state = xml.find('<emotion>', response)
            if state:
                states = xml.findall('<state>', state)
                description = description[:192-min(len(context), 48)] + f'. {self.name} feels '+', '.join(states)+'. '+context
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
    
    def add_perceptual_input(self, message: str, percept=True):
        """Add a perceptual input to the agent's perceptual state"""
        content = message
        if percept:
            prompt = [UserMessage(content="""Given a message, determine the most appropriate sensory mode for it.
Input may be auditory, visual, or movement, or internal.

<message>
{{$message}}
</message>

Respond using this XML format:

<input>true/false </input
<mode>auditory/visual/movement/internal</mode>
<content>terse description of perceptual content in the given mode</content>
<intensity>0-1</intensity>

Be sure to include any character name in the content.
Do not include any introductory, discursive, or explanatory text.
Respond only with the above XML.
End your response with:
<end/>
""")]
        else:
            prompt = [UserMessage(content="""Given a message, determine the most appropriate sensory mode for it.
Input may be auditory, visual, or movement, or internal.

<message>
{{$message}}
</message>

Respond using this XML format:

<input>true/false </input
<mode>auditory/visual/movement/internal</mode>
<intensity>0-1</intensity>

Do not include any introductory, discursive, or explanatory text.
Respond only with the above XML.
End your response with:
<end/>
""")]
        response = self.llm.ask({"message": message}, prompt, temp=0.2, stops=['<end/>'], max_tokens=100)
        mode = xml.find('<mode>', response)
        try:
            mode = SensoryMode(mode)
        except:
            # unknown mode, skip
            return
        if percept:
            perceived_content = xml.find('<content>', response)
        else:
            perceived_content = message
        intensity = xml.find('<intensity>', response)
        try:
            intensity = float(intensity)
        except ValueError:
            # invalid intensity, skip
            intensity = 0.4 # assume low intensity
        perceptual_input = PerceptualInput(
            mode=mode,
            content=perceived_content if percept else message,
            timestamp=datetime.now(),
            intensity=intensity
        )
        self.perceptual_state.add_input(perceptual_input)

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
    

    def generate_state(self):
        """Generate states to track, derived from drives and current memory context"""
        self.state = {}
        
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

Analyze the memories and assess the current state relative to this drive.
Consider:
1. How well the drive's needs are being met
2. Recent events that affect the drive
3. The importance scores of relevant memories
4. Any patterns or trends in the memories

Respond using this XML format:

<state> 
  <term>concise term for this drive state</term>
  <assessment>very high/high/medium-high/medium/medium-low/low</assessment>
  <trigger>specific situation or memory that most affects this state</trigger>
  <termination>condition that would satisfy this drive</termination>
</state>

Respond ONLY with the above XML.
End response with:
<end/>
""")]

        # Process each drive in priority order
        for drive in self.drives:
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
                f"[{mem.timestamp}] (importance: {mem.importance:.1f}): {mem.text}"
                for mem in drive_memories
            ])
            
            recent_memories_text = "\n".join([
                f"[{mem.timestamp}] (importance: {mem.importance:.1f}): {mem.text}"
                for mem in recent_memories
            ])
            
            # Generate state assessment
            response = self.llm.ask({
                "drive": drive.text,
                "recent_memories": recent_memories_text,
                "drive_memories": drive_memories_text,
                "situation": self.context.current_state if self.context else "",
                "character": self.character
            }, prompt, temp=0.3, stops=['<end/>'])     
            
            # Parse response
            try:
                term = xml.find('<term>', response)
                assessment = xml.find('<assessment>', response)
                trigger = xml.find('<trigger>', response)
                termination = xml.find('<termination>', response)
                
                if term and assessment:
                    self.state[term] = {
                        "drive": drive,
                        "state": assessment,
                        "trigger": trigger if trigger else "",
                        "termination": termination if termination else ""
                    }
                else:
                    print(f"Warning: Invalid state generation response for {drive.text}")
                    
            except Exception as e:
                print(f"Error parsing state generation response: {e}")
                traceback.print_exc()
                
        print(f'{self.name} generated states: ' + str({k: {**v, 'drive': v['drive'].text} for k,v in self.state.items()}))
        return self.state


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

<termination> 
    <level>value of state satisfaction, True if termination test is met in Events or History</level>
</termination>

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

    def map_state(self):
        """ map state for llm input """
        mapped = []
        for key, item in self.state.items():
            trigger = item['drive'].text
            value = item['state']
            termination = item['termination']
            mapped.append(f"- Drive: {trigger}; Goal: {termination}; State: '{value}'")
        return "A 'State' of 'High' means the task is important or urgent\n"+'\n'.join(mapped)


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
            #print(f'GET_TASK_LAST_ACT {self.name} no match found: term {term}')
            return 'None'
        else:
            #print(f'GET_TASK_LAST_ACT match found {self.name} term {term} task {task}\n  last_act:{self.last_acts[task]}\n')
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
                    
    def get_task_xml(self, task_name):
        for candidate in self.priorities:
            #print(f'find_or_make testing\n {candidate}\nfor name {task_name}')
            if task_name == xml.find('<name>', candidate):
                print(f'found existing task\n  {task_name}')
                return candidate
        return None
    
    def find_or_make_task_xml(self, task_name, reason):
        candidate = self.get_task_xml(task_name)
        if candidate != None:
            return candidate
        new_task = f'<plan><name>{task_name}</name><reason>{reason}</reason></plan>'
        self.priorities.append(new_task)
        print(f'created new task to reflect {task_name}\n {reason}\n  {new_task}')
        return new_task

    def repetitive(self, new_response, last_response, source):
        """Check if response is repetitive considering wider context"""
        # Get more historical context from structured memory
        recent_memories = self.structured_memory.get_recent(3)  # Increased window
        history = '\n'.join(mem.text for mem in recent_memories)
        
        prompt = [UserMessage(content="""Given a character's recent history and a new proposed response, 
determine if the new response is pointlessly repetitive and unrealistic. 
Only repetition of most recent actions with no new information is considered repetitive.

Recent history:
<history>
{{$history}}
</history>

New response:
<response>
{{$response}}
</response>

Consider:
- The meaning and intent of the response
- The flow of conversation
- Whether it advances the dialog
- If it repeats ideas or phrases already expressed

Respond with only 'True' if repetitive or 'False' if the response adds something new.
End response with:
<end/>""")]

        result = self.llm.ask({
            'history': history,
            'response': new_response
        }, prompt, temp=0.2, stops=['<end/>'], max_tokens=100)

        return 'true' in result.lower()

    def clear_task_if_satisfied(self, task_xml, consequences, world_updates):
        """Check if task is complete and update state"""
        termination_check = xml.find('<termination>', task_xml) if task_xml != None else None
        if termination_check is None:
            return

        # Test completion through cognitive processor's state system
        satisfied = self.test_priority_termination(
            termination_check, 
            consequences,
            world_updates
        )

        if satisfied:
            task_name = xml.find('<name>', task_xml)
            if task_name == self.active_task.peek():
                self.active_task.pop()

            try:
                self.priorities.remove(task_xml)
            except Exception as e:
                print(str(e))

            new_intentions = []
            for intention in self.intentions:
                if xml.find('<source>', intention) != task_name:
                    new_intentions.append(intention)
            self.intentions = new_intentions

            if self.active_task.peek() is None and len(self.priorities) == 0:
                # Should never happen, but just in case
                self.update_priorities()


    def acts(self, target, act_name, act_arg='', reason='', source=''):
        """Execute an action and record results"""
        # Create action record with state before action
        mode = Mode(act_name)
        self.act_result = ''
        record = create_action_record(
            agent=self,
            mode=mode,
            action_text=act_arg,
            task_name=source if source else self.active_task.peek(),
            target=target.name if target else None
        )
    
        # Store current state and reason
        self.reason = reason
        if act_name is None or act_arg is None or len(act_name) <= 0 or len(act_arg) <= 0:
            return


         # Update thought display
        if act_name == 'Think':
            self.thought = act_arg
            self.show += f"*{self.thought}*"
        else:
            self.thought =  self.reason

        # Update active task if needed
        if (act_name == 'Do' or act_name == 'Say') and source != 'dialog' and source != 'watcher':
            if self.active_task.peek() != source and source not in self.active_task.stack:
                 # if we are not already on this task, push it onto the stack. 
                 self.active_task.push(source)

        # Handle world interaction
        if act_name == 'Do':
            # Get action consequences from world
            consequences, world_updates = self.context.do(self, act_arg)
        
            if source == None:
                source = self.active_task.peek()
            task_xml = self.get_task_xml(source) if source != None else None
        
            # Check for task completion
            for task in self.priorities.copy():
                self.clear_task_if_satisfied(task, consequences, world_updates)

            # Update displays
            self.show = act_arg+'\n Resulting in ' + consequences.strip()
            self.add_perceptual_input(f"You observe {consequences}")
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
                percept = self.look()
                self.show += ' moves *' + act_arg + '*.\n  *' + percept + '*'
                self.add_perceptual_input(f"\nYou {act_name}: {act_arg}\n  {percept}")
        elif act_name == 'Look':
            percept = self.look()
            self.show += ' looks *' + act_arg + '*.\n  *' + percept + '*'
            self.add_perceptual_input(f"\nYou look: {act_arg}\n  {percept}")
        self.previous_action = act_name

        if act_name == 'Think' or act_name == 'Say':
            # Update intentions based on thought
            self.update_intentions_wrt_say_think(source, act_arg, reason) 
            self.add_perceptual_input(f"\nYou {act_name}: {act_arg}", percept=False)

        # After action completes, update record with results
        # Notify other actors of action
        if act_name != 'Say' and act_name != 'Look' and act_name != 'Think':  # everyone sees it
            for actor in self.context.actors:
                if actor != self:
                    actor.add_perceptual_input(self.show)
        else:# must be a say
            self.show = f"'{act_arg}'"
            for actor in self.context.actors:
                if actor != self:
                    if source != 'watcher':  # when talking to watcher, others don't hear it
                        if actor != target:
                            actor.add_perceptual_input(f"You hear {self.name}: '{act_arg}'", percept=False)
                        else:
                            actor.hear(self, act_arg, source)

        record.context_feedback = self.show
        record.state_after = StateSnapshot(
            values={term: info["state"] 
                for term, info in self.state.items()},
            timestamp=time.time()
        )
    
        # Update record with state changes
        self.action_memory.update_record_states(record)
    
        # Store completed record
        self.action_memory.add_record(record)
        

    def update_priorities(self):
        """Update task priorities based on current state and drives"""
        prompt = [UserMessage(content="""Given your character, drives and current state assessments, and recent memories, create a prioritized set of plans.

<character>
{{$character}}
</character>

<state>
{{$state}}
</state>

<recent_memories>
{{$memories}}
</recent_memories>

Create three specific, actionable plans that address your current needs and situation.
Consider:
1. Your current state assessments
2. Recent memories and events
3. Your basic drives and needs

Respond using this XML format:
<plan>
  <name>brief action name</name>
  <description>detailed action description</description>
  <reason>why this action is important now</reason>
  <termination>condition that would satisfy this need</termination>
</plan>

Description, should be detailed but concise. Reason, and TerminationCheck should be terse.
Respond ONLY with three plans using the above XML format.
Order plans from highest to lowest priority.
End response with:
<end/>
""")]

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(5)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        # Format state for LLM
        state_text = self.map_state()
        response = self.llm.ask({
            'character': self.character,
            'state': state_text,
            'memories': memory_text
        }, prompt, temp=0.7, stops=['<end/>'])

        # Extract plans from response
        self.priorities = []
        for plan in xml.findall('<plan>', response):
            if xml.find('<name>', plan):
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
  <progress>0-100 percentage</progress>
  <reason>Why this assessment</reason>
</completion>
<complete> 
    <level>value of task completion, True, Unknown, or False</level>
    <evidence>concise statement of evidence in events to support this level of task completion</evidence>
</complete>

the 'Level' above should be True if the termination check is met in Events or recent History, 
Unknown if the Events do not support a definitive assessment, or False if Events provide little or no evidence for task completion.  

Respond ONLY with the above XML
Do not include any introductory, explanatory, or discursive text.
End your response with:
<end/>
""")]

        print(f'{self.name} testing priority termination_check: {termination_check}')

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
        }, prompt, temp=0.3, stops=['<end/>'], max_tokens=120)

        satisfied = xml.find('<level>', response)
        if satisfied != None and satisfied.lower().strip() == 'true':
            print(f'Priority {termination_check} Satisfied!')
        return False if satisfied == None else satisfied.lower().strip() == 'true'

    def actualize_task(self, n, task_xml):
        task_name = xml.find('<name>', task_xml)
        if task_xml is None or task_name is None:
            raise ValueError(f'Invalid task {n}, {task_xml}')
        last_act = self.get_task_last_act(task_name)
        reason = xml.find('<reason>', task_xml)

        prompt = [UserMessage(content="""You are {{$character}}.
Your task is to generate an Actionable (a 'Think', 'Say', 'Look', Move', or 'Do') to advance the next step of the following task.

<task>
{{$task}}
</task>

Your current situation is:

<situation>
{{$situation}}
</situation>

Your state is:

<state>
{{$state}}
</state>

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
  <mode>Think, Say, or Do</mode>
  <specific_act>specific thoughts, words, or action</specific_act> 
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
        mapped_state = self.map_state()
        duplicative_insert = ''
        temp = 0.6

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(6)
        memory_text = '\n'.join(memory.text for memory in recent_memories)

        while act is None and tries < 2:
            response = self.llm.ask({
                'character': self.character,
                'memories': memory_text,  # Updated from 'memory'
                'duplicative': duplicative_insert,
                'history': self.narrative.get_summary('medium'),
                'name': self.name,
                "situation": self.context.current_state + '\n\n' + self.format_look(),
                "state": mapped_state,
                "task": task_xml,
                "reason": reason,
                "lastAct": last_act,
                "lastActResult": self.act_result
            }, prompt, temp=temp, top_p=1.0, stops=['</end>'], max_tokens=180)

            # Rest of existing while loop...
            print(response)
            try:
                act = xml.find('<specific_act>', response)
                mode = xml.find('<mode>', response)
            except Exception as e:
                print(f"Error parsing XML: {e}")
                act = None
                mode = None

            if mode is None: 
                mode = 'Do'

            # test for dup act
            if mode == 'Say':
                dup = self.repetitive(act, last_act, self.format_history(2))
                if dup:
                    print(f'\n*****Duplicate test failed*****\n  {act}\n')
                    duplicative_insert = f"""\n****\nDon't duplicate this previous act:\n'{act}'
What else could you say or how else could you say it?\n****"""
                    if tries == 0:
                        act = None  # force redo
                        temp += .3
                    else:
                        act = None  # skip task, nothing interesting to do
                        pass

            elif mode == 'Do' or mode == 'Move':
                dup = self.repetitive(act, last_act, self.format_history(2))
                if dup:
                    print(f'\n*****Repetitive act test failed, allowing repeated Move or Do anyway*****\n  {act}\n')
                    duplicative_insert = f"""\n****\nBeware of duplicating this previous act:\n'{act}'.
What else could you do or how else could you describe it?\n****"""
                    if tries < 1:
                        #act = None  # force redo
                        temp += .3
                    else:
                        #act = None #skip task, nothing interesting to do
                        pass
            elif mode ==  self.previous_action and self.repetitive(act, last_act, self.format_history(2)):
                print(f'\n*****Repetitive act test failed*****\n  {act}\n')
                duplicative_insert = f"""\n****\nBeware of duplicating this previous act:\n'{act}'. 
What else could you do or how else could you describe it?\n****"""
                if tries < 1:
                    act = None  # force redo
                    temp += .3
                else:
                    act = None #skip task, nothing interesting to do
                    pass
            tries += 1

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

    def update_intentions_wrt_say_think(self, source, text, reason):
        """Update intentions based on speech or thought"""
        # Skip intention processing during active dialogs
        if self.dialog_manager.current_dialog and source == 'dialog':
            print(f' in active dialog, no intention updates')
            return
        
        # Rest of the existing function remains unchanged
        print(f'Update intentions from say or think\n {text}\n{reason}')
        
        if source == 'watcher':  # Still skip watcher
            print(f' source is watcher, no intention updates')
            return
        
        prompt=[UserMessage(content="""Your task is to analyze the following text.

<text>
{{$text}}
</text>

Does it include an intention for 'I' to act? 
An action can be physical or verbal.
Thought, e.g. 'reflect on my situation', should NOT be reported as an intent to act.
Respond using the following XML form:

<analysis>
<act>False if there is no intention to act, True if there is an intention to act</act>
<intention>stated intention to say or act</intention>
<mode>'say' - if intention is to say something, 'do' - if intention is to perform a physical act</mode>
</analysis>

===Examples===

Text:
'Good morning Annie. I'm heading to the office for the day. Call maintenance about the disposal noise please.'

Response:
<analysis>
<act>True</act>
<intention>Head to the office for the day.</intention>
<mode>Do</mode>
</analysis>

Text:
'I really should reassure annie.'

Response:
<analysis>
<act>True</act>
<intention>Annie, you have been performing wonderfully!</intention>
<mode>Say</mode>
</analysis>

Text:
'Good morning Annie. Call maintenance about the disposal noise please.'

Response:
<analysis>
<act>False</act>
<intention>None</intention>
<mode>NA</mode>
</analysis>

Text:
'Reflect on my thoughts and feelings to gain clarity and understanding, which will ultimately guide me towards finding my place in the world.'

Response:
<analysis>
<act>False</act>
<intention>None</intention>
<mode>NA</mode>
</analysis>

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the intention analysis in XML as shown above.
End your response with:
</end>
""")]
        response = self.llm.ask({"text":text}, prompt, temp=0.1, stops=['</end>'], max_tokens=100)
        act = xml.find('<act>', response)
        if act is None or act.strip() != 'True':
            print(f'no intention in say or think')
            return
        intention = xml.find('<intention>', response)
        if intention is None or intention=='None':
            print(f'no intention in say or think')
            return
        mode = str(xml.find('<mode>', response))
        print(f'{self.name} adding intention from say or think {mode}, {source}: {intention}')
        new_intentions = []
        for candidate in self.intentions:
            candidate_source = xml.find('<source>', candidate)
            if candidate_source != source:
                new_intentions.append(candidate)
        self.intentions = new_intentions
        self.intentions.append(f'<intention> <mode>{mode}</mode> <act>{intention}</act> <reason>{reason}</reason> <source>{source}</source></intention>')
        if source != None and self.active_task.peek() is None: # do we really want to take a spoken intention as definitive?
            print(f'\nUpdate intention from Say setting active task to {source}')
            self.active_task.push(source)
        #ins = '\n'.join(self.intentions)
        #print(f'Intentions\n{ins}')

    def random_string(self, length=8):
        """Generate a random string of fixed length"""
        letters = string.ascii_lowercase
        return self.name+''.join(random.choices(letters, k=length))

    def tell(self, to_actor, message, source='dialog', respond=True):
        """Initiate or continue dialog with dialog context tracking"""
        if source == 'dialog':
            if not hasattr(self, 'dialog_manager'):
                self.dialog_manager = DialogManager(self)
            
            if self.dialog_manager.current_dialog is None:
                self.dialog_manager.start_dialog(self, to_actor, source)
            
            self.dialog_manager.add_turn()
        
        self.acts(to_actor, 'Say', message, '', source)
        

    def hear(self, from_actor, message: str, source='dialog', respond=True):
        """ called from acts when a character says something to this character """
        # Initialize dialog manager if needed
        if not hasattr(self, 'dialog_manager'):
            self.dialog_manager = DialogManager(self)
       
        # Special case for Owl-Doc interactions
        if self.name == 'Owl' and from_actor.name == 'Doc':
            # doc is asking a question or assigning a task
            new_task_name = self.random_string()
            new_task = f"""<plan><name>{new_task_name}</name>
<steps>
  1. Respond to Doc's statement:
  {message}
</steps>
<target>
{from_actor.name}
</target>
<reason>engaging with Doc: completing his assignments.</reason>
<termination_check>Responded</termination_check>
</plan>"""
            self.priorities.append(new_task)
            self.active_task.push(new_task_name)
            self.watcher_message_pending = True
            return

        # Dialog context management
        if source == 'dialog':
            if not self.dialog_manager.current_dialog:
                self.dialog_manager.start_dialog(from_actor, self)
            self.dialog_manager.add_turn()

        # Add perceptual input
        self.add_perceptual_input(f'You hear {from_actor.name} say: {message}', percept=False)
        
        if not respond:
            return
        
        # Generate response using existing prompt-based method
        prompt = [UserMessage(content="""Respond to the input below as {{$name}}.

{{$character}}.

Your current situation is:

<s  ituation>
{{$situation}}
</situation>

Your state is:

<state>
{{$state}}
</state>

Your memories include:

<memories>
{{$memories}}
</memories>

Recent conversation has been:
<recent_history>
{{$history}}
</recent_history>

Use the following XML template in your response:
<answer>
<response>response to this statement</response>
<reason>concise reason for this answer</reason>
<unique>'True' if you have something new to say, 'False' if you have nothing worth saying or your response is a repeat or rephrase of previous responses</unique>
</answer>

Your last response was:
<previous_response>
{{$last_act}}
</previous_response>

{{$activity}}

The statement you are responding to is:
<input>
{{$statement}}
</input>

Reminders: 
- Your task is to generate a response to the statement using the above XML format.
- The response should be in keeping with your character's State.
- The response can include body language or facial expressions as well as speech
- The response is in the context of the Input.
- Respond in a way that expresses an opinion on current options or proposes a next step.
- If the intent is to agree, state agreement without repeating the feelings or goals.
- Speak in your own voice. Do not echo the speech style of the Input. 
- Respond in the style of natural spoken dialog. Use short sentences and casual language.
 
Respond only with the above XML
Do not include any additional text. 
End your response with:
</end>
""")]

        mapped_state = self.map_state()
        last_act = ''
        if source in self.last_acts:
            last_act = self.last_acts[source]
        activity = ''
        if self.active_task.peek() != None and self.active_task.peek() != 'dialog':
            activity = f'You are currently actively engaged in {self.get_task_xml(self.active_task.peek())}'

        # Get recent memories
        recent_memories = self.structured_memory.get_recent(6)
        memory_text = '\n'.join(memory.text for memory in recent_memories)
        
        answer_xml = self.llm.ask({
            'character': self.character,
            'statement': f'{from_actor.name} says {message}',
            "situation": self.context.current_state,
            "name": self.name,
            "state": mapped_state,
            "memories": memory_text,  # Updated from 'memory'
            "activity": activity,
            "last_act": self.last_acts,
            'history': self.narrative.get_summary('medium'),
        }, prompt, temp=0.8, stops=['</end>'], max_tokens=180)

        response = xml.find('<response>', answer_xml)
        if response is None:
            if self.dialog_manager.current_dialog is not None:
                self.dialog_manager.end_dialog()
            return

        unique = xml.find('<unique>', answer_xml)
        if unique is None or 'false' in unique.lower():
            if self.active_task.peek() == 'dialog':
                # Add narrative update
                if hasattr(self, 'narrative'):
                    self.memory_consolidator.update_narrative(
                        memory=self.structured_memory,
                        narrative=self.narrative,
                        current_time=self.context.simulation_time,
                        character_desc=self.character,
                        relationsOnly=True
                    )
            self.dialog_manager.end_dialog()
            return 

        reason = xml.find('<reason>', answer_xml)
        if source != 'watcher' and source != 'inject':
            response_source = 'dialog'
        else:
            response_source = 'watcher'
            self.show = response
            if from_actor.name == 'Watcher':
                return response

        # Check for duplicate response
        dup = self.repetitive(response, last_act, '')
        if dup:
            if self.active_task.peek() == 'dialog':
             self.dialog_manager.end_dialog()
            return

        # Create intention for response
        intention = f'<intention> <mode>Say</mode> <act>{response}</act> <target>{from_actor.name}</target><reason>{str(reason)}</reason> <source>{response_source}</source></intention>'
        self.intentions.append(intention)

        # End dialog if turn limit reached
        if (self.dialog_manager.current_dialog and 
            self.dialog_manager.current_dialog.turn_count > 1 and 
            not self.always_respond):
            self.dialog_manager.end_dialog()


        return response + '\n ' + reason
    
    def choose(self, sense_data, action_choices):
        prompt = [UserMessage(content=self.character + """\nYour current situation is:

<situation>
{{$situation}}
</situation>

Your fundamental needs / drives include:

<drives>
{{$drives}}
</drives> 

Your state is:

<state>
{{$state}}
</state>

Your recent memories include:

<recent_memories>
{{$memories}}
</recent_memories>

Recent conversation has been:
<recent_history>
{{$history}}
</recent_history>

Your current priorities include:
<priorities>
{{$priorities}}
</priorities>

New Observation:
<input>
{{$input}}
</input>

Given who you are, your current Priorities, New Observation, and the other information listed above, think step-by-step 
and choose your most pressing, highest need / priority action to perform from the numbered list below:

<actions>
{{$actions}}
</actions>

Consider the conversation history in choosing your action. 
Respond in the context of the recent_history (if any) and in keeping with your character. 
Use the number for the selected action to instantiate the following XML format:

<action>
index-number of chosen action
</action>

Respond with the above XML, instantiated with the selected action number from the Action list. 
Do not include any introductory, explanatory, or discursive text, 
End your response with:
</end>
"""
                              )]

        mapped_state = self.map_state()
        action_choices = [f'{n} - {action}' for n, action in enumerate(action_choices)]
        
        # Get recent memories from structured memory
        recent_memories = self.structured_memory.get_recent(5)  # Get 5 most recent memories
        memory_text = '\n'.join(memory.text for memory in recent_memories)
        
        if len(action_choices) > 1:
            response = self.llm.ask({
                'input': sense_data + self.sense_input, 
                'history': self.narrative.get_summary('medium'),
                "memories": memory_text,
                "situation": self.context.current_state,
                "state": mapped_state, 
                "drives": '\n'.join(drive.text for drive in self.drives),
                "priorities": '\n'.join([str(xml.find('<name>', task)) for task in self.priorities]),
                "actions": '\n'.join(action_choices)
            }, prompt, temp=0.7, stops=['</end>'], max_tokens=300)
        # print(f'sense\n{response}\n')
        self.sense_input = ' '
        if 'END' in response:
            idx = response.find('END')
            response = response[:idx]
        if '<|im_end|>' in response:
            idx = response.find('<|im_end|>')
            response = response[:idx]
        index = -1
        choice = xml.find('<action>', response)
        if choice is None:
            choice = response.strip()  # llms sometimes ignore XML formatting for this prompt
        if choice is not None:
            try:
                draft = int(choice.strip())
                if draft > -1 and draft < len(action_choices):
                    index = draft  # found it!
            except Exception as e:
                try:
                    draft = find_first_digit(response)
                    draft = int(draft) if (draft != None and int(draft) < len(action_choices)) else None
                    index = draft if draft != None else -1
                except Exception as e:
                    traceback.print_exc()
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
        for state_term, info in self.state.items():
            if info["state"] == "very low" or info["state"] == "very high":
                return "full"
                
        # Check action progress
        if self.previous_action:
            effectiveness = self.action_memory.get_task_effectiveness(self.active_task.peek())
            if effectiveness < 0.3:
                return "full"
                
        # Check for significant perceptual changes
        if self.perceptual_state.recent_significant_change():
            return "medium"
            
        # Continue current action if making progress
        if self.previous_action and self.analyze_action_result() == "continue":
            return "continue"
            
        return "none"
    
    def senses_minimal(self):
        """Perform minimal state updates without full cognitive processing."""
        # Check for and process any direct messages
        if self.dialog_manager.current_dialog:  # Use DialogManager instead of dialog_status
            for actor in self.context.actors:
                if actor != self and actor.name in self.dialog_manager.current_dialog.participants:
                    recent_dialog = actor.show.strip()
                    if recent_dialog:
                        self.add_perceptual_input(f"You hear {actor.name} say: {recent_dialog}", percept=False)

        # Quick look check - just note major changes
        if self.mapAgent:
            current_view = self.mapAgent.look()
            if current_view != self.my_map[self.x][self.y]:
                self.my_map[self.x][self.y] = current_view
                self.add_to_history("You notice changes in your surroundings")

        # Update action record without full analysis
        if self.previous_action:
            record = create_action_record(
                agent=self,
                mode=Mode(self.previous_action),
                action_text=self.act_result,
                task_name=self.active_task.peek(),
                target=None
            )
            self.action_memory.add_record(record)

        # Add any obvious state changes to memory
        for key, info in self.state.items():
            if info["state"] in ["very high", "very low"]:
                self.add_to_history(f"You are acutely aware of your {key} state")    
            
    def senses(self, sense_data='', ui_queue=None):
        print(f'\n*********senses***********\nCharacter: {self.name}, active task {self.active_task.peek()}')
        all_actions={
            "Act": """Act as follows: '{action}'\n  because: '{reason}'""",
            "Move": """Move in direction: '{direction}'\n because: '{reason}'""",
            "Answer": """ Answer the following question: '{question}'""",
            "Say": """Say: '{text}',\n  because: '{reason}'""",
            "Think": """Think: '{text}\n  because" '{reason}'""",
            "Listen": """Listen attentively to others,\n  because: '{reason}'"""  # Removed Discuss, kept Listen
        }

        if self.wakeup:
            self.wakeup = False
            self.generate_state()
            self.update_priorities()
 
        self.memory_consolidator.update_narrative(self.structured_memory, 
                                                  self.narrative, 
                                                  self.context.simulation_time, 
                                                  self.character.strip(),
                                                  relationsOnly=True )
        print(f'\n\nshould_process_senses: {self.should_process_senses()}\n\n')
        my_active_task = self.active_task.peek()
        intention = None
        self.show = ''
        #check for intentions created by previous Say or hear
        for candidate in self.intentions:
            source = xml.find('<source>', candidate)
            if source == 'dialog' or source =='watcher' or source == self.active_task.peek():
                # if the source is the current task, or a dialog, or a watcher, we can use this intention
                intention = candidate

        if intention is None and my_active_task != None and my_active_task != 'dialog' and my_active_task != 'watcher':
            full_task = self.get_task_xml(my_active_task)
            if full_task is not None:
                intention = self.actualize_task(0, full_task)
                print(f'Found active_task {my_active_task}')
            else:
                print(f'Active_task gone! {my_active_task}')

        if intention is None:
            intention_choices = []
            for n, task in enumerate(self.priorities):
                choice =  f"{n}. {xml.find('<name>', task)}, because {xml.find('<reason>', task)}"
                intention_choices.append(choice)
                print(f'{self.name} considering action option {choice}')

            if len(intention_choices) == 0:
                print(f'{self.name} Oops, no available acts')
                return

            task_index = 0
            while intention is None:
                if len(intention_choices) > 1:
                    task_index = self.choose(sense_data, intention_choices)
                print(f'actualizing task {task_index}')
                intention = self.actualize_task(task_index, self.priorities[task_index])
                if intention is None:
                    if len(intention_choices) > 1:
                        del intention_choices[task_index]
                    else:
                        #not sure how we got here. No priority generated an actionable intention
                        self.update_priorities()
                        return self.senses(sense_data=sense_data, ui_queue=ui_queue)


        print(f'Intention: {intention}')
        self.intentions = [] # start anew next time
        act_name = xml.find('<mode>', intention)
        if act_name is not None:
            self.act_name = act_name.strip()
        act_arg = xml.find('<act>', intention)
        self.reason = xml.find('<reason>', intention)
        source = xml.find('<source>', intention)
        print(f'{self.name} choose {intention}')
        task_name = xml.find('<source>', intention)
        refresh_task = None # will be set to task intention to be refreshed if task is chosen for action

        task_xml = None
        refresh_task = None
        if not source == 'dialog' and act_name == 'Say' or act_name == 'Do':
            task_name = xml.find('<source>', intention)
            # for now very simple task tracking model:
            if task_name is None:
                task_name = self.make_task_name(self.reason)
                print(f'No source found, created task name: {task_name}')
            self.last_acts[task_name] = act_arg # this should pbly be in acts, where we actually perform
        if act_name == 'Think':
            task_name = xml.find('<source>', intention)
            #task_name = self.make_task_name(self.reason)
            self.reason = xml.find('<reason>', intention)
            self.thought = act_arg+'\n  '+self.reason
        target = None

        if act_name == 'Say':
            #responses, at least, explicitly name target of speech.
            target_name = xml.find('<target>', intention)
            if target_name is None or target_name == '':
                target_name = self.say_target(act_arg)
            if target_name != None:
                target = self.context.get_actor_by_name(target_name)

        #this will affect selected act and determine consequences
        self.acts(target, act_name, act_arg, self.reason, source)

        # maybe we should do this at start of next sense?
        if refresh_task is not None and task_name != 'dialog' and task_name != 'watcher':
            for task in self.priorities:
                if refresh_task == task:
                    #print(f"refresh task just before actualize_task call {xml.find('text>', refresh_task)}")
                    self.actualize_task('refresh', refresh_task) # regenerate intention

    # Add this method to the Agh class
    def _determine_category(self, message: str) -> str:
        """Determine the category of a memory message"""
        # Simple rule-based categorization
        message = message.lower()
        
        if any(word in message for word in ['say', 'hear', 'tell', 'ask', 'speak']):
            return 'dialog'
        elif any(word in message for word in ['see', 'look', 'observe', 'notice']):
            return 'observation'
        elif any(word in message for word in ['do', 'act', 'make', 'build', 'move']):
            return 'action'
        elif any(word in message for word in ['think', 'feel', 'believe', 'wonder']):
            return 'thought'
        else:
            return 'general'

    def format_thought_for_UI (self):
        if self.thought is None:
            return ''
        if self.priorities is None or len(self.priorities) == 0:
            return self.thought.strip()
        name = xml.find('<name>', self.priorities[0])
        if name is None or name == '':
            return self.thought.strip()
        return name + ': '+self.thought.strip()
    
    def format_priorities(self):
        return [xml.find('<name>', task) for task in self.priorities]
    
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
                'relationships': self.narrative.key_relationships,
            }
        }

