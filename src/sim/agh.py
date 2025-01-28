from datetime import datetime
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
    def __init__(self, name, character_description):
        print(f"Initializing Character {name}")  # Debug print
        self.name = name
        self.character = character_description
        self.llm = None  # will be init by Worldsim
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
        self.dialog_status = 'Waiting'  # Waiting, Pending
        self.dialog_length = 0
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
            background="",
            last_update=datetime.now(),  # Will be updated to simulation time
            key_relationships={},
            active_drives=[]
        )

        self.drives: List[Drive] = []  # Initialize empty drive list
        self.perceptual_state = PerceptualState(self)
        

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
                character_desc=self.character
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
            if task_name == xml.find('<Name>', candidate):
                print(f'found existing task\n  {task_name}')
                return candidate
        return None

    def find_or_make_task_xml(self, task_name, reason):
        candidate = self.get_task_xml(task_name)
        if candidate is not None:
            return candidate
        new_task = f'<Plan><Name>{task_name}</Name><Reason>{reason}</Reason></Plan>'
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
        self.add_to_history(f"You hear {from_actor.name} say: {message}")

    def say_target(self, text):
        """Determine the intended recipient of a message"""
        prompt = [UserMessage(content="""Determine the intended hearer of the following message spoken by you.
{{$character}}

You're recent history has been:

<History>
{{$history}}
</History>
        
Known other actors include:
        
<Actors>
{{$actors}}
</Actors>
        
The message is:
        
<Message>
{{$message}}
</Message>

Respond using the following XML format:

<Target>
  <Name>intended recipient name</Name>
</Target>

End your response with:
</End>
""")]
        response = self.llm.ask({
            'character': self.character,
            'history': self.narrative.get_summary('medium'),
            'actors': '\n'.join([actor.name for actor in self.context.actors]),
            "message": text
        }, prompt, temp=0.2, stops=['</End>'], max_tokens=180)

        return xml.find('<Name>', response)

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
            if dir == 'Current':
                del dir_obs['visibility']
            view[dir] = dir_obs
        self.my_map[self.x][self.y] = view
        return self.my_map

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
{json.dumps(obs, indent=2).strip('\'"{}')}
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
                self.add_to_history(f'You see {actor.name}')

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
        super().__init__(name, character_description)
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
        description = self.character
        try:
            context = ''
            i = 0
            candidates = self.context.current_state.split('.')
            while len(context) < 84 and i < len(candidates):
                context += candidates[i]+'. '
                i +=1
            context = context[:96]
            description = self.name + ', '+'. '.join(self.character.split('.')[:2])[6:] +', '+\
                self.show.replace(self.name, '')[-128:].strip()
            prompt = [UserMessage(content="""Following is a description of a character in a play. 

<Description>
{{$description}}
</Description>
            
Extract from this description two or three words that describe the character's emotional state.
Use common adjectives like happy, sad, frightened, worriedangry, curious, aroused, cold, hungry, tired, disoriented, etc.
The words should each describe a different aspect of the character's emotional state, and should be distinct from each other.

Respond using this XML format:

<EmotionalState>
  <State>adjective</State>
</EmotionalState>

End your response with:
</End>
""")]
            concerns = ''
            for priority in self.priorities:
                concern = xml.find('<State>', priority) + '. '+xml.find('<Reason>', priority)
                concerns = concerns + '; '+concern
            state = description + '.\n '+concerns +'\n'+ context
            response = self.llm.ask({ "description": state}, prompt, temp=0.2, stops=['</End>'], max_tokens=100)
            state = xml.find('<EmotionalState>', response)
            if state:
                states = xml.findall('<State>', state)
                description = description[:192-min(len(context), 48)] + ', '+', '.join(states)+'.'
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
   
    def add_to_history_perceptual(self, text: str, intensity: float = 0.7):
        """Legacy method - routes through perceptual system"""
        self.perceptual_state.add_input(PerceptualInput(
            mode=SensoryMode.EXTERIOR,  # Default for legacy calls
            content=text,
            timestamp=self.context.simulation_time,
            intensity=intensity
        ))
        
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

    def look(self, height=5):
        if self.mapAgent is None:
            return ''  # Return empty string if no map agent exists
        obs = self.mapAgent.look()
        view = {}
        for dir in ['Current', 'North', 'Northeast', 'East', 'Southeast', 
                   'South', 'Southwest', 'West', 'Northwest']:
            dir_obs = map.extract_direction_info(obs, dir)
            if dir == 'Current':
                del dir_obs['visibility']  # empty, since visibility means how far one can see
            view[dir] = dir_obs
        self.my_map[self.x][self.y] = view
        return self.my_map
    

    def generate_state(self):
        """Generate states to track, derived from drives and current memory context"""
        self.state = {}
        
        prompt = [UserMessage(content="""Given a Drive and related memories, assess the current state relative to that drive.

<Drive>
{{$drive}}
</Drive>

<RecentMemories>
{{$recent_memories}}
</RecentMemories>

<DriveMemories>
{{$drive_memories}}
</DriveMemories>

<Situation>
{{$situation}}
</Situation>

<Character>
{{$character}}
</Character>

Analyze the memories and assess the current state relative to this drive.
Consider:
1. How well the drive's needs are being met
2. Recent events that affect the drive
3. The importance scores of relevant memories
4. Any patterns or trends in the memories

Respond using this XML format:

<State> 
  <Term>concise term for this drive state</Term>
  <Assessment>very high/high/medium-high/medium/medium-low/low</Assessment>
  <Trigger>specific situation or memory that most affects this state</Trigger>
  <Termination>condition that would satisfy this drive</Termination>
</State>

Respond ONLY with the above XML.
End response with:
</End>
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
            }, prompt, temp=0.3, stops=['</End>'])
            
            # Parse response
            try:
                term = xml.find('<Term>', response)
                assessment = xml.find('<Assessment>', response)
                trigger = xml.find('<Trigger>', response)
                termination = xml.find('<Termination>', response)
                
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

<State>
{{$state}}
</State>

<Situation>
{{$situation}}
</Situation>

<Character>
{{$character}}
</Character>

<RecentMemories>
{{$memories}}
</RecentMemories>

<History>
{{$history}}
</History>
 
<Events>
{{$events}}
</Events>

<Termination_check>
{{$termination_check}}
</Termination_check>

Respond using this XML format:

<Termination> 
    <Level>value of state satisfaction, True if termination test is met in Events or History</Level>
</Termination>

The 'Level' above should be True if the termination test is met in Events or recent History, and False otherwise.  

Respond ONLY with the above XML.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</End>
"""
                                  )]

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
        }, prompt, temp=0.3, stops=['</End>'], max_tokens=60)

        satisfied = xml.find('<Level>', response)
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

<Name>task-name</Name>

<Motivation>
{{$reason}}
</Motivation>

Respond only with your task-name using the above XML
Do not include your reasoning in your response.
Do not include any introductory, discursive, or explanatory text.
End your response with:
</End>
""")]
        response = self.llm.ask({"reason":reason}, instruction, temp=0.3, stops=['</End>'], max_tokens=12)
        return xml.find('<Motivation', response)
                    
    def get_task_xml(self, task_name):
        for candidate in self.priorities:
            #print(f'find_or_make testing\n {candidate}\nfor name {task_name}')
            if task_name == xml.find('<Name>', candidate):
                print(f'found existing task\n  {task_name}')
                return candidate
        return None
    
    def find_or_make_task_xml(self, task_name, reason):
        candidate = self.get_task_xml(task_name)
        if candidate != None:
            return candidate
        new_task = f'<Plan><Name>{task_name}</Name><Reason>{reason}</Reason></Plan>'
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

Recent history:
<History>
{{$history}}
</History>

New response:
<Response>
{{$response}}
</Response>

Consider:
- The meaning and intent of the response
- The flow of conversation
- Whether it advances the dialog
- If it repeats ideas or phrases already expressed

Respond with only 'True' if repetitive or 'False' if the response adds something new.
End response with:
<End/>""")]

        result = self.llm.ask({
            'history': history,
            'response': new_response
        }, prompt, temp=0.2, stops=['<End/>'], max_tokens=100)

        return 'true' in result.lower()

    def clear_task_if_satisfied(self, task_xml, consequences, world_updates):
        """Check if task is complete and update state"""
        termination_check = xml.find('<TerminationCheck>', task_xml) if task_xml != None else None
        if termination_check is None:
            return

        # Test completion through cognitive processor's state system
        satisfied = self.test_priority_termination(
            termination_check, 
            consequences,
            world_updates
        )

        if satisfied:
            task_name = xml.find('<Name>', task_xml)
            if task_name == self.active_task.peek():
                self.active_task.pop()

            try:
                self.priorities.remove(task_xml)
            except Exception as e:
                print(str(e))

            new_intentions = []
            for intention in self.intentions:
                if xml.find('<Source>', intention) != task_name:
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

        # Determine visibility of action
        self_verb = 'hear' if act_name == 'Say' or act_name == 'Listen' else 'see'
        visible_arg = 'Thinking ...' if act_name == 'Think' else act_arg
        if act_name == 'Listen': visible_arg = 'Listening ...'

        # Update main display
        intro = f'{self.name}:' if (act_name == 'Say' or act_name == 'Think' or act_name == 'Listen') else f'{self.name}'
        visible_arg = f"'{visible_arg}'" if act_name == 'Say' else visible_arg

        if act_name != 'Do':
            self.show += f"\n{intro} {visible_arg}"
        self.add_to_history(f"\nYou {act_name}: {act_arg} \n  why: {reason}")

        # Update thought display
        if act_name == 'Think':
            self.thought = act_arg + '\n ... ' + self.reason
        else:
            self.thought =  self.reason

        # Update active task if needed
        if (act_name == 'Do' or act_name == 'Say') and source != 'dialog' and source != 'watcher':
            if self.active_task.peek() != source:
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
            self.show += '\n  ' + consequences.strip() + '\n' + world_updates.strip()
            self.add_to_history(f"You observe {consequences}")
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
                self.look()
        elif act_name == 'Look':
            self.look()

        self.previous_action = act_name

        if act_name == 'Think':
            # Update intentions based on thought
            self.update_intentions_wrt_say_think(source, act_arg, reason)
 
        # After action completes, update record with results
        # Notify other actors of action
        if act_name != 'Say':  # everyone sees it
            for actor in self.context.actors:
                if actor != self:
                    actor.add_to_history(self.show)
        else:
            for actor in self.context.actors:
                if actor != self:
                    if source != 'watcher':  # when talking to watcher, others don't hear it
                        if actor != target:
                            actor.add_to_history(f'You hear {self.name} say: {act_arg}')
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

<Character>
{{$character}}
</Character>

<State>
{{$state}}
</State>

<RecentMemories>
{{$memories}}
</RecentMemories>

Create three specific, actionable plans that address your current needs and situation.
Consider:
1. Your current state assessments
2. Recent memories and events
3. Your basic drives and needs

Respond using this XML format:
<Plan>
  <Name>brief action name</Name>
  <Description>detailed action description</Description>
  <Reason>why this action is important now</Reason>
  <TerminationCheck>condition that would satisfy this need</TerminationCheck>
</Plan>

Description, should be detailed but concise. Reason, and TerminationCheck should be terse.
Respond ONLY with three plans using the above XML format.
Order plans from highest to lowest priority.
End response with:
</End>
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
        }, prompt, temp=0.7, stops=['</End>'])

        # Extract plans from response
        self.priorities = []
        for plan in xml.findall('<Plan>', response):
            if xml.find('<Name>', plan):
                self.priorities.append(plan)

    def test_priority_termination(self, termination_check, consequences, updates=''):
        """Test if consequences of recent acts (or world update) have satisfied priority"""
        prompt = [UserMessage(content="""A CompletionCriterion is provided below. 
Reason step-by-step to establish whether this CompletionCriterion has now been met as a result of recent Events,
using the CompletionCriterion as a guide for this assessment.
Consider these factors in determining task completion:
- Sufficient progress towards goal for intended purpose
- Diminishing returns on continued effort
- Environmental or time constraints
- "Good enough" vs perfect completion
                    
<History>
{{$history}}
</History>

<Events>
{{$events}}
</Events>

<RecentMemories>
{{$memories}}
</RecentMemories>

<CompletionCriterion>
{{$termination_check}}
</CompletionCriterion>

Respond using this XML format:

Respond with both completion status and progress indication:
<Completion>
  <Status>Complete|Partial|Insufficient</Status>
  <Progress>0-100 percentage</Progress>
  <Reason>Why this assessment</Reason>
</Completion>
<Complete> 
    <Level>value of task completion, True, Unknown, or False</Level>
    <Evidence>concise statement of evidence in events to support this level of task completion</Evidence>
</Complete>

the 'Level' above should be True if the termination check is met in Events or recent History, 
Unknown if the Events do not support a definitive assessment, or False if Events provide little or no evidence for task completion.  

Respond ONLY with the above XML
Do not include any introductory, explanatory, or discursive text.
End your response with:
</End>
"""
                                  )]

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
            "history": self.format_history()
        }, prompt, temp=0.3, stops=['</End>'], max_tokens=120)

        satisfied = xml.find('<Level>', response)
        if satisfied != None and satisfied.lower().strip() == 'true':
            print(f'Priority {termination_check} Satisfied!')
        return False if satisfied == None else satisfied.lower().strip() == 'true'

    def actualize_task(self, n, task_xml):
        task_name = xml.find('<Name>', task_xml)
        if task_xml is None or task_name is None:
            raise ValueError(f'Invalid task {n}, {task_xml}')
        last_act = self.get_task_last_act(task_name)
        reason = xml.find('<Reason>', task_xml)

        prompt = [UserMessage(content="""You are {{$character}}.
Your task is to generate an Actionable (a 'Think', 'Say', 'Look', Move', or 'Do') to advance the next step of the following task.

<Task>
{{$task}}
</Task>

Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your state is:

<State>
{{$state}}
</State>

Your recent memories include:

<RecentMemories>
{{$memories}}
</RecentMemories>

Recent history includes:
<RecentHistory>
{{$history}}
</RecentHistory>

The Previous specific act for this Task, if any, was:

<PreviousSpecificAct>
{{$lastAct}}
</PreviousSpecificAct>

And the observed result of that was:
<ObservedResult>
{{$lastActResult}}.
</ObservedResult>

Respond with an Actionable, including its Mode and SpecificAct. 

In choosing an Actionable (see format below), you can choose from three Mode values:
- 'Think' - reason about the current situation wrt your state and the task.
- 'Say' - speak, to motivate others to act, to align or coordinate with them, to reason jointly, or to establish or maintain a bond. 
    Say is especially appropriate when there is an actor you are unsure of, you are feeling insecure or worried, or need help.
    For example, if you want to build a shelter with Samantha, it might be effective to Say 'Samantha, let's build a shelter.'
- 'Look' - observe your surroundings, gaining information on features, actors, and resources at your current location and for the eight compass
    points North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- 'Move' - move in any one of eight directions: North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- 'Do' - perform an act (other than move) with physical consequences in the world.

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
- If speaking (mode is 'Say'), then:
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
<Actionable>
  <Mode>'Think', 'Say', 'Move', or 'Do', corresponding to whether the act is a reasoning, speech, or physical act</Mode>
  <SpecificAct>thoughts, words to speak, DIRECTION (if 'Move' mode), or physical action description</SpecificAct>
</Actionable>

===Examples===

Task:
Situation: increased security measures; State: fear of losing Annie

Response:
<Actionable>
  <Mode>Do</Mode>
  <SpecificAct>Call a meeting with the building management to discuss increased security measures for Annie and the household.</SpecificAct>
</Actionable>

----

Task:
Establish connection with Joe given RecentHistory element: "Who is this guy?"

Response:
<Actionable>
  <Mode>Say</Mode>
  <SpecificAct>Hi, who are you?</SpecificAct>
</Actionable>

----

Task:
Find out where I am given Situation element: "This is very very strange. Where am I?"

Response:
<Actionable>
  <Mode>Look</Mode>
  <SpecificAct>Samantha starts to look around for any landmarks or signs of civilization</SpecificAct>
</Actionable>

----

Task:
Find food.


Response:
<Actionable>
  <Mode>Move</Mode>
  <SpecificAct>SouthWest</SpecificAct>
  <Reason>I need to find food, and my previous Look showed berries one move SouthWest.</Reason>
</Actionable>

===End Examples===

Use the XML format:

<Actionable> 
  <Mode>Think, Say, or Do<Mode>
  <SpecificAct>specific thoughts, words, or action</SpecificAct> 
</Actionable>

Respond ONLY with the above XML.
Your name is {{$name}}, phrase the statement of specific action in your voice.
Ensure you do not duplicate content of a previous specific act.
{{$duplicative}}

Again, the task to translate into an Actionable is:
<Task>
{{$task}} given {{$reason}}
</Task>

Do not include any introductory, explanatory, or discursive text.
End your response with:
</End>
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
            }, prompt, temp=temp, top_p=1.0, stops=['</End>'], max_tokens=180)

            # Rest of existing while loop...
            print(response)
            act = xml.find('<SpecificAct>', response)
            mode = xml.find('<Mode>', response)

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

            elif mode == 'Do':
                dup = self.repetitive(act, last_act, self.format_history(2))
                if dup:
                    print(f'\n*****Repetitive act test failed*****\n  {act}\n')
                    duplicative_insert = f"""\n****\nBeware of duplicating this previous act:\n'{act}'.
What else could you do or how else could you describe it?\n****"""
                    if tries < 1:
                        act = None  # force redo
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
                candidate_source = xml.find('<Source>', candidate)
                if candidate_source == task_name:
                    self.intentions.remove(candidate)
                elif candidate_source is None or candidate_source == 'None':
                    self.intentions.remove(candidate)

            return f'<Intent> <Mode>{mode}</Mode> <Act>{act}</Act> <Reason>{reason}</Reason> <Source>{task_name}</Source> </Intent>'
        else:
            print(f'No intention constructed, presumably duplicate')
            return None

    def update_intentions_wrt_say_think(self, source, text, reason):
        # determine if text implies an intention to act, and create a formatted intention if so
        print(f'Update intentions from say or think\n {text}\n{reason}')

        if source == 'dialog' or source=='watcher':
            print(f' source is dialog or watcher, no intention updates')
            return
        prompt=[UserMessage(content="""Your task is to analyze the following text.

<Text>
{{$text}}
</Text>

Does it include an intention for 'I' to act? 
An action can be physical or verbal.
Thought, e.g. 'reflect on my situation', should NOT be reported as an intent to act.
Respond using the following XML form:

<Analysis>
<Act>False if there is no intention to act, True if there is an intention to act</Act>
<Intention>stated intention to say or act</Intention>
<Mode>'Say' - if intention is to say something, 'Do' - if intention is to perform a physical act/Mode>
</Analysis>

===Examples===

Text:
'Good morning Annie. I'm heading to the office for the day. Call maintenance about the disposal noise please.'

Response:
<Analysis>
<Act>True</Act>
<Intention>Head to the office for the day.</Intention>
<Mode>Do</Mode>
</Analysis>

Text:
'I really should reassure annie.'

Response:
<Analysis>
<Act>True</Act>
<Intention>Annie, you have been performing wonderfully!</Intention>
<Mode>Say</Mode>
</Analysis>

Text:
'Good morning Annie. Call maintenance about the disposal noise please.'

Response:
<Analysis>
<Act>False</Act>
<Intention>None</Intention>
<Mode>NA</Mode>
</Analysis>

Text:
'Reflect on my thoughts and feelings to gain clarity and understanding, which will ultimately guide me towards finding my place in the world.'

Response:
<Analysis>
<Act>False</Act>
<Intention>None</Intention>
<Mode>NA</Mode>
</Analysis>

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the intention analysis in XML as shown above.
End your response with:
</End>
""")]
        response = self.llm.ask({"text":text}, prompt, temp=0.1, stops=['</End>'], max_tokens=100)
        act = xml.find('<Act>', response)
        if act is None or act.strip() != 'True':
            print(f'no intention in say or think')
            return
        intention = xml.find('<Intention>', response)
        if intention is None or intention=='None':
            print(f'no intention in say or think')
            return
        mode = str(xml.find('<Mode>', response))
        print(f'{self.name} adding intention from say or think {mode}, {source}: {intention}')
        new_intentions = []
        for candidate in self.intentions:
            candidate_source = xml.find('<Source>', candidate)
            if candidate_source != source:
                new_intentions.append(candidate)
        self.intentions = new_intentions
        self.intentions.append(f'<Intent> <Mode>{mode}</Mode> <Act>{intention}</Act> <Reason>{reason}</Reason> <Source>{source}</Source></Intent>')
        if source != None and self.active_task.peek() is None: # do we really want to take a spoken intention as definitive?
            print(f'\nUpdate intention from Say setting active task to {source}')
            self.active_task.push(source)
        #ins = '\n'.join(self.intentions)
        #print(f'Intentions\n{ins}')

    def tell(self, to_actor, message, source='dialog', respond=True):
        if self.active_task.peek() != 'dialog':
            self.active_task.push('dialog')
        self.acts(to_actor,'Say', message, '', 'dialog')
        return

        #generate response intention

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
        

 
    def hear(self, from_actor, message, source='dialog', respond=True):
        # Initialize dialog manager if needed
        if not hasattr(self, 'dialog_manager'):
            self.dialog_manager = DialogManager(self)

        # Special case for Owl-Doc interactions
        if self.name == 'Owl' and from_actor.name == 'Doc':
            # doc is asking a question or assigning a task
            new_task_name = self.random_string()
            new_task = f"""<Plan><Name>{new_task_name}</Name>
<Steps>
  1. Respond to Doc's statement:
  {message}
</Steps>
<Target>
{from_actor.name}
</Target>
<Reason>engaging with Doc: completing his assignments.</Reason>
<TerminationCheck>Responded</TerminationCheck>
</Plan>"""
            self.priorities.append(new_task)
            self.active_task.push(new_task_name)
            self.watcher_message_pending = True
            return

        # Dialog context management
        if source == 'dialog':
            if self.dialog_manager.current_dialog is None:
                self.dialog_manager.start_dialog(from_actor, self, source)
            self.dialog_manager.add_turn()

        # Add to history
        self.add_to_history(f'You hear {from_actor.name} say {message}')
    
        if not respond:
            return

        # Generate response using existing prompt-based method
        prompt = [UserMessage(content="""Respond to the input below as {{$name}}.

{{$character}}.

Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your state is:

<State>
{{$state}}
</State>

Your memories include:

<Memories>
{{$memories}}
</Memories>

Recent conversation has been:
<RecentHistory>
{{$history}}
</RecentHistory>

Use the following XML template in your response:
<Answer>
<Response>response to this statement</Response>
<Reason>concise reason for this answer</Reason>
<Unique>'True' if you have something new to say, 'False' if you have nothing worth saying or your response is a repeat or rephrase of previous responses</Unique>
</Answer>

Your last response was:
<PreviousResponse>
{{$last_act}}
</PreviousResponse>

{{$activity}}

The statement you are responding to is:
<Input>
{{$statement}}
</Input>>

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
</End>
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
        }, prompt, temp=0.8, stops=['</End>'], max_tokens=180)

        response = xml.find('<response>', answer_xml)
        if response is None:
            if self.dialog_manager.current_dialog is not None:
                self.dialog_manager.end_dialog()
            return

        unique = xml.find('<Unique>', answer_xml)
        if unique is None or 'false' in unique.lower():
            if self.active_task.peek() == 'dialog':
             self.dialog_manager.end_dialog()
            return 

        reason = xml.find('<Reason>', answer_xml)
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
        intention = f'<Intent> <Mode>Say</Mode> <Act>{response}</Act> <Target>{from_actor.name}</Target><Reason>{str(reason)}</Reason> <Source>{response_source}</Source></Intent>'
        self.intentions.append(intention)

        # End dialog if turn limit reached
        if (self.dialog_manager.current_dialog and 
            self.dialog_manager.current_dialog.turn_count > 1 and 
            not self.always_respond):
            self.dialog_manager.end_dialog()

        return response + '\n ' + reason
    
    def choose(self, sense_data, action_choices):
        prompt = [UserMessage(content=self.character + """\nYour current situation is:

<Situation>
{{$situation}}
</Situation>

Your fundamental needs / drives include:

<Drives>
{{$drives}}
</Drives> 

Your state is:

<State>
{{$state}}
</State>

Your recent memories include:

<RecentMemories>
{{$memories}}
</RecentMemories>

Recent conversation has been:
<RecentHistory>
{{$history}}
</RecentHistory>

Your current priorities include:
<Priorities>
{{$priorities}}
</Priorities>

New Observation:
<Input>
{{$input}}
</Input>

Given who you are, your current Priorities, New Observation, and the other information listed above, think step-by-step 
and choose your most pressing, highest need / priority action to perform from the numbered list below:

<Actions>
{{$actions}}
</Actions>

Consider the conversation history in choosing your action. 
Respond in the context of the RecentHistory (if any) and in keeping with your character. 
Use the number for the selected action to instantiate the following XML format:

<Action>
index-number of chosen action
</Action>

Respond with the above XML, instantiated with the selected action number from the Action list. 
Do not include any introductory, explanatory, or discursive text, 
End your response with:
</End>
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
                "priorities": '\n'.join([str(xml.find('<Name>', task)) for task in self.priorities]),
                "actions": '\n'.join(action_choices)
            }, prompt, temp=0.7, stops=['</End>'], max_tokens=300)
        # print(f'sense\n{response}\n')
        self.sense_input = ' '
        if 'END' in response:
            idx = response.find('END')
            response = response[:idx]
        if '<|im_end|>' in response:
            idx = response.find('<|im_end|>')
            response = response[:idx]
        index = -1
        choice = xml.find('<Action>', response)
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
 
        dialog_active = False
        my_active_task = self.active_task.peek()
        intention = None
        self.show = ''
        #check for intentions created by previous Say or hear
        for candidate in self.intentions:
            source = xml.find('<Source>', candidate)
            if source == 'dialog' or source =='watcher':
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
                choice =  f"{n}. {xml.find('<Name>', task)}, because {xml.find('<Reason>', task)}"
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
        act_name = xml.find('<Mode>', intention)
        if act_name is not None:
            self.act_name = act_name.strip()
        act_arg = xml.find('<Act>', intention)
        self.reason = xml.find('<Reason>', intention)
        source = xml.find('<Source>', intention)
        print(f'{self.name} choose {intention}')
        task_name = xml.find('<Source>', intention)
        refresh_task = None # will be set to task intention to be refreshed if task is chosen for action

        task_xml = None
        refresh_task = None
        if not dialog_active and act_name == 'Say' or act_name == 'Do':
            task_name = xml.find('<Source>', intention)
            # for now very simple task tracking model:
            if task_name is None:
                task_name = self.make_task_name(self.reason)
                print(f'No source found, created task name: {task_name}')
            self.last_acts[task_name] = act_arg # this should pbly be in acts, where we actually perform
        if act_name == 'Think':
            task_name = xml.find('<Source>', intention)
            #task_name = self.make_task_name(self.reason)
            self.reason = xml.find('<Reason>', intention)
            self.thought = act_arg+'\n  '+self.reason
        target = None

        if act_name == 'Say':
            #responses, at least, explicitly name target of speech.
            target_name = xml.find('<Target>', intention)
            if target_name == None:
                target_name = self.say_target(act_arg)
            if target_name != None:
                target = self.context.get_actor_by_name(target_name)

        #this will affect selected act and determine consequences
        self.acts(target, act_name, act_arg, self.reason, source)

        # maybe we should do this at start of next sense?
        if refresh_task is not None and task_name != 'dialog' and task_name != 'watcher':
            for task in self.priorities:
                if refresh_task == task:
                    #print(f"refresh task just before actualize_task call {xml.find('<Text>', refresh_task)}")
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
        name = xml.find('<Name>', self.priorities[0])
        if name is None or name == '':
            return self.thought.strip()
        return name + ': '+self.thought.strip()
    
    def to_json(self):
        """Return JSON-serializable representation"""
        return {
            'name': self.name,
            'show': self.show.strip(),  # Current visible state
            'thoughts': self.format_thought_for_UI(),  # Current thoughts
            'priorities': [p for p in self.priorities],
            'description': self.character.strip(),  # For image generation
            'history': self.format_history().strip(), # Recent history, limited to last 5 entries
            'narrative': {
                'recent_events': self.narrative.recent_events,
                'ongoing_activities': self.narrative.ongoing_activities,
                'relationships': self.narrative.key_relationships,
                'background': self.narrative.background
            }
        }

