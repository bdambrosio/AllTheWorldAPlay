from enum import Enum
from weakref import WeakValueDictionary
import os, sys
from datetime import timedelta, datetime
import json
import numpy as np
#from sentence_transformers import SentenceTransformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import hash_utils

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

def datetime_handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # More standard format
    elif isinstance(obj, timedelta):
        return obj.total_seconds()
    else:
        return str(obj) 


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
    
    def __init__(self, name, actors, description, preconditions=None, termination=None, signalCluster=None, drives=None):
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
        self.completion_statement = ''

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
        return f"Goal {self.name}: {self.description}; actors: {', '.join([actor.name for actor in self.actors])}; preconditions: {self.preconditions};  termination: {self.termination}"
    
from datetime import timedelta

def parse_duration(duration_str: str) -> timedelta:
    """Convert duration string to timedelta
    Args:
        duration_str: Either minutes as int ("5") or with units ("2 minutes")
    Returns:
        timedelta object
    """
    if isinstance(duration_str, timedelta):
        return duration_str
    
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
    
    def __init__(self, name, description, reason, actors, goal, termination=None, start_time=None, duration=None):
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
        self.completion_statement = ''
        self.progress = 0

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
    def __init__(self, act=True, scene=True, signal=True, goal=True, task=True, action=True):
        self.act = act
        self.scene = scene
        self.signal = signal
        self.goal = goal
        self.task = task
        self.action = action

class CentralNarrative:
    def __init__(self, question, why, reason, others, risks):
        self.question = question
        self.why = why
        self.reason = reason
        self.others = others
        self.risks = risks

    def to_string(self):
        return f'CentralNarrative: Question - {self.question}; Why - {self.why}; Reason - {self.reason}; Others Roles - {self.others}; Risks - {self.risks}'
    
    @classmethod
    def parse_from_hash(cls, hash_string):
        """Parse a hash-formatted string into a CentralNarrative object"""
        """Validate an XML act and create an Act object
        
        Args:
            hash_string: Hash-formatted act definition
            task: Task this act is for
        """
        try:
            question = hash_utils.find('question', hash_string)
            why = hash_utils.find('why', hash_string)
            reason = hash_utils.find('reason', hash_string)
            others = hash_utils.find('others', hash_string)
            risks = hash_utils.find('risks', hash_string)
            if question and why and reason and others and risks:
                return CentralNarrative(question, why, reason, others, risks)
            else:
                print(f"Error parsing CentralNarrative from hash: {hash_string}")
                return None
        except Exception as e:
            print(f"Error parsing CentralNarrative from hash: {e}")
            return None       