from __future__ import annotations
import base64
import os, json, math, time, requests, sys


from sim.SpeechStylizer import SpeechStylizer
from sim.trace import trace
import zlib
import sim.survival as survival
import sim.stage as stage
import sim.ablation as ablation
import sim.record as record

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
from sim.cognitive import drives
from sim.cognitive.drives import Drive
from sim.memory.consolidation import MemoryConsolidator
from sim.memory.core import MemoryEntry, NarrativeSummary, StructuredMemory
from sim.memory.core import MemoryRetrieval
from sim.cognitive.EmotionalStance import EmotionalStance
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
from sim.character_dataclasses import Mode, Stack, Act, Goal, Task, Autonomy, datetime_handler     
if TYPE_CHECKING:
    from sim.context import Context  # Only imported during type checking

logger = logging.getLogger('simulation_core')

# Acts a character may spend on one task before step_task returns (the loop allowed 2 but
# returned after the first, so each task got one act and a Say took the whole beat).
MAX_ACTS_PER_TASK = 3



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
    def __init__(self, name, character_description, reference_description='', init_x=None, init_y=None, server_name='local', mapAgent=True):
        print(f"Initializing Character {name}")  # Debug print
        self.name = name.strip().capitalize()
        self.character = character_description
        self.original_character = character_description
        self.llm = llm_api.LLM(server_name)
        self.llm.owner = self.name  # for the run trace (sim.trace)
        # voice/appearance traits, filled by set_character_traits(); defaults so prompts that
        # read them don't crash when that LLM call fails
        self.gender = ''
        self.age = ''
        self.accent = ''
        self.personality = []
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
        self.recent_exchanges = []   # (other, what I opened with, the last thing they answered): see note_exchange
        # the drives of a character whose play gives it none
        self.drives = [
            Drive( "immediate physiological needs: survival, water, food, clothing, shelter, rest."),
            Drive("safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events."),
            Drive("love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friendship, intimacy, trust, acceptance."),
            Drive("self-esteem, including recognition, respect, and achievement.")
        ]
            
        self.always_respond = True
        
        # Initialize memory systems
        # (CognitiveToolInterface was constructed here but never used; it built an OpenAI client
        # that required OPENAI_API_KEY even for local vLLM)
        self.structured_memory:StructuredMemory = StructuredMemory(owner=self)
        self.memory_consolidator:MemoryConsolidator = MemoryConsolidator(self, self.llm, self.context)
        self.memory_retrieval:MemoryRetrieval = MemoryRetrieval()
        self.new_memory_cnt = 0
        self.next_task = None  # Add this line
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

    def _restamp_pre_context_entries(self, start_time):
        """Percepts and memories added before the character joined a Context (e.g. a play's
        opening add_perceptual_input calls) were stamped with the wall clock, which is years
        after the play's simulation clock, so they sorted as 'most recent' for the whole play.
        Anything stamped later than the play's start is such an entry: move it to the start."""
        for mem in self.structured_memory.concrete_memories:
            if mem.timestamp and mem.timestamp > start_time:
                mem.timestamp = start_time
        for mem in self.structured_memory.abstract_memories:
            if mem.start_time and mem.start_time > start_time:
                mem.start_time = start_time
            if mem.end_time and mem.end_time > start_time:
                mem.end_time = start_time
        for inputs in self.perceptual_state.current_inputs.values():
            for percept in inputs:
                if percept.timestamp and percept.timestamp > start_time:
                    percept.timestamp = start_time

    def set_context(self, context: Context):
        if getattr(self, 'context', None) is context and getattr(self, 'speech_stylizer', None) is not None:
            # Already initialized for this context (owner NPCs are set up in get_npc_by_name and
            # again in Context.__init__'s actor loop); redoing it rebuilds the managers and
            # repeats three LLM calls.
            return
        self.context = context
        self._restamp_pre_context_entries(context.simulation_time)
        self.actor_models = KnownActorManager(self, context)
        self.resource_models = KnownResourceManager(self, context)
        self.resourceRefManager = ResourceReferenceManager(self, context, self.llm)
        self.narrative = NarrativeSummary(
            recent_events="",
            ongoing_activities="",
            last_update=self.context.simulation_time, 
            active_drives=[]
        )
        if self.memory_consolidator:
            self.memory_consolidator.context = context
        self.emotionalStance = EmotionalStance.for_character(self)
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
        stylizer = getattr(self, 'speech_stylizer', None)
        if stylizer is not None and stylizer.voice_line():
            description += f'\nHow you speak: {stylizer.voice_line()}.'
        if self.actor_models.get_known_actor_relationship(self.name):
            description += '\n\n' + self.actor_models.get_known_actor_relationship(self.name)
        if type(description) != str:
            description = ''
        return description
    
    def validate_and_create_goal(self, goal_hash):
        """Validate a goal hash and create a goal object

        Args:
            goal_hash: Hash-formatted goal definition
        """
        name = hash_utils.find('goal', goal_hash)
        description = hash_utils.find('description', goal_hash)
        other_actor_name = hash_utils.find('otherActorName', goal_hash)
        preconditions = hash_utils.find('preconditions', goal_hash)
        termination = hash_utils.find('termination', goal_hash)
        drive = drives.find(self, hash_utils.find('drive', goal_hash))   # the drive this goal serves

        other_actor = None
        if other_actor_name and other_actor_name.strip().lower() != 'none':
            other_actor,_ = self.actor_models.resolve_or_create_character(other_actor_name)

        if name and description and termination:
            goal = Goal(name=name, 
                        actors=[self, other_actor] if other_actor else [self],
                        description=description, 
                        preconditions=preconditions,
                        termination=termination.replace('##','').strip(),
                        drives=[drive] if drive else []
            )
            trace.event('goal_created', character=self.name, goal_id=goal.id, goal=goal.name,
                        description=goal.description, termination=goal.termination)
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
        # Split targets on commas / ' and ' only: findList splits on whitespace, so a Look at
        # "tree line" created two NPC characters, "Tree" and "Line".
        target_form = hash_utils.find('target', hash_string) or ''
        target_names = [n.strip() for part in target_form.split(',') for n in part.split(' and ')]
        target_names = [n for n in target_names if n and n.lower() not in name_stop_list]
        is_say = bool(mode) and mode.strip().capitalize() == 'Say'
        targets = []
        for name in target_names:
            if is_say:
                # speech implies a listener: an unknown addressee becomes an NPC
                target, _ = self.actor_models.resolve_or_create_character(name.capitalize())
            else:
                # Look/Move/Do/Think targets are often places or things: resolve existing
                # characters only, never invent one
                target, _ = self.actor_models.resolve_character(name.capitalize())
            if target is not None:
                targets.append(target)
        # (an #actors list was parsed here too, possibly creating NPCs, then overwritten below)

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
                # act prompts ask for no reason; this copied the action text into it
                reason=task.reason if task is not None and getattr(task, 'reason', None) else '',
                duration=duration,
                source=task,
                target=targets
            )
            # survival plays: the offered action this Do uses, by its listed name (drink@Spring3)
            act.affordance = (hash_utils.find('affordance', hash_string) or '').strip() or None
            # does the character leave the scene with this act? its writer says (sim/stage.py presence)
            act.leaves = (hash_utils.find('leaves', hash_string) or '').strip().lower().startswith('y')
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

The original description stays as it is. Respond only with what has changed for your character since it was written: one to three sentences, 50 words at most,
on their present appearance and condition, and any lasting change of mind or commitment. Plain statements of fact about the character; do not restate the original, and write no slogans or morals.
Do not include any reasoning, introductory, explanatory, or discursive text, or any markdown or other formatting. 

"""
        response = default_ask(self, system_prompt=system_prompt, prefix=mission, suffix=suffix, 
                                addl_bindings={"original_description":self.original_character, "current_description":self.character}, 
                                tag='Agh.update', max_tokens=300)
        if response:
            # the author's description is kept whole (rewriting all of it each scene wore away
            # traits such as how the character speaks); only the 'Lately' paragraph changes
            self.character = f'{self.original_character.strip()}\nLately: {response.strip()}'
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
            target, _ = self.actor_models.resolve_character(candidate.strip())   # (context.resolve_reference never existed)
            if target:
                return target.name
        return None
    # World interaction methods

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
        for resource_id in resources:  # what the {{$resources}} prompt block lists (it was never filled)
            self.resource_models.add_resource_model(resource_id)
        # the raw #view dump is the current view (also kept as self.look_percept), not an event
        # worth remembering: storing it filled structured memory with map dumps
        view_percept = self.add_perceptual_input(view_text, mode=SensoryMode.VISUAL, remember=False)
        # only an actual Look act has an intent to judge relevance against; setup/scene-start
        # looks passed act_arg None/'' and spent an LLM call on "X does None and notices ..."
        notices = self.process_consequences(act_arg, reason, view_text) if act_arg else ''
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

    # Utility methods
    def other(self):
        """Get the other actor in the context"""
        for actor in self.context.actors+self.context.npcs:
            if actor != self:
                return actor
        return None

    # Initialization and persistence
    def initialize(self, ui=None):
        """Called from worldsim once everything is set up"""
        self.ui = ui
        if hasattr(self, 'mapAgent'):
            self.look()
        """Initialize agent state"""
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

    def set_drives(self, drive_texts: List[str]) -> None:
        """The character's drives, most important first (sim/cognitive/drives.py)."""
        self.drives = [Drive(text) for text in drive_texts]



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

        description = f'Portrait of {self.name}'  # fallback if anything below fails
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
            # how the character looks, how it feels, where it is (sim/stage.py). It was the first
            # two sentences of the description, then 'Background:' and the opening of the Situation
            # cut off at 96 characters.
            description = stage.portrait(self.context, self, response or '')

        except Exception as e:
            traceback.print_exc()
        return description

    def update_actor_image(self):
        image_path = None
        try:
            description = self.generate_image_description()
            prompt = description
            #print(f' actor image prompt len {len(prompt)}')
            # one seed for a character, so that it keeps a face from one portrait to the next
            trace.event('image', of=self.name, prompt=prompt)
            image_path = llm_api.generate_image(self.llm, prompt, size='384x384', filepath=self.name + '.png',
                                                seed=zlib.crc32(self.name.encode()))
        except Exception as e:
            traceback.print_exc()
        return image_path

    def update_world_image(self):
        raise NotImplementedError("Derived classes must implement update_world_image")
    
    def add_perceptual_input(self, message: str, percept=True, mode=None, remember=True):
        """Add a perceptual input to the agent's perceptual state.
        remember=False keeps it out of structured memory (e.g. raw #view map dumps)."""
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
        self.perceptual_state.add_input(perceptual_input, remember=remember)
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

        weighted_acts = len(task.acts) + 2*len([act for act in task.acts if act.mode in ('Say', Mode.Say)])
        trace.event('task_check', character=self.name, task_id=task.id, task=task.name,
                    satisfied=bool(satisfied), progress=progress, acts=len(task.acts))
        # basic timeout on task; Says weigh 3, so a talk-only task ends after ~2 Says
        if satisfied or weighted_acts > random.randint(MAX_ACTS_PER_TASK + 2, MAX_ACTS_PER_TASK + 4):
            trace.event('task_end', character=self.name, task_id=task.id, task=task.name,
                        outcome='satisfied' if satisfied else 'timeout', acts=len(task.acts), progress=progress,
                        goal_id=task.goal.id if task.goal else None)
            timed_out = not satisfied
            satisfied = True # force completion
            if task == self.focus_task.peek():
                self.focus_task.pop()
                self.focus_action = None
            if self.focus_goal and task in self.focus_goal.task_plan:
                self.focus_goal.task_plan.remove(task)
            if self.focus_goal:
                pass #self.focus_goal.tasks_completed.append(task)
            if timed_out:   # it was reported as satisfied, so the character believed it had succeeded
                self.add_perceptual_input(f"You have set this aside, not done: {task.short_string()}", mode='internal')
            else:
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

        trace.event('goal_check', character=self.name, goal_id=goal.id, goal=goal.name,
                    satisfied=bool(satisfied), progress=progress)
        if satisfied:
            self.goal_history.append(goal)
            if goal.name != 'preconditions':
                # remove preconditions goal if it is satisfied
                # (this loop reused 'goal', so the rest of the method recorded, removed and
                # un-focused whichever goal the loop ended on instead of the satisfied one)
                for g in self.goals.copy():
                    if g.name == 'preconditions' and g.description == goal.preconditions:
                        self.goals.remove(g)
            #self.add_perceptual_input(f"Following goal has been satisfied. This may invalidate parts of the above:\n{goal.short_string()}", mode='internal')
            self.achievments.append(goal.termination)
            if goal in self.goals:
                self.goals.remove(goal)           
            self.update()

            await self.context.update(local_only=True) # remove confusing obsolete data, goal completion is a big event
            drives.goal_satisfied(self, goal)
            if self.focus_goal is goal:
                self.focus_goal_history.append(goal)
                self.focus_goal = None
        return satisfied


    def move(self, where, reason='', alone=False):
        """Walk somewhere: a named place, a person, or a compass direction. Scene partners
        standing with the character come along, unless the act or its reason says it goes alone
        (stage.walk). Returns the words for the audience, or None when `where` names nowhere."""
        where = re.sub(r'^(?:move|go|walk)s?\s+(?:to\s+)?', '', (where or '').strip(), flags=re.I)
        where = re.sub(r'^towards?\s+', '', where).strip()
        if self.mapAgent is None or not where:
            return None
        world = self.context.map
        # a named resource wins ('Walk west toward Spring3, scanning ...'), then a person, then a direction
        for token in [where] + stage.places_named(self.context, where):
            resource = world.get_resource_by_id(token.strip())
            if isinstance(resource, dict):   # it returns the string "path" for the path type
                return stage.walk(self, tuple(resource['location']), resource['name'], alone)
        for other in self.context.actors + self.context.extras + self.context.npcs:
            if other is not self and other.mapAgent is not None and re.search(rf'\b{re.escape(other.name)}\b', where, re.I):
                return stage.walk(self, stage.here(other), other.name, alone=True)
        if map.Direction.from_string(where):
            return ''   # a bare compass direction leads nowhere: a walk goes to a place or a person
        resource, _ = self.resourceRefManager.resolve_reference_with_llm(where)
        if resource:
            return stage.walk(self, tuple(resource['location']), resource['name'], alone)
        return None

    def _first_character(self, target, act):
        """The character an act is aimed at, from acts()'s target argument or the act's own."""
        for source in (target, getattr(act, 'target', None)):
            for candidate in (source if isinstance(source, (list, tuple)) else [source]):
                if isinstance(candidate, Character) and candidate is not self:
                    return candidate
        return None

    async def approach(self, target):
        """Walk to a scene partner who is out of reach before acting on or speaking to them."""
        line = stage.approach(self, target)
        if line:
            self.look()
            self.context.add_event(f'{self.name} {line}')
            self.context.message_queue.put({'name': self.name, 'text': line, 'line_kind': 'action'})
            self.context.transcript.append(f'{self.name} {line}')
            await asyncio.sleep(0)

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
        self.visible_result = ''
        self.attempt_only = False
            
        leaving = bool(getattr(act, 'leaves', False)) and self.context.current_scene is not None
        if act_mode == 'Move':
            line = None
            try:
                # with the others, unless the mover leaves them: then it goes alone, and is gone
                line = self.move(act_arg, reason, alone=leaving)
            except Exception as e:
                logger.error(f'{self.name} move failure: {e}\n{traceback.format_exc()}')
            if leaving and line is not None:
                line = re.sub(r'^walks to ', 'leaves for ', line) if line else 'leaves.'
                stage.leave(self)
            if line == '':
                pass   # already there: nothing to tell
            elif line:
                self.look(act_mode, act_arg, reason)
                self.context.add_event(f'{self.name} {line}')
                self.context.message_queue.put({'name': self.name, 'text': line, 'line_kind': 'action'})
                self.context.transcript.append(f'{self.name} {line}')
                await asyncio.sleep(0)
            else:   # it names nowhere on the map: the narrator decides what the attempt comes to
                act_mode = 'Do'
                act_arg = 'move to ' + act_arg
                act.action = act_arg  # Context.do reads act.action

        # Handle world interaction
        if act_mode == 'Do':
            if not survival.acts_at_a_distance(self, act):
                await self.approach(self._first_character(target, act))
            # Get action consequences from world
            await asyncio.sleep(0)
            consequences, world_updates, character_updates = await self.context.do(self, act)
            if character_updates and len(character_updates) > 0:
                self.add_perceptual_input(f'{character_updates}', mode='internal')  
            notices = self.process_consequences(act_arg, reason, consequences)
            await asyncio.sleep(0)
            # the act text often starts with the character's own name ('Xue pulls a tablet ...')
            deed = re.sub(rf'^\s*{re.escape(self.name)}\s+', '', act_arg)
            if notices and len(notices) > 0:
                self.add_perceptual_input(f'{notices}', mode='internal')
            # the audience sees the deed and what it physically did; what the character makes of
            # it ('notices': how the result bears on its goal) is the character's own
            result = (consequences or '').strip()
            refused = getattr(self, 'last_affordance', None)
            if refused and not refused['applied'] and refused.get('fact') and result.startswith(refused['fact']):
                result = result[len(refused['fact']):].strip()   # the world's reason is for the character, not the audience
            elif refused and refused['applied'] and refused.get('fact') and result.startswith(refused['fact']):
                # the world's own sentence ('Xue has a meal at Cafe.') stood between the deed and
                # a narration that told the same thing; it is shown only when nothing else tells it
                told_by_narration = result[len(refused['fact']):].strip()
                result = told_by_narration or result
            # the world refused it, by its rules or because the deed rested on something that is
            # not there: others saw a try, and the page tells what happened, not what was meant
            rested_on_nothing = getattr(self, 'deed_refused', False)
            self.attempt_only = bool(refused and not refused['applied']) or rested_on_nothing
            self.last_affordance = None
            for note in getattr(self, 'settled_notes', []):   # the world's own wording is for the characters
                result = result.replace(note, '').strip()
            told = stage.stage_line(self.name, '' if rested_on_nothing else act_arg, result)
            if rested_on_nothing:
                told = re.sub(rf'^{re.escape(self.name)}\s+', '', told) or f"tries: {deed.rstrip('. ')}, and it comes to nothing."
            self.context.message_queue.put({'name':self.name, 'text': told, 'line_kind':'action'})
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
            # what others present can see happen (they used to get the action text only)
            # what the world settled and the narration it admitted (sim/record.py); it was an
            # LLM's summary of unchecked narration, which is how a pile of berries became an event
            if leaving and not self.attempt_only:
                stage.leave(self)   # the deed has told it
            visible_result = stage.stage_line(self.name, '', result)
            self.visible_result = visible_result
            self.context.add_event(f"{self.name} {'tries: ' if self.attempt_only else ''}{deed}"
                                   + (' (it does not happen)' if self.attempt_only else '')
                                   + (f' -> {visible_result}' if visible_result else ''))

        elif act_mode == 'Look':
            percept = self.look(act_mode, act_arg=act_arg,reason=reason)
            consequences = await self.context.look(self, act)
            if  consequences and len(consequences) > 0:
                self.add_perceptual_input(f'{consequences}', mode='internal')  
            self.lastActResult = consequences or 'You look, and see nothing new.'
            self.show += act_arg + '.\n ' + consequences
            if consequences.strip():   # a look that finds nothing is not a line: four of them in two acts drained the dread they were meant to build
                self.context.message_queue.put({'name':self.name, 'text': re.sub(rf'^{re.escape(self.name)}\s+', '', stage.stage_line(self.name, '', consequences)), 'line_kind':'action'})
            else:
                trace.event('look_empty', character=self.name)
            await asyncio.sleep(0)

        elif act_mode == 'Think': # Say is handled below
            self.thought = act_arg
            self.show += f" \n...{self.thought}..."
            #self.add_perceptual_input(f"\nYou {act_mode}: {act_arg}", percept=False, mode='internal')
            elevenlabs_params = self.speech_stylizer.voice_params(self)
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
            await self.approach(self._first_character(target, act))
            act_arg = act_arg.strip().strip('"“”').strip()   # the words, not the words in quotes
            # A line is held to what its speaker cannot be mistaken about (its hands, its body,
            # what it is doing, where it is, the hour) before anyone hears it, unless the
            # character means it as a lie (sim/record.py speech). What is heard, remembered and
            # answered is the line as it is finally said.
            meant_as_lie = bool(re.search(r'\(\s*lie\s*\)', act_arg, re.I))
            act_arg = re.sub(r'\s*\(\s*lie\s*\)\s*', ' ', act_arg, flags=re.I).strip()
            handed = None
            if meant_as_lie:
                trace.event('speech_check', character=self.name, outcome='lie', claims=[], before=act_arg)
            else:
                act_arg, handed = record.speech(self.context, self, act_arg, self._first_character(target, act))
            act.action = act_arg
            self.show += f"{act_arg}'"
            elevenlabs_params = self.speech_stylizer.voice_params(target)   # the line is spoken as written
            #print(f"Queueing message for {self.name}: {act_arg}")  # Debug
            self.context.message_queue.put({'name':self.name, 'text':f"'{act_arg}'", 'elevenlabs_params': json.dumps(elevenlabs_params)})
            self.context.transcript.append(f'{self.name}: "{act_arg}"')
            await asyncio.sleep(0)
            done = survival.spoken_deed(self.context, self, act_arg)
            if not done and handed:   # 'Here, take the branch': the words hand it over
                done = survival.resolve_do(self.context, self, f'hands {handed[1].name} the {handed[0]}', targets=[handed[1]])
                if not (getattr(self, 'last_affordance', None) or {}).get('applied'):
                    done = None   # out of reach, or not the speaker's to give: it was only said
                self.last_affordance = None
            if done:
                self.context.add_event(f'{self.name} -> {done}')
                self.add_perceptual_input(done, percept=False, mode='internal')
                self.context.message_queue.put({'name': self.name, 'text': re.sub(rf'^{re.escape(self.name)}\s+', '', stage.stage_line(self.name, '', done).lstrip('. ')), 'line_kind': 'action'})
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
            if leaving:
                # a parting line: the others hear it, the speaker goes, and nobody answers the door
                for other in self.bystanders():
                    other.add_perceptual_input(f"{self.name} says: '{content}', and leaves.", percept=False, mode='auditory')
                self.context.message_queue.put({'name': self.name, 'text': 'leaves.', 'line_kind': 'action'})
                stage.leave(self)
                target = None
            elif target is not None and isinstance(target, Character) and stage.off_stage(target):
                target = None   # said to someone who has gone: nobody hears it
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
                    target.emotionalStance = EmotionalStance.for_character(target)
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

                for other in self.bystanders(target):
                    other.add_perceptual_input(f"You hear {self.name} say to {target.name}: '{content}'", percept=False, mode='auditory')
                # self is speaker, don't get confused about first arg to add_turn
                self.actor_models.get_actor_model(target.name).dialog.add_turn(self, content)
                target.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.add_turn(self, content)
                await target.hear(self, act_arg, source)

         # After action completes, update record with results
        # Notify other actors of action
        if act_mode != 'Say' and act_mode != 'Look' and act_mode != 'Think':  # everyone you do or move or look if they are visible
            seen = f"You see {self.name}: '{act_arg}'"
            if act_mode == 'Do' and getattr(self, 'attempt_only', False):
                # the others were told the deed as done when the world had refused it
                seen = f"You see {self.name} try: '{act_arg}'. It does not happen"
            if act_mode == 'Do' and getattr(self, 'visible_result', '').strip(' .'):
                seen += f". Result: {self.visible_result.strip(' .')}"
            for actor in self.context.actors+self.context.extras:
                if actor != self:
                    # (the target was excluded here and only got an unread sense_input)
                    actor_model = self.actor_models.get_actor_model(actor.name)
                    if actor_model != None and actor_model.visible:
                        percept = actor.add_perceptual_input(seen, percept=False, mode='visual')
                        actor.actor_models.get_actor_model(self.name, create_if_missing=True).infer_goal(percept)
        self.previous_action_mode = act_mode           

    def instantiate_narrative_goal(self, goal_statement, generate_conditions=True):
        """instantiate a goal from a narrative"""
        drives.appraise(self)   # what has reached the character since it last acted moves its drives

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
Drives are the character's basic motivations, most pressing first: each is pressing, present or quiet right now, with what the character has just noticed about it.

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
    preconditions - a statement of conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any
    termination  - a condition (5-6 words) that would mark satisficing achievement of the goal.
    drive - the id (d..) of the one drive listed above that this goal most serves.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Each goal should begin with a #goal tag, and should end with ## on a separate line as shown below:
be careful to insert line breaks only where shown, separating a value from the next tag:

#goal terse (5-8) words) name for this goal
#description concise (8-14) words) further details of this goal
#otherActorName name of the other actor involved in this goal, or None if no other actor is involved
#preconditions (3-4 words) statement of conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any
#termination terse (5-6 words) statement of condition that would mark achievement or partial achievement of this goal
#drive id (d..) of the drive this goal most serves
##


Respond ONLY with the above hash-formatted text.
End response with:
</end>
""")]

        response = self.llm.ask({"name":self.name,
                                 "character": self.get_character_description(),
                                 "goal_statement":goal_statement,
                                 "central_narrative": self.story_note(),
                                 "act_specific_narrative": self.act_direction(),
                                 "scene": self.director_notes(story=False),
                                 "drives": drives.drive_lines(self, ids=True),
                                 "situation": record.seen_by(self) if self.context else "",
                                 "surroundings": self.look_percept,
                                 "character": self.get_character_description(),
                                 "recent_events": self.narrative.get_summary('medium'),
                                 "goal_history":'\n'.join([f'{g.name} - {g.description}\n\t{g.completion_statement}' for g in self.goal_history]) if self.goal_history else 'None to report',
                                 "relationships": self.actor_models.format_relationships(include_transcript=True),
                                 "recent_memories": "\n".join([m.to_string() for m in self.structured_memory.get_recent(16)]),
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
    preconditions - a statement of situational conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any. 
        These should be necessary pre-conditions for the expected initial tasks of this goal.

Be sure to include a goal that responds to the character's most pressing drive in the context of the central narrative and the information above.
Do NOT duplicate or significantly overlap the completed goal or generate a goal for objectives duplicating the completion statement of the completed goal.
Other goals must not duplicate or significantly overlap this goal or the completed goal. 
Nothing in this or other instructions limits your use of deception or surprise.

#Drives, most pressing first
{{$ranked_drives}}
##

                              
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Each goal should begin with a #goal tag, and should end with ## on a separate line as shown below:
be careful to insert line breaks only where shown, separating a value from the next tag:

#goal terse (5-8) words) name for this goal
#description concise (8-14) words) further details of this goal
#otherActorName name of the other actor involved in this goal, or None if no other actor is involved
#preconditions (3-4 words) statement of conditions necessary before attempting this goal (eg, sunrise, must be alone, etc), if any
#termination terse (5-6 words) statement of condition that would mark achievement or partial achievement of this goal
#drive id (d..) of the drive this goal most serves
##


Respond ONLY with the above hash-formatted text.
End response with:
</end>
"""


        goal_memories = []
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
                                   "ranked_drives": drives.drive_lines(self, ids=True),
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
            if new_task and not goal.task_plan:  # was 'task_plan and len(task_plan)==0': never true
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

        goal_memories = []
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
                                "act_specific_narrative": self.act_direction(),
                                "central_narrative": self.story_note(),
                                "my_narrative": self.central_narrative if self.central_narrative else '',
                                "scene": self.director_notes(story=False),
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
                                "situation": record.seen_by(self),
                                "world": record.seen_by(self),
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
            "situation": record.seen_by(self),
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

#The act should be consistent with the character's drives and emotional stance.

{{$director_notes}}

In light of the director's notes, your acts should serve story momentum as well as character psychology.

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
{{$already_said}}

Respond with two alternative acts, including their Mode, action, target, and expected duration. 
The acts should vary in mode and action.

In choosing each act (see format below), you can choose from these Modes:
- Say - speak, to obtain or share information, to align, coordinate with, or manipulate others in service of your goals or drives; to reason jointly, or to establish or maintain a bond. 
    For example, if you want to build a shelter with Mirela, it might be effective to Say: 'Mirela, let's build a shelter.'
- Look - observe your surroundings, gaining information on features, actors, and resources at your current location and for the eight compass
    points North, NorthEast, East, SouthEast, South, SouthWest, West, or NorthWest.
- Move - walk to a named place or person: give its name (a place listed under 'You could do here', such as Spring3, or an actor's name).
    The walk takes time and those with you come along, unless you say you go alone. A bare compass direction is not a Move.
- Do - perform an act (other than move) to achieve physical consequences in the world or demonstrate to others your commitment to a goal or drive (whether real or performative). 
    This is often appropriate when the task involves interacting physically with a resource or actor. See below for actor thoughts on physical actions.
- Think - mental reasoning about the current situation wrt your state and the task.
    Often useful when you need to plan or strategize, or when you need to understand your own motivations and emotions, but beware of overthinking. Thoughts are not revealed to others.

############
#Consider the following thoughts the actor animating this character has had about physical action ('Do' actions). 
#Remember, these are suggestive, neither limiting nor exhaustive, and should be used as a guide, not a constraint:

{{$do_actions}}
############

#What is physically going on right now (where you are, your body, what just happened, what you could do here).
{{$act_note}}
{{$here_and_now}}
##

Review your character and current emotional stance when choosing Mode and action. 
Emotional tone and orientation can (and should) heavily influence the boldness, mode, phrasing and style of expression for an Act.
A Say is one spoken turn as your character would say it aloud: one to three plain sentences, no slogans or speeches.

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
- Do not name what you and the other are to each other or what the moment means ("you're my anchor", "we're bound now"); ask, refuse, offer or tease, and let that show it.
- Words do nothing physical: to eat, drink, take, give or go somewhere, choose a Do or a Move. Never report as done what has not been done.
- If your character is deliberately saying something untrue about itself (a lie or a bluff, meant to deceive), end the line with the mark (lie); it will not be shown. Otherwise say nothing of what you hold, are doing, or your body's state that is not so.
- 'Your body' above is the truth about your thirst, hunger and tiredness: do not complain of a need it says is met, or speak as if one were met that it says is not.
- Speak only of what your character has seen, been told or can reasonably guess; do not invent facts about this place or its dangers.
- Speak as a person of this world: distances in paces or minutes, things by their ordinary names ('the spring', not 'Spring2'). Never 'tiles', 'inventory', or anything about a game or its workings.
{{$tempo}}
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
#duration expected duration of the action in minutes. Use a fraction of task duration according to the expected progress towards completion.{{$affordance_line}}
#leaves yes or no. yes only when with this act your character leaves the others and goes out of the scene: walks out, goes off alone, departs. Going somewhere together with the others, or moving about the place, is no.
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
Establish connection with Tomas given RecentHistory element: "Who is this guy?"
Actors: Mirela

Response:
#mode Say
#action Hi, who are you?
#duration 1 minute
#target Tomas
#actors Mirela
##

----

Task:
Find out where I am given Situation element: "This is very very strange. Where am I?"
Actors: Mirela

Response:
#mode Look
#action look around for landmarks or signs of civilization
#duration 1 minute
#actors Mirela
##

----

Task:
Find food.
Actors: Tomas, Mirela

Response:
#mode Move
#action berries#2
#duration 15 minute
#target None
#actors Tomas, Mirela
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
        self.emotionalStance = EmotionalStance.for_character(self)
        memories = self.perceptual_state.get_information_items(goal.short_string(), threshold=0.1, max_results=3)
        for t in goal.task_plan:
            memories = self.perceptual_state.get_information_items(t.short_string(), threshold=0.1, max_results=3)
            for tm in memories:
                if tm not in goal_memories:
                    goal_memories.append(tm)
        task = self.focus_task.peek()
        if self.context.scene_post_narrative:
            scene_post_narrative = f"\n\nThe scene is heading toward: {self.context.scene_post_narrative}"
        else:
            scene_post_narrative = ''

        response = default_ask(self, prefix=mission, suffix=suffix, 
                               addl_bindings={"goal_string":goal.to_string(), "task":task, "task_string":task.to_fullstring(),
                                              "do_actions": self.do_actions,
                                              "here_and_now": self.physical_context(),
                                              "affordance_line": survival.affordance_prompt_line(self),
                                              "act_note": survival.act_note(self),
                                              "already_said": self.already_said(task),
                                              "tempo": stage.speech_tempo(self),
                                              "goal_memories": "\n".join([m.content for m in goal_memories]),
                                              "scene_post_narrative": '',   # where the scene is heading is in the director's notes
                                              "director_notes": self.director_notes(),
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
    
    def _pending_task_names(self, other):
        """Tasks still ahead for self and other: the rest of the scene plan, else their goals' plans."""
        plan = self.context.scene_integrated_task_plan
        if self.context.current_scene and plan:
            ahead = plan[self.context.scene_integrated_task_plan_index + 1:]
            return '\n'.join(f"{e['actor'].name}: {e['task'].name}" for e in ahead if e['actor'] in (self, other))
        names = []
        for c in (self, other):
            if c.focus_goal:
                names += [f'{c.name}: {t.name}' for t in c.focus_goal.task_plan]
        return '\n'.join(names)

    async def extract_commitments_following_conversation(self, other:Character, transcript:str):
        """One pass over a finished dialog for new commitments by either party, individual or joint.
        Replaces a joint pass plus up to two individual passes (3 LLM calls, joint tasks recorded
        for one party only). Each commitment is rated once (Context.evaluate_commitment_significance)
        and, in a scene, placed for each committing character (Context.place_commitment); outside
        a scene it is added to that character's goal plan. Returns the tasks added."""
        if self.check_if_in_dialog_subtask():
            return []
        prompt=[UserMessage(content="""Your task is to analyze the following transcript of a conversation between {{$name}} and {{$other_name}},
and identify any new commitments to act now made in it, by {{$name}}, by {{$other_name}}, or by both together. Otherwise respond None.
Any commitment reported must be definite, immediate, non-hypothetical, and remain unfulfilled at the end of the transcript.
A commitment must come from a clear, firm statement of intent to act, not from a question, an opinion, a statement of fact, or other conversational noise.
Be careful to distinguish between hypotheticals and actual commitments. Respond only with actual commitments.

<hypothetical commitments example>
Interviewer says: Alex, suppose you're thrown into a high-stakes project using unfamiliar tech,
     and the timeline's slashed in half - how would you keep things from unraveling without your usual backup?
Alex says: If that happened, I'd quickly get up to speed on the essentials and trim the project down.
    I'd be clear about the risks from the start, tackle key features first, and adjust as needed.
</hypothetical commitments example>

Any new commitment should be consistent with the scene being played, as given below.

<central_narrative>
{{$central_narrative}}
</central_narrative>

<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

<scene>
{{$scene}}
</scene>

<transcript>
{{$transcript}}
</transcript>

Tasks already planned (a commitment that repeats one of these is not new):
<pending_tasks>
{{$pending_tasks}}
</pending_tasks>

An action can be physical or verbal. Thought, e.g. 'reflect on my situation', is NOT a commitment to act.
For each commitment, #actors names who committed to act: one name for an individual commitment, both names for a joint one.
Report at most 3 commitments, most concrete and immediate first, in the temporal order implied by the transcript.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Close each commitment with ## on a separate line. Be careful to insert line breaks only where shown, separating a value from the next tag:

#name brief (4 words max) task name
#description terse (6 words max) statement of the action to be taken
#actors names of the committing character(s), comma separated
#reason (4 words max) on why this action is important now
#duration (2 words) expected duration of the action in minutes
#termination (5 words max) easily satisfied termination test to quickly end this task
##

Always refer to characters by name, never by pronoun or label ('he', 'she', 'other', 'Agent').

===Example===

<transcript>
Jean says: Hey Francoise, what needs doin' on the farm today?
Francoise says: Waterin' the south field's a priority, Jean. Crops are gettin' parched.
Jean says: Aye, let's get the waterin' done first. I'll grab the buckets and meet you by the well.
Francoise says: I'll meet you by the well then, Jean.
Jean says: See you by the well, Francoise. I'll bring the ropes too, just in case.
</transcript>

Response:

#name water south field
#description Meet by the well, water field
#actors Jean, Francoise
#reason crops are parched
#duration 30 minutes
#termination south field watered
##
#name bring ropes
#description bring ropes to the well
#actors Jean
#reason in case a handle breaks
#duration 10 minutes
#termination ropes at the well
##

===End Example===

Do NOT include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]
        response = self.llm.ask({"transcript": transcript,
                                 "pending_tasks": self._pending_task_names(other),
                                 "name": self.name,
                                 "other_name": other.name,
                                 "central_narrative": self.story_note(),
                                 "act_specific_narrative": self.act_direction(),
                                 "scene": self.director_notes(story=False)},
                                prompt, tag='extract_commitments_following_conversation', temp=0.1, stops=['</end>'], max_tokens=300)
        added = []
        hashes = hash_utils.findall_forms(response) if response else []
        if not hashes:
            print(f'no new commitments in conversation {self.name} / {other.name}')
            return added
        in_scene = self.__class__.__name__ == 'NarrativeCharacter' and self.context.current_scene
        for task_hash in hashes:
            task = self.validate_and_create_task(task_hash)
            if not task:
                continue
            owners = [c for c in (self, other) if c in (task.actors or [])] or [self]
            significance = None
            if in_scene:
                significance = self.context.evaluate_commitment_significance(owners[0], other if owners[0] is self else self, task)
            owner_tasks = []
            for i, owner in enumerate(owners):
                # a joint commitment becomes a task for each party (it ran for one party only)
                owner_task = task if i == 0 else Task(task.name, task.description, task.reason, task.actors, None,
                                                      termination=task.termination,
                                                      start_time=self.context.simulation_time, duration=task.duration)
                owner_tasks.append((owner.name, owner_task.id))
                if in_scene:
                    if self.context.place_commitment(owner, owner_task, significance):
                        added.append(owner_task)
                elif owner.focus_goal:
                    await owner.generate_task_plan(owner.focus_goal, new_task=owner_task, replan=True)
                    added.append(owner_task)
                else:
                    print(f'{owner.name}: no focus goal, commitment {owner_task.name} dropped')
            trace.event('commitment_extracted', task=task.name, description=task.description,
                        speaker=other.name, listener=self.name, joint=len(owners) > 1,
                        significance=significance, in_scene=bool(in_scene), owner_tasks=owner_tasks)
        return added

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
        transcript = from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog() # this is dialog from the perspective of self.
        if f'{self.name} says: ' not in transcript:
            # whoever is spoken to answers at least once: ending here left the speaker to repeat
            # the same proposal three times to someone who never replied
            return False
        if stage.tempo(self) == 'urgent':
            return True   # one exchange, then back to what is happening
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
        trace.event('dialog_turn', speaker=from_actor.name, listener=self.name, text=str(message)[:1000])
       
        # Special case for Owl-Doc interactions
        if self.name == 'Owl' and from_actor.name == 'Doc':
            text, response_source, _ = self.generate_dialog_turn(from_actor, message, self.focus_task.peek())
            self.show = text
            return
        

        # Remove text between ellipses - thoughts don't count as dialog
        message = re.sub(r'\.\.\..*?\.\.\.', '', message)

        if (self.name == 'Viewer') or self.natural_dialog_end(from_actor): # test if the dialog has reached a natural end, or hearer is viewer.
            # close the dialog
            dialog = self.actor_models.get_actor_model(from_actor.name).dialog.get_current_dialog()
            trace.event('dialog_end', speaker=from_actor.name, listener=self.name,
                        turns=self.actor_models.get_actor_model(from_actor.name).dialog.turn_count)
            self.add_perceptual_input(f'Conversation with {from_actor.name}:\n {dialog}', percept=False, mode='auditory')
            dialog_as_list = dialog.split('\n') 
            self.note_exchange(from_actor, dialog_as_list)
            self.actor_models.get_actor_model(from_actor.name).short_update_relationship(dialog_as_list, use_all_texts=True)
    
            self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
            self.last_task = self.focus_task.peek()
            self._pop_dialog_task(from_actor)
            # it would probably be better to have the other actor deactivate the dialog itself
            other_dialog = from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog()
            from_actor.add_perceptual_input(f'Conversation with {self.name}:\n {other_dialog}', percept=False, mode='auditory')
            from_actor.note_exchange(self, other_dialog.split('\n'))
            from_actor.actor_models.get_actor_model(self.name).short_update_relationship(other_dialog.split('\n'), use_all_texts=True)
            from_actor.actor_models.get_actor_model(self.name).dialog.deactivate_dialog()
            from_actor.last_task = from_actor.focus_task.peek()
            from_actor._pop_dialog_task(self)

            if self.name != 'Viewer' and from_actor.name != 'Viewer' and ablation.enabled('commitments'):
                # once both dialog tasks are off the stacks: one pass for both parties' commitments
                await self.extract_commitments_following_conversation(from_actor, dialog)
                await asyncio.sleep(0)
            return

        cut_in = self.bystander_cuts_in(from_actor, message)
        if cut_in:
            message = f'{message}\n{cut_in}'
        text, response_source, reason = self.generate_dialog_turn(from_actor, message, self.focus_task.peek()) # Generate response using existing prompt-based method
        if not text or not text.strip():
            # no <response> (generate_dialog_turn already deactivated the dialog); an empty Say
            # was performed anyway, shown as '' and heard by from_actor, which kept talking
            return
        action = Act(mode='Say', action=text, actors=[self, from_actor], reason=reason or '', duration=1, source=response_source, target=[from_actor])
        action.leaves = bool(getattr(self, '_reply_leaves', False))
        await self.act_on_action(action, response_source)
        if self.focus_action == action:
            self.focus_action = None
        await asyncio.sleep(0)

    def own_recent_lines(self, other):
        """Guidance built from this character's own last two spoken lines: a habit of speech
        (a metaphor, a pet phrase, one sentence shape) should colour some lines, not every one."""
        model = self.actor_models.get_actor_model(other.name)
        mine = [t.split(' says: ', 1)[1] for t in (model.dialog.transcript if model else [])
                if t.startswith(f'{self.name} says: ')][-2:]
        if not mine:
            return ''
        return ("- Your own last lines were: " + ' / '.join(f"'{m[:140]}'" for m in mine)
                + ". Do not reuse their imagery or their sentence shape. If either used a figure of speech, speak plainly this time.")

    def note_exchange(self, other, dialog_lines):
        """Remember how a finished dialog went: what this character opened with and the last
        thing the other answered, so its next act does not open the same exchange again."""
        mine = [l.split(' says: ', 1)[1] for l in dialog_lines if l.startswith(f'{self.name} says: ')]
        theirs = [l.split(' says: ', 1)[1] for l in dialog_lines if l.startswith(f'{other.name} says: ')]
        if mine and theirs:
            self.recent_exchanges = (self.recent_exchanges + [(other.name, mine[0][:160], theirs[-1][:160])])[-4:]

    def already_said(self, task):
        """For the act prompt: recent exchanges and what was already said in this task. A scene
        used to circle one demand, each act re-opening it as a new dialog with no memory that
        it had just been asked and answered."""
        parts = []
        if self.recent_exchanges:
            parts.append("#Your recent exchanges: what you opened with, and the last answer you got.\n"
                         + '\n'.join(f"- to {other}: '{asked}' -> {other} answered: '{answer}'"
                                     for other, asked, answer in self.recent_exchanges)
                         + "\nDo not open the same exchange again: asked the same way, it gets the same answer. "
                           "Take the answer as given and do something different: act on it, concede, raise the price, or turn to someone else.\n##")
        spoken = [a.action for a in (task.acts or []) if a.mode == 'Say']
        if spoken:
            parts.append("#In this task you have already said:\n" + '\n'.join(f"- '{line[:160]}'" for line in spoken[-2:])
                         + "\nYou have had your say in this task: now choose a Do, Move, Look or Think, "
                           "unless someone has just said something you must answer.\n##")
        return '\n'.join(parts)

    EARSHOT_TILES = 4
    MAX_CUT_INS_PER_SCENE = 2     # per character
    CUT_IN_GAP_LINES = 4          # transcript lines before the same character may cut in again

    def bystanders(self, other=None):
        """Characters in the current scene, other than self and `other`, close enough to hear."""
        scene = self.context.current_scene
        if scene is None or self.mapAgent is None:
            return []
        here = (self.mapAgent.x, self.mapAgent.y)
        return [c for c in self.context.actors + self.context.extras + self.context.npcs
                if c is not self and c is not other and c.name != 'Viewer'
                and getattr(c, 'current_scene', None) is scene and c.mapAgent is not None
                and max(abs(c.mapAgent.x - here[0]), abs(c.mapAgent.y - here[1])) <= self.EARSHOT_TILES]

    def bystander_cuts_in(self, from_actor, message):
        """A dialog is between two characters, but others in the scene hear it. Before self
        answers from_actor, one bystander (the one just named, else the one silent longest) is
        asked whether it cuts in. Its line is shown, joins both parties' record of the dialog,
        and is returned ('Hu cuts in, to Xue: ...') for self's answer to take into account.
        Returns '' when nobody speaks."""
        if self is from_actor or 'Viewer' in (self.name, from_actor.name):
            return ''
        scene, now = self.context.current_scene, len(self.context.transcript)

        def may(c):
            in_scene, count, last = getattr(c, '_cut_ins', (None, 0, 0))
            if in_scene is not scene:
                count, last = 0, -self.CUT_IN_GAP_LINES
            return count < self.MAX_CUT_INS_PER_SCENE and now - last >= self.CUT_IN_GAP_LINES
        candidates = [c for c in from_actor.bystanders(self) if may(c)]
        if not candidates:
            return ''
        speaker = min(candidates, key=lambda c: (c.name.lower() not in message.lower(), getattr(c, '_cut_ins', (None, 0, 0))[2]))
        dialog = from_actor.actor_models.get_actor_model(self.name).dialog
        prompt = [SystemMessage(content="""You are an actor in an improvised play. Your character is present while two others talk, and hears every word.
Most of the time a bystander lets them talk. Cut in only if staying silent now would cost your character something it wants in this scene, or you were just spoken about or lied about."""),
                  UserMessage(content="""<your_character>
{{$character}}
</your_character>

<scene>
{{$scene}}
</scene>

<emotional_state>
{{$emotional_state}}
</emotional_state>

<your_drives>
{{$drives}}
</your_drives>

The exchange between {{$a}} and {{$b}} so far:
<dialog>
{{$dialog}}
</dialog>

{{$a}} has just said to {{$b}}:
<statement>
{{$statement}}
</statement>

Does {{$name}} cut in now, before {{$b}} answers? Respond with these lines only:
#cut_in yes or no
#to {{$a}} or {{$b}} (whom the line is addressed to)
#line the spoken line: one or two plain sentences, 30 words at most, in your character's own way of speaking; speech only
</end>
""")]
        reply = speaker.llm.ask({'character': speaker.get_character_description(), 'scene': speaker.scene_brief(),
                                 'emotional_state': speaker.emotionalStance.to_definition(), 'name': speaker.name,
                                 'drives': drives.drive_lines(speaker),
                                 'a': from_actor.name, 'b': self.name, 'statement': message,
                                 'dialog': dialog.get_transcript(max_turns=8)},
                                prompt, tag='bystander_cut_in', temp=0.7, stops=['</end>'], max_tokens=120) or ''
        line = (hash_utils.find('line', reply) or '').strip().strip('"\'')
        if not (hash_utils.find('cut_in', reply) or '').strip().lower().startswith('y') or not line:
            return ''
        to = from_actor if (hash_utils.find('to', reply) or '').strip().lower().startswith(from_actor.name.lower()) else self
        line = record.speech(self.context, speaker, line, to)[0] or line
        speaker._cut_ins = (scene, (getattr(speaker, '_cut_ins', (None, 0, 0))[1] if getattr(speaker, '_cut_ins', (None,))[0] is scene else 0) + 1, now)
        self.context.message_queue.put({'name': speaker.name, 'text': f"'{line}'",
                                        'elevenlabs_params': json.dumps(speaker.speech_stylizer.voice_params(to))})
        self.context.transcript.append(f'{speaker.name}: "{line}"')
        trace.event('dialog_turn', speaker=speaker.name, listener=to.name, text=line[:1000], cut_in=True)
        dialog.add_turn(speaker, line)
        self.actor_models.get_actor_model(from_actor.name).dialog.add_turn(speaker, line)
        speaker.add_perceptual_input(f"You cut in, saying to {to.name}: '{line}'", percept=False, mode='internal')
        for other in speaker.bystanders():
            if other not in (self, from_actor):
                other.add_perceptual_input(f"You hear {speaker.name} say to {to.name}: '{line}'", percept=False, mode='auditory')
        return f"{speaker.name} cuts in, to {to.name}: '{line}'"

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
        system_prompt = """You are an actor in an improvised play, speaking your character's next line.
You are in the scene, not writing it: say what this person would actually say now, to this listener, to get what they want.
A good line does one thing (asks, refuses, offers, concedes, threatens, changes the subject) and leaves the other actor something to answer.
People in conversation speak plainly and often incompletely; they do not deliver slogans, aphorisms or speeches.
""" + ("Bodily needs, the place the characters are in, and what has just happened are real; let them shape what is said."
       if getattr(self, 'body', None) is not None else
       "The place the characters are in and what has just happened are real; let them shape what is said.") + """
Keep your character's own way of speaking (see 'How you speak') from line to line.
        """
        prompt_string = """Be faithful to your character as presented in the following.
Disagreement is not only allowed but expected when confidence is low or relationship is in question, fear is high, or the character perceives conflicting goals or a threat.

##The scene you are playing:
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

        prompt_string += """What is physically going on right now (where you are, your body, what just happened, what you could do here):
<here_and_now>
{{$here_and_now}}
</here_and_now>

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
<leaves>yes or no: yes only when with these words your character walks out on the others and leaves the scene</leaves>

{{$duplicative_insert}}

Guidance: 
- Use the appropriate pronoun (he, she, him, her) according to declared gender of each character. {{$target}} gender is {{$target_gender}}.
    Actor genders:
    {{$actor_genders}}
- Speech only: do not narrate actions, gestures or body language in the response; physical actions are separate Do acts.
- Respond in a way that advances the dialog. E.g., express an opinion or propose a next step. Don't hesitate to disagree with the speaker if consistent with the character's personality and goals.
- Do not respond to a question with a question.
- If the intent is to agree, state agreement without repeating the statement.
- Above all, speak in your own voice. Do not echo the speech style of the statement.
- Do not open by repeating the speaker's word or phrase back as a question ("A trap?", "Numbers?"). Answer it.
- One turn: one to three sentences, 40 words at most. Make one point; leave the rest for later turns.
- No slogans, paired contrasts ("X is ephemeral. Y is law.") or flourishes; say it the way the character would across a table.
- Let your emotional state colour the line without announcing it.
- Do not name what the two of you are to each other or what the moment means ("you're my anchor", "we're bound now", "trust is survival"): ask, refuse, offer, tease or change the subject, and let that show it.
- Words do nothing physical. Do not say you are eating, drinking, taking or handing over something unless you are doing it this moment and it is within reach; never report as done what has not been done.
- If your character is deliberately saying something untrue about itself (a lie or a bluff, meant to deceive), end the line with the mark (lie); it will not be shown. Otherwise say nothing of what you hold, are doing, or your body's state that is not so.
- 'Your body' above is the truth about your thirst, hunger and tiredness: do not complain of a need it says is met, or speak as if one were met that it says is not.
- Speak only of what your character has seen, been told or can reasonably guess, and say a guess as a guess. Do not invent facts about this place or its dangers.
- Speak as a person of this world: distances in paces or minutes, things by their ordinary names ('the spring', not 'Spring2'). Never 'tiles', 'inventory', or anything about a game or its workings.
- Avoid phrases and images already used in the dialog to this point.
{{$own_recent}}
{{$tempo}}
 
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

        actor_genders = ''
        for actor in self.context.actors+self.context.extras:
            actor_genders += f'\t{actor.name}: {actor.gender}\n'

        answer_xml = self.llm.ask({
            'character': self.get_character_description(),
            'emotionalState': emotionalState.to_definition(),
            'statement': f'{from_actor.name} says {message}' if self is not from_actor else message,
            "situation": record.seen_by(self),
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
            'own_recent': self.own_recent_lines(from_actor),
            "tempo": stage.speech_tempo(self),
            # own view of the scene only: the full scene JSON revealed the other character's goal
            "scene": self.scene_brief(),
            "here_and_now": self.physical_context(),
            }, prompt, tag='generate_dialog_turn', temp=0.8, stops=['</end>'], max_tokens=300)
        response = xml.find('<response>', answer_xml)
        # The line ends where anything that is not the line begins. (One model closed its
        # response with the reason's tag, and 'Polite deferral, no commitment yet' was spoken
        # aloud three times in a scene.)
        if response:
            response = re.split(r'</?\s*(?:reason|response|leaves)\s*>', response, maxsplit=1, flags=re.I)[0].strip()
        self._reply_leaves = (xml.find('<leaves>', answer_xml) or '').strip().lower().startswith('y')
        if response is None:
            print(f'No response to hear')
            self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
            return '', source, ''
 
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
                return response, response_source, reason

        # (response, source task, the LLM's <reason> -- parsed but dropped before)
        return response, response_source, reason
    
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
                    admissible = (xml.find('admissible', response) or '').strip()
                    impossible = (xml.find('impossible', response) or '').strip()
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
                                        termination=goal.preconditions, drives=goal.drives)
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
                        'signal_cluster': '',  # (signal clusters removed; this raised when it was None)
                        'emotional_stance': {
                            'arousal': str(self.emotionalStance.arousal.value),
                            'tone': str(self.emotionalStance.tone.value),
                            'orientation': str(self.emotionalStance.orientation.value)
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
                                    self.focus_goal = Goal(name=custom_data['name'], description=custom_data['description'], 
                                                           actors=actors, termination=custom_data['termination'], 
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
                            'signal_cluster': '',
                            'emotional_stance': {
                            'arousal': str(self.emotionalStance.arousal.value),
                            'tone': str(self.emotionalStance.tone.value),
                            'orientation': str(self.emotionalStance.orientation.value)
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
            if not action.source or not action.source.goal:
                print(f'{self.name} request_action_choice: act {action.mode} {action.action} has no source or goal')
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


    def scene_brief(self):
        """The scene in progress, for a prompt that performs it: the director's notes to this actor."""
        scene = self.context.current_scene
        if not scene:
            return ''
        return self.director_notes()   # the scene is the director's to describe, and is given as notes

    # ------------------------------------------------------------------ the director's notes
    # What the director tells the actor playing this character: the story, what this act is
    # about and the actor's part in it, this scene's set-up and where it should arrive, the
    # character's objective, and what the character does not know. They are outside the play.
    # They are set for a scene (Context.run_scene fills self.direction) and are never stored as
    # the character's memory or in anything made from memory: the plan used to be given to the
    # character itself, as 'internal:' memories and in its situation, so it knew the scene
    # before it happened.
    NOTES_HEAD = ("######## DIRECTOR'S NOTES TO YOU, THE ACTOR PLAYING {name} ########\n"
                  "These notes are outside the play. {name} does not know them. Use them to shape how you play the scene: "
                  "what to want, press for, resist and arrive at. Never have {name} say or act on them as something {name} knows. "
                  "What {name} has not seen or been told, {name} finds out in the scene, or not at all: being surprised, "
                  "mistaken or in the dark is part of the performance.\n")
    NOTES_FOOT = "\n######## END OF DIRECTOR'S NOTES ########"

    def story_note(self):
        """The story the company is telling (the play's central question), for the actor."""
        story = getattr(self.context, 'central_narrative', None) if self.context else None
        return str(story).strip() if story else ''

    def act_direction(self):
        """What this act is about and this actor's part in it. Not the other characters' parts,
        and not the scenes still to come: an actor told the act's later beats played them at once."""
        frame = (getattr(self.context, 'act_frame', None) or {}) if self.context else {}
        role = next((r for n, r in (frame.get('roles') or {}).items() if n.lower() == self.name.lower()), '')
        return '\n'.join(filter(None, [f"What this act is about: {frame['focus']}" if frame.get('focus') else '',
                                       f"What rides on it: {frame['stakes']}" if frame.get('stakes') else '',
                                       f"Your part in it: {role}" if role else '']))

    def does_not_know(self):
        """What this character has still to find out: the director's list for the scene, and the
        names of people it met as strangers and has not heard named (Context.strangers)."""
        d = getattr(self, 'direction', None) or {}
        names = sorted(getattr(self, 'unknown_names', None) or ())
        named = [f"the name of {n}. What you are shown calls this person '{n}' so that you can aim an act; "
                 f"{self.name} has not heard that name and cannot use it" for n in names]
        return named + list(d.get('unknown') or [])

    def director_notes(self, story=True):
        """The notes as a block for a prompt, with the standing instruction on how to use them.
        story=False for a template that gives the story and the act in slots of its own."""
        d = getattr(self, 'direction', None) or {}
        parts = []
        if story:
            parts += [f"The story the company is telling:\n{self.story_note()}" if self.story_note() else '', self.act_direction()]
        if d.get('setup') or d.get('title'):
            parts.append(f"This scene, '{d.get('title', '')}', at {d.get('place', '')}: {d.get('setup', '')}".rstrip(': '))
        if d.get('ending'):
            parts.append(f"It should arrive at: {d['ending']}")
        if d.get('goal'):
            parts.append(f"Your objective in it: {d['goal']}")
        if d.get('decision'):
            choice, reason = d['decision']
            parts.append(f"In the course of the scene {self.name} comes to choose: {choice}" + (f" ({reason})" if reason else '')
                         + f". That is not yet chosen when the scene opens; {self.name} gets there.")
        missing = self.does_not_know()
        if missing:
            parts.append(f"What {self.name} does not know (it is found out in the scene, or not at all):\n" + '\n'.join(f'- {u}' for u in missing))
        body = '\n'.join(p for p in parts if p) or 'None for this moment: play from the character.'
        return self.NOTES_HEAD.format(name=self.name) + body + self.NOTES_FOOT

    def physical_context(self):
        """Where the character is, its body, recent physical events, and what it could do here."""
        parts = [f'Surroundings: {self.look_percept or "unknown"}']
        others = stage.whereabouts(self)
        if others:
            parts.append(f'Where the others are: {others}. To act on or speak to someone out of reach you first walk to them, which takes time.')
        business = stage.pending_business(self)
        if business:
            parts.append(f"Still to be done by you in this scene (a Do with that #affordance; saying it is not doing it): {', '.join(business)}")
        body = getattr(self, 'body', None)
        if body is not None:
            parts.append(f'Your body: {body.summary()}')
        parts.append(f'It is now {stage.clock(self.context)}.')
        parts.append('Recent events:\n' + self.context.recent_events_text(6))
        actions = survival.available_actions(self)
        if actions:
            parts.append('You could do here: ' + actions)
        return '\n'.join(parts)

    def _pop_dialog_task(self, other):
        """Pop the 'dialog with other' task if it is on top (the pop was unconditional and could
        remove the scene task when the dialog task was already gone)."""
        top = self.focus_task.peek()
        if top is not None and top.name.startswith('dialog with'):
            self.focus_task.pop()

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
                self.context.message_queue.put({'name':self.name, 'text':f'world_update', 'data':await self.context.world_image_json()})
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

        survival.tick(self)   # needs rise with elapsed simulation time (survival scenarios only)
        print(f'\n{self.name} decides, focus task {self.focus_task.peek().name}')
        logger.info(f'{self.name} step_task: focus task {self.focus_task.peek().name}')
        task: Task = self.focus_task.peek()
        # iterate over task until it is no longer the focus task. 
        # This is to allow for multiple acts on the same task, clear_task_if_satisfied will stop the loop with poisson timeout or completion
        act_count=0
        subtask_count = 0
        while act_count < MAX_ACTS_PER_TASK and self.focus_task.peek() and not stage.off_stage(self):
            drives.appraise(self)   # what the character has just taken in moves its drives before it acts
            action_alternatives: List[Act] = self.generate_acts(task)
            if not action_alternatives:
                # generate_acts returns [] when the LLM call fails; pick_weighted([]) raised
                logger.warning(f'{self.name} step_task: no actions generated for {task.name}')
                break
            # one opening Say and one Look to a task, and no Look straight after a Look: take the
            # alternative that is not more of the same (Samantha scanned east for tusks five times)
            spent = {m for m in ('Say', 'Look') if any(a.mode == m for a in (task.acts or []))}
            if getattr(self, 'previous_action_mode', None) == 'Look':
                spent.add('Look')
            action_alternatives = [a for a in action_alternatives if a.mode not in spent] or action_alternatives
            # a deed is looked at before it is done: one that rests on a thing the actor does not
            # have is passed over for another it offered, or written again (sim/record.py deeds)
            action_alternatives = record.sound_deeds(self.context, self, action_alternatives)
            await self.request_action_choice(action_alternatives) # sets self.focus_action
            record.mend_deed(self.context, self, self.focus_action)
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
                if not action_alternatives:
                    logger.warning(f'{self.name} step_task: no actions generated for subtask of {task.name}')
                    break
                action_alternatives = record.sound_deeds(self.context, self, action_alternatives)
                await self.request_action_choice(action_alternatives) # sets self.focus_action
                record.mend_deed(self.context, self, self.focus_action)
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
                break
            # (an unconditional 'return True' here ended every task after its first act)

        if self.focus_task.peek() is task:
            return False
            #self.focus_task.pop() # didn't get done by turn limit pretend task is done            return True # probably need to replan around step failure, but we're not going to do that here
        else: # clear assigned task and all intensions created by it, out of turns for this task
            while task in self.focus_task.stack:
                self.focus_task.pop()
        return True # probably need to replan around step failure, maybe we pushed an intension and exhausted our turns, who knows?


    async def act_on_action(self, action: Act, task: Task):
        if task and task.goal:
            drives.goal_attempted(task.goal)
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
        trace.event('act', character=self.name, mode=act_mode, action=act_arg, reason=action.reason,
                    targets=[t.name for t in (action.target or []) if hasattr(t, 'name')],
                    task_id=task.id if task else None, task=task.name if task else None,
                    goal_id=task.goal.id if task and task.goal else None,
                    affordance=getattr(action, 'affordance', None))
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
        began = self.context.simulation_time
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
        # the act takes its own duration unless the world already took longer (a walk, a meal,
        # the replies it drew): the two were added together, so a ten-minute meal took twenty
        self.context.simulation_time = max(self.context.simulation_time, began + act_duration)

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

    async def to_json_with_image(self):
        """to_json plus a fresh portrait. The description (an LLM call, reading character state)
        runs here like any other call; only the render goes to a thread."""
        data = self.to_json()
        try:
            description = self.generate_image_description()
            if description:
                image_path = await asyncio.to_thread(generate_image, self.llm, description, filepath=self.name+'.png')
                with open(image_path, 'rb') as f:
                    data['image'] = base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.error(f'{self.name} portrait failed: {e}')
        return data

    def to_json(self):
        """Return JSON-serializable representation (portrait: to_json_with_image)"""
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

    def _explorer_debug(self):
        """Explore panel 'Debug' sections: what the character is doing and why, down to the
        LLM calls it made. Each section is guarded so one bad field cannot blank the panel."""
        def section(fn):
            try:
                return fn()
            except Exception as e:
                return {'error': f'{type(e).__name__}: {e}'}

        def focus():
            goal, task, act = self.focus_goal, self.focus_task.peek(), self.focus_action
            return {
                'goal': {'id': goal.id, 'name': goal.name, 'description': goal.description,
                         'termination': goal.termination, 'progress': goal.progress} if goal else None,
                'task': {'id': task.id, 'name': task.name, 'description': task.description,
                         'termination': task.termination, 'progress': task.progress,
                         'acts_done': len(task.acts)} if task else None,
                'task_stack': [t.name for t in (self.focus_task.stack or [])],
                'action': act.to_string() if act else None,
                'goal_history': [g.name for g in (self.goal_history or [])][-8:],
            }

        def location():
            agent = getattr(self, 'mapAgent', None)
            return {'x': agent.x, 'y': agent.y} if agent else None

        def memory_stats():
            mem = self.structured_memory
            last = getattr(self.memory_consolidator, 'last_semantic_clustering', None)
            return {
                'concrete': len(mem.concrete_memories),
                'abstract': len(mem.abstract_memories),
                'recent_abstractions': [
                    {'summary': a.summary[:200], 'instances': len(a.instances),
                     'type': getattr(a, 'abstraction_type', 'singleton'), 'drive': str(a.drive) if a.drive else ''}
                    for a in mem.get_recent_abstractions(6)],
                'information_items': len(self.perceptual_state.information_items),
                'last_semantic_clustering': last.isoformat() if last else None,
            }

        def narrative():
            return {
                'central_narrative': getattr(self, 'central_narrative', None),
                'act': self.get_current_act_info(),
                'scene': self.get_current_scene_info(),
                'act_index': getattr(self, 'current_act_index', None),
                'scene_index': getattr(self, 'current_scene_index', None),
            }

        def llm_calls():
            calls = list(getattr(self.llm, 'recent_calls', []))
            return {'recent': calls[::-1], 'failed': sum(1 for c in calls if not c['ok'])}

        def social():
            return {'visible': [m.name for m in self.actor_models.known_actors.values() if m.visible]}

        return {
            'focus': section(focus),
            'narrative': section(narrative),
            'autonomy': section(lambda: dict(vars(self.autonomy))),
            'location': section(location),
            'social': section(social),
            'memory_stats': section(memory_stats),
            'llm_calls': section(llm_calls),
            'look_percept': getattr(self, 'look_percept', ''),
            'body': self.body.summary() if getattr(self, 'body', None) is not None else None,
            'available_actions': survival.available_actions(self),
            'recent_events': list(self.context.recent_events)[-8:] if self.context else [],
        }

    def get_explorer_state(self):
        """Return detailed state for explorer UI"""
        focus_task = self.focus_task.peek()
        # act_on_action records self.last_act (action_history was never appended to)
        last_act = getattr(self, 'last_act', None)

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
            'drives': [{'text': drive.text or '', 'activation': drive.activation} for drive in drives.ranked(self)],
            # raw percept log, newest first (this was mislabelled 'emotional_state')
            'percepts': [
                {
                    'text': percept.content or '',
                    'mode': percept.mode.name if percept.mode else 'unknown',
                    'time': percept.timestamp.isoformat() if percept.timestamp else self.context.simulation_time.isoformat()
                }
                for percept in (self.perceptual_state.get_current_percepts(chronological=True) or [])[:20]
            ],
            'emotional_stance': self.emotionalStance.to_string() if getattr(self, 'emotionalStance', None) else '',
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
                        'id': goal.id,
                        'focused': goal is self.focus_goal,
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
                        'needs': p.needs or '',
                        'termination': p.termination,
                        'progress': p.progress,
                        'acts_done': len(p.acts),
                    } for p in (self.focus_task.stack or [])
                ],
                'intensions': [intension.name for intension in self.intensions]  # List[str]
            },
            # what the character has lately noticed about each drive (drives.appraise)
            'signals': drives.for_ui(self),
            'debug': self._explorer_debug(),
        }

