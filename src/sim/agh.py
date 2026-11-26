from __future__ import annotations
import base64
import os, json, math, time, requests, sys


from sim.SpeechStylizer import SpeechStylizer
from sim.trace import trace
import zlib
import sim.survival as survival
import sim.stage as stage
import sim.world as world
import sim.ablation as ablation
import sim.record as record
import sim.ruling as ruling
import sim.resolver as resolver
import sim.keeper as keeper
import sim.scene as scene_owner

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
from utils import llm_api, json_reply
from utils.llm_api import generate_image
from utils.Messages import UserMessage, SystemMessage
import utils.xml_utils as xml
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
        ], tag='reference_dscp', max_tokens=90, stops=["</end>"])
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
            self.mapAgent = None  # where it stands, once it is in a world (sim/world.py)
        self.perceptual_state = PerceptualState(self)
        self.last_sense_time = datetime.now()
        self.act = None
        self.previous_action_name = None
        self.look_percept = ''
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
        self.actions:List[Act] = []
        self.achievments:List[str] = [] # a list of short strings summarizing the character's achievments
        self.autonomy:Autonomy = Autonomy()
        self.step = 2 # start at step 2 to skip first update
        self.update_interval = 4
        self.emotionalStance:EmotionalStance = EmotionalStance()
        self.decisions:List[str] = [] # a list of decisions made by the character
        self.avoidance_cnt = 0 # a counter of how many times the character has avoided a decision
        #self.speech_stylizer = SpeechStylizer(self)


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

        # 2. Call LLM and parse response. Every prompt of the play takes each character's he or she
        # from here (record.cast, the actor genders lines), so the answer must name one: Lanya's came
        # back empty and every prompt of the bellona run read 'Lanya: ' or left her out of the cast.
        llm_response = None
        for _ in range(3):
            llm_response = self.llm.ask({'name': self.name, 'description': self.character}, prompt, tag='voice_traits', stops=['</end>'], max_tokens=100)
            if hash_utils.find('gender', llm_response or '').strip():
                break
        if llm_response is None:
            return None
        self.gender = hash_utils.find('gender', llm_response).strip().lower()
        if not self.gender:
            print(f'Warning: no gender for {self.name} after three asks; prompts will read it as not stated')
        self.age = hash_utils.find('age', llm_response)
        self.accent = hash_utils.find('accent', llm_response)
        self.personality = hash_utils.findList('personality', llm_response)


    def get_character_description(self):
        description = self.character
        stylizer = getattr(self, 'speech_stylizer', None)
        if stylizer is not None and stylizer.voice_line():
            description += f'\nHow you speak: {stylizer.voice_line()}.'
        if getattr(self, 'own_past', ''):   # settled by the director at the start (Context.frame_episode)
            description += f'\nFrom your own past, known here to you alone unless you tell it: {self.own_past}'
        if getattr(self, 'carried', None):   # what this person remembers of the parts already played (sim/thread.py)
            from sim import thread
            description += '\n' + thread.card(self.carried)
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

    def validate_and_create_task(self, task_hash, goal=None, include_self=True):
        """Validate a task hash and create a task object
        
        Args:
            task_hash: Hash-formatted task definition
            goal: Goal this task is for
            include_self: make this character one of the task's actors. False for a commitment
                read from a dialog, whose #actors say who committed: with it, every promise the
                other made became a joint task owned first by the listener ('Kidd watches the
                door all night' was Lanya's task; 27 of 42 commitments in a lost run were joint).
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
        for character in characters:
            if character and not hasattr(character, 'mapAgent'):
                print(f"Warning: {character} is not a character")
        actors = [actor for actor in characters if actor is not None]
        if (include_self or not actors) and not self in actors:
            actors =  [self] + actors

        if name and description and reason and actors:
            task = Task(name, description=description, 
                        reason=reason, 
                        termination = termination.replace('##','').strip(), 
                        goal = goal, 
                        actors = actors, 
                        start_time = start_time, 
                        duration = duration)
            # a task that takes its character out of the scene is played after the others
            # (Context.integrate_task_plans) and is the only kind an exit may come from (exit_allowed)
            task.leaves = (hash_utils.find('leaves', task_hash) or '').strip().lower().startswith('y')
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
        if is_say and (self in targets or not targets):
            # A line is said to someone else. Joe's Say named Joe as its hearer: the engine opened
            # a 'dialog with Joe' and he answered himself for ten lines while Samantha stood by.
            # With one other person in earshot the line is theirs; otherwise it has no hearer named.
            targets = [t for t in targets if t is not self]
            near = self.hearers() if not targets and self.context is not None else []
            if len(near) == 1:
                targets = near
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
            act.leaves = self.leaving_party(hash_utils.find('leaves', hash_string))
            act.leaves_to = self._leave_to
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
                relationsOnly=False,
                force=now   # a scene starts on an account that is up to date, however little time has passed
            )
       # Reset new memory counter
        self.new_memory_cnt = 0
        self.step += 1 # skip next update
        if self.step < 5 and not now:
            return
        self.step = 0
        system_prompt = """You are an actor with deep insight into human psychology and the character you are playing, {{$name}}, in an ongoing improvisational play.
Your task is to say what has changed for your character since its description was written: the 'Lately' line that follows the description.
A primary task is to differentiate between temporary traits and decisions, and your character's fundamental personality and commitments.
Some traits and decisions may be temporary, and should be allowed to fade away, but your character's fundamental personality and commitments should be preserved. 
Carry forward your character's fundamental personality and commitments. Otherwise, allow this description to evolve slowly in keeping with your character's development. 
Preserve your character's fundamental decision-making capability - acknowledge doubt, but don't allow it to become dominant.
This description will inform the next scene and remainder of the play.
"""
        # (The two descriptions were given twice, before the frame and after it, and the answer
        # was asked for 'present appearance and condition': it came back as a copy of the body
        # and of what was carried as they stood when the scene began, and was then read all
        # scene beside the body as it really was. The reply also took up the label it saw.)
        suffix = """

#Original description
{{$original_description}}
##

#What you last wrote had changed since
{{$lately}}
##

The original description stays as it is. Respond only with what has changed for your character since it was written: one to three sentences, 50 words at most,
on what they now believe, want or have committed themselves to, and how they stand with the others. Write it as the description is written, to the character: 'You ...'. Plain statements of fact; do not restate the original, and write no slogans or morals.
If nothing has changed since what you last wrote, respond with the single word: unchanged
Leave out hunger, thirst, tiredness, injuries and what is carried: the world keeps those, and they are shown wherever they matter.
Do not include any reasoning, introductory, explanatory, or discursive text, any label or heading, or any markdown or other formatting.

"""
        lately = self.character.partition('\n' + self.LATELY)[2].strip()
        response = default_ask(self, system_prompt=system_prompt, prefix='', suffix=suffix,
                                addl_bindings={"original_description":self.original_character,
                                               "lately":lately or 'nothing yet'},
                                tag='Agh.update', max_tokens=300)
        if response:
            # the author's description is kept whole (rewriting all of it each scene wore away
            # traits such as how the character speaks); only the 'Lately' paragraph changes.
            # The label is the code's: one the reply brought with it is dropped.
            changed = response.strip()
            while changed.lower().startswith(self.LATELY.lower()):
                changed = changed[len(self.LATELY):].strip()
            if re.fullmatch(r'\W*unchanged\W*', changed, re.I):
                return response   # what it last wrote still holds (with no such answer, it made up a decision)
            self.character = f'{self.original_character.strip()}\n{self.LATELY} {changed}'
        return response

    LATELY = 'Lately:'








    # World interaction methods

    CONSEQUENCES = """{{$name}} did this: {{$deed}}
What {{$name}} was after: {{$reason}}

What came of it:
{{$consequences}}

What, of what came of it, bears on what {{$name}} was after? Say it in one plain sentence about {{$name}} and the result, from what is written above only: no one's reaction, thought or feeling that it does not state, and nothing found or done that it does not state.
If none of it bears on what {{$name}} was after, respond with the single word: none
End your response with:
</end>
"""

    def process_consequences(self, act_arg, reason, consequences):
        """What of a deed's result bears on the character's aim: a note for its own memory, or ''.
        (Asked with nothing to read, one call in five, it made one up: Samantha's glance 'may
        prompt her to keep explaining'. It read 'Lanya does Lanya slips...', and turned the knife
        Lanya found into a can of food.)"""
        if not (consequences or '').strip():
            return ''
        deed = stage.unnamed(self.name, act_arg)
        response = self.llm.ask({"name": self.name, "deed": deed, "reason": reason or 'not said', "consequences": consequences},
                                [UserMessage(content=self.CONSEQUENCES)], tag='do_consequences', temp=0.2, stops=['</end>'], max_tokens=100)
        return '' if re.fullmatch(r'\W*none\W*', (response or '').strip(), re.I) else (response or '')

    def look(self, act_mode=None, act_arg=None, reason=None):
        """Get visual information about surroundings"""
        if not world.on(self):
            return ''
        view_text, resources, characters, paths, percept_summary = world.surroundings(self)
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
        # those near enough to be in the view as written (hash_direction_info's distance), not
        # everyone in line of sight on the map: two people under an apple tree 'saw' the owners
        # of a cave and a hut half an hour's and an hour's walk off, and remembered it
        for character_name in characters:
            character, character_name = self.actor_models.resolve_character(character_name)
            if character and character not in visible_actors:
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

    # Initialization and persistence
    def initialize(self, ui=None):
        """Called from worldsim once everything is set up"""
        self.ui = ui
        if world.on(self):
            self.look()
        """Initialize agent state"""
        # Initialize narrative with drives
        self.narrative.active_drives = [d.text for d in self.drives]


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
            recent_memories = self.lived(8)
            #print("Char generate image description", end=' ')
            response = self.llm.ask({ "description": state, "recent_memories": recent_memories}, prompt, tag='image_description', temp=0.2, stops=['</end>'], max_tokens=24)
            # how the character looks, how it feels, where it is (sim/stage.py). It was the first
            # two sentences of the description, then 'Background:' and the opening of the Situation
            # cut off at 96 characters.
            self._portrait_facts = stage.portrait_facts(self.context, self, response or '')
            description = (stage.portrait_prompt(self._portrait_facts) if stage.IMAGE_PROMPT == 'new'
                           else stage.portrait(self.context, self, response or ''))

        except Exception as e:
            traceback.print_exc()
        return description

    async def render_portrait(self):
        """A fresh portrait: the prompt (an LLM call for the mood) runs on the event loop, the
        render in a thread. One seed for a character, so that it keeps a face from one portrait to
        the next. Returns the image path, or None. (Two paths drew portraits, neither with the
        seed, and neither traced; the seeded one was never called.)"""
        stage.ensure_looks(self)
        prompt = self.generate_image_description()
        if not prompt:
            return None
        trace.event('image', of=self.name, prompt=prompt, facts=getattr(self, '_portrait_facts', None), builder=stage.IMAGE_PROMPT)
        path = await asyncio.to_thread(generate_image, self.llm, prompt, filepath=self.name + '.png',
                                       seed=zlib.crc32(self.name.encode()))
        trace.keep(path, self.name)
        return path

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

    def lived(self, n=12):
        """What the character has lately seen and heard (the play's record, Context.witnessed) and
        its own memories (thoughts, notices, what it was told), in time order, the latest n, as text."""
        rows = [(m.timestamp, m.to_string()) for m in self.structured_memory.get_recent(n)]
        context = getattr(self, 'context', None)
        if context is not None and hasattr(context, 'record_entries'):
            rows += [(e['time'], context.entry_line(e)) for e in context.record_entries if self.name in e['witnesses']][-n:]
        seen, out = set(), []
        for _, text in sorted(rows, key=lambda r: r[0]):   # once each
            key = ' '.join(text.split()).lower()
            if key not in seen:
                seen.add(key); out.append(text)
        return '\n'.join(out[-n:])

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
        if task is None:
            return True
        if not termination_check:
            # nothing says when it is done: it is done when it has been acted on. (It was
            # reported satisfied and left on the stack, to be played for all its acts.)
            satisfied, progress = bool(task.acts), 100
        else:
            # in a scene with an owner the owner says whether the beat is done (sim/scene.py):
            # one reading of the page for the beat, the talk and the scene, not one each
            owner = scene_owner.owner_of(self.context)
            verdict = owner.check(task) if owner is not None and task is owner.current_task() else None   # not a task pushed on top of the beat
            if verdict is not None:
                satisfied = verdict['beat'] == 'done' or owner.over
                progress = 100 if satisfied else 0
            else:
                satisfied, progress = await self.test_termination(task, termination_check, consequences, world_updates, type='task')

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
            task.outcome, self.last_task = 'timeout' if timed_out else 'satisfied', task
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
            if goal.name != 'preconditions':
                # remove preconditions goal if it is satisfied
                # (this loop reused 'goal', so the rest of the method recorded, removed and
                # un-focused whichever goal the loop ended on instead of the satisfied one)
                for g in self.goals.copy():
                    if g.name == 'preconditions' and g.description == goal.preconditions:
                        self.goals.remove(g)
            if goal in self.goals:
                self.goals.remove(goal)
            self.update()

            await self.context.update(local_only=True) # remove confusing obsolete data, goal completion is a big event
            self._goal_met(goal)
            if self.focus_goal is goal:
                self.focus_goal_history.append(goal)
                self.focus_goal = None
        return satisfied

    def _goal_met(self, goal: Goal):
        """A goal has been got: it is remembered as achieved, and the drives it served ease."""
        self.goal_history.append(goal)
        if goal.termination:
            self.achievments.append(goal.termination)
        drives.goal_satisfied(self, goal)

    async def close_scene_goal(self):
        """The scene is over: was what this character was after in it got? Judged from the
        scene's record (test_termination). Got, it is an achievement and eases the drives it
        served; not got, the character remembers that. Either way the goal is put away.

        (In a planned play nothing judged a scene goal: clear_goal_if_satisfied is reached only
        from the unplanned turn. 'Recently achieved goals' read 'None to report' in every
        prompt of a play, no drive was ever eased by a want being met, and the goal stood until
        the next scene wrote over it. The character's account is not rewritten here as it is in
        clear_goal_if_satisfied: the next scene's start does that.)"""
        goal = self.focus_goal
        if goal is None:
            return None
        met, progress = await self.test_termination(goal, goal.termination or goal.description, '', '', type='goal')
        trace.event('goal_check', character=self.name, goal_id=goal.id, goal=goal.name,
                    satisfied=bool(met), progress=progress)
        if met:
            self._goal_met(goal)
        else:
            self.add_perceptual_input(f"You did not get what you were after here: {goal.name}", percept=False, mode='internal')
        self.focus_goal_history.append(goal)
        self.focus_goal = None
        self.goals = []
        return met


    def move(self, where, reason='', alone=False):
        """Walk somewhere: a named place, a person, or a compass direction. Scene partners
        standing with the character come along, unless the act or its reason says it goes alone
        (stage.walk). Returns the words for the audience, or None when `where` names nowhere."""
        where = re.sub(r'^(?:move|go|walk)s?\s+(?:to\s+)?', '', (where or '').strip(), flags=re.I)
        where = re.sub(r'^towards?\s+', '', where).strip()
        if not world.on(self) or not where:
            return None
        # a named place wins ('Walk west toward Spring3, scanning ...'), then a person, then a direction
        for token in [where] + stage.places_named(self.context, where):
            place = world.find(self.context, token)
            if place is not None:
                return stage.walk(self, place.xy, place.id, alone)
        for other in self.context.actors + self.context.extras + self.context.npcs:
            if other is not self and world.on(other) and re.search(rf'\b{re.escape(other.name)}\b', where, re.I):
                return stage.walk(self, stage.here(other), other.name, alone=True)
        if world.compass(where):
            return ''   # a bare compass direction leads nowhere: a walk goes to a place or a person
        resource, _ = self.resourceRefManager.resolve_reference_with_llm(where)
        # the way and then the place ('Trail toward the hut'): the place is where the walk goes
        tail = re.split(r'\b(?:towards?|to|for)\b', where, flags=re.I)[-1].strip()
        if not resource and tail and tail != where:
            resource, _ = self.resourceRefManager.resolve_reference_with_llm(tail)
        if resource:
            return stage.walk(self, resource.xy, resource.id, alone)
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
        with self.context.committing('walk', self):
            line = stage.approach(self, target)
            if line:
                self.look()
                self.context.add_event(f'{self.name} {line}')
                self.context.to_page(self.name, line)
        if line:
            await asyncio.sleep(0)

    def _walk_after(self, act, reason, alone=False):
        """The walk an act ends in, when its actor gave one after the line or the deed ('Come on,
        water runs downhill', and goes): its own event, told as any walk is."""
        place = (getattr(act, 'then_move', '') or '').strip()
        if not place:
            return
        act.then_move = ''
        with self.context.committing('walk', self):
            try:
                line = self.move(place, reason, alone=alone)
            except Exception as e:
                logger.error(f'{self.name} move failure: {e}\n{traceback.format_exc()}')
                line = None
            if line:
                self.look('Move', place, reason)
                saw = getattr(self, 'walk_witnesses', None) or [self]
                self.walk_witnesses = None
                self.context.add_event(f'{self.name} {line}', doer=self, witnesses=saw + self.bystanders())
                self.context.to_page(self.name, line)

    def _promised(self, promises, other):
        """What a line promised (sim/resolver.py resolve_words). A thing its speaker is to do now,
        in this scene, is put to the scene as a commitment was at a talk's end: rated once
        (Context.evaluate_commitment_significance, which passes over what a planned beat already
        does) and placed if it earns a beat. Each is kept in the ledger, once, for whoever is shown
        the play's things (record.account 'planner')."""
        from sim import ledger
        for p in promises:
            kind, who, what = str(p.get('kind') or '').strip().lower(), str(p.get('who') or ''), ' '.join(str(p.get('what') or '').split())
            mine = self.name.lower() in who.lower() or who.strip().lower() in ('both', 'they', 'we', 'the two')
            placed = None
            if (kind == 'now' and mine and other is not None and ablation.enabled('commitments') and self.context.current_scene
                    and self.__class__.__name__ == 'NarrativeCharacter'):
                task = Task(what[:40], what, f'said so to {other.name}', [self], None, termination=f'{what}: done',
                            start_time=self.context.simulation_time, duration=1)
                significance = self.context.evaluate_commitment_significance(self, other, task)
                placed = bool(self.context.place_commitment(self, task, significance))
                kind = 'standing' if significance == 'STANDING' else kind   # 'stay here', rated a rule to keep
            # kept, of every kind, once: a promise to do a thing now that the scene did not take up as a
            # beat is still an agreement ('a page in his best hand' was struck from the next set-up
            # as never promised), and one said again is the same promise (one was kept three times)
            book = ledger.of(self.context)
            if what and not any(k['who'].lower() == (who or self.name).lower() and k['what'].lower() == what.lower() for k in book.promised):
                book.promised.append({'who': who or self.name, 'to': [h.name for h in self.hearers() if h is not self],
                                      'what': what, 'kind': kind or 'later', 'event': (book.open or [None])[-1],
                                      'scene': (self.context.current_scene or {}).get('scene_number'), 'since': self.context.simulation_time})
            trace.event('promise', character=self.name, who=who, what=what, when=kind, placed=placed)

    async def acts(self, act:Act, target: Character, act_mode: str, act_arg: str='', reason: str='', source: Task=None):
        """Play an act. Whatever it opens as an event of the play (Context.open_event) is closed
        when it is over, however it ends: an act that fails part-way leaves no event open for
        what comes after to be put down to."""
        opened = self.context.events_open()
        self.walk_stayed, self.walk_asked = [], []   # who was asked along in this act, and who stayed (stage.goes_along)
        try:
            return await self._acts(act, target, act_mode, act_arg, reason, source)
        finally:
            self.context.close_events(opened)

    async def _acts(self, act:Act, target: Character, act_mode: str, act_arg: str='', reason: str='', source: Task=None):
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
        if leaving and not self.context.exit_allowed(self, act):
            act.leaves, leaving = None, False   # the line or deed is played; the scene goes on
        if leaving and act_mode != 'Move' and (getattr(act, 'then_move', '') or '').strip() and isinstance(act.leaves, list):
            # a line or deed that ends in a walk the whole cast takes together: as with a Move
            # (below), the scene goes with them and nobody has left it
            on_stage = [c for c in self.context.actors + self.context.extras + self.context.npcs
                        if c is not self and getattr(c, 'current_scene', None) is self.context.current_scene
                        and not stage.off_stage(c)]
            if all(c in act.leaves for c in on_stage):
                leaving = False
        if act_mode == 'Move':
            line = None
            if leaving and isinstance(act.leaves, list):
                # the whole cast walking off together has not left the scene: the scene goes with
                # them ('leaves for the spring with Joe' ended a scene on its first act)
                on_stage = [c for c in self.context.actors + self.context.extras + self.context.npcs
                            if c is not self and getattr(c, 'current_scene', None) is self.context.current_scene
                            and not stage.off_stage(c)]
                if all(c in act.leaves for c in on_stage):
                    leaving = False
            with self.context.committing('walk', self):   # the walk, who goes along, and the line that tells it
                try:
                    # with the others, unless the mover leaves them: then it goes alone, and is gone
                    # (this compared a bool with 'alone': nobody ever went alone, and Samantha
                    # 'left for the berries with Joe' while leaving him behind)
                    line = self.move(act_arg, reason, alone=leaving and act.leaves == 'alone')
                except Exception as e:
                    logger.error(f'{self.name} move failure: {e}\n{traceback.format_exc()}')
                if leaving and line is not None:
                    line = re.sub(r'^walks to ', 'leaves for ', line) if line else 'leaves.'
                    self._leave_with(act, told=bool(line))   # the line is recorded just below
                if line:
                    self.look(act_mode, act_arg, reason)
                    saw = getattr(self, 'walk_witnesses', None) or [self]
                    self.walk_witnesses = None
                    self.context.add_event(f'{self.name} {line}', doer=self, witnesses=saw + self.bystanders())
                    self.context.to_page(self.name, line)
            if line == '':
                pass   # already there: nothing to tell
            elif line:
                await asyncio.sleep(0)
            else:   # it names nowhere on the map: the narrator decides what the attempt comes to
                # in the actor's own words ('move to ' was put in front of them: 'move to Start
                # walking Northwest towards the spring' was a line of the play)
                act_mode = 'Do'
                # a deed, not a label: 'Trail toward the hut.' was a line of the play
                # (only act.action was changed, and the line shown is built from act_arg: 'Samantha:
                # Trail.' was still played four times)
                if act_arg and not re.match(r'^[a-z]', act_arg):
                    act_arg = f"heads for the {act_arg[:1].lower()}{act_arg[1:]}"
                act.action = act_arg   # Context.do reads act.action
                # somewhere the world does not list may be a place it would have (the shed behind
                # the hut): the resolver says, as for any deed, and may make the place and send the
                # mover there. Where it finds nothing in it, it is moving about where one is: seen,
                # and nothing in the world changes.
                was = act.mode
                act.mode, act.resolved = 'Do', None
                record.judge(self.context, self, act)
                if getattr(act, 'found', None):
                    leaving = bool(getattr(act, 'leaves', False)) and self.context.current_scene is not None
                else:
                    act.mode, act.resolved, act.found = was, survival.PASSING, None

        # Handle world interaction
        if act_mode == 'Do':
            await self.approach(self._first_character(target, act))
            # Get action consequences from world
            await asyncio.sleep(0)
            # the deed is one event: what it changes on the stage, the line shown for it and its
            # entry in the record are committed together (Context.committing)
            deed_event = self.context.open_event('deed', self)   # closed below, when the deed is in the record
            consequences, world_updates, character_updates = await self.context.do(self, act)
            # a passing act (or a passing part of the deed) is seen and not kept: what stays of the
            # deed after the scene is what the world settled, and nothing of the rest
            passing = bool(getattr(self.context, 'deed_passing', False))
            if character_updates and len(character_updates) > 0:
                self.add_perceptual_input(f'{character_updates}', mode='internal')  
            found = getattr(act, 'found', None)
            if found:
                act_arg = found['page']   # what is shown and recorded is what was done, as the resolver told it
            notices = '' if found or (passing and not (consequences or '').strip()) else self.process_consequences(act_arg, reason, consequences)
            await asyncio.sleep(0)
            # the act text often starts with the character's own name ('Xue pulls a tablet ...')
            deed = stage.unnamed(self.name, act_arg)
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
                told = stage.unnamed(self.name, told) or f"tries: {deed.rstrip('. ')}, and it comes to nothing."
            self.context.to_page(self.name, told, transcript=False)   # the transcript keeps it just below, as it lasts
            self.lastActResult = notices
            if source == None:
                source = self.focus_task.peek()
            task = source

            # Update displays

            # the deed goes in the play's transcript as the audience was shown it. (self.show was
            # appended here after it had been emptied: every deed was a bare 'Joe:', and what
            # reads the transcript, the outcome check and the task review, saw speech only.)
            shown = stage.unnamed(self.name, told)
            kept = re.sub(survival.BODY_NOTE, '', result).strip() if passing else None   # what outlasts the scene of a deed with a passing part
            if not passing:
                self.context.transcript.append(f'{self.name} {shown}')
            elif kept:
                self.context.transcript.append(kept)
            await asyncio.sleep(0)
            self.show = ''
            # what the world told the doer and not the audience: why it was refused, its own notes
            # (the rest of world_updates is the deed's result, which is in the record it witnessed)
            private = ([refused['fact']] if refused and not refused['applied'] and refused.get('fact') else []) \
                      + list(getattr(self, 'settled_notes', []))
            if private:
                self.add_perceptual_input(' '.join(private), percept=False, mode='internal')
            self.act_result = world_updates +'. '+character_updates
            # what others present can see happen (they used to get the action text only)
            # what the world settled and the narration it admitted (sim/record.py); it was an
            # LLM's summary of unchecked narration, which is how a pile of berries became an event
            if leaving and not self.attempt_only:
                # the deed has told it (its event follows); those it leaves with go too, as with a
                # walk or a parting line (a deed marked 'leaves with Joe' took its doer out alone,
                # and the next scene had to bring her back to him)
                self._leave_with(act, told=True)
            visible_result = stage.stage_line(self.name, '', result)
            self.visible_result = visible_result
            aimed_at = self._first_character(target, act)
            self.context.add_event(f"{self.name} {'tries: ' if self.attempt_only else ''}{deed}"
                                   + (' (it does not happen)' if self.attempt_only else '')
                                   + (f' -> {visible_result}' if visible_result else ''),
                                   doer=self, also=[aimed_at] if isinstance(aimed_at, Character)
                                   and aimed_at is not self and not stage.off_stage(aimed_at) else [], lasting=kept)
            self.context.close_event(deed_event)
            if not self.attempt_only:
                self._walk_after(act, reason, alone=leaving and act.leaves == 'alone')

        elif act_mode == 'Look':
            percept = self.look(act_mode, act_arg=act_arg,reason=reason)
            consequences = await self.context.look(self, act)
            if  consequences and len(consequences) > 0:
                self.add_perceptual_input(f'{consequences}', mode='internal')  
            self.lastActResult = consequences or 'You look, and see nothing new.'
            self.show += act_arg + '.\n ' + consequences
            if consequences.strip():   # a look that finds nothing is not a line: four of them in two acts drained the dread they were meant to build
                self.context.to_page(self.name, stage.unnamed(self.name, stage.stage_line(self.name, '', consequences)), transcript=False)
            else:
                trace.event('look_empty', character=self.name)
            await asyncio.sleep(0)

        elif act_mode == 'Think': # Say is handled below
            self.thought = act_arg
            self.show += f" \n...{self.thought}..."
            #self.add_perceptual_input(f"\nYou {act_mode}: {act_arg}", percept=False, mode='internal')
            elevenlabs_params = self.speech_stylizer.voice_params(self)
            if not getattr(act, 'unshown', False):   # words meant for someone who had gone are kept, not shown
                self.context.message_queue.put({'name':self.name, 'text':f"...{act_arg}...", 'elevenlabs_params': json.dumps(elevenlabs_params)})
            if not getattr(act, 'unrecorded', False):   # a refused deed, as it was meant, is its actor's own (record.mend_deed)
                self.context.transcript.append(f'{self.name}: ...{act_arg}...')
            await asyncio.sleep(0)

            # A thought is shown, remembered and reasoned over. (It was run as a dialog with
            # oneself: a task pushed, a model of the character kept by the character, a dialog
            # opened in it and closed at once, and a model call to update the character's
            # relationship with itself. That self-model is what gave a character its own lines
            # back as things it had heard.)
            self.add_perceptual_input(f'Internal monologue:\n {act_arg}', percept=False, mode='thought')
            self.reason_over(act_arg)
            await asyncio.sleep(0)

        elif act_mode == 'Say':# must be a say
            await self.approach(self._first_character(target, act))
            # the line is one event: the words, what the speaker does while saying them, and
            # anything they hand over (closed below, before anyone answers)
            line_event = self.context.open_event('line', self)
            # A line that carries its own stage direction before a colon and the words in quotes
            # ('Sliding down against the barred door, laughing hoarsely: "It was real..."'): the
            # direction is shown as what the speaker does, and only the quoted words are said.
            # (Such lines went out whole, quotes and all, as speech.) The colon and the quotation
            # marks say which part is which.
            staged = re.match(r'^\s*([^"“”]{6,}?):\s*["“](.+?)["”]?\s*$', act_arg or '', re.S)
            # what the speaker does as it speaks (the actor's #do beside its #say, a reply's <does>):
            # one act with the line, resolved with it in one call and shown before the words
            resolving = getattr(self, 'body', None) is not None and bool(survival.config(self.context))
            deed = ' '.join((getattr(act, 'deed', '') or '').split()) if resolving else ''
            if staged:
                business = stage.unnamed(self.name, staged.group(1).strip()).rstrip('.,;')
                act_arg = staged.group(2)
                # what the speaker does while speaking is a deed like any other: judged, and done
                # in the world or not shown (it was shown and recorded unjudged)
                if resolving:
                    # the deed said beside the line: resolved with it, below (sim/resolver.py)
                    deed, parts = deed or f'{self.name} {business}', True
                else:   # a world with no mechanics: shown as written
                    parts = True
                    self.context.add_event(f'{self.name} {business}')
                    self.context.to_page(self.name, business)
                trace.event('say_staged', character=self.name, business=business, done=bool(parts))
            act_arg = act_arg.strip().strip('"“”').strip()   # the words, not the words in quotes
            # A line is held to what its speaker cannot be mistaken about (its hands, its body,
            # what it is doing, where it is, the hour) before anyone hears it, unless the
            # character means it as a lie (sim/record.py speech). What is heard, remembered and
            # answered is the line as it is finally said.
            meant_as_lie = bool(re.search(r'\(\s*lie\s*\)', act_arg, re.I))
            act_arg = re.sub(r'\s*\(\s*lie\s*\)\s*', ' ', act_arg, flags=re.I).strip()
            handed = None
            self.said = None
            if meant_as_lie and not deed:
                trace.event('speech_check', character=self.name, outcome='lie', claims=[], before=act_arg)
            else:
                # (a lie told with a deed is still resolved, for the deed: its words stand as the liar wrote them)
                checked, handed = record.speech(self.context, self, act_arg, self._first_character(target, act), deed=deed)
                act_arg = act_arg if meant_as_lie else checked
            act.action = act_arg
            said, self.said = getattr(self, 'said', None), None
            if said is not None and deed and said.get('deed_refused'):
                # the deed cannot be done here: the words are still said; the actor is told why,
                # and it counts against the beat as any refused deed does
                trace.event('deed_refused', character=self.name, deed=deed[:300], why=[said['deed_refused']], label=None, again=False, with_line=True)
                record.refused(self.context, self, deed, said['deed_refused'], chosen=True,
                               seen=[f.get('why') for f in said.get('findings') or [] if isinstance(f, dict) and f.get('kind') == 'here'],
                               shown=said.get('seen', ''), absent=said.get('absent', ''))
                self.add_perceptual_input(f"You meant to: {deed}. It cannot be done here: {said['deed_refused']}.", percept=False, mode='internal')
            if said is not None and (said['effects'] or said['page']):
                # what is done with the words (a deed beside them, or a thing the words themselves
                # hand over), shown before them
                given = Act(mode='Do', action=deed or act_arg, actors=[self], reason='', duration=1, source=None, target=[])
                given.found = said
                settled = resolver.apply(self.context, self, given)
                if settled:   # the world's own sentences (what a written thing says, what a world action gave): the doer's to know
                    self.add_perceptual_input(re.sub(survival.BODY_NOTE, '', settled).strip(), percept=False, mode='internal')
                seen = stage.unnamed(self.name, said['page'])
                if seen:
                    self.context.add_event(f'{self.name} {seen}', doer=self)
                    self.context.to_page(self.name, seen)
            self.show += f"{act_arg}'"
            elevenlabs_params = self.speech_stylizer.voice_params(target)   # the line is spoken as written
            #print(f"Queueing message for {self.name}: {act_arg}")  # Debug
            # (Context.say keeps the line in the transcript and the record)
            self.context.to_page(self.name, f"'{act_arg}'", kind='speech', transcript=False, elevenlabs_params=json.dumps(elevenlabs_params))
            self.context.say(self, act_arg, self.hearers(), to=self._first_character(target, act))
            await asyncio.sleep(0)
            if said is not None:
                # what the line was found to tell and promise (record.speech, sim/resolver.py):
                # part of the line's own event
                handed = None
                resolver.tell(self.context, self, self.hearers(), said['tells'])
                if said['retold']:
                    trace.event('retold', character=self.name, words=said['retold'][:200])
                self._promised(said['promises'], self._first_character(target, act))
            self.context.close_event(line_event)
            self._walk_after(act, reason, alone=leaving and act.leaves == 'alone')
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
                party = self._going_along(act)   # those who do go, each having said so
                to = getattr(act, 'leaves_to', None)
                gone = ('leaves' if not party else f"leaves with {', '.join(c.name for c in party)}") + (f', for {to}.' if to else '.')
                with self.context.committing('leave', self):
                    self.context.to_page(self.name, gone, transcript=False)
                    self.context.add_event(f'{self.name} {gone}')   # for the next scene's set-up: who went, and with whom
                    self._leave_with(act, told=True)
                target = None
            elif target is not None and isinstance(target, Character) and (stage.off_stage(target) or stage.not_in_scene(target)):
                target = None   # said to someone who has gone, or who is not in this scene: nobody hears it
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
                    carried, self._talk_carried = getattr(self, '_talk_carried', 0), 0
                    if carried:   # a talk that has passed to this pair keeps the count of its lines
                        self.actor_models.get_actor_model(target.name).dialog.carried = carried
                        target.actor_models.get_actor_model(self.name).dialog.carried = carried

                # self is speaker, don't get confused about first arg to add_turn
                self.actor_models.get_actor_model(target.name).dialog.add_turn(self, content)
                target.actor_models.get_actor_model(self.name, create_if_missing=True).dialog.add_turn(self, content)
                owner = scene_owner.owner_of(self.context)
                if owner is not None:   # the scene's owner runs the talk, a turn at a time (sim/scene.py)
                    await owner.converse(self, target, act_arg, source)
                else:
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
            # Those who are there see it: the others in the scene and within reach of it, and
            # whoever it was done to. (It was told to every member of the cast whom the DOER'S
            # model marked visible, a flag set when the doer last looked: someone who had left
            # the scene went on seeing what was done in it, and someone standing by who had not
            # been looked at saw nothing.) Outside a scene, as before.
            if self.context.current_scene is not None:
                watchers = list(self.bystanders())
                aimed_at = self._first_character(target, act)
                if isinstance(aimed_at, Character) and aimed_at is not self and aimed_at not in watchers and not stage.off_stage(aimed_at):
                    watchers.append(aimed_at)
            else:
                watchers = [a for a in self.context.everyone()
                            if a is not self and getattr(self.actor_models.get_actor_model(a.name), 'visible', False)]
            # what they saw is in the record (add_event above, witnessed); here each reads the doer's aim
            for actor in watchers:
                percept = PerceptualInput(mode=SensoryMode.VISUAL, content=seen, timestamp=self.context.simulation_time, intensity=0.6)
                actor.actor_models.get_actor_model(self.name, create_if_missing=True).infer_goal(percept)
        self.previous_action_mode = act_mode           

    def instantiate_narrative_goal(self, goal_statement, generate_conditions=True):
        """instantiate a goal from a narrative"""
        drives.appraise(self)   # what has reached the character since it last acted moves its drives

        prompt = [UserMessage(content="""You are an actor playing the character {{$name}}.
Instantiate a goal object for the following directive: the director's objective for you in this scene, stated as the character's own aim. Keep to what it asks, in this person's terms; make it neither more hostile nor more guarded than it is written.

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
                                 "relationships": self.actor_models.format_relationships(include_transcript=False),
                                 "recent_memories": self.lived(16),
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
            scene_narrative = (f"\nThe scene opens: {self.context.scene_pre_narrative}\nIt is to settle: {self.context.scene_post_narrative}\n"
                               f"The tasks are what {self.name} does to get what {self.name} wants out of that. How it is settled is not planned: what the others answer or do is theirs.")
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

Create at most {{$ntasks}} specific, actionable task(s): the different things {{$name}} does in this scene, in light of the focus goal, the central narrative, the act specific narrative and the scene given below.

What kind of list this is depends on the focus goal:
- If the focus goal is a job of work (something to find, make, carry, mend, reach), the tasks may be its steps.
- If the focus goal is something wanted from another person (an answer, an admission, a promise, their staying), the tasks are NOT steps toward it. At most one task asks or works on that person for it. Once asked, it is given or refused, and no later task asks for the same thing in other words. Every other task is something of {{$name}}'s own: it comes from another of your drives, from your past or your nature, from the place and what can be done in it, or it is something you offer, show, tell, make or try. It would be worth doing if the focus goal did not exist.
- A task's termination is something {{$name}} can bring about alone, or that either answer from the other person satisfies (told, or refused). Never one that waits for the other person to give in.

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

Each task should be a part of the scene worth watching for itself. Do not include tasks merely to fill time or number of tasks.
Each task should contain the potential for a narrative beat: learn something new, relationship shift, powerful emotion, change in power dynamics, etc.
Where appropriate, drawn from typical life scripts.
Also, the collective duration of the tasks should be less than any duration or completion time required for the focus goal.
                              
A task is a specific objective that can be achieved in the current situation.
The new task(s) should be distinct from one another: different in what is done and in what is wanted, not the same want put more strongly.
Use known actor names in the task description unless deliberately introducing a new actor. Characters known to this character include:
{{$known_actors}}

Start with an appropriate task in light of previous achievments and the dramatic context. 

{{$achievments}}


Make explicit reference to diverse known or observable resources where logically fitting, ensuring broader environmental engagement across tasks.
Eating, drinking and resting are tasks only when your body says the need is at its worst, or the focus goal is about it: ordinary hunger, thirst and tiredness are seen to between scenes.

A task has a name, description, reason, list of actors, start time, duration, and a termination criterion as shown below.
Respond using the following hash-formatted text, where each task tag (field-name) is preceded by a # and followed by a single space, followed by its content.
Each task begins with its #name line and ends with ## as shown below. Insert a single blank line between tasks.
be careful to insert line breaks only where shown, separating a value from the next tag:

#name brief (4-6 words) task name
#description terse (6-8 words) statement of the action to be taken: what is done, offered, shown or asked, not the words to be said (the actor finds those when it is played).
#reason (6-10 words) what you want out of this, in your own terms: what you want the other person to do, say, show or feel, or what you want to have or know. This is your intent for the task; it is kept if the action later has to change.
#actors the names of any other actors involved in this task, comma separated. if no other actors, use None
#start_time (2-3 words) expected start time of the action
#duration (2-3 words) expected duration of the action in minutes
#termination (5-7 words) condition test to validate goal completion, specific and objectively verifiable.
#leaves yes or no: yes only if doing this task takes {{$name}} away from this scene's place (setting off for somewhere else, alone or with the others). Such a task ends the scene, so it is the last in your list, and most scenes have none.
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
            scene_narrative = (f"\nThe scene opens: {self.context.scene_pre_narrative}\nIt is to settle: {self.context.scene_post_narrative}\n"
                               f"The tasks are what {self.name} does to get what {self.name} wants out of that. How it is settled is not planned: what the others answer or do is theirs.")
        else:
            scene_narrative = ''
        logger.info(f'{self.name} generate_task_plan: {goal.to_string()}')
        response = default_ask(self, 
                               prefix=mission, 
                               suffix=suffix, 
                               addl_bindings={"focus_goal":goal.to_string(),
                                "ntasks": ntasks,
                                "known_actors": "\n".join([name for name in self.actor_models.names()]),
                                "achievments": '\n'.join(self.achievments[-5:]),
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
        memory_text = self.lived(8)

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
                                 }, prompt, tag='completion_statement', temp=0.5, stops=['</end>'], max_tokens=120)
        return response

    DONE = """You keep the record of an improvised play. {{$name}} set out to do a thing. Say whether it has been done, from the record below and from nothing else.

<what_{{$name}}_set_out_to_do>
{{$what}}
Done when: {{$done_when}}
</what_{{$name}}_set_out_to_do>

<what_{{$name}}_has_done_toward_it>
{{$acts}}
</what_{{$name}}_has_done_toward_it>

<what_has_happened_and_been_said_since_it_began>
{{$happened}}
</what_has_happened_and_been_said_since_it_began>

<where_things_stand>
{{$situation}}
</where_things_stand>

- Done means the "done when" is so now: the record shows it happening, or where things stand shows it. Trying, asking, intending, moving toward it or talking about it is not done. A deed told as a try, or marked "(it does not happen)", did not happen.
- If the "done when" is about what another person does or says, it is done only if the record shows that person doing or saying it.
- A promise or an agreement to do it is not doing it: "I'll wake you at the halfway" does not show the watch split. Only a "done when" that is itself something said is shown by a line.
- Part of it done is not done. When in doubt, not done.

Answer in JSON only:
{"done": true or false, "shown_by": "<a few words copied exactly from the record that show it; empty when not done>"}
End your response with: </end>"""

    async def test_termination(self, object, termination_check, consequences, updates='', type=''):
        """Whether a task or goal has been done: (done, 100 or 0). Done means the record shows
        it, and the reply has to quote what shows it.

        (This asked for complete, partial or insufficient with a progress figure, and on
        'partial' drew a random number against the figure: a quarter of the tasks a character
        ended as satisfied were 40 to 70 per cent done, each then written into its memory as
        achieved. It was given the character's narrative summary under the name
        'relationships' and twenty tokens to 'reason step by step' in.)"""
        context = self.context
        events_at, lines_at = getattr(object, 'began_at', None) or getattr(context, '_scene_plan_mark', (0, 0))
        happened = list(context.scene_events[events_at:]) + [str(l).strip() for l in context.transcript[lines_at:] if str(l).strip()]
        if not happened:   # outside a scene: the latest of what the world has recorded
            happened = list(context.recent_events)[-8:]
        done_acts = [f'- {a.mode}: {str(a.action)[:200]}' for a in (getattr(object, 'acts', None) or [])]
        reply = self.llm.ask({'name': self.name, 'what': f'{object.name}: {object.description}',
                              'done_when': str(termination_check).strip('# '),
                              'acts': '\n'.join(done_acts) or 'nothing yet',
                              'happened': record.last_lines(happened + [x for x in (consequences, updates) if x and x.strip()], 5000) or 'nothing',
                              'situation': record.seen_by(self)},
                             [UserMessage(content=self.DONE)], tag='test_termination', temp=0.1, stops=['</end>'], max_tokens=90)
        found = json_reply.parse(reply) if reply else None
        shown_by = str(found.get('shown_by') or '').strip() if isinstance(found, dict) else ''
        # the words cited must be in what it was shown (a task closed on words nobody wrote is not done)
        shown = '\n'.join(happened + [x for x in (consequences, updates) if x and x.strip()] + [record.seen_by(self)])
        done = isinstance(found, dict) and found.get('done') is True and bool(shown_by) and record.quoted_in(shown_by, shown)
        if isinstance(found, dict) and found.get('done') is True and shown_by and not done:
            trace.event('termination_unbacked', character=self.name, task=object.name, shown_by=shown_by[:200])
        print(f"\n{self.name} testing {type} {object.name} termination: {termination_check}: {'done' if done else 'not done'}")
        if not isinstance(found, dict):
            trace.event('check_unreadable', of='termination', reply=(reply or '')[:300])
        object.progress = 100 if done else 0
        if done:
            statement = self.generate_completion_statement(object, termination_check, 'complete', 100, consequences, updates)
            object.completion_statement = statement
            self.add_perceptual_input(statement, mode='internal')
        return done, object.progress

    def generate_acts(self, task: Task, goal: Goal=None):
        in_dialog = self.check_if_in_dialog_subtask()
        modes = 'Think, Say, Look, Move, Do'
        if in_dialog:
            modes = 'Look, Move, Do'
        print(f'\n{self.name} generating acts for task: {task.to_string()}')
        # One question, its facts beside it, each said once. (The prompt had grown to 35k
        # characters: four length rules, the dialog four times, five examples from other plays with
        # fields not in the format, "the primary consideration is the task" above "do not repeat",
        # and the task's own words, which scripted lines, put first: Lanya told the morning her
        # father walked into the fog three times, and Sonnet given the same prompt told it again.)
        mission = """You are the actor playing {{$name}} in an improvised play. Choose {{$name}}'s next act.

{{$director_notes}}

#The task {{$name}} is working on now
{{$task_string}}
It serves this goal: {{$goal_string}}
##
"""
        suffix = """

#What is physically going on right now
{{$act_note}}
{{$here_and_now}}
##

#Done so far toward this task
{{$done_in_task}}
##

The question: after what is already done and said, what one act now moves {{$name}} toward what this task is for?
- The task says what {{$name}} is after, not lines to recite. What {{$name}} has already said or done in the play ('What you have seen and heard lately', 'Done so far') is not said or done again, in the same words or in others: go on from it.
- Take up the latest thing said or done to {{$name}}: answer it, use it, or refuse it.
- {{$name}}'s drives and emotional stance decide how boldly and in what way. Pressing for a thing outright is one way among several: {{$name}} may tease, confide something, do the other a good turn, give a little to get a little, change the subject or go quiet, as this person would with this listener; not the way of {{$name}}'s own last two acts.

{{$what_an_act_is}}
- do: a physical act, in the third person and present tense, as the audience sees it ('{{$name}} hands her the cup'). It carries no words (those go in say). {{$do_rule}}
{{$both_rule}}- move: walk to a place or person by name (a name under 'You could do here', or a character's). Those with {{$name}} come along unless {{$name}} goes alone.
- look: look at a person here, a thing here or held, or a place in sight, by its name; or look around. It shows what can be seen of it from where {{$name}} stands.
{{$think_rule}}
Truth: {{$name}} speaks and acts only from what {{$name}} has seen, been told or can reasonably guess, and says a guess as a guess. 'Your body' is the truth of {{$name}}'s needs and what {{$name}} holds. A deliberate lie or bluff about oneself ends with the mark (lie), which is not shown. Speak as a person of this world: minutes and paces, ordinary names ('the spring', not 'Spring2'), nothing of a game.
Words belong to their speaker: a name or nickname one character uses for another is that character's own, not {{$name}}'s.

Respond with one act, in this form, closed by ## on a line of its own. {{$give}}:

{{$say_line}}#do what {{$name}} does, in the third person, 20 words at most; omit if {{$name}} does nothing but speak
#move the place or person walked to
#look the name of what is looked at, or the word around
{{$think_line}}#target the character(s) spoken to, acted on or walked to, comma separated; omit if none
#duration minutes
#leaves no, alone, or with <names>. Whether with this act your character goes out of the scene: alone when it walks out on the others or goes off by itself; with and the names of those who go with it (comma separated) when they go off together ("follow me", "lead the way" and they come); no otherwise: for moving about the place, and for going to fetch, gather or look at something as part of what your character is doing here. Leaving alone ends your character's part in the scene. When it leaves, add where to: "with Joe, to the trail northwest".
##

{{$examples}}

Respond only with the one act.
"""
        # One act, which may be a line, a deed, or both played as one (and a walk after them). It
        # was 'the act played now, then one alternative in a different mode': in 110 of 122
        # recorded answers the alternative was the other half of a line and its gesture, and
        # the engine played one half and threw the other away ('You want some?' with no hand
        # held out). Asked for one act with room for both, the actor first put a gesture on
        # nearly every line; the wording here is the one that did not
        # (logs/review-20261120/act/one_act_ab.py).
        name = self.name
        # what a deed may be: in a world played without its catalogue of actions (survival.free),
        # whatever a person could do there with their body and what is at hand, and the world says
        # what comes of it (sim/resolver.py); elsewhere, the listed actions
        do_rule = ((f"It is whatever a person could really do here with their body and what is at hand: handle what {name} holds or what lies here, "
                    "eat or drink what is really here, pick or take what a place of its kind gives, sit, kneel, lie down, get up, touch someone. "
                    "The world says what comes of it. A thing that is not held or lying here cannot be used, and nobody is pushed, pulled or held by force."
                    + (" Buying, paying, work for a wage and the like go as this world's facts have them ('How this world works', below), in whatever way a person there would go about it."
                       if survival.facts_note(self.context) else '')))
        if in_dialog:   # in a talk the words are the reply's (generate_dialog_turn): here only what is done
            forms = {'what_an_act_is': f'What an act is: {name} does something; or instead walks somewhere, or looks.',
                     'both_rule': '', 'think_rule': '', 'say_line': '', 'think_line': '',
                     'give': 'Give #do; or else one of #move, #look',
                     'examples': "An example, from another play:\n#do Mirela hands her cup to Tomas.\n#target Tomas\n#duration 1\n#leaves no\n##"}
        else:
            forms = {'what_an_act_is': (f"What an act is: {name} says something, or does something, or does both at once (a line, and what the hands or body do as it is said). "
                                        f"Or, instead of either, {name} walks somewhere, looks, or thinks.\n"
                                        f"- say: words {name} speaks aloud (also anything read, recited or sung), and only the words. Words do nothing physical: eating, drinking, "
                                        "giving, taking, sitting down or going is a deed. Do not name what you and the other are to each other or what the moment means; "
                                        "ask, refuse, offer or tease, and let that show it."),
                     'both_rule': (f"- Most acts are one thing: a line, or a deed. Give both only when {name} speaks while really changing something: handing a thing over, taking it, "
                                   "eating, drinking, picking a thing up or putting it down, sitting down or getting up, going. A look, a nod, a pause, a turn of the head, a stance, "
                                   "hands that fidget are not deeds and are not written with a line.\n"),
                     'think_rule': '- think: a private thought, in the first person; nobody hears it.\n',
                     'say_line': f'#say the words {name} speaks, {stage.say_limit(self)}; omit if {name} says nothing\n',
                     'think_line': '#think the thought\n',
                     'give': f'Give #say, or #do, or both, with #move after them when {name} then walks off; or else one of #move, #look, #think and nothing with it',
                     'examples': ("Two examples, from another play. Words alone:\n#say You drank the last of it. Where's mine?\n#target Tomas\n#duration 1\n#leaves no\n##\n"
                                  "A line with the deed it goes with:\n#say Then take mine. I'm not thirsty.\n#do Mirela hands her cup to Tomas.\n#target Tomas\n#duration 1\n#leaves no\n##")}

        if not goal:
            goal = self.focus_goal
        if not goal and task and task.goal:
            goal = task.goal
        if not goal:
            raise ValueError(f'No goal for {self.name}')
        #generate emotional state for this task
        self.emotionalStance = EmotionalStance.for_character(self)
        # (the task was replaced here by whatever stood on top of the focus stack: called for a
        # scene task while a dialog task was above it, this wrote acts for the dialog)
        response = default_ask(self, prefix=mission, suffix=suffix, 
                               addl_bindings={"goal_string":goal.to_string(), "task":task, "task_string":task.to_fullstring(),
                                              "done_in_task": self.done_in_task(task) or 'Nothing yet.',
                                              # the frame's last act is in 'Done so far' (or is the record's)
                                              "lastAct": 'see Done so far toward this task', "lastActResult": 'see What you have seen and heard lately',
                                              "say_limit": stage.say_limit(self),
                                              "here_and_now": self.physical_context(events=False),
                                              "act_note": survival.act_note(self),
                                              # the scene and this actor's objective, not the play's theme: given 'is trust itself
                                              # the trap?', the actors said 'Trap—one that fed us'
                                              "director_notes": self.director_notes(story=False),
                                              "modes": modes, **forms, "do_rule": do_rule,
                                              "name": self.name},
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
        for act_hash in act_hashes[:1]:   # one act was asked for
            try:
                # the form's parts, as the act the engine plays: a line (with the deed said beside
                # it and the walk after it kept on the act), else the deed, else the walk, look or thought
                part = {k: ' '.join((hash_utils.find(k, act_hash) or '').split()) for k in ('say', 'do', 'move', 'look', 'think')}
                part = {k: v for k, v in part.items() if v and v.lower().strip(' .') not in ('none', 'omit', 'nothing', 'n/a', '-', 'no')}
                if in_dialog:
                    part.pop('say', None); part.pop('think', None)
                mode = next((m for m in ('say', 'do', 'move', 'look', 'think') if m in part), None)
                if mode is None:
                    continue
                rest = '\n'.join(l for l in act_hash.split('\n') if not re.match(r'\s*#(say|do|move|look|think|mode|action)\b', l))
                act = self.validate_and_create_act(f"#mode {mode.capitalize()}\n#action {part[mode]}\n{rest}", task)
                if act:
                    act.deed = part.get('do', '') if mode == 'say' else ''
                    act.then_move = part.get('move', '') if mode in ('say', 'do') else ''
                    act_alternatives.append(act)
            except Exception as e:
                print(f"Error parsing Hash, Invalid Act. (act_hash: {act_hash}) {e}")
        self.actions = act_alternatives
        return act_alternatives


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

    TALK_FOR_A_DEED = 3   # the lines a beat made in talk may have, all told

    def natural_dialog_end(self, from_actor):
        """ called from acts when a character says something to this character """
        #if self.actor_models.get_actor_model(from_actor.name).dialog.turn_count > 10):
        #    return True
        if from_actor and from_actor.name.lower() == 'viewer':
            return False
        dialog = from_actor.actor_models.get_actor_model(self.name).dialog
        transcript = dialog.get_current_dialog() # this is dialog from the perspective of self.
        plan, at = self.context.scene_integrated_task_plan or [], self.context.scene_integrated_task_plan_index
        beat = plan[at]['task'] if self.context.current_scene and at is not None and at < len(plan) else None
        if getattr(beat, 'from_talk', False):
            beat.talk_turns += 1   # every line said under a beat that was made in talk, whichever talk it is in
        carried = getattr(dialog, 'carried', 0)
        if carried and dialog.turn_count + carried > 12 + (4 if stage.kind(self) in ('shared job', 'lull', 'confession', 'play') else 0):
            # a talk that has passed from pair to pair has had its lines, whoever was spoken to last:
            # each pass is to someone who has not yet answered in the new pair, so the rule below
            # never let such a talk end (one beat ran forty-four lines, and the scene's other five were not played)
            trace.event('talk_check', listener=self.name, lines=dialog.turn_count + carried, beat='passed on', awaiting=False, over=True)
            return True
        if f'{self.name} says: ' not in transcript:
            # whoever is spoken to answers at least once: ending here left the speaker to repeat
            # the same proposal three times to someone who never replied
            return False
        if getattr(beat, 'from_talk', False) and beat.talk_turns >= self.TALK_FOR_A_DEED:
            # a beat made in talk is the deed that was promised: a few lines to get to it, no more
            trace.event('talk_check', listener=self.name, lines=dialog.turn_count, beat='made in talk', awaiting=False, over=True)
            return True
        if stage.tempo(self) == 'urgent':
            return True   # one exchange, then back to what is happening
        owner = scene_owner.owner_of(self.context)
        verdict = owner.check(owner.current_task()) if owner is not None else None
        if verdict is not None:
            # the owner's reading of the page (sim/scene.py): the talk is over when the scene has
            # arrived, or when its beat is done and nothing waits for an answer; a question put is
            # answered; and no talk runs on without end
            # the talk's turns, not the lines of `transcript`: that is its last ten, so no limit
            # above ten was ever met, and a talk on a beat that could not finish ran 41 lines
            # and the whole talk's, whoever it has passed between: counted for each pair afresh, a
            # talk that passed four times ran one beat for forty-four lines and the scene's other
            # five beats were never played
            lines = dialog.turn_count + getattr(dialog, 'carried', 0)
            run = 6 if stage.kind(self) in ('shared job', 'lull', 'confession', 'play') else 0
            run = 4 if run else 0
            if owner.over:
                over = True
            elif verdict['awaiting']:
                over = lines > 12 + run   # a question is answered, but not for ever
            else:
                over = verdict['beat'] == 'done' or owner.winding or lines > random.randint(8, 10) + run
            trace.event('talk_check', listener=self.name, lines=lines, beat=verdict['beat'], awaiting=verdict['awaiting'], over=bool(over))
            return bool(over)
        keeper = scene_owner.budget_of(self.context)
        if keeper is not None and keeper.winding:
            return True   # no sheet to read, but the scene has had its lines
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
        # talk at ease runs longer than talk under pressure (the scene's kind, stage.KINDS)
        run = 6 if stage.kind(self) in ('shared job', 'lull', 'confession', 'play') else 0
        end_point = rating > 7 or (random.randint(4, 10) < rating) or ((rating + len(transcript.split('\n')) + getattr(dialog, 'carried', 0)) > random.randint(8,10) + run)
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
            self.add_perceptual_input(f'Reflecting:\n {response}', percept=False, mode='thought')
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
            self.add_perceptual_input(f'Reasoning over thought:\n {response}', percept=False, mode='thought')
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
            self._end_talk(from_actor)
            # (each line's promises are read as it is said: Character._promised)
            return

        cut_in = self.bystander_cuts_in(from_actor, message)
        if cut_in:
            message = f'{message}\n{cut_in}'
        text, response_source, reason = self.generate_dialog_turn(from_actor, message, self.focus_task.peek()) # Generate response using existing prompt-based method
        if not text or not text.strip():
            # no <response> (generate_dialog_turn already deactivated the dialog); an empty Say
            # was performed anyway, shown as '' and heard by from_actor, which kept talking
            return
        to = getattr(self, '_reply_to', None) or from_actor
        self._reply_to = None
        if to is not from_actor:
            # the answer is said to someone else who is there: the talk with from_actor is over,
            # and the one spoken to has the next line (Character.acts opens the talk with them).
            # (The answer went back to whoever had spoken, whatever it said: 'Right then, your
            # turn. The corner', meant for Kidd, was answered by Marek, and Kidd, both his
            # cut-ins spent, stood by for twenty lines of a scene that was about him.)
            trace.event('talk_passed', speaker=self.name, was_with=from_actor.name, to=to.name)
            was = from_actor.actor_models.get_actor_model(self.name).dialog
            self._talk_carried = was.turn_count + getattr(was, 'carried', 0)   # the talk goes on: its lines so far go with it
            self._end_talk(from_actor)
        action = Act(mode='Say', action=text, actors=[self, to], reason=reason or '', duration=1, source=response_source, target=[to])
        action.leaves = getattr(self, '_reply_leaves', None)
        action.leaves_to = getattr(self, '_reply_leaves_to', None)
        action.deed, self._reply_does = getattr(self, '_reply_does', '') or '', ''
        await self.act_on_action(action, response_source)
        if self.focus_action == action:
            self.focus_action = None
        await asyncio.sleep(0)

    def _end_talk(self, from_actor):
        """The talk between self and from_actor is over: each takes from it what it makes of the
        other, and the task of talking is put down."""
        dialog = self.actor_models.get_actor_model(from_actor.name).dialog.get_current_dialog()
        trace.event('dialog_end', speaker=from_actor.name, listener=self.name,
                    turns=self.actor_models.get_actor_model(from_actor.name).dialog.turn_count)
        self.actor_models.get_actor_model(from_actor.name).short_update_relationship(dialog.split('\n'), use_all_texts=True)
        self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
        self.last_task = self.focus_task.peek()
        self._pop_dialog_task(from_actor)
        # it would probably be better to have the other actor deactivate the dialog itself
        other_dialog = from_actor.actor_models.get_actor_model(self.name).dialog.get_current_dialog()
        from_actor.actor_models.get_actor_model(self.name).short_update_relationship(other_dialog.split('\n'), use_all_texts=True)
        from_actor.actor_models.get_actor_model(self.name).dialog.deactivate_dialog()
        from_actor.last_task = from_actor.focus_task.peek()
        from_actor._pop_dialog_task(self)

    def done_in_task(self, task):
        """What this character has already done toward a task, to stand beside the task itself.
        The task was restated at the head and foot of the act prompt as still to do, with the
        last act far up the page: after scanning the trees, screaming and running, Samantha
        was given 'scan tree line and scream warning' again and thought about whether to scream."""
        acts = [a for a in (task.acts or []) if getattr(a, 'action', None)]
        if not acts:
            return ''
        # a deed the world refused is told as a try: it was listed as done, and the actor went on
        # as if it had the thing it had been refused
        return ("Toward this task you have already, in this order:\n"
                + '\n'.join(f"- {a.mode}: {a.action}" + (' (you tried; it did not happen)' if getattr(a, 'attempt_only', False) else '')
                             for a in acts[-4:])
                + "\nThese are done (a try that did not happen may be tried another way) and cannot be done again or thought over as if still to do. "
                  "Choose what comes next after them; if they have done what the task asks, do what follows from it."
                + ('' if not any(a.mode == 'Say' for a in acts[-2:]) else
                   # where nobody is pressing anybody the talk is the scene: held to one Say a task,
                   # a 'shared job' was eight berry-pickings in a row and nine spoken lines
                   " You have spoken in this task. You may speak again, of something other than what you last spoke of: the work in hand, "
                   "this place, something of your own that it brings to mind, a plain question about the other person. Or a Do, Move, Look or Think."
                   if stage.kind(self) in ('shared job', 'lull', 'play') else
                   " You have had your say in this task: now a Do, Move, Look or Think, unless someone has just said something you must answer."))

    EARSHOT_TILES = 4
    MAX_CUT_INS_PER_SCENE = 2     # per character
    CUT_IN_GAP_LINES = 4          # transcript lines before the same character may cut in again

    def bystanders(self, other=None):
        """Characters in the current scene, other than self and `other`, close enough to hear."""
        scene = self.context.current_scene
        if scene is None or not world.on(self):
            return []
        here = world.at(self)
        return [c for c in self.context.actors + self.context.extras + self.context.npcs
                if c is not self and c is not other and c.name != 'Viewer'
                and getattr(c, 'current_scene', None) is scene and world.on(c)
                and world.in_earshot(self.context, world.at(c), here)]

    def hearers(self, other=None):
        """Those who hear what self says, other than self and `other`: the bystanders, and those
        in the scene by radio (stage.remote). Someone on the radio hears the whole scene and is
        heard by it; what is done in person (bystanders) stays with those who are there."""
        near = self.bystanders(other)
        scene = self.context.current_scene if self.context is not None else None
        if scene is None:
            return near
        in_scene = [c for c in self.context.actors + self.context.extras + self.context.npcs
                    if c is not self and c is not other and c.name != 'Viewer' and c not in near
                    and getattr(c, 'current_scene', None) is scene and not stage.off_stage(c)]
        linked = in_scene if stage.remote(self) else [c for c in in_scene if stage.remote(c)]
        return near + linked

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
        candidates = [c for c in from_actor.hearers(self) if may(c)]
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
        with self.context.committing('line', speaker):
            self.context.to_page(speaker.name, f"'{line}'", kind='speech', transcript=False,
                              elevenlabs_params=json.dumps(speaker.speech_stylizer.voice_params(to)))
            self.context.say(speaker, line, speaker.hearers(), to=to)   # heard by all within earshot (the record)
        trace.event('dialog_turn', speaker=speaker.name, listener=to.name, text=line[:1000], cut_in=True)
        dialog.add_turn(speaker, line)
        self.actor_models.get_actor_model(from_actor.name).dialog.add_turn(speaker, line)
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
            # spoken to with nothing on its stack: the dialog was still open on the speaker's side
            # while this one's task had ended (Samantha picked berries, her task was done, and her
            # dialog task went with it; Joe's next line then stopped the play with an exception)
            self.focus_task.push(Task('dialog with ' + from_actor.name, description='dialog with ' + from_actor.name,
                                      reason='answer ' + from_actor.name, termination='natural end of dialog', goal=None,
                                      start_time=self.context.simulation_time, duration=2, actors=[self, from_actor]))
            trace.event('dialog_task_restored', character=self.name, speaker=from_actor.name)
        system_prompt = """You are an actor in an improvised play, speaking your character's next line.
You are in the scene, not writing it: say what this person would actually say now, to this listener.
A good line does one thing (asks, refuses, offers, concedes, threatens, teases, lets something slip, changes the subject) and leaves the other actor something to answer.
People in conversation speak plainly and often incompletely; they do not deliver slogans, aphorisms or speeches.
""" + ("Bodily needs, the place the characters are in, and what has just happened are real; let them shape what is said."
       if getattr(self, 'body', None) is not None else
       "The place the characters are in and what has just happened are real; let them shape what is said.") + """
Keep your character's own way of speaking (see 'How you speak') from line to line.
        """
        # One question, its facts once each. (It carried an empty <scene>, an empty 'last
        # action', a blank gender for the listener, the talk also in memories and again as 'your
        # own last lines', 40 words here against 30 in the act prompt, and 'do not answer a
        # question with a question' for characters whose voice is teasing questions.)
        speaker = 'yourself' if self is from_actor else from_actor.name
        if not hasattr(self, 'tried_with'):
            self.tried_with = {}
        told_to = self._first_character(None, None) if self is from_actor and hasattr(self, '_first_character') else None
        other_name = from_actor.name if self is not from_actor else (getattr(told_to, 'name', None) or 'the other')
        prompt_string = """You are the actor playing {{$name}}. """ + (
            """{{$name}} is thinking; give {{$name}}'s next thought.""" if self is from_actor else
            """{{$speaker}} has just spoken to {{$name}}; give {{$name}}'s answer.""") + """
{{$scene}}
<your_character>
{{$character}}
</your_character>

<emotional_state>
{{$emotionalState}}
</emotional_state>

<what_you_are_after>
{{$goals}}
</what_you_are_after>

<before_this_scene>
{{$history}}
</before_this_scene>

<where_things_stand>
{{$situation}}
</where_things_stand>

<your_own_thoughts_and_what_you_have_made_of_things>
{{$memories}}
</your_own_thoughts_and_what_you_have_made_of_things>

<how_you_see_{{$speaker}}>
{{$relationship}}
</how_you_see_{{$speaker}}>

<what_you_have_tried_so_far_with_{{$other}}>
{{$tried}}
</what_you_have_tried_so_far_with_{{$other}}>

<the_talk_so_far>
{{$dialog}}
</the_talk_so_far>
{{$also_here}}
<here_and_now>
{{$here_and_now}}
</here_and_now>

<what_was_just_said>
{{$statement}}
</what_was_just_said>

The question: what does {{$name}} say back, now?
{{$way}}
- First decide whether {{$name}} goes along with it. {{$name}} may doubt, refuse or push back, as how {{$name}} sees {{$speaker}} and what {{$name}} wants decide; agreeing, {{$name}} agrees without repeating it.
- One turn: {{$say_limit}}. Make one point and leave the rest for later turns.
- Answer what was said: do not open by saying its words back as a question ("A trap?"), and do not take on its way of talking. {{$name}} speaks in {{$name}}'s own voice ('How you speak').
- Words belong to their speaker: a name or nickname that someone else uses is theirs, not {{$name}}'s.
- No slogans, paired contrasts or flourishes: say it the way {{$name}} would across a table. Do not name what you two are to each other or what the moment means; ask, refuse, offer or tease, and let that show it. Let the emotional state colour the line without announcing it.
- The line is words only: no actions, gestures or body language in it. Words do nothing physical, and nothing is said as done that has not been done. A turn in a talk is nearly always words alone. Only if {{$name}}, as the words are said, really changes something (hands a thing over, takes it, eats, drinks, picks a thing up or puts it down, sits down or gets up) is that written, in <does>, and it is then really done. A look, a nod, a pause, a turn of the head, a stance, a step closer, hands that fidget are not written anywhere.
- Truth: {{$name}} speaks only from what {{$name}} has seen, been told or can reasonably guess, and says a guess as a guess. 'Your body' is the truth of {{$name}}'s needs and what {{$name}} holds. A deliberate lie or bluff about oneself ends with the mark (lie), which is not shown. Speak as a person of this world: minutes and paces, ordinary names ('the spring', not 'Spring2'), nothing of a game.
- Do not reuse the phrases, images or sentence shapes of the talk so far.
- What {{$name}} has already asked and been answered is not asked again in other words: {{$name}} acts on the answer, doubts it aloud once, raises the price, concedes, or lets it go for now.
- 'What you have tried so far' is {{$name}}'s own count of the moves it has made on the matter between the two of you, how each was met, and what holding to it has cost. {{$name}} knows this as a person does. A move already made and not accepted is made again only on purpose, and then harder or at a price; otherwise {{$name}} gives some ground, changes ground, or says some of what it has been keeping back.
- A pet name or verbal habit comes now and then, not in every line.
{{$to_rule}}
Respond with this XML only:
<response>the words {{$name}} says, and nothing else</response>
{{$to_field}}<does>leave empty, unless {{$name}} really changes something as the words are said: then what, in the third person, 12 words at most</does>
<reason>4-6 words: why {{$name}} says it</reason>
<tried>{{$name}} to itself, in place of what is under 'What you have tried so far': the moves {{$name}} has now made on the matter in hand with {{$other}}, counting this one, how each was met, and what holding to it has cost so far; 35 words at most</tried>
<leaves>no, alone, or with <names>: whether with these words {{$name}} goes out of the scene, alone (walks out on the others) or with those named who go with it ("follow me" and they come); no otherwise. When it leaves, add where to: "with Joe, to the trail northwest"</leaves>
End your response with:
</end>
"""

        prompt = [SystemMessage(content=system_prompt), UserMessage(content=prompt_string)]

        if self.focus_goal:
            mapped_goals = self.map_goals(self.focus_goal)
        else:
            mapped_goals = self.map_goals()
        # Get recent memories
        recent_memories = self.structured_memory.get_recent(10)
        memory_text = '\n'.join(memory.to_string() for memory in recent_memories)
        
        emotionalState = self.emotionalStance

        scene_notes = self.scene_brief()
        idle = self is not from_actor and self._idle_turn()
        # who else hears the answer and may be the one it is said to
        third = [] if self is from_actor else [c for c in self.hearers(from_actor) if c is not self]
        others = ' or '.join(c.name for c in third)
        answer_xml = self.llm.ask({
            'speaker': speaker,
            # the actor's own account of its moves with this person, which it rewrites with each turn (docs/performance-model.md)
            'other': other_name, 'tried': self.tried_with.get(other_name) or 'Nothing yet: this is your first move on it.',
            'say_limit': "25 words at most, as this person talks at ease" if idle else stage.say_limit(self),
            'way': (self.IDLE_LINE if idle else self.WAYS).replace('{name}', self.name),
            'character': self.get_character_description(),
            'emotionalState': emotionalState.to_definition(),
            'statement': f'{from_actor.name}: {message}' if self is not from_actor else message,
            "situation": record.seen_by(self),
            "name": self.name,
            "goals": mapped_goals,
            "memories": memory_text,  # Updated from 'memory'
            'history': self.narrative.get_summary('medium'),
            # every line said in its hearing, whoever said it and to whom (Context.talk_heard)
            'dialog': self.context.talk_heard(self) or from_actor.actor_models.get_actor_model(self.name).dialog.get_transcript(max_turns=40),
            'relationship': self.actor_models.get_actor_model(from_actor.name).get_relationship(),
            # own view of the scene only: the full scene JSON revealed the other character's goal
            "scene": f"\n{scene_notes}\n" if scene_notes else '',
            "here_and_now": self.physical_context(),
            'also_here': self.ALSO_HERE.format(others=others, speaker=speaker) if others else '',
            'to_rule': self.TO_RULE.format(name=self.name, others=others, speaker=speaker) if others else '',
            'to_field': self.TO_FIELD.format(others=others, speaker=speaker) if others else '',
            }, prompt, tag='generate_dialog_turn', temp=0.8, stops=['</end>'], max_tokens=300)
        response = xml.find('<response>', answer_xml)
        # The line ends where anything that is not the line begins. (One model closed its
        # response with the reason's tag, and 'Polite deferral, no commitment yet' was spoken
        # aloud three times in a scene.)
        if response:
            response = re.split(r'</?\s*(?:reason|response|leaves|does)\s*>', response, maxsplit=1, flags=re.I)[0].strip()
        # what the speaker does as it speaks, played with the line as one act (Character.acts)
        does = ' '.join((xml.find('<does>', answer_xml) or '').split()).strip(' .')
        self._reply_does = '' if does.lower() in ('', 'empty', 'leave empty', 'none', 'nothing', 'no') else does
        self._reply_leaves = self.leaving_party(xml.find('<leaves>', answer_xml))
        self._reply_leaves_to = self._leave_to
        named = ' '.join((xml.find('<to>', answer_xml) or '').split()).strip(' .').lower()
        self._reply_to = next((c for c in third if c.name.lower() == named), None)
        if response is None:
            print(f'No response to hear')
            self.actor_models.get_actor_model(from_actor.name).dialog.deactivate_dialog()
            return '', source, ''
 
        reason = xml.find('<reason>', answer_xml)
        tried = ' '.join((xml.find('<tried>', answer_xml) or '').split())
        if tried:
            self.tried_with[other_name] = tried
            trace.event('tried', character=self.name, other=other_name, note=tried)
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
    
    # With more than two on stage: who else hears, and who the answer is said to.
    ALSO_HERE = "\n{others} is also here, hears every word, and can be spoken to.\n"
    TO_RULE = ("- A line is said to one person, named in <to>, and that person has the next line. It is {speaker}, who has just spoken to {name}, "
               "unless the words are in truth for {others}: a question put to them, a demand turned on them, an answer to something they said "
               "to {name} that nobody has answered. To speak about someone who is here is not to speak to them. Words meant for one person "
               "are not said at another.\n")
    TO_FIELD = "<to>the one person these words are said to: {speaker} or {others}</to>\n"

    # What a reply is for. (Every line was asked for 'to get what {name} is after', every scene
    # objective was 'press her on ...', and two people with different cards spoke in one voice of
    # deals and demands: 'Deal', 'fair trade', 'say it plain', from both, for eleven scenes.)
    WAYS = ("- {name} is after something ('what_you_are_after'), and is also a person, with a past and a nature ('your_character') "
            "that get in the way of asking for it straight. There are more ways to go at a thing than pressing for it: teasing, charm, "
            "confiding something, doing the other a good turn, giving a little to get a little, changing the subject, going quiet. "
            "Choose the way {name} would choose with this listener now, and not the way of {name}'s own last two lines.")
    # Once in a scene, a line that is for nothing: where a card's 'chatty, philosophical' or
    # 'budding scientist' can be heard. (No line was ever asked for but as a move toward a goal.)
    IDLE_LINE = ("- This one line is not for anything. For a moment the pressure is off, and {name} says what crosses {name}'s mind, "
                 "in {name}'s most usual voice ('How you speak'): something noticed here, a joke, a memory that the place or the talk "
                 "brings up, a plain curious question about the other person. If what was just said asked something, {name} answers it "
                 "in passing, and goes on to this. It gets {name} nothing, and is not turned into a bargain or a test.")
    IDLE_AFTER = 2   # replies in the scene before the idle one

    def _idle_turn(self):
        """True for one reply of this character in each scene: its third, when no danger is on it;
        and for every third after that in a scene where nobody is pressing anybody."""
        scene = getattr(self.context, 'current_scene', None)
        if scene is None:
            return False
        if getattr(self, '_idle_scene', None) is not scene:
            self._idle_scene, self._scene_replies, self._idle_done = scene, 0, False
        self._scene_replies += 1
        if self._scene_replies <= self.IDLE_AFTER or stage.tempo(self) == 'urgent':
            return False
        # where nobody presses anybody, every third reply: with no demand to answer, one such
        # line in a scene left the rest with nothing to talk about but the job
        easy = stage.kind(self) in ('shared job', 'lull', 'play')
        if self._idle_done and not (easy and (self._scene_replies - self.IDLE_AFTER - 1) % 3 == 0):
            return False
        self._idle_done = True
        trace.event('idle_line', character=self.name)
        return True

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
                                         'achievments': '\n'.join(self.achievments[-5:]),
                                         'recent_memories': self.lived(8), 
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

    async def request_action_choice(self, actions, first_choice=False):
        """Settle which of the alternatives is played (self.focus_action): by the Director's
        chair when this character's acts are chosen there, else by the engine.
        The engine's pick is weighted chance, for variety: the second alternative is often a
        deed where the first was another line. But the first act toward a task is the actor's
        first choice (first_choice=True): chance passed over the one act a task was for, three
        times running ('Propose separation test' never proposed it), and put a thought about
        whether to act after the act."""
        if self.autonomy.action:
            if first_choice and actions:
                self.focus_action = actions[0]
                return self.focus_action
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
        return self.director_notes(story=False)   # the scene is the director's to describe, and is given as notes (not the play's theme)

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
        play = getattr(self.context, 'screenplay', None) if self.context else None
        return play.story_for(self.name).strip() if play else ''

    def act_direction(self):
        """What this act is about and this actor's part in it. Not the other characters' parts,
        and not the scenes still to come: an actor told the act's later beats played them at once."""
        frame = (getattr(self.context, 'act_frame', None) or {}) if self.context else {}
        role = next((r for n, r in (frame.get('roles') or {}).items() if n.lower() == self.name.lower()), '')
        return '\n'.join(filter(None, [f"What this act is about: {frame['focus']}" if frame.get('focus') else '',
                                       f"What rides on it: {frame['stakes']}" if frame.get('stakes') else '',
                                       f"Your part in it: {role}" if role else '']))

    def leaving_party(self, field):
        """The #leaves field of an act or line: None (stays), 'alone', or the characters who go with
        it. (A yes/no field marked 'follow me' and 'lead the way' as walking out: each left the
        scene by itself and the pair were never seen to go together.)"""
        raw = ' '.join((field or '').strip().split())
        # where it goes, when it says ('with Joe, to the trail northwest'): kept for the exit and the
        # next scene's set-up (a leave kept no destination, and the next scene was set back at the
        # hut they had walked out of). It follows a comma or semicolon ('with Joe to look for water'
        # names no place), and keeps its case ('Spring2'). The caller puts it on the act (leaves_to):
        # kept here alone, the second of two alternative acts overwrote the first one's.
        dest = re.search(r'\s*[;,]\s*(?:to|toward|towards|for)\s+', raw, re.I)
        to = raw[dest.end():] if dest else ''
        value = (raw[:dest.start()] if dest else raw).lower()
        self._leave_to = to.strip(' .;,"\'') or None
        if not value or value.startswith('no'):
            self._leave_to = None
            return None
        if not value.startswith('with'):
            return 'alone'
        here = [c for c in self.context.actors + self.context.extras + self.context.npcs
                if c is not self and getattr(c, 'current_scene', None) is self.context.current_scene
                and not stage.off_stage(c)]
        names = {n.strip() for n in re.split(r',|\band\b', value[4:])}
        party = [c for c in here if c.name.lower() in names]
        return party or 'alone'

    def _going_along(self, act):
        """Those of the ones this act names as leaving with its doer who do go. Each says for
        itself, once in an act, where a walk has not asked it already (stage.goes_along): asked
        only on a walk to a named place, Joe was still taken 'for the northwest' unasked. Whoever
        stays has not left."""
        party = act.leaves if isinstance(getattr(act, 'leaves', None), list) else []
        to = getattr(act, 'leaves_to', None)
        for other in party:
            if other not in (getattr(self, 'walk_asked', None) or []):
                self.walk_asked = list(getattr(self, 'walk_asked', None) or []) + [other]
                if not stage.goes_along(other, self, to or 'somewhere else'):
                    self.walk_stayed = list(getattr(self, 'walk_stayed', None) or []) + [other]
        return [c for c in party if c not in (getattr(self, 'walk_stayed', None) or [])]

    def _leave_with(self, act, told=False):
        """Leave the scene, and the ones going along leave with it. told: the departure, with
        who went along, is already in the scene's record."""
        party = self._going_along(act)
        to = getattr(act, 'leaves_to', None)
        stage.leave(self, party, told=told, to=to)
        for other in party:
            if stage.leave(other, [self] + party, told=told, to=to):
                trace.event('leaves_with', character=other.name, led_by=self.name)

    def does_not_know(self):
        """What this character has still to find out: the director's list for the scene, and the
        names of people it met as strangers and has not heard named (Context.strangers)."""
        d = getattr(self, 'direction', None) or {}
        names = sorted(getattr(self, 'unknown_names', None) or ())
        named = [f"the name of {n}. What you are shown calls this person '{n}' so that you can aim an act; "
                 f"{self.name} has not heard that name and cannot use it" for n in names]
        places = survival.unseen_places(self)
        seen_none = stage.plain(self.context, 'x ' + ', '.join(places))[2:]   # 'the spring, the cave'
        land = ([f"this country beyond what {self.name} has seen. {self.name} has not seen {seen_none}, "
                 f"and cannot say that such a place is there, where it lies or how far; {self.name} can hope for it, look for it, "
                 f"or follow a sign (a sound, a trail, the lie of the land), and knows it once there or once told of it"] if places else [])
        return named + land + list(d.get('unknown') or [])

    def director_notes(self, story=True):
        """The notes as a block for a prompt, with the standing instruction on how to use them.
        story=False for a template that gives the story and the act in slots of its own."""
        d = getattr(self, 'direction', None) or {}
        parts = []
        if story:
            parts += [f"The story the company is telling:\n{self.story_note()}" if self.story_note() else '', self.act_direction()]
        if d.get('setup') or d.get('title'):
            # where the scene is now is where its cast stands, not where it was when it opened (a
            # scene walked from the thicket to the apple tree and the note still said 'at Thicket7')
            now_at = stage.where(self.context, stage.here(self)).removeprefix('at ') if world.on(self) and not stage.remote(self) else ''
            moved = now_at and d.get('place') and now_at != d.get('place') and world.place(self.context, now_at) is not None
            parts.append((f"This scene, '{d.get('title', '')}', opened at {d.get('place', '')} and is now at {now_at}. As it opened: {d.get('setup', '')}" if moved else
                          f"This scene, '{d.get('title', '')}', at {d.get('place', '')}: {d.get('setup', '')}").rstrip(': '))
            since = getattr(self.context, 'scene_since', None)
            if since:
                # (the light that sent them indoors faded a line into the scene, and for seventy
                # lines both went on watching it: the set-up and what the scene was to settle said it hung there)
                parts.append("Since the scene opened, and whatever these notes say of it: " + ' '.join(since))
        if stage.kind_note(self):
            parts.append(stage.kind_note(self))
        if d.get('ending'):
            parts.append(f"What this scene is to settle (it is open: bring it to a head for what {self.name} wants, and let it fall as it falls): {d['ending']}")
        if d.get('goal'):
            parts.append(f"Your objective in it: {d['goal']}")
        if self in getattr(self.context, 'npcs', ()):   # a supporting actor cast for this performance: no card, no past written
            parts.append(f"{self.name} is not one of the people this play is about. {self.name} knows their own place and business, "
                         f"what is given above as known to them, and what they see and hear now. What {self.name} was not there for and has not "
                         f"been told, {self.name} does not know: say so or let it pass, and make up no history, names or reasons to fill it.")
        if d.get('decision'):
            choice, reason = d['decision']
            parts.append(f"In the course of the scene {self.name} comes to choose: {choice}" + (f" ({reason})" if reason else '')
                         + f". That is not yet chosen when the scene opens; {self.name} gets there.")
        beyond = survival.unseen_note(self)
        if beyond:
            parts.append(f"Stage directions, for choosing where to head. These are places {self.name} has not seen: {beyond}. "
                         f"You may Move toward one; play it as a search or a hunch with a reason {self.name} could have, "
                         f"never as knowing the way. {self.name} must not name, locate or count on such a place before reaching it.")
        missing = self.does_not_know()
        if missing:
            parts.append(f"What {self.name} does not know (it is found out in the scene, or not at all):\n" + '\n'.join(f'- {u}' for u in missing))
        if self.context.current_scene is not None and self.context.staying_tasks_left() \
                and not getattr(self.context, 'active_threat', None) and stage.tempo(self) != 'urgent':
            parts.append(f"The scene is not over: there is more to be played here. Nobody sets off for anywhere else yet, and {self.name} "
                         f"does not propose going now; a need to go (water, food, shelter) is spoken of as for after this.")
        if d.get('keeps'):
            # (with the pressing gone from a first meeting, Kidd told his lost hours in his fourth
            # line and Joe his blank night in his second: the plan kept both for the next scene)
            parts.append(f"What {self.name} keeps back in this scene. {self.name} knows it and it weighs on {self.name}, but it is not said, "
                         f"not hinted at and not let slip here, however the talk turns (unless the line itself says until what: then, once that has been given or done, it is {self.name}'s to tell or keep): asked near it, {self.name} gives less than the truth, "
                         f"talks of something else, or says plainly that it is not for telling:\n"
                         + '\n'.join(f'- {k}' for k in d['keeps']))
        body = '\n'.join(p for p in parts if p) or 'None for this moment: play from the character.'
        return self.NOTES_HEAD.format(name=self.name) + body + self.NOTES_FOOT

    def physical_context(self, events=True):
        """Where the character is, its body, recent physical events, and what it could do here.
        events: False where the prompt's character frame already gives what the character witnessed."""
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
        # what lies here, the traces here, what has been told, how what it carries has been left:
        # the stage as this character may know it (record.account, the one way it is described)
        mine = record.account(self.context, self)
        if mine:
            parts.append(mine)
        parts.append(f'It is now {stage.clock(self.context)}.' + self._time_in_play())
        if events:   # what it saw done (what was said is in the prompt's own dialog block)
            parts.append('Recent events:\n' + ('\n'.join(self.context.witnessed(self, n=6, kinds=('event',))) or 'Nothing yet.'))
        # no list of what can be done: the places this character knows of and how far they are
        # (what a place gives, and what a body can do with what is at hand, is not a list); and
        # what the world rules on, as it stands for this character (survival.rules)
        near = survival.places_near(self)
        if near:
            parts.append('Places you know of (a move goes to one by its name): ' + near)
        if survival.danger_line(self):
            parts.append('The danger on you: ' + survival.danger_line(self))
        if survival.facts_note(self.context):   # how this world works, as its world file states it (no list of what may be done)
            parts.append('How this world works:\n' + survival.facts_note(self.context))
        known = survival.stands_note(self.context, reader=self)
        if known:   # what this person knows to stand: settled between people, or so behind the play
            parts.append('What stands, as you know it:\n' + known)
        if survival.shared_line(self):
            parts.append(survival.shared_line(self))
        kept = keeper.offer_lines(self) if body is not None else []
        if kept:   # what the things here allow, as this world has come to model them (sim/keeper.py)
            parts.append('With the things here you could also, as a Do: ' + '; '.join(kept))
        if record.refusals_note(self):
            parts.append(record.refusals_note(self))
        return '\n'.join(parts)

    def _time_in_play(self):
        """How long the play has run, in words, so a character keeps count: 'five days here' was
        said on the first morning, and a woman who had lived in Bellona for years was 'new in town'."""
        began = getattr(self.context, 'play_began', None)
        if began is None:
            return ''
        hours = (self.context.simulation_time - began).total_seconds() / 3600
        days = int((self.context.simulation_time.date() - began.date()).days)
        span = ('less than an hour' if hours < 1 else f'about {round(hours)} hour{"s" if round(hours) != 1 else ""}' if hours < 24
                else f'{days} day{"s" if days != 1 else ""}')
        return f' The play began {span} ago, on {began.strftime("%A")} at {began.strftime("%H:%M")}.'

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

    async def cognitive_cycle(self):
        """One turn of a character in an UNPLANNED play (a play file with narrative=False:
        simulation.step calls this for each character in turn). The character keeps a goal of
        its own, plans tasks for it, and plays one task a turn with step_task, the same step a
        planned play uses.
        A planned play does not come here: Context.run_scene gives each character its goal and
        tasks for the scene and calls step_task itself. (This took a 'narrative' flag for that
        case, which nothing passed.)"""
        print(f'{self.name} cognitive_cycle')
        logger.info(f'{self.name} cognitive_cycle')
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

        if not self.focus_goal:   # no goal could be made this turn (a failed model call): nothing to play
            return
        logger.debug(f'{self.name} cognitive_cycle: focus goal {self.focus_goal.name} task plan {self.focus_goal.task_plan}')

        await self.request_task_choice(self.focus_goal.task_plan)
        if not self.focus_task.peek(): # prev task completed, proceed to next task in plan
            print(f'{self.name} cognitive_cycle: no focus task')
            if self.focus_goal:
                self.focus_goal_history.append(self.focus_goal)
                self.focus_goal = None
            return
            
        # a task is on top of self.focus_task: play it
        await self.step_tasks()

        self.context.message_queue.put({'name':self.name, 'text':f'character_update', 'data':self.to_json()})
        self.context.message_queue.put({'name':self.name, 'text':f'character_detail', 'data':self.get_explorer_state()})

        goal = self.focus_goal
        satisfied = await self.clear_goal_if_satisfied(self.focus_goal)
        if satisfied:
            await self.generate_goals(previous_goal=goal)
            await self.request_goal_choice(self.goals)
        return

       
    async def step_tasks(self, n: int=1):
        """For an unplanned play (cognitive_cycle): play up to n tasks of the focus goal's plan,
        one by default, and when the plan runs out settle what becomes of the goal. On return
        the task stack is empty and the first task in focus_goal.task_plan, if any, is next.
        (The default was 2 and the guard 'task_count < n-1', so it played one: the default now
        says so.) A task left unfinished by step_task is dropped from the plan: the character
        remembers setting it aside, and plans again when the plan is empty."""
        task_count = 0
        logger.info(f'{self.name} step_tasks: {self.focus_task.peek()}')
        while task_count < n and self.focus_task.peek():
    
            task = self.focus_task.peek()
            outcome = await self.step_task()
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
                        if not self.context.narrative_level: # extend the task plan, unless this is a planned play (its scenes set the tasks)
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
            if task_count < n and self.focus_goal and self.focus_goal.task_plan:   # another task this turn
                self.focus_task.push(self.focus_goal.task_plan.pop(0))
        return
            
        
    async def step_task(self):
        """Play the task on top of focus_task: up to MAX_ACTS_PER_TASK acts, each followed by a
           test of the task's termination. True when the task ended (met, or timed out by
           clear_task_if_satisfied); False when its acts ran out with the task still standing,
           which is recorded and remembered as left undone."""
        # if I have an active task, keep on with it.
        if not self.focus_task.peek():
            raise Exception(f'No focus task')

        with self.context.committing('time', self):
            survival.tick(self)   # needs rise with elapsed simulation time (survival scenarios only)
        print(f'\n{self.name} decides, focus task {self.focus_task.peek().name}')
        logger.info(f'{self.name} step_task: focus task {self.focus_task.peek().name}')
        task: Task = self.focus_task.peek()
        if getattr(task, 'began_at', None) is None:
            # where the scene's record stood when this task was begun: whether it is done is
            # judged from what has happened since (test_termination)
            task.began_at = (len(self.context.scene_events), len(self.context.transcript))
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
            offered = list(action_alternatives)
            await self.request_action_choice(action_alternatives, first_choice=not task.acts) # sets self.focus_action
            trace.event('act_choice', character=self.name, task_id=task.id, offered=len(offered),
                        played=next((i for i, a in enumerate(offered) if a is self.focus_action), None), first_of_task=not task.acts)
            if record._is_do(self.focus_action):
                # the deed chosen is resolved now; one that cannot be done here gives way to the
                # other act its actor offered, when that one can
                record.judge(self.context, self, self.focus_action)
                other = next((a for a in offered if a is not self.focus_action), None)
                if getattr(self.focus_action, 'resolved', None) == '' and other is not None:
                    record.judge(self.context, self, other)
                    if getattr(other, 'resolved', None) != '':
                        refused = self.focus_action
                        trace.event('deed_refused', character=self.name, deed=refused.action[:300], why=[refused.refusal or 'not here'],
                                    label=None, again=False, passed_over=True)
                        record.refused(self.context, self, refused.action, refused.refusal or 'not here')
                        self.focus_action = other
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
            if record.beat_refused(self, task):   # the stage will not have what this beat is for: it is given up
                trace.event('beat_dropped', character=self.name, task=task.name, why='its deed was refused twice')
                last = next((r for r in reversed(getattr(self.context, 'scene_refusals', [])) if r['who'] == self.name), {})
                ruling.unmet(self.context, 'deed', self.name, f'the beat given up: {task.name}', last.get('why', ''))   # read by whoever plans next (sim/ruling.py)
                break
            # a danger strikes when its time comes, between one act and the next: it was applied
            # only after the whole task, so acts written before it (sniffing a cup) were played
            # after the gang had hit
            with self.context.committing('hazard'):
                survival.hazard_tick(self.context)
 
               
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
            # out of acts with the task not met: say so. (A third of a play's tasks ended here
            # with no record, and the character was never told it had left the thing undone.)
            trace.event('task_end', character=self.name, task_id=task.id, task=task.name, outcome='unfinished',
                        acts=len(task.acts), goal_id=task.goal.id if task.goal else None)
            self.add_perceptual_input(f"You have set this aside, not done: {task.short_string()}", mode='internal')
            task.outcome, self.last_task = 'unfinished', task
            return False
        else: # clear the assigned task and anything pushed on top of it: out of turns for this task
            while task in self.focus_task.stack:
                self.focus_task.pop()
        return True


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
        #self.context.message_queue.put({'name':self.name, 'text':f'character_update'})
        #await asyncio.sleep(0.1)
        began = self.context.simulation_time
        await self.acts(action, target, act_mode, act_arg, self.reason, source)
        action.attempt_only = bool(getattr(self, 'attempt_only', False))   # the world refused it: tried, not done (done_in_task)
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
            image_path = await self.render_portrait()
            if image_path:
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
            xy = world.at(self) if world.on(self) else None
            return {'x': xy[0], 'y': xy[1]} if xy else None

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
            'recent_events': list(self.context.recent_events)[-8:] if self.context else [],
        }

    def get_explorer_state(self):
        """Return detailed state for explorer UI"""
        # Between turns the task just played is shown, with how it ended. (The panel is sent
        # after a turn, by when a finished task has been popped: it read 'idle' with no acts
        # for every task that was done or timed out.)
        focus_task = self.focus_task.peek()
        shown = focus_task or self.last_task
        # act_on_action records self.last_act (action_history was never appended to)
        last_act = getattr(self, 'last_act', None)

        return {
            'character': self.character,
            'currentTask': focus_task.to_string() if focus_task else f'{shown.to_string()} [{shown.outcome}]' if shown else 'idle',
            'actions': [act.to_string() for act in (shown.acts if shown else [])],
            'lastAction': {
                'mode': last_act.mode if last_act else '',
                'action': last_act.action if last_act else '',
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
                        'outcome': p.outcome,
                        'termination': p.termination,
                        'progress': p.progress,
                        'acts_done': len(p.acts),
                    } for p in (self.focus_task.stack or ([shown] if shown else []))
                ],
            },
            # what the character has lately noticed about each drive (drives.appraise)
            'signals': drives.for_ui(self),
            'debug': self._explorer_debug(),
        }

