from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Act, Character, Goal, Task
    from sim.context import Context  # Only imported during type checking
    from sim.narrativeCharacter import NarrativeCharacter

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Messages import UserMessage, SystemMessage
from sim.cognitive.EmotionalStance import EmotionalStance
from utils import llm_api
import utils.xml_utils as xml
import sim.map as map
import sim.survival as survival
import sim.record as record
from sim.cognitive import drives
import utils.hash_utils as hash_utils
import utils.choice as choice   
#from sim.cognitive.DialogManager import Dialog

# The frame every prompt of a character shares.
#
# A character's LLM is asked two different things. As PLAYWRIGHT it helps write the plan (its
# arc, the acts): it then sees the plan. Everywhere else it is the ACTOR playing the character,
# or the character's own mind, and this frame is the character: who it is, what it wants, feels,
# remembers and can perceive. Nothing of the plan belongs in it. (The frame used to open with
# the central dramatic question and to give the play's Situation text, which carries the scene's
# title and set-up; the plan was also stored as the character's memories. So the first thing a
# woman said to a stranger who had not spoken was 'You really don't remember anything?')
# A call that performs or plans a scene adds the director's notes in its own template
# (Character.director_notes), marked as outside the play.
PLAYWRIGHT_FRAME = """    

#The central dramatic question driving the play
{{$central_narrative}}
##

#Your character (personality and role in the play)
{{$character}}
##

#Your drives, most pressing first (what you want; each is pressing, present or quiet right now, with what you have just noticed about it)
{{$drives}}
##

#Emotional stance
{{$emotional_stance}}
##

#Before this scene: your character's state and recent activities, as of the scene's start (what has happened since is below)
{{$narrative}}
##

The current situation in the play
#Situation 
{{$situation}}
##

#Your immediate surroundings
{{$surroundings}}
##
                      
#Actor genders for pronoun matching
{{$actor_genders}}
##

#Known other actors 
{{$actors}}
##

#Known resources
{{$resources}}
##

#Your recently achieved goals and their outcomes
{{$goal_history}}
##

#Your recent memories
{{$memories}}
##

#Previous act if any
{{$lastAct}}
##

#Observed result of that act
{{$lastActResult}}
##


"""

CHARACTER_FRAME = """    

######## {{$name}}, THE CHARACTER ########
Everything down to the closing line of this block is {{$name}}: who {{$name}} is, and what {{$name}} wants, feels, remembers and can perceive. {{$name}} speaks and acts from this, and from nothing else.

#Who you are
{{$character}}
##

#Your drives, most pressing first (what you want; each is pressing, present or quiet right now, with what you have just noticed about it)
{{$drives}}
##

#Emotional stance
{{$emotional_stance}}
##

#Before this scene: your state and recent activities, as of the scene's start (what has happened since is below)
{{$narrative}}
##

#Where things stand, as you can see for yourself
{{$situation}}
##

#What you have seen and heard lately, oldest first
{{$witnessed}}
##

#Your immediate surroundings
{{$surroundings}}
##

#The others, as you know them: what you can see of each, what has passed between you, what you make of them
{{$actors}}
##

#Places and things you know of
{{$resources}}
##

#Your recently achieved goals and their outcomes
{{$goal_history}}
##

#Your own recent thoughts and what you have made of things
{{$memories}}
##

#Previous act if any
{{$lastAct}}
##

#Observed result of that act
{{$lastActResult}}
##
######## END OF {{$name}}, THE CHARACTER ########


"""


def ask (character:Character, system_prompt:str=None, prefix:str=None, suffix:str=None, addl_bindings:dict={}, max_tokens:int=100, log:bool=True, tag:str='', view:str='character'):
    """view: 'playwright' for a call in which the character's LLM helps write the plan;
    'character' (the default) for every other call."""
    prompt = []
    if system_prompt:
        prompt.append(SystemMessage(content=system_prompt))
    if prefix:
        prompt.append(UserMessage(content=prefix+"""\nYou are {{$name}}."""))
    
    prompt.append(UserMessage(content=PLAYWRIGHT_FRAME if view == 'playwright' else CHARACTER_FRAME))
    if suffix:
        prompt.append(UserMessage(content=suffix+'\n\nend your response with </end>'))
    

    # memory holds what the record does not: thoughts, notices, task outcomes, what one was told
    recent_memories = character.structured_memory.get_recent(10)
    memory_text = '\n'.join(memory.to_string() for memory in recent_memories)
    
    emotionalState = character.emotionalStance        

    task = addl_bindings.get("task", None)
    if not task:
        task = character.focus_task.peek()
    
    if task and task.acts and len(task.acts) > 0:
        lastAct = task.acts[-1].mode + ' ' + task.acts[-1].action
        # what came of it: what the character took from its last act
        # (a Look that found something was followed by 'Observed result: None to report')
        lastActResult = getattr(character, 'lastActResult', '')
    else:
        lastAct = character.focus_action if character.focus_action else None
        if hasattr(character, 'lastActResult'):
            lastActResult = character.lastActResult if character.lastActResult else ''
        else:
            lastActResult = ''
    actor_genders = ''
    for actor in character.context.actors+character.context.extras:
        actor_genders += f'\t{actor.name}: {actor.gender}\n'

    # goal and task planning see what can physically be done here; the act prompt has it in
    # its own here-and-now block
    could_do = ''
    if 'here_and_now' not in addl_bindings:
        actions = survival.available_actions(character, part='known')
        could_do = f'\nYou could do here: {actions}' if actions else ''
    bindings = {"name":character.name,
                "character":character.get_character_description(),
                "drives": drives.drive_lines(character),
                "central_narrative": character.central_narrative if character.central_narrative else 'None yet established',
                "narrative":character.narrative.get_summary('medium'),
                # the act and dialog prompts give the surroundings in their own here-and-now block
                "surroundings":(character.look_percept if 'here_and_now' not in addl_bindings else 'see the here-and-now block below') + could_do,
                "time": character.context.simulation_time.isoformat(),
                "situation": (character.context.current_state if view == 'playwright' else record.seen_by(character)) if character.context else "",
                "actor_genders":actor_genders,
                # what passed between them is in 'witnessed' (the record), not repeated here
                "actors":character.actor_models.format_relationships(include_transcript=view == 'playwright'),
                "witnessed": character.context.witnessed_block(character) if character.context else '',
                "resources":character.resource_models.to_string(),
                "goal_history":'\n'.join([f'{g.name} - {g.description}\n\t{g.completion_statement}' for g in character.goal_history]) if character.goal_history else 'None to report',
                "memories": memory_text,
                "lastAct": lastAct.action if hasattr(lastAct, 'action') else lastAct,
                "lastActResult": lastActResult if lastActResult else 'None to report',
                "emotional_stance":emotionalState.to_definition()}
    for key, value in addl_bindings.items():
        bindings[key]=value
    
    response = character.llm.ask(bindings, prompt, tag=tag, max_tokens=max_tokens, stops=['</end>'], log=log)
    return response

