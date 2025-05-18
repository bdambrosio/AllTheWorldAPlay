from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Act, Character, Goal, Task
    from sim.context import Context  # Only imported during type checking

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Messages import UserMessage, SystemMessage
from sim.cognitive.EmotionalStance import EmotionalStance
from utils import llm_api
import utils.xml_utils as xml
import sim.map as map
import utils.hash_utils as hash_utils
import utils.choice as choice   
#from sim.cognitive.DialogManager import Dialog

def ask (character:Character, system_prompt:str=None, prefix:str=None, suffix:str=None, addl_bindings:dict={}, max_tokens:int=100, log:bool=False, tag:str=''):

    prompt = []
    if system_prompt:
        prompt.append(SystemMessage(content=system_prompt))
    if prefix:
        prompt.append(UserMessage(content=prefix+"""\nYou are {{$name}}."""))
    
    prompt.append(UserMessage(content="""    

#Character
{{$character}}
##
                     
#Drives
{{$drives}}
##

#Character Narrative
{{$narrative}}
##

Your current situation is:

#Situation 
{{$situation}}
##

#Surroundings 
{{$surroundings}}
##
                      
#Known other actors 
{{$actors}}
##

#Known resources
{{$resources}}
##

#Recently achieved goals
{{$goal_history}}
##

#Recent memories
{{$memories}}
##

#Recent history
{{$history}}
##

#Previous act if any
{{$lastAct}}
##

#Observed result of that act
{{$lastActResult}}
##

#Emotional stance
{{$emotional_stance}}
##

"""))
    if suffix:
        prompt.append(UserMessage(content=suffix+'\n\nend your response with </end>'))
    
    ranked_signalClusters = character.driveSignalManager.get_scored_clusters()
    focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order

    recent_memories = character.structured_memory.get_recent(16)
    memory_text = '\n'.join(memory.text for memory in recent_memories)
    
    emotionalState = character.emotionalStance        

    task = addl_bindings.get("task", None)
    if not task:
        task = character.focus_task.peek()
    
    if task and task.acts and len(task.acts) > 0:
        lastAct = task.acts[-1].mode + ' ' + task.acts[-1].action
        lastActResult = task.result
    else:
        lastAct = character.focus_action if character.focus_action else ''
        if hasattr(character, 'lastActResult'):
            lastActResult = character.lastActResult if character.lastActResult else ''
        else:
            lastActResult = ''

    bindings = {"name":character.name,
                "character":character.get_character_description(),
                "drives":'\n'.join([f'{d.id}: {d.text}; activation: {d.activation:.2f}' for d in character.drives]),
                "narrative":character.narrative.get_summary('medium'),
                "signals": "\n".join([sc.to_string() for sc in focus_signalClusters]),
                "surroundings":character.look_percept,
                "time": character.context.simulation_time.isoformat(),
                "situation":character.context.current_state if character.context else "",
                "actors":character.actor_models.format_relationships(include_transcript=True),
                "resources":character.resource_models.to_string(),
                "goal_history":'\n'.join([f'{g.name} - {g.description}' for g in character.goal_history]),
                "memories": memory_text,
                "history": "", #character.history,
                "lastAct": lastAct,
                "lastActResult": lastActResult,
                "emotional_stance":emotionalState.to_definition(),
                "lastAct":lastAct,
                "lastActResult":lastActResult}
    for key, value in addl_bindings.items():
        bindings[key]=value
    
    response = character.llm.ask(bindings, prompt, tag=tag, max_tokens=max_tokens, stops=['</end>'], log=log)
    return response

