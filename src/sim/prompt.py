from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Act, Character, Goal, Task
    from sim.context import Context  # Only imported during type checking

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Messages import UserMessage, SystemMessage
from sim.cognitive.driveSignal import Drive, DriveSignalManager, SignalCluster
from sim.memory.consolidation import MemoryConsolidator
from sim.memory.core import MemoryEntry, NarrativeSummary, StructuredMemory
from sim.memory.core import MemoryRetrieval
from src.sim.cognitive.EmotionalStance import EmotionalStance
from utils import llm_api
from utils.Messages import UserMessage, SystemMessage
import utils.xml_utils as xml
import sim.map as map
import utils.hash_utils as hash_utils
import utils.choice as choice   
from sim.cognitive.DialogManager import Dialog
from sim.cognitive import knownActor
from sim.cognitive import perceptualState
from sim.cognitive.perceptualState import PerceptualInput, PerceptualState, SensoryMode
from sim.cognitive.knownActor import KnownActor, KnownActorManager


def ask (character:Character, mission:str, suffix:str, addl_bindings:dict, max_tokens:int=100):

    prompt = [UserMessage(content="""You are {{$name}}.
Please {{$mission}}

Your current situation is:

Character
{{$character}}
##
                     
Drives
{{$drives}}
##

Character Narrative
{{$narrative}}
##

#Surroundings {{$surroundings}}
##
                      
#Situation {{$situation}}

known other actors 
{{$actors}}
##

recently achieved goals are:
{{$goal_history}}
##

recent memories:
{{$memories}}
##

Recent history:
{{$history}}
##

previous act if any:
{{$lastAct}}
##

observed result of that was:
{{$lastActResult}}
##

emotional stance:
{{$emotional_stance}}
##

""")]
    full_prompt = prompt.append(UserMessage(content=suffix+'\n\nend your response with <end/>'))
    
    ranked_signalClusters = character.driveSignalManager.get_scored_clusters()
    focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order

    recent_memories = character.structured_memory.get_recent(8)
    memory_text = '\n'.join(memory.text for memory in recent_memories)
    
    emotionalState = EmotionalStance.from_signalClusters(character.driveSignalManager.clusters, character)        

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
                "mission":mission,
                "character":character.character,
                "drives":"\n".join([d.text for d in character.drives]),
                "narrative":character.narrative.get_summary('medium'),
                "signals": "\n".join([sc.to_string() for sc in focus_signalClusters]),
                "surroundings":character.look_percept,
                "time": character.context.simulation_time.isoformat(),
                "situation":character.context.current_state if character.context else "",
                "actors":character.actor_models.format_relationships(include_transcript=False),
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
    
    response = character.llm.ask(bindings, prompt, max_tokens=max_tokens, stops=['<end/>'])
    return response

if __name__ == "__main__":
    from sim.context import Context
    from sim.agh import Character
    from sim.memory.core import NarrativeSummary
    from sim.memory.core import MemoryEntry
    from sim.memory.core import MemoryRetrieval
    from sim.memory.core import StructuredMemory
    from sim.cognitive.DialogManager import Dialog
    from sim.cognitive import knownActor
    from sim.cognitive import perceptualState
    from sim.cognitive.perceptualState import PerceptualInput, PerceptualState, SensoryMode
    from sim.cognitive.knownActor import KnownActor, KnownActorManager
    