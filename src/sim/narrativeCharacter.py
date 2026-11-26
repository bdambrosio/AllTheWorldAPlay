from __future__ import annotations
import sim.survival as survival
from sim.cognitive import drives
import sim.stage as stage
import sim.record as record
from sim.trace import trace
import asyncio
import copy
from datetime import datetime
import logging
from typing import Optional, Dict, List, Any
import os, sys, re, traceback, requests, json

from sim.character_dataclasses import Act, Goal, Task, CentralNarrative, datetime_handler
from sim.cognitive.EmotionalStance import EmotionalStance
from utils.Messages import UserMessage
from utils import hash_utils, json_reply
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from enum import Enum
from typing import TYPE_CHECKING
from sim.prompt import ask as default_ask
from sim.agh import Character

if TYPE_CHECKING:
    from sim.memory.core import NarrativeSummary

logger = logging.getLogger('simulation_core')



class NarrativeCharacter(Character):
    def __init__(self, name, character_description= '', reference_description='', init_x=None, init_y=None, server_name='local'):
        super().__init__(name, character_description, reference_description=reference_description, init_x=init_x, init_y=init_y, server_name=server_name)
        # Narrative files
        self.play_file = None 
        self.map_file = None
        self.play_file_content = None
        self.map_file_content = None
        # Narrative
        self.act_wants = ''   # what it wants out of the act being planned (propose_act)
        self.current_act = None
        self.current_scene = None
        print(f'{self.name}: {self.__class__.__name__}')
        self.previous_proposal = None
        self.current_proposal = None
        self.round_number = 0 # central narrative negotiation round number
        self.central_narrative = None

    def validate_scene_time(self, scene) -> Optional[datetime]:
        """
        Validate and parse the scene time from ISO 8601 format.
        Returns a datetime object if valid, None if invalid or missing.
        """
        time_str = scene.get('time')
        if not time_str:
            return self.context.simulation_time
        time_str = time_str.strip().replace('x', '0')
        time_str = time_str.replace('00T', '01T') # make sure day is not 00
        try:
            # Parse ISO 8601 format
            scene_time = datetime.fromisoformat(time_str)
            return scene_time
        except (ValueError, TypeError):
            # Handle invalid format or type
            return self.context.simulation_time



    def reserialize_narrative_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the narrative JSON structure and returns (is_valid, error_message)
        """
        # Check top-level structure
        if not isinstance(json_data, dict):
            return False, "Root must be a JSON object"
        reserialized = copy.deepcopy(json_data)
        for act in reserialized["acts"]:
            if "scenes" not in act or not isinstance(act["scenes"], list):
                continue
            for scene in act["scenes"]:
                if isinstance(scene["time"], datetime):
                    scene["time"] = scene["time"].isoformat()
                elif isinstance(scene["time"], str):
                    scene["time"] = datetime.fromisoformat(scene["time"])
                else:
                    print(f'Invalid time format in scene {scene["scene_number"]}: {scene["time"]}')
                    scene["time"] = self.context.simulation_time
        return reserialized
    
    async def introduce_myself(self, play_file_content, map_file_content=None):

        system_prompt = """Imagine you are addressing the audience of a play. 
Create a short introduction (no more than 60 tokens) to your character, including your name, terse description, role in the play, drives, and most important relationships with other characters. 
Leave some mystery. This should be in the first person, you will speak to the audience.
{{$strangers}}
For example:
#####
Hi, I'm Maya, a brilliant artist. I live in a small coastal town with my partner Elijah.
But things may be changing....
#####
"""

        mission = """The overall cast of characters and setting are given below in Play.py and Map.py.

<Scenario>
{{$scenario}}
</Scenario>

<Map>
{{$map}}
</Map>


## 1.  INPUT FILES

You have been provided two Python source files above.

▶ play.py  
    • Contains Character declarations
    • At the end, a Context is defined:  
      `context.Context([CharA, CharB, ...], "world-blurb", scenario_module=<map>)`  

▶ map.py  
    • Defines Enum classes for terrain, infrastructure, resource, and property types.  
    • Contains `resource_rules`, `terrain_rules`, etc.  (You may cite these names when choosing locations.)

    After processing the above files and in light of the following information, generate a medium-term narrative arc for yourself.

"""
                            
        suffix="""
Respond only with your introduction.
Do not include any other explanatory, discursive, or formatting text.
End your response with </end>
"""

        introduction = default_ask(self, view='playwright', system_prompt=system_prompt, prefix = mission, suffix = suffix,
                                addl_bindings={"scenario": play_file_content, 
                                 "map": map_file_content if map_file_content else '',
                                 # a stranger is not named to the audience by someone who has not met them
                                 "strangers": (f"You have not met {', '.join(sorted(self.unknown_names))} and do not know their name: speak of a stranger, never by name."
                                               if getattr(self, 'unknown_names', None) else ''),
                                 "name": self.name,
                                 "start_time": self.context.simulation_time.isoformat()},
                                 max_tokens=200, tag='NarrativeCharacter.introduction')
        if introduction:
            self.context.message_queue.put({'name':self.name, 'text':introduction+'\n'})
            await asyncio.sleep(0)
    


    # The act level works as the scene level does: each character says what it wants, from what it knows, and the
    # director, who knows everything, writes the act (Context.integrate_narratives). Until 2026-11-26 each character
    # wrote a whole arc of acts before the play and a whole draft of each act, scenes and everyone's goals, and was
    # given everything to do it with: every other character's sketch with what they tell nobody, the director's facts,
    # the hidden things and what a writing said. The two drafts of Bellona's first act were the same three scenes at
    # the same places, and the goal 'see what Kidd hurries to pocket' was written for Lanya in Kidd's draft
    # (bellona-20261126-115753). Nothing here is shown what its character has not seen or been told.
    WANTS = """The next act of the play is about to be planned, and each character is asked what it is after.
Answer for {{$name}}, from what {{$name}} knows and nothing else: its own past and wants, what it has seen and been told so far, the people as it has found them, the places it knows. {{$name}} does not know what the others keep to themselves, what lies where it has not been, or what is going to happen."""
    WANTS_FORM = """

Say what {{$name}} wants out of the next stretch of the play: one to three wants, the most pressing first. Each is a want {{$name}} could own to now: something to get, learn, settle, keep or be rid of, and whom or what it has to do with. No scenes, nothing of what anyone else will do or answer, and nothing of how it comes out.

Respond in this form and nothing else:
#want <the want, twelve words at most> | <who or what it has to do with> | <why it presses now, ten words at most>
#means <what {{$name}} is ready to do or give for the first of them, fifteen words at most>
#keeps <what {{$name}} will not tell or give up while going after them, fifteen words at most; or the word nothing>
##
"""

    def propose_act(self, number):
        """What this character wants out of the act about to be planned, from its own view: the lines of its answer as
        one text for the director, or '' when it gave none."""
        reply = default_ask(self, prefix=self.WANTS, suffix=self.WANTS_FORM, addl_bindings={"name": self.name},
                            max_tokens=220, tag='NarrativeCharacter.act_wants') or ''
        lines = [' '.join(l.split()) for l in reply.split('\n') if l.strip().startswith(('#want', '#means', '#keeps'))]
        self.act_wants = '\n'.join(lines)
        trace.event('act_wants', character=self.name, act=number, wants=lines)
        return self.act_wants

    async def negotiate_central_narrative(self, round:int, play_file_content, map_file_content=None):
        """Negotiate the central dramatic question with other characters"""

        mission="""You are {{$name}} engaging in collaborative story development with other characters to establish the central dramatic question that will drive the upcoming improvisational play.
The dramatic question usually revolves around how one or more characters will resolve a conflict, either internal (among conflicting drives or desires) or external (between or among characters).
Rarely, conflict can be achieved through compromise. More often, it must be resolved through making a choice between starkly defined alternatives and living with the resulting gains and losses. """

        suffix = """
# Your Character Profile
{{$character_description}}

## Cast & Setting
**Other Characters:**
{{$other_characters_profiles}}

**Setting & Context:**
{{$setting_background}}

## Negotiation Round {{$round_number}} of 2

**Previous Proposals (if Round 2):**
{{$previous_proposals}}

**Current Proposals on the Table:**
{{$current_proposals}}

** Your Task
Propose or advocate for a central dramatic question that:
1. **Engages your drives** - Creates opportunities/obstacles relevant to your core motivations
2. **Involves other characters** - Requires meaningful interaction with at least 2-3 others present
3. **Fits the setting** - Makes sense given the current context and environment
4. **Creates genuine conflict** - Poses a question where the outcome genuinely matters and isn't predetermined
5. **Has clear stakes** - Everyone understands what's at risk if the question isn't resolved


** Strategic Considerations
- In Round 1: Be bold and advocate for your interests
- In Round 2: Consider compromise positions that serve multiple characters
- Remember: The question you choose will shape the entire narrative arc
- Think about whether you want to be protagonist, antagonist, or catalyst in this conflict

Respond using the following hash-formatted text, where each line consists of a # (hash mark) followed by a tag (text, no spaces), a single space, and the content.
For this request the tags are Question, Why, Reason, Others, Risks.
Close the hash-formatted text with ##  on a separate line, as shown below.
be careful to insert line breaks only where shown, separating a value from the next tag:

#Question central question or conflict terse (6-10 words)
#Why dramatic potential (6-8 words)
#Reason my character's stake in this question or conflict (6-10 words)
#Others roles others play in this central question or conflict (8-12 words)
#Risks weaknesses in this proposal(4-6 words)
##

"""


        response = default_ask(self, view='playwright', prefix=mission, suffix=suffix,
                              addl_bindings={"name": self.name, 
                               "character_description": self.character,
                               "other_characters_profiles": '\n'.join([char.character for char in self.context.actors if char.name != self.name]),
                               "setting_background": self.context.current_state,
                               "previous_proposals": '\n'.join([f'{char.name}: {char.previous_proposal.to_string() if char.previous_proposal else ""}' for char in self.context.actors if char.name != self.name]),
                               "current_proposals": '\n'.join([f'{char.name}: {char.current_proposal.to_string() if char.current_proposal else ""}' for char in self.context.actors if char.name != self.name]),
                               "round_number": round,
                               "play_file_content": play_file_content,
                               "map_file_content": map_file_content if map_file_content else ''},
                              max_tokens=200, tag='NarrativeCharacter.negotiate_central_narrative')
        central_narrative = CentralNarrative.parse_from_hash(response) if response else None
        if central_narrative is None and response:
            # parse_from_hash returns None on a malformed reply; .to_string() below crashed play load
            logger.warning(f'{self.name}: unparseable central narrative proposal in round {round}')
        if central_narrative is not None:
            self.previous_proposal = self.current_proposal
            self.current_proposal = central_narrative
            self.central_narrative = central_narrative.to_string() # store as string, since that is how global central narrative is stored
            #self.context.message_queue.put({'name':self.name, 'text':f'proposes {self.current_proposal.to_string()}\n'})
            await asyncio.sleep(0.4)
        return response
    

        
        
    REVIEW = """You are the actor playing {{$name}} in an improvised play. Before this scene began you planned what {{$name}} would do in it; that plan is your own idea for the scene. Earlier beats have now been played. Look again at the task you planned to do next.

{{$notes}}

<character>
{{$character}}
</character>

<what_{{$name}}_is_after_in_this_scene>
{{$goal}}
</what_{{$name}}_is_after_in_this_scene>

<planned_task>
What you meant to do: {{$task}}
Why (your intent): {{$intent}}
Done when: {{$done_when}}
</planned_task>

<your_later_tasks_in_this_scene>
{{$later}}
</your_later_tasks_in_this_scene>

<what_has_happened_in_the_scene_since_you_planned>
{{$happened}}
</what_has_happened_in_the_scene_since_you_planned>

The intent is yours and it is kept: it is what you brought to this scene. Decide only whether the planned action is still the way to pursue it.
- keep: the action has not been done and can still be played as planned. When in doubt, keep. The same action in other words is not a change: answer keep.
- adapt: the action has already been done or said (by you or by someone else, in this scene or before it), has been overtaken by what happened, or cannot be done now. Give a new action that pursues the same intent from where things now stand. First say, in "serves", how the new action gets {{$name}} what the intent is after; if you cannot say that plainly, it is not an adaptation.
- respond: the intent itself has been met, or what another character has just done makes it pointless, or {{$name}} has turned to something else. Give what {{$name}} now wants in answer to what happened, and why. If the action you have in mind does not serve the planned intent, this is the honest answer, not adapt.
{{$final}}
Answer in JSON only. For keep, the verdict alone is enough.
{"verdict": "keep" | "adapt" | "respond", "serves": "<for adapt only: how the new action gets what the intent is after>", "action": "<4 to 8 words naming the new action>", "how": "<one plain sentence: what {{$name}} does or presses for; not the words to be said, which the actor finds in the moment>", "done_when": "<5 to 8 words: something that can be seen or heard>", "why": "<for respond only: the new intent, 6 to 10 words>"}
End your response with: </end>"""

    def refresh_task(self, task, scene_task_plan, final_task=False):
        """Look again at a planned task just before it is played, in the light of what has
        happened in the scene since the plans were made. Returns the task to play.

        Each character plans its tasks for a scene on its own, before the scene: that is where
        the scene's surprises come from, and the plan is the character's idea of the scene.
        This step is to keep that idea playable, not to have another one. It answers keep,
        adapt (the same intent, another action) or respond (the intent is spent: answer what
        just happened). On keep and adapt the task's reason, the character's own statement of
        why, is carried over by this code and cannot be rewritten; only respond gives a new one.

        (The step before this one was asked to keep, revise or replace, and rewrote four tasks
        in five: most of them the same task in other words, some with the reason softened or
        hardened on the way, one in eight cut off at its token limit and silently dropped. It
        was shown the names of the tasks begun, not what had come of them, and it ran even
        when nothing had yet been played.)"""
        context = self.context
        events_at, lines_at = getattr(context, '_scene_plan_mark', (0, 0))
        events, lines = context.scene_events[events_at:], [str(l).strip() for l in context.transcript[lines_at:] if str(l).strip()]
        if not events and not lines:
            trace.event('task_review', character=self.name, task_id=task.id, verdict='keep', why='nothing has happened since it was planned')
            return task
        goal = task.goal or self.focus_goal
        later = [t['task'].to_string() for t in scene_task_plan if t['actor'] is self and t['task'] is not task
                 and scene_task_plan.index(t) > context.scene_integrated_task_plan_index]
        asked = {'name': self.name, 'notes': self.director_notes(story=False),
                 'character': self.get_character_description(),
                 'goal': f'{goal.name}: {goal.description}' if goal else 'not said',
                 'task': f'{task.name}: {task.description}', 'intent': task.reason, 'done_when': task.termination or 'not said',
                 'later': '\n'.join(later) or 'none',
                 'happened': record.last_lines(events + lines, 5000),
                 'final': 'This is your last task in the scene.\n' if final_task else ''}
        for attempt in (1, 2):
            reply = self.llm.ask(asked, [UserMessage(content=self.REVIEW)], tag='review_task', temp=0.2, stops=['</end>'], max_tokens=280)
            found = json_reply.parse(reply) if reply else None
            verdict = str(found.get('verdict') or '').strip().lower() if isinstance(found, dict) else ''
            if verdict in ('keep', 'adapt', 'respond'):
                break   # an answer with no verdict (one came back empty) is asked for once more
        action, how = (str(found.get(k) or '').strip() for k in ('action', 'how')) if isinstance(found, dict) else ('', '')
        serves = str(found.get('serves') or '').strip() if isinstance(found, dict) else ''
        why = str(found.get('why') or '').strip() if isinstance(found, dict) else ''
        if verdict == 'respond' and action and not how:
            # the new action and why, with no sentence of how: the action is what is done. (This
            # answer was dropped as unreadable, and the beat it replaced, a trade already made,
            # was played again.)
            how = action
        if verdict not in ('adapt', 'respond') or not action or not how or (verdict == 'adapt' and not serves):
            # keep; or a reply that cannot be read, names no action, or changes the action
            # without saying how the new one serves the intent: the plan stands
            trace.event('task_review', character=self.name, task_id=task.id,
                        verdict='keep' if verdict == 'keep' else 'unserved' if verdict == 'adapt' and action and how else 'unreadable',
                        reply='' if verdict == 'keep' else (reply or '')[:300])
            return task
        new_task = Task(action, how, why if verdict == 'respond' and why else task.reason, task.actors, goal,
                        termination=str(found.get('done_when') or '').strip() or task.termination,
                        start_time=task.start_time, duration=task.duration)
        trace.event('task_review', character=self.name, task_id=task.id, verdict=verdict, new_task_id=new_task.id,
                    before=f'{task.name}: {task.description}', after=f'{action}: {how}', reason=new_task.reason,
                    reason_kept=new_task.reason == task.reason, serves=serves)
        if goal is not None:
            goal.task_plan = [new_task]
        return new_task

    async def decide(self, decision, act=None, scene=None):
        """Decide on a choice for a decision"""
        system_prompt = """You are {{$name}}, a character in an improvisational play.
You are given a decision to make, and a list of choices.
Your task is to decide which choice to make, and why.
"""
        prefix = """In the scene about to be played you are after: {{$scene_goal}}

The decision to make is:
#Decision
{{$decision}}
##

The choices are:
#Choices
{{$choices}}
##

Making one of the choices above may well involve denying one of you drives listed below. 
Denying a drive means the choice you are making is incompatible with hope of achieving that drive in the future.
You will be asked to identify the drive you are denying in the choice you are making.

"""
        suffix = """    
#this is act {{$act_number}}. your avoidance count is {{$avoidance_cnt}}. 

#your previous decisions have been:
{{$previous_decisions}}
##

Choose as this person would: from who you are, what you want, and what has passed between you and the others so far.
The choice is how you go after what you are after in this scene; it is not a break with it. It should fit what you have already decided.
Respond ONLY with the choice you have decided on, and why. The why should be terse, and will be revealed to no-one but yourself.
Use the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Each choice should be closed by a ## tag on a separate line.
Be careful to insert line breaks only where shown, separating a value from the next tag.

#choice choice
#delay True / False - you made a choice to delay or to pursue compromise.
#deny drive denied to make this choice
#motivation terse (6-8 words statement why you have decided on this choice
##

Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
"""
        response = default_ask(self, system_prompt=system_prompt, prefix=prefix, suffix=suffix, 
                               # ('make the hard choice, and don't be afraid to be ruthless' stood in this
                               # prompt: every character was ruthless, in every scene)
                               addl_bindings={"decision": decision.get('decision', ''), "choices": '\n\t'+'\n\t'.join(decision.get('choices', '')),
                                               "scene_goal": (getattr(self, 'direction', None) or {}).get('goal') or 'not said',
                                               "act_number": act['act_number'] if act else 0, "avoidance_cnt": self.avoidance_cnt,
                                               "previous_decisions": '\n'.join(json.dumps(d) for d in self.decisions if d['delay'] == False)}, 
                               tag = 'NarrativeCharacter.decide',
                               max_tokens=400, log=True)
        if response is not None:
            response = response.strip()
        choice = hash_utils.find('choice', response)
        avoidance = hash_utils.find('delay', response)
        drive_denied = hash_utils.find('deny', response)
        if avoidance and avoidance.strip().lower() == 'true':
            avoidance = True
            self.avoidance_cnt += 1
        else:
            avoidance = False
            self.avoidance_cnt = 0 # reset avoidance counter
        reason = hash_utils.find('motivation', response)
        if choice and reason:
            # what was asked is kept with the answer (the answer alone was kept, so neither the
            # next decision nor the Explore panel could tell what had been chosen between)
            decision = {'decision': decision.get('decision', ''), 'choices': list(decision.get('choices') or []),
                        'choice': choice, 'delay': avoidance, 'drive_denied': drive_denied, 'reason': reason}
            self.decisions.append(decision)
            return decision
        else:
            return None