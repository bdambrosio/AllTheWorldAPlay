from __future__ import annotations
import sim.survival as survival
from sim.cognitive import drives
import sim.stage as stage
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
from utils import hash_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from enum import Enum
from typing import TYPE_CHECKING
from sim.prompt import ask as default_ask
from sim.agh import Character

if TYPE_CHECKING:
    from sim.memory.core import NarrativeSummary

logger = logging.getLogger('simulation_core')

# What each kind of act is for (sim/screenplay.py KINDS), for whoever drafts one. They were all
# shown for every act, numbered 1 to 4, whichever act was being written.
ACT_RULES = {'opening': "    1. This act should be short and to the point. Your target is 3 scenes maximum.\n    2. Sequence scenes to introduce characters and create unresolved tension.\n    3. Establish the central dramatic question clearly: {{$central_narrative}}\n    4. Act post-state must specify: what the characters now know, what they've agreed together, and what specific tension remains unresolved.\n    5. Final scene post-narrative must show characters making a concrete commitment or decision about their shared challenge.\n    6. Ensure act post-state represents measurable progress from act pre-state, not just mood shifts.\n", 'middle': '    1. Each scene must advance the central dramatic question: {{$central_narrative}}\n    2. Midpoint should fundamentally complicate the question (make it harder to answer or change its nature). Your target is 4 scenes maximum.\n    3. Prevent lateral exploration - every scene should move closer to OR further from resolution..\n    5. Avoid pointless repetition of scene intentions, but allow characters to develop their characters.\n    6. Sequence scenes for continuously building tension, perhaps with minor temporary relief, E.G., create response opportunities (e.g., Character A\'s revelation triggers Character B\'s confrontation)\n    7. Ensure each scene raises stakes higher than the previous scene - avoid cycling back to earlier tension levels.\n    8. Midpoint scene post-narrative must specify: what discovery/setback fundamentally changes the characters\' approach to the central question.\n    9. Act post-state must show: what new obstacles emerged, what the characters now understand differently, and what desperate action they\'re forced to consider.\n    10. Each scene post-narrative must demonstrate measurable escalation from the previous scene - not just "tension increases" but specific new problems or revelations.\n', 'resolving': '    1. Directly answer the central dramatic question, as far as this episode is to settle it: {{$central_narrative}}\n    2. No scene should avoid engaging with the question\'s resolution. Your target is 3 scenes maximum.\n    3. Sequence scenes for maximum tension (eg, alternate elation/disappointment, tension/relief, etc. beats)\n    4. create response opportunities (e.g., Character A\'s revelation triggers Character B\'s confrontation)  \n    5. Act post-state must explicitly state: whether the General dramatic question was answered YES or NO, what specific outcome was achieved, and what the characters\' final status is.\n    6. Final scene post-narrative must show the concrete resolution - not "they find peace" but "they escape the forest together" or "they remain trapped but united."\n    7. No Scene may end with ambiguous outcomes - each must show clear progress toward or away from resolving the central question.\n', 'closing': '    1. Show the immediate aftermath and consequences of how the last act left: {{$central_narrative}}\n    2. Maximum two scenes - focus on essential closure only, avoid extended exploration.\n    3. Preserve character scene intentions where possible. Combine overlapping scene intentions from different characters where possible.\n    4. First scene should show the immediate practical consequences of the resolution (what changes in their situation).\n    5. Second scene (if needed) should show the emotional/relational aftermath (how characters have transformed).\n    6. No new conflicts or dramatic questions - only reveal the implications of what was already resolved.\n    7. Act post-state must specify: the characters\' new equilibrium, what they\'ve learned or become, and their final emotional state.\n    8. Final scene post-narrative must provide definitive closure - show the "new normal" that results from their journey.\n    9. No ambiguity about outcomes - the coda confirms and completes the resolution, not reopens questions.\n'}
ACT_KIND_WORDS = {'opening': 'the opening act', 'middle': 'a middle act', 'resolving': 'the act that resolves the question', 'closing': 'the closing act, after the question is settled'}


class NarrativeCharacter(Character):
    def __init__(self, name, character_description= '', reference_description='', init_x=None, init_y=None, server_name='local'):
        super().__init__(name, character_description, reference_description=reference_description, init_x=init_x, init_y=init_y, server_name=server_name)
        # Narrative files
        self.play_file = None 
        self.map_file = None
        self.play_file_content = None
        self.map_file_content = None
        # Narrative
        self.plan = None
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
    


    async def write_narrative(self, play_file_content:dict, map_file_content:str, central_narrative:CentralNarrative):

        system_prompt = """You are a seasoned writer designing a {{$act_count}} act narrative arc for a 30 minute screenplay.
Every act should push dramatic tension higher: give the protagonist a clear want, place obstacles in the way, and end each act changed by success or setback.
Keep the stakes personal and specific — pledge of a commitment, revelation of a secret, a deadline that can’t be missed — so the audience feels the pulse of consequence.
Let conflict emerge through spoken intention and subtext, as well as through the characters' actions and reactions with the world and each other.
Characters hold real agency; they pursue goals, make trade-offs, and can fail. Attempts at compromise often results in loss of both possibilities. 
""" + survival.stakes_text(self.context, 'Survival chores are background unless they expose or escalate the core mystery.') + """
Use vivid but economical language, vary emotional tone, and avoid repeating imagery.
By the final act, resolve — or intentionally leave poised — the protagonist’s primary drive.
        """

        mission = """ 
The overall setting, cast of characters, and physical world are given below.

#Setting
{{$scenario}}

#Physical World
{{$map}}
##


A prior negotiation process has established the following central dramatic question that will drive the overall play*:
{{$central_narrative}}

You also have your own plans that may or may not be consistent with the central dramatic question.
Focus your personal lifeplan on what you see as your role in the play*: {{$my_narrative}}

* narrative is a paragraph or two including the following elements:
1. Central Dramatic Question: One clear sentence framing the core conflict
2. Stakes: What happens if this question isn't resolved - consequences that matter to all
3. Character Roles: for each character, what is their role in the conflict? Protagonist/Antagonist/Catalyst/Obstacle - with 1-2 sentence role description
4. Dramatic Tension Sources: The main opposing forces that will drive the narrative
5. Success Scenario: Brief description of what "winning" looks like
6. Failure Scenario: Brief description of what "losing" looks like

The company has also settled what this episode of the play is for, and what really lies in its world to be found:
{{$episode}}

After processing the above information and in light of the following information, generate a {{$act_count}} act outline for the play.

Consider the following guidelines:
** For act 1:
    1. This act should be short and to the point.
    2. Sequence scenes to introduce characters and create unresolved tension.
    3. Establish the central dramatic question clearly.
    4. Act post-state must specify: what the characters now know, what they've agreed to do together, and what specific tension remains unresolved.
    5. Final act post-state must show characters making a concrete commitment or decision about their shared challenge.
    6. Ensure act post-state represents measurable progress from act pre-state, not just mood shifts.
 
** For act 2:
    1. Each scene must advance the central dramatic question.
    2. Midpoint should fundamentally complicate the question (make it harder to answer or change its nature).
    3. Prevent lateral exploration - every scene should move closer to OR further from resolution..
    4. Allow characters to develop their characters.
    5. Ensure each scene raises stakes higher than the previous scene - avoid cycling back to earlier tension levels.
    6. Midpoint scene post-narrative must specify: what discovery/setback fundamentally changes the characters' approach to the central question.
    7. Act post-state must show: what new obstacles emerged, what the characters now understand differently, and what desperate action they're forced to consider.
    8. Each scene post-narrative must demonstrate measurable escalation from the previous scene - not just "tension increases" but specific new problems or revelations.

** For act 3:
    1. Directly answer the central dramatic question.
    2. Act post-state must explicitly state: whether the General dramatic question was answered YES or NO, what specific outcome was achieved, and what the characters' final status is.
    3. Final act post-state must show the concrete resolution - not "they find peace" but "they escape the forest together" or "they remain trapped but united."

"""

        suffix = """

Remember, you are {{$name}}

## TASK: Generate a narrative arc for yourself
Generate a single JSON document that outlines the narrative arc for yourself.
The narrative should start at the current time: {{$start_time}}
""" + survival.stakes_text(self.context, 'Survival tasks (food, water, shelter) are assumed handled off-stage unless they advance the mystery or dramatic tension.') + """
By the end of the narrative, what the episode is to settle is settled, one way or the other; what it leaves open is still open.

### Structure
Return exactly one JSON object with these keys:

* "title"    – a short, evocative play title.  
* "acts" – an array of act objects.  Each act object has  
  - "act_number" (int, 1-based)  
  - "act_title"   (string)  
  - "act_description" (string, concise (15-20 word) description of the act, focusing on it's dramatic tension and how it fits into the overall narrative arc)
  - "act_goals" {"primary": "primary goal", "secondary": "secondary goal"} (string, concise (about 8 words each) description of the goals of the act)
  - "act_pre_state" (string, description of the situation / goals / tensions as the act starts. concise, about 10 words. Must be concrete and specific. e.g., 'I am pregnant', not 'there is a secret.')
  - "act_post_state" (string, description of the situation / goals / tensions as the act ends. concise, about 10 words)
  - "tension_points": [
      {"characters": ["<Name>", ...], "issue": (string, concise (about 8 words) description of the issue), "resolution_requirement": (string, "partial" / "complete")}
      ...
    ],

This is your plan, others may or may not agree with it. 
You may later choose to share some or all of it with specific other characters, in hopes of getting them to cooperate.
You will have the opportunity to revise your plan as you go along, observe the results, and as you learn more about the other characters.

### 2.2  Guidelines
1. Base every character’s *stated goal* on their drives, any knowledge you have of them and any percepts. Keep it actionable for the scene (e.g., “Convince Dana to stay”, not “Seek happiness”).  
2. Craft exactly {{$act_count}} acts - keep the momentum going. By the end there should be some resolution of the dramatic tension: of what the episode settles, not of what it leaves open.
3. Escalate tension act-to-act; expect characters to be challenged, and to be forced to reconsider their goals and perhaps change them in future.  
4. Set scenes at the named places listed under Physical World, where their action can happen.  
5. Aim for <u>dialogue-forward theatre</u>: lean on conflict & objective, not big visuals, although an occasional hi-impact visual can be used to enhance the drama.  
6. Vary imagery and emotional tone; avoid repeating the same metaphor scene-to-scene.  
7. Do **NOT** invent new characters unless absolutely necessary, and never break JSON validity.  
8. Keep the JSON human-readable (indent 2 spaces).

Return **only** the JSON.  No commentary, no code fences.

"""

        narrative = default_ask(self, view='playwright', system_prompt=system_prompt, prefix = mission, suffix = suffix, 
                                addl_bindings={"scenario": json.dumps(self.context.summarize_scenario(), indent=2, default=datetime_handler), 
                                 "map": self.context.summarize_map(),
                                 "name": self.name,
                                 "central_narrative": central_narrative,
                                 "my_narrative": self.central_narrative,
                                 "episode": self.context.episode_note() or 'Nothing further.',
                                 "act_count": self.context.screenplay.act_count,
                                 "start_time": self.context.simulation_time.isoformat()},
                                 max_tokens=5000, tag='NarrativeCharacter.write_narrative')
        try:
            self.plan = json.loads(narrative.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            print(f'Error parsing narrative: {e}')
            self.plan = self.context.repair_json(narrative, e)
        valid, reason = self.context.validate_narrative_json(self.plan, require_scenes=False)
        if not valid:
            # an unusable plan is an empty one: its acts are headings, filled in act by act
            print(f'Invalid narrative: {reason}')
            trace.event('plan_refused', character=self.name, reason=str(reason))
            self.plan = {"title": f"{self.name}'s part", "acts": []}
        # every character's plan has the acts the screenplay counts on, numbered 1..n: a longer
        # plan is cut, a shorter one gets an act with a heading only, which the rewrite before
        # that act is played fills in (a character with a shorter plan was left out of the
        # later acts altogether)
        count = self.context.screenplay.act_count
        acts = self.plan["acts"][:count]
        for n in range(len(acts), count):
            acts.append({"act_number": n + 1, "act_title": f"Act {n + 1}"})
            self.context.validate_narrative_act(acts[-1], require_scenes=False)
        for n, act in enumerate(acts):
            act["act_number"] = n + 1
        self.plan["acts"] = acts
        self.current_act = acts[0]
        self.current_scene = None
        return self.plan

    def get_act_from_plan(self, act_number):
        for act in self.plan["acts"]:
            if act["act_number"] == act_number:
                return act
        return None


    def update_act(self, new_act):
        """Put a rewritten (and already validated) act in this character's plan, in place of the
        act with its number. False when the plan has no such act (the closing act)."""
        for i, act in enumerate(self.plan["acts"]):
            if act["act_number"] == new_act["act_number"]:
                self.plan["acts"][i] = new_act
                return True
        return False
    
    def replan_narrative_act(self, act, act_central_narrative:CentralNarrative, previous_act_post_state:str, kind='middle'):
        """This character's draft of an act, written from where the play now stands. kind: what
        the act is for (opening, middle, resolving, closing)."""

        system_prompt = """You are a seasoned writer rewriting act {{$act_number}} for a play.
Every act should push dramatic tension higher: give the protagonist a clear want, place obstacles in the way, and end each act changed by success or setback.
Keep the stakes personal and specific — pledge of a commitment, revelation of a secret, a deadline that can’t be missed, a broken promise — so the audience feels the pulse of consequence.
Let conflict emerge through spoken intention and subtext, as well as through the characters' actions and reactions with the world and each other.
Characters hold real agency; they pursue goals, make trade-offs, and can fail.
""" + survival.stakes_text(self.context, 'Survival chores are background unless they expose or escalate the core mystery.') + """
Use vivid but economical language, vary emotional tone, and avoid repeating imagery.
        """
        mission = """ You have already created a initial outline for the following act, and now need to update it based on the actual performance so far and the guiding central dramatic question.
Note the tensions and relationships with other characters, as these may have changed.
Note also the final scene and post-state of the previous act.
Check consistency with the choices made in this updated act, including it's title, pre-state, location, goals, description, and character goals in the initial scene.
Based on prior conversations and events recorded below, update your plan for the following act. 
The act may be empty, ie have no scenes. In this case, you should flesh out the act with scenes based on it's title and place in the play. Make sure the overall narrative arc is maintained.
If there are scenes in the act, they may have already occurred in the performance so far, or may no longer be relevant (e.g. other actor's plans have changed), and if so should be removed or rewritten to reflect the new information.
Other scenes may need to be added or modified to better reflect the new information. Analyze the act description and design the new act to achieve the dramatic tension described. Extend with new scenes if needed.
Note also that time has passed since you originally formulated this act. For example, you might have been planning to have breakfast, but it may now be past noon.
the current time is {{$time}}
your original act was:

{{$act}}

and it was number {{$act_number}} in the your original narrative plan.
####
Critically, the play has progressed since you originally formulated this act, perhaps in unanticipated directions. The previous act post-state (ie, the actual situation after the previous act) is now:
{{$previous_act_post_state}}
####

What has happened in the acts already played (the record of events, not what was planned):

{{$previous_acts}}

"""
        suffix = """

Pay special attention to the goals you have already attempted and their completion status in planning your new act. 
Do not repeat goals that have already been attempted and completed, unless you have a new reason to attempt them again or it is important for dramatic tension.

Note that in the current situation any assumptions or preconditions on which your original act and plan were based may no longer be valid.
However, the original short-term plan above should still be used to locate your new plan within the overall narrative arc of the performance.

The following guidelines for this act supplement the general guidelines above, and override them where necessary. This is act {{$act_number}}, {{$act_kind}}:

{{$act_rules}}

{{$scene_rules}}

#Places
{{$places}}
##

Respond with the updated act, using the following format:

```json
updated act
```

Act format is as follows:

An Act is a single JSON document that outlines a short-term plan for yourself
###  Structure
Return exactly one JSON object with these keys:

- "act_number" (int, copied from the original act)  
- "act_title"   (string, copied from the original act or rewritten as appropriate)  
- "act_description" (string, concise (about 15 words) description of the act, focusing on it's dramatic tension and how it fits into the overall narrative arc)
- "act_goals" {"primary": "primary goal", "secondary": "secondary goal"} (string, concise (about 8 words each) description of the goals of the act)
- "act_pre_state": (string, description of the situation / goals / tensions before the act starts. concise, about 10 words). Must be concrete and specific. e.g., 'I am pregnant', not 'there is a secret.'
- "act_post_state": (string, description of the situation / goals / tensions after the act ends. concise, about 10 words)
- "tension_points": [
    {"characters": ["<Name>", ...], "issue": (string, concise (about 8 words) description of the issue), "resolution_requirement": (string, "partial" / "complete")}
    ...
  ]
- "scenes"      (array) 

Each **scene** object must have:
{ "scene_number": int, // sequential within the play 
 "scene_title": string, // concise descriptor 
 {{$location_field}}
 "time": "2025-01-01T08:00:00", // the start time of the scene, in ISO 8601 format
 "characters": { "<Name>": { "goal": "<one-line playable goal>" }, … }, 
 "action_order": [ "<Name>", … ], // each name occurrence is a 'beat' in the scene lead by the named character. list only characters present in the scene 'characters' list.
 "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors. Must be concrete and specific. e.g., 'Laura doesnt love you', not 'there is a secret.'", 
 "post_narrative": "Short prose (≤20 words) summarising end state and what emotional residue or new tension lingers." 
 // OPTIONAL: 
 "task_budget": 2 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the scene to avoid rambling or repetition, or slightly higer for scenes with complex goals or interactions.
 }

 === Example ===
##
```json
{
  "act_number": {{$act_number}},
  "act_title": "rewritten act title",
  "act_description": "concise (about 10 words) description of the act, focusing on it's dramatic tension and how it fits into the overall narrative arc",
  "act_goals" {"primary": "primary goal", "secondary": "secondary goal"},
  "act_pre_state": "description of the situation / goals / tensions before the act starts. concise, about 10 words. Must be concrete and specific. e.g., 'I am pregnant', not 'there is a secret.'",
  "act_post_state": "description of the situation / goals / tensions after the act ends. concise, about 10 words",
  "tension_points": [
    {"characters": ["<Name>", ...], "issue": (string, concise (about 8 words) description of the issue), "resolution_requirement": (string, "partial" / "complete")}
    ...
  ],
  "scenes": [
    {
      "scene_number": 1,  
      "scene_title": "Scene 1",
      "time": "2025-01-01T09:00:00", // the start time of the scene, in ISO 8601 format
      "location": "Location 1",
      "characters": { "Character 1": { "goal": "Goal 1" }, ... },
      "action_order": [ "<Name>", … ], // each name occurrence is a 'beat' in the scene lead by the named character. list only characters present in the scene 'characters' list.
      "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors. Must be concrete and specific. e.g., 'I killed Joe', not 'there is a secret.'", 
      "post_narrative": "Short prose (≤20 words) summarising end state and what emotional residue or new tension lingers." 
      // OPTIONAL: 
      "task_budget": 2 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the action_order or slightly higher for scenes with complex goals or interactions. 
    },
    ...
  ]
}

Again, the act central narrative is:

{{$act_central_narrative}}

```
=== End of Example ===

"""
        acts = {"acts": [act]} # reserialize expects a narrative json object
        reserialized_act = self.context.reserialize_acts_times(acts)
        act_number = act['act_number']
        response = default_ask(self, view='playwright', system_prompt=system_prompt, prefix=mission, suffix=suffix,
                              addl_bindings={"name": self.name, "act": json.dumps(reserialized_act['acts'][0], indent=1, default=datetime_handler), 
                               "act_number": act['act_number'],
                               "act_kind": ACT_KIND_WORDS[kind],
                               "act_rules": ACT_RULES[kind].replace('{{$central_narrative}}', str(self.context.central_narrative or '')),
                               "location_field": stage.LOCATION_FIELD, "scene_rules": stage.SCENE_RULES,
                               "places": stage.gazetteer(self.context),
                               "previous_act_post_state": previous_act_post_state,
                               "act_central_narrative": act_central_narrative,
                               "central_narrative": self.context.central_narrative,
                               "previous_acts": self.context.screenplay.so_far() or 'None: this is the first act.'},
                              max_tokens=1400, tag='NarrativeCharacter.replan_narrative_act')
        try:
            updated_act = None
            act_id = act['act_number']
            if not response:
                return None
            response = response.replace("```json", "").replace("```", "").strip()
            updated_act = json.loads(response)
        except Exception as e:
            print(f'Error parsing updated act: {e}')
            logger.error(f'Error parsing updated act: {e}')
            logger.error(traceback.format_exc())
            updated_act = self.context.repair_json(response, e)
        # A rewrite that is not a usable act is not used: the act as it stood goes to the
        # integrator. (It was returned whatever update_act said, and a scene without a time
        # ended a play three calls later.)
        valid, reason = self.context.validate_narrative_act(updated_act, require_scenes=True) if isinstance(updated_act, dict) else (False, 'no act in the reply')
        if not valid:
            print(f'{self.name} keeps act {act_id} as it was: {reason}')
            trace.event('act_rewrite_refused', character=self.name, act=act_id, reason=str(reason))
            return act
        print(f'{self.name} updates act {act_id}')
        self.update_act(updated_act)
        self.current_act = updated_act
        return updated_act

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
    

        
        
    def refresh_task(self, task, scene_task_plan, final_task=False):
        """Refresh the task to ensure it is up to date with the latest information
        adapted from Character.generate_task_plan"""

        if not self.focus_goal:
            raise ValueError(f'No focus goal for {self.name}')
        """generate task alternatives to achieve a focus goal"""
        goal = self.focus_goal        
        system_prompt = """You are an actor in an improvisational play. Review the following planned task for the character you are playing {{$name}} in light of recent events. 

##Previously Planned Task
{{$planned_task}}

##Instructions
Review your planned task against current conditions and decide if it should be:
1. **KEPT** - Still appropriate and achievable as planned
2. **REVISED** - Core intent is good but details need adjustment  
3. **REPLACED** - Circumstances have made a different approach better

Consider these factors:
- Has the scene dynamic shifted from your expectations?
- Are the planned actors still available/relevant?
- Do recent events create better opportunities?
- Does your task conflict with or duplicate others in the scene plan?

Respond with either:
- "KEEP" (if no changes needed)
- The revised task in hash format (if changes would improve effectiveness)

Following will be a large set of background information about the play, the scene, and the task.

"""

        mission = """

You previously planned to pursue this task next:

{{$planned_task}}

The scene will end with this task:
{{$final_task}}

You are performing a in play with the following Dramatic Context:

##Dramatic Context
<central_narrative>
{{$central_narrative}}
</central_narrative>

<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

You have your own plans that may or may not be consistent with the central dramatic question. 
Potential conflict between your own plans and the central dramatic question is real, only you can decide.

<my_narrative>
{{$my_narrative}}
</my_narrative>

##Scene
The current scene you will be acting in is:
<scene>
{{$scene}}
</scene>

and, importantly, the scene narrative is:
<scene_narrative>
{{$scene_narrative}}
</scene_narrative>

The overall task plan for this scene is as follows, although as yet unperformed tasks may be revised as events unfold:
<scene_task_plan>
{{$scene_task_plan}}
</scene_task_plan>

And the following set of tasks has already been performed within the scene:
<scene_history>
{{$scene_history}}
</scene_history>

##Previous Achievments
{{$achievments}}

"""

        suffix = """    
##Instructions
Review your planned task against current conditions and decide if it should be kept, revised, or replaced. Keep your thinking to yourself. Do not output your thinking.
1. Choose to keep the task if it is still appropriate and achievable as planned
2. Choose to revise the task if the core intent is good but details need significant adjustment  
3. Choose to replace the task if circumstances have made a different approach better

Consider these factors:
- Has the scene dynamic shifted significantlyfrom your expectations?
- Have the relationships among planned actors changed in ways that make your task no longer appropriate?
- Do recent events create better opportunities?
- Does your task conflict with or duplicate others in the scene plan? Avoid duplication. Review past goals and their completion status to ensure you are not repeating yourself.

Respond ONLY with either:
- "KEEP" (if no changes needed) *OR*
- The revised /replaced task in hash format (if changes would improve effectiveness)

Remember, if you choose to make any changes:                             
 - A task is a specific objective that can be achieved in the current situation and which is a major step in satisfying the focus goal.
 - The task(s) should be distinct from one another, and each should advance the focus goal.
 - Use known actor names in the task description unless deliberately introducing a new actor. Characters known to this character include:
    {{$known_actors}}

 - Make explicit reference to diverse known or observable resources where logically fitting, ensuring broader environmental engagement across tasks.
 - In refering to other actors, always use their name, without other labels like 'Agent', 
 and do not use pronouns or referents like 'he', 'she', 'that guy', etc.


##Task Format
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

Respond ONLY with KEEP, if you choose to leave the proposed task unchanged, or with the revised task in hash-formatted-text format and ending with ## as shown above.

End response with:
</end>
"""

        if self.context.scene_post_narrative:
            scene_narrative = f"\nThe narrative arc of the scene is from:  {self.context.scene_pre_narrative} to  {self.context.scene_post_narrative}\nThe task sequence should be consistent with this theme."
        else:
            scene_narrative = ''
        # the actor's own tasks in the scene: what the others mean to do is theirs to play
        scene_task_plan = '\n'.join([f'{t["actor"].name}: {t["task"].to_string()}' for t in scene_task_plan if t["actor"] is self])
        logger.info(f'{self.name} generate_task_plan: {goal.to_string()}')
        response = default_ask(self, system_prompt=system_prompt,
                               prefix=mission, 
                               suffix=suffix, 
                               addl_bindings={"focus_goal":goal.to_string(),
                                "planned_task": task.to_string(),
                                "final_task": "\nThis is your final task in this scene." if final_task else '',
                                "known_actors": "\n".join([name for name in self.actor_models.names()]),
                                "achievments": '\n'.join(self.achievments[:5]),
                                "scene_narrative": scene_narrative,
                                "act_specific_narrative": self.act_direction(),
                                "central_narrative": self.story_note(),
                                "my_narrative": self.central_narrative if self.central_narrative else '',
                                "scene": self.director_notes(story=False),
                                "scene_task_plan": scene_task_plan, 
                                "scene_history": '\n'.join(self.context.scene_history)},
                               tag = 'Character.generate_task_plan',
                               max_tokens=100, log=True)


        # add each new task, but first check for and delete any existing task with the same name
        forms = hash_utils.findall_forms(response)
        if forms and len(forms) > 0:
            task_hash = forms[0]
            print(f'\n{self.name} new task: {task_hash.replace('\n', '; ')}')
            if not self.focus_goal:
                print(f'{self.name} generate_plan: no focus goal, skipping task')   
            new_task = self.validate_and_create_task(task_hash, goal)
            if new_task:
                goal.task_plan = [new_task] # only use the first task, that's all we asked for.
                return new_task
            else:
                print(f'{self.name} generate_plan: no new task')
        return task
    
    async def decide(self, decision, act=None, scene=None):
        """Decide on a choice for a decision"""
        system_prompt = """You are {{$name}}, a character in an improvisational play.
You are given a decision to make, and a list of choices.
Your task is to decide which choice to make, and why.
"""
        prefix = """The decision to make is:
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

Any decision you make must be consisten with previous irrevocable decisions. 
If act is greater than 2 or avoidance is greater than 5 you must make an irrevocable decision, rather than choose to delay, avoid, or persue compromise.
Don't be fooled: a 'firm commitment to balanced future' is avoidance / compromise, not a hard irrevocable decision. Make the hard choice, and don't be afraid to be ruthless.
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
                               addl_bindings={"decision": decision.get('decision', ''), "choices": '\n\t'+'\n\t'.join(decision.get('choices', '')),
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
            decision = {'choice': choice, 'delay': avoidance, 'drive_denied': drive_denied, 'reason': reason}
            self.decisions.append(decision)
            return decision
        else:
            return None