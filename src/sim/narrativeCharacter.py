from __future__ import annotations
import asyncio
import copy
from datetime import datetime
import logging
from typing import Optional, Dict, List, Any
import os, sys, re, traceback, requests, json

from sim.character_dataclasses import Act, Goal, Task, CentralNarrative, datetime_handler
from src.sim.cognitive.EmotionalStance import EmotionalStance
from src.utils.Messages import UserMessage
from utils import hash_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from enum import Enum
from typing import TYPE_CHECKING
from sim.prompt import ask as default_ask
from sim.agh import Character
from sim.cognitivePrompt import CognitiveToolInterface

if TYPE_CHECKING:
    from sim.memory.core import NarrativeSummary

logger = logging.getLogger('simulation_core')

class NarrativeCharacter(Character):
    def __init__(self, name, character_description= '', reference_description='', init_x=50, init_y=50, server_name='local'):
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
        self.current_act_index = 0
        self.current_scene_index = 0
        print(f'{self.name}: {self.__class__.__name__}')
        self.previous_proposal = None
        self.current_proposal = None
        self.round_number = 0 # central narrative negotiation round number
        self.central_narrative = None
        self.cognitive_tools = CognitiveToolInterface(self)

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
    
    def reserialize_act_to_string(self, act: Dict[str, Any]) -> str:
        """Reserialize the act to a string"""
        serialized_str = self.reserialize_narrative_json({"acts":[act]})
        return json.dumps(serialized_str['acts'][0], indent=2, default=datetime_handler)

    async def introduce_myself(self, play_file_content, map_file_content=None):

        system_prompt = """Imagine you are addressing the audience of a play. 
Create a short introduction (no more than 60 tokens) to your character, including your name, terse description, role in the play, drives, and most important relationships with other characters. 
Leave some mystery. This should be in the first person, you will speak to the audience.
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

        introduction = default_ask(self, system_prompt=system_prompt, prefix = mission, suffix = suffix,
                                addl_bindings={"scenario": play_file_content, 
                                 "map": map_file_content if map_file_content else '',
                                 "name": self.name,
                                 "start_time": self.context.simulation_time.isoformat(),
                                 "primary_drive": f'{self.drives[0].id}: {self.drives[0].text}; activation: {self.drives[0].activation:.2f}',
                                 "secondary_drive": f'{self.drives[1].id}: {self.drives[1].text}; activation: {self.drives[1].activation:.2f}'}, 
                                 max_tokens=200, tag='NarrativeCharacter.introduction')
        if introduction:
            self.context.message_queue.put({'name':self.name, 'text':introduction+'\n'})
            await asyncio.sleep(0)
    


    async def write_narrative(self, play_file_content:dict, map_file_content:str, central_narrative:CentralNarrative):

        system_prompt = """You are a seasoned writer designing a 2-3 act narrative arc for a 30 minute screenplay.
Every act should push dramatic tension higher: give the protagonist a clear want, place obstacles in the way, and end each act changed by success or setback.
Keep the stakes personal and specific—loss of trust, revelation of a secret, a deadline that can’t be missed—so the audience feels the pulse of consequence.
Let conflict emerge through spoken intention and subtext, as well as through the characters' actions and reactions with the world and each other.
Characters hold real agency; they pursue goals, make trade-offs, and can fail. Survival chores are background unless they expose or escalate the core mystery.
Use vivid but economical language, vary emotional tone, and avoid repeating imagery.
By the final act, resolve —or intentionally leave poised— the protagonist’s primary drive.
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

After processing the above information and in light of the following information, generate a 3 act outline for the play.

Consider the following guidelines:
** For act 1:
    1. This act should be short and to the point..
    2. Sequence scenes to introduce characters and create unresolved tension.
    3. Establish the central dramatic question clearly: {{$central_narrative}}
    4. Act post-state must specify: what the characters now know, what they've agreed to do together, and what specific tension remains unresolved.
    5. Final scene post-narrative must show characters making a concrete commitment or decision about their shared challenge.
    6. Ensure act post-state represents measurable progress from act pre-state, not just mood shifts.
 
** For act 2:
    1. Each scene must advance the central dramatic question: {{$central_narrative}}
    2. Midpoint should fundamentally complicate the question (make it harder to answer or change its nature).
    3. Prevent lateral exploration - every scene should move closer to OR further from resolution..
    5. Avoid pointless repetition of scene intentions, but allow characters to develop their characters.
    6. Sequence scenes for continuously building tension, perhaps with minor temporary relief, E.G., create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)
    7. Ensure each scene raises stakes higher than the previous scene - avoid cycling back to earlier tension levels.
    8. Midpoint scene post-narrative must specify: what discovery/setback fundamentally changes the characters' approach to the central question.
    9. Act post-state must show: what new obstacles emerged, what the characters now understand differently, and what desperate action they're forced to consider.
    10. Each scene post-narrative must demonstrate measurable escalation from the previous scene - not just "tension increases" but specific new problems or revelations.

** For act 3:
    1. Directly answer the central dramatic question: {{$central_narrative}}
    2. No scene should avoid engaging with the question's resolution.
    3. Sequence scenes for maximum tension (alternate trust/mistrust beats)
    4. create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)  
    5. Act post-state must explicitly state: whether the General dramatic question was answered YES or NO, what specific outcome was achieved, and what the characters' final status is.
    6. Final scene post-narrative must show the concrete resolution - not "they find peace" but "they escape the forest together" or "they remain trapped but united."
    7. No Scene may end with ambiguous outcomes - each must show clear progress toward or away from resolving the central question.

"""

        suffix = """

Remember, you are {{$name}}

## TASK: Generate a narrative arc for yourself
Generate a single JSON document that outlines the narrative arc for yourself.
The narrative should start at the current time: {{$start_time}}
Survival tasks (food, water, shelter) are assumed handled off-stage unless they advance the mystery or dramatic tension.
By the end of the narrative, the primary drive should be resolved.

### Structure
Return exactly one JSON object with these keys:

* "title"    – a short, evocative play title.  
* "acts" – an array of act objects.  Each act object has  
  - "act_number" (int, 1-based)  
  - "act_title"   (string)  
  - "act_description" (string, concise (15-20 word) description of the act, focusing on it's dramatic tension and how it fits into the overall narrative arc)
  - "act_goals" {"primary": "primary goal", "secondary": "secondary goal"} (string, concise (about 8 words each) description of the goals of the act)
  - "act_pre_state" (string, description of the situation / goals / tensions before the act starts. concise, about 10 words)
  - "act_post_state" (string, description of the situation / goals / tensions after the act ends. concise, about 10 words)
  - "tension_points": [
      {"characters": ["<Name>", ...], "issue": (string, concise (about 8 words) description of the issue), "resolution_requirement": (string, "partial" / "complete")}
      ...
    ],

This is your plan, others may or may not agree with it. 
You may later choose to share some or all of it with specific other characters, in hopes of getting them to cooperate.
You will have the opportunity to revise your plan as you go along, observe the results, and as you learn more about the other characters.

### 2.2  Guidelines
1. Base every character’s *stated goal* on their drives, any knowledge you have of them and any percepts. Keep it actionable for the scene (e.g., “Convince Dana to stay”, not “Seek happiness”).  
2. Craft 3 acts - keep the momentum going, don't add acts just to add acts. By the end there should be some resolution of the dramatic tension and the characters' primary drives.
3. Escalate tension act-to-act; expect characters to be challenged, and to be forced to reconsider their goals and perhaps change them in future.  
4. Place scenes in plausible locations drawn from `map.py` resources/terrain.  
5. Aim for <u>dialogue-forward theatre</u>: lean on conflict & objective, not big visuals, although an occasional hi-impact visual can be used to enhance the drama.  
6. Vary imagery and emotional tone; avoid repeating the same metaphor scene-to-scene.  
7. Do **NOT** invent new characters unless absolutely necessary, and never break JSON validity.  
8. Keep the JSON human-readable (indent 2 spaces).

Return **only** the JSON.  No commentary, no code fences.

"""

        narrative = default_ask(self, system_prompt=system_prompt, prefix = mission, suffix = suffix, 
                                addl_bindings={"scenario": json.dumps(self.context.summarize_scenario(), indent=2, default=datetime_handler), 
                                 "map": self.context.summarize_map(),
                                 "name": self.name,
                                 "central_narrative": central_narrative,
                                 "my_narrative": self.central_narrative,
                                 "start_time": self.context.simulation_time.isoformat(),
                                 "primary_drive": f'{self.drives[0].id}: {self.drives[0].text}; activation: {self.drives[0].activation:.2f}',
                                 "secondary_drive": f'{self.drives[1].id}: {self.drives[1].text}; activation: {self.drives[1].activation:.2f}'}, 
                                 max_tokens=5000, tag='NarrativeCharacter.write_narrative')
        try:
            self.plan = json.loads(narrative.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            print(f'Error parsing narrative: {e}')
            self.plan = self.context.repair_json(narrative, e)
        valid, reason = self.context.validate_narrative_json(self.plan, require_scenes=False)
        if not valid:
            print(f'Invalid narrative: {reason}')
            return None
        # initialize the current act and scene
        self.current_act_index = 0
        self.current_scene_index = 0    
        self.current_act = self.plan["acts"][self.current_act_index]
        self.current_scene = None
        return self.plan

    async def share_narrative(self):
        """Share the narrative with other characters"""
        for character in self.context.actors+self.context.extras:
            if character != self:
                mission = """Decide what, if anything, about your plans you would like to share with {{$name}} to help coordinate your actions. 
You may well decide to share nothing, especially with those you perceive as adversaries, in which case you should respond with </end>
Note that the current time is {{$time}}, if you wish to share a future plan, phrase your message accordingly. (e.g. "I'll be in the park tomorrow" or "I'll be in the park on May 15th")
Take note of any prior conversations you have already had with {{$name}}."""
                suffix = """

Your plans at the moment are:
{{$narrative}}

If you have nothing you want to tell {{$name}}, simply respond with </end>
Otherwise, respond with the information you would like to share, using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag. Specifically, DO NOT insert a newline between the #share tag and the information you want to share:

#share ...
##

=== Example ===
#share Hi John, I'm going to be in the park tomorrow.
##
=== End of Example ===

End your response with </end>
"""

                response = default_ask(self, prefix=mission, suffix=suffix, 
                                addl_bindings={"name": character.name, 
                                 "plan": json.dumps(self.context.reserialize_acts_times(self.plan), default=datetime_handler)}, 
                                 max_tokens=240, tag='NarrativeCharacter.share_narrative')
                try:
                    say_arg = hash_utils.find('share', response)
                    if say_arg:
                        #say_arg = say_arg.replace(f'information', '').strip()
                        if say_arg and say_arg != '':
                            print(f'{self.name} says to {character.name}: {say_arg}')
                            act = Act(mode='Say', action=say_arg, actors=[self, character], reason=say_arg, duration=1, source=None, target=[character])
                        await self.act_on_action(act, None)
                except Exception as e:
                    print(f'Error parsing shared narrative: {e}')
                    traceback.print_exc()
        return None

    def get_act_from_plan(self, act_number):
        for act in self.plan["acts"]:
            if act["act_number"] == act_number:
                return act
        return None


    def update_act(self, new_act):
        valid, reason = self.context.validate_narrative_act(new_act, require_scenes=True)
        if not valid:
            print(f'Invalid act: {reason}')
            return False
        act_num = new_act["act_number"]
        for i, act in enumerate(self.plan["acts"]):
            if act["act_number"] == act_num:
                self.plan["acts"][i] = new_act
                if self.current_act_index == i:
                    self.current_act = new_act
                    self.current_scene = new_act["scenes"][0]
                    self.current_scene_index = 0
                return True
        return False        
    
    def update_narrative_from_shared_info(self):
        """Update the narrative with the latest shared information"""

        system_prompt = """You are a seasoned writer designing medium-term arcs for a movie.
Every act should push dramatic tension higher: give the protagonist a clear want, place obstacles in the way, and end each act changed by success or setback.
Keep the stakes personal and specific—loss of trust, revelation of a secret, a deadline that can’t be missed—so the audience feels the pulse of consequence.
Let conflict emerge through spoken intention and subtext, as well as through the characters' actions and reactions with the world and each other.
Characters hold real agency; they pursue goals, make trade-offs, and can fail. Survival chores are background unless they expose or escalate the core mystery.
Use vivid but economical language, vary emotional tone, and avoid repeating imagery.
By the final act, resolve—or intentionally leave poised—the protagonist’s primary drive.
        """
        mission = """Based on prior conversations recorded below, do you see need to update an act in your plan?"""
        suffix = """

Recent dialogs:
{{$dialogs}}

Your plan at the moment is:
{{$plan}}

if you choose to update an act, remember:

Respond with the single word 'yes' or 'no', and, if yes, the act number of the first act you would like to update, and the updated act, using the following format:

#update yes or no
#actId (int) number of the first act to update
##
```json
updated act
```

Act format is as follows:

An Act is a single JSON document that outlines a short-term plan for yourself
###  Structure
Return exactly one JSON object with these keys:

 
* "acts" – an array of act objects.  Each act object has  
  - "act_number" (int, 1-based)  
  - "act_title"   (string)  
  - "act_description" (string, concise (about 10 words) description of the act, focusing on it's dramatic tension and how it fits into the overall narrative arc)
  - "act_goals" {"primary": "primary goal", "secondary": "secondary goal"} (string, concise (about 8 words each) description of the goals of the act)
  - "act_pre_state" (string, description of the situation / goals / tensions before the act starts. concise, about 10 words)
  - "act_post_state" (string, description of the situation / goals / tensions after the act ends. concise, about 10 words)
  - "tension_points": [
      {"characters": ["<Name>", ...], "issue": (string, concise (about 8 words) description of the issue), "resolution_requirement": (string, "partial" / "complete")}
      ...
    ],
  - "scenes"      (array) (only for the first act)

Each **scene** object must have:
{"scene_number": int, // sequential within the play 
 "scene_title": string, // concise descriptor 
 "location": string, // pick from resource or terrain names in the map file
 "time": YYYY-MM-DDTHH:MM:SS, // the start time of the scene, in ISO 8601 format
 "duration": int, // in minutes
 "characters": { "<Name>": { "goal": "<one-line playable goal>" }, … }, 
 "action_order": [ "<Name>", … ], // each name occurrence is a 'beat' in the scene lead by the named character. list only characters present in the scene 'characters' list.
 "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors.", 
 "post_narrative": "Short prose (≤20 words) summarising end state and what emotional residue or new tension lingers." 
 // OPTIONAL: 
 "task_budget": 4 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the scene to avoid rambling or repetition. 
 }

 === Example ===
#update yes or no
#actId (int) number of the first act to update
##
```json
{
  "act_number": (int),
  "act_title": (string),
  "act_description": (string),
  "act_goals": {"primary": "primary goal", "secondary": "secondary goal"},
  "act_pre_state": (string),
  "act_post_state": (string),
  "tension_points": [
    {"characters": ["<Name>", ...], "issue": (string, concise description of the issue), "resolution_requirement": (string, "partial" / "complete")}
    ...
  ],
  "scenes": [
    {
      "scene_number": 1,  
      "scene_title": "Scene 1",
      "time": YYYY-MM-DDTHH:MM:SS, // the start time of the scene, in ISO 8601 format
      "duration": int, // in minutes
      "location": "Location 1",
      "characters": { "Character 1": { "goal": "Goal 1" }, ... },
      "action_order": [ "<Name>", … ], // each name occurrence is a 'beat' in the scene lead by the named character. list only characters present in the scene 'characters' list.
      "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors.", 
      "post_narrative": "Short prose (≤20 words) summarising end state and what emotional residue or new tension lingers." 
       // OPTIONAL: 
      "task_budget": 4 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the action_order to avoid rambling or repetition. 
    },
    ...
  ]
}
```
=== End of Example ===

Remember that your primary and secondary drives are:

Primary
{{$primary_drive}}

Secondary
{{$secondary_drive}}

Respond with your decision, and if yes, the act number of the first act you would like to update and the updated act, using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#update yes or no
#actId (int) number of the first act to update
##
```json
updated act
```
"""
        dialogs = self.actor_models.dialogs()

        response = default_ask(self, system_prompt=system_prompt, prefix=mission, suffix=suffix,
                              addl_bindings={"name": self.name, 
                               "dialogs": '\n'.join(dialogs),
                               "plan": json.dumps(self.context.reserialize_acts_times(self.plan), default=datetime_handler),
                               "primary_drive": f'{self.drives[0].id}: {self.drives[0].text}; activation: {self.drives[0].activation:.2f}',
                               "secondary_drive": f'{self.drives[1].id}: {self.drives[1].text}; activation: {self.drives[1].activation:.2f}'},
                              max_tokens=1600, tag='NarrativeCharacter.update_narrative')
        try:
            updated_act = None
            act_id = -1
            if "```json" in response:
                update = response.split("```json")
                flag = hash_utils.find('update', update[0])
                act_id_str = hash_utils.find('actId', update[0])
                act_id=1
                if flag and flag == 'yes':
                    try:
                        act_id = int(act_id_str.strip())
                    except Exception as e:
                        print(f'Error parsing actId: {e}')
                        logger.error(f'Error parsing actId: {e}')
                        logger.error(traceback.format_exc())
                    try:
                        updated_act = json.loads(update[1].replace("```", "").strip())
                    except Exception as e:
                        print(f'Error parsing updated act: {e}')
                        logger.error(f'Error parsing updated act: {e}')
                        logger.error(traceback.format_exc())
            else:
                flag = hash_utils.find('update', response)
                act_id=1
                if flag and flag == 'yes':
                    try:
                        act_id = int(hash_utils.find('actId', response).strip())
                    except Exception as e:
                        print(f'Error parsing actId: {e}')
                        logger.error(f'Error parsing actId: {e}')
                        logger.error(traceback.format_exc())
                        act_id = updated_act['act_number']
                    update = response.split("##")    
                    if len(update) > 1:
                        updated_act = json.loads(update[1].replace("```json", "").replace("```", "").strip())
            if act_id != -1 and updated_act:
                print(f'{self.name} updates act {act_id}')
                self.update_act(updated_act)
        except Exception as e:
            print(f'Error parsing update narrative: {e}')
            logger.error(f'Error parsing update narrative: {e}')
            logger.error(traceback.format_exc())
        return None

    def replan_narrative_act(self, act, previous_act, act_central_narrative:CentralNarrative, previous_act_post_state:str):
        """Rewrite the act with the latest shared information"""

        system_prompt = """You are a seasoned writer rewriting act {{$act_number}} for a play.
Every act should push dramatic tension higher: give the protagonist a clear want, place obstacles in the way, and end each act changed by success or setback.
Keep the stakes personal and specific—loss of trust, revelation of a secret, a deadline that can’t be missed—so the audience feels the pulse of consequence.
Let conflict emerge through spoken intention and subtext, as well as through the characters' actions and reactions with the world and each other.
Characters hold real agency; they pursue goals, make trade-offs, and can fail. Survival chores are background unless they expose or escalate the core mystery.
Use vivid but economical language, vary emotional tone, and avoid repeating imagery.
By the final act, resolve—or intentionally leave poised—the protagonist’s primary drive.
        """
        mission = """ You have already created a initial outline for the following act, and now need to update it based on the actual performance so far and the guiding central dramatic question.
Note the tensions and relationships with other characters, as these may have changed.
Note also the post-narrative of the previous act, and check consistency with the assumptions made in this act, including it's title, pre-narrative, goals, description, and character goals in the initial scene.
Based on prior conversations and events recorded below, update your plan for the following act. 
The act may be empty, ie have no scenes. In this case, you should flesh out the act with scenes based on it's title and place in the play. Make sure the overall narrative arc is maintained.
If there are scenes in the act, they may have already occurred in the performance so far, or may no longer be relevant (e.g. other actor's plans have changed), and if so should be removed or rewritten to reflect the new information.
Other scenes may need to be added or modified to better reflect the new information. Analyze the act description and design the new act to achieve the dramatic tension described. Extend with new scenes if needed.
Note also that time has passed since you originally formulated this act. For example, you might have been planning to have breakfast, but it may now be past noon.
the current time is {{$time}}
your original act was:

{{$act}}

and it was number {{$act_number}} in the your original narrative plan.

Critically, the play has progressed since you originally formulated this act, perhaps in unanticipated directions. The previous act post-state (ie, the actual situation after the previous act) is now:
{{$previous_act_post_state}}

The actual previous acts performed were an integration across all characters's plans, and are:

{{$previous_acts}}

The overall play Dramatic Context (central Question / Conflict driving all dramatic action is:
{{$central_narrative}}

The dramatic context agreed upon for the current act (ie, this act being replanned) is:
{{$act_central_narrative}}

"""
        suffix = """

Pay special attention to the goals you have already attempted and their completion status in planning your new act. 
Do not repeat goals that have already been attempted and completed, unless you have a new reason to attempt them again or it is important for dramatic tension.

Note that in the current situation any assumptions or preconditions on which your original act and plan were based may no longer be valid.
However, the original short-term plan above should still be used to locate your new plan within the overall narrative arc of the performance.

The following act-specific guidelines supplement the general guidelines above, and override them where necessary. Again, the current act number is {{$act_number}}:

** For act 1:
    1. This act should be short and to the point..
    2. Sequence scenes to introduce characters and create unresolved tension.
    3. Establish the central dramatic question clearly: {{$central_narrative}}
    4. Act post-state must specify: what the characters now know, what they've agreed together, and what specific tension remains unresolved.
    5. Final scene post-narrative must show characters making a concrete commitment or decision about their shared challenge.
    6. Ensure act post-state represents measurable progress from act pre-state, not just mood shifts.
 
** For act 2 (midpoint act):
    1. Each scene must advance the central dramatic question: {{$central_narrative}}
    2. Midpoint should fundamentally complicate the question (make it harder to answer or change its nature).
    3. Prevent lateral exploration - every scene should move closer to OR further from resolution..
    5. Avoid pointless repetition of scene intentions, but allow characters to develop their characters.
    6. Sequence scenes for continuously building tension, perhaps with minor temporary relief, E.G., create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)
    7. Ensure each scene raises stakes higher than the previous scene - avoid cycling back to earlier tension levels.
    8. Midpoint scene post-narrative must specify: what discovery/setback fundamentally changes the characters' approach to the central question.
    9. Act post-state must show: what new obstacles emerged, what the characters now understand differently, and what desperate action they're forced to consider.
    10. Each scene post-narrative must demonstrate measurable escalation from the previous scene - not just "tension increases" but specific new problems or revelations.

** For act 3 (final act):
    1. Directly answer the central dramatic question: {{$central_narrative}}
    2. No scene should avoid engaging with the question's resolution.
    3. Sequence scenes for maximum tension (alternate trust/mistrust beats)
    4. create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)  
    5. Act post-state must explicitly state: whether the General dramatic question was answered YES or NO, what specific outcome was achieved, and what the characters' final status is.
    6. Final scene post-narrative must show the concrete resolution - not "they find peace" but "they escape the forest together" or "they remain trapped but united."
    7. No Scene may end with ambiguous outcomes - each must show clear progress toward or away from resolving the central question.

** For act 4 (coda):
    1. Show the immediate aftermath and consequences of Act 3's resolution of: {{$central_narrative}}
    2. Maximum two scenes - focus on essential closure only, avoid extended exploration.
    3. Preserve character scene intentions where possible. Combine overlapping scene intentions from different characters where possible.
    4. First scene should show the immediate practical consequences of the resolution (what changes in their situation).
    5. Second scene (if needed) should show the emotional/relational aftermath (how characters have transformed).
    6. No new conflicts or dramatic questions - only reveal the implications of what was already resolved.
    7. Act post-state must specify: the characters' new equilibrium, what they've learned or become, and their final emotional state.
    8. Final scene post-narrative must provide definitive closure - show the "new normal" that results from their journey.
    9. Avoid ambiguity about outcomes - the coda confirms and completes the resolution, not reopens questions.


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
- "act_pre_state": (string, description of the situation / goals / tensions before the act starts. concise, about 10 words)
- "act_post_state": (string, description of the situation / goals / tensions after the act ends. concise, about 10 words)
- "tension_points": [
    {"characters": ["<Name>", ...], "issue": (string, concise (about 8 words) description of the issue), "resolution_requirement": (string, "partial" / "complete")}
    ...
  ]
- "scenes"      (array) 

Each **scene** object must have:
{ "scene_number": int, // sequential within the play 
 "scene_title": string, // concise descriptor 
 "location": string, // pick from resource or terrain names in the map file
 "time": "2025-01-01T08:00:00", // the start time of the scene, in ISO 8601 format
 "characters": { "<Name>": { "goal": "<one-line playable goal>" }, … }, 
 "action_order": [ "<Name>", … ], // each name occurrence is a 'beat' in the scene lead by the named character. list only characters present in the scene 'characters' list.
 "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors.", 
 "post_narrative": "Short prose (≤20 words) summarising end state and what emotional residue or new tension lingers." 
 // OPTIONAL: 
 "task_budget": 4 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the scene to avoid rambling or repetition, or to 1.67*len(characters) for scenes with complex goals or interactions.
 }

 === Example ===
##
```json
{
  "act_number": {{$act_number}},
  "act_title": "rewritten act title",
  "act_description": "concise (about 10 words) description of the act, focusing on it's dramatic tension and how it fits into the overall narrative arc",
  "act_goals" {"primary": "primary goal", "secondary": "secondary goal"},
  "act_pre_state": "description of the situation / goals / tensions before the act starts. concise, about 10 words",
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
      "pre_narrative": "Short prose (≤20 words) describing the immediate setup & stakes for the actors.", 
      "post_narrative": "Short prose (≤20 words) summarising end state and what emotional residue or new tension lingers." 
      // OPTIONAL: 
      "task_budget": 4 (integer) – the total number of tasks (aka beats) for this scene. set this to the number of characters in the action_order to avoid rambling or repetition. 
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
        response = default_ask(self, system_prompt=system_prompt, prefix=mission, suffix=suffix,
                              addl_bindings={"name": self.name, "act": json.dumps(reserialized_act['acts'][0], indent=1, default=datetime_handler), 
                               "act_number": act['act_number'],
                               "previous_act_post_state": previous_act_post_state,
                               "act_central_narrative": act_central_narrative,
                               "central_narrative": self.context.central_narrative,
                               "play": json.dumps(self.context.reserialize_acts_times(self.plan), indent=1, default=datetime_handler),
                               "previous_acts": '\n'.join([json.dumps(self.context.reserialize_act_times(act), default=datetime_handler) for act in self.context.previous_acts]) if self.context.previous_acts else ''},
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
        if updated_act:
            print(f'{self.name} updates act {act_id}')
            self.update_act(updated_act)
        else:
            print(f'{self.name} failed to update act {act_id}')
            updated_act = act
        self.current_act = updated_act
        self.current_act_index = act_number
        return updated_act

    async def negotiate_central_narrative(self, round:int, play_file_content, map_file_content=None):
        """Negotiate the central dramatic question with other characters"""

        mission="""You are {{$name}} engaging in collaborative story development with other characters to establish the central dramatic question that will drive the upcoming narrative.
"""

        suffix = """
# Your Character Profile
{{$character_description}}

## Cast & Setting
**Other Characters:**
{{$other_characters_profiles}}

**Setting & Context:**
{{$setting_background}}


## Negotiation Round {{round_number}} of 2

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


        response = default_ask(self, prefix=mission, suffix=suffix,
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
        if response:
            central_narrative = CentralNarrative.parse_from_hash(response)
            self.previous_proposal = self.current_proposal
            self.current_proposal = central_narrative
            self.central_narrative = central_narrative.to_string() # store as string, since that is how global central narrative is stored
            self.context.message_queue.put({'name':self.name, 'text':f'proposes {self.current_proposal.to_string()}\n'})
            await asyncio.sleep(0.4)
        return response
    
    def refresh_task(self, task, scene_task_plan, final_task=False):
        """Refresh the task to ensure it is up to date with the latest information
        adapted from Character.generate_task_plan"""

        if not self.focus_goal:
            raise ValueError(f'No focus goal for {self.name}')
        """generate task alternatives to achieve a focus goal"""
        goal = self.focus_goal        
        system_prompt = """You are an actor in an improvisational play. Review the following planned task in light of recent events. 

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
Review your planned task against current conditions and decide if it should be kept, revised, or replaced.
1. Choose to keep the task if it is still appropriate and achievable as planned
2. Choose to revise the task if the core intent is good but details need significant adjustment  
3. Choose to replace the task if circumstances have made a different approach better

Consider these factors:
- Has the scene dynamic shifted significantlyfrom your expectations?
- Have the relationships among planned actors changed in ways that make your task no longer appropriate?
- Do recent events create better opportunities?
- Does your task conflict with or duplicate others in the scene plan? Avoid duplication. Review past goals and their completion status to ensure you are not repeating yourself.

Respond with either:
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
        scene_task_plan = '\n'.join([f'{t["actor"].name}: {t["task"].to_string()}' for t in scene_task_plan])
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
                                "act_specific_narrative": self.context.act_central_narrative if self.context.act_central_narrative else '',
                                "central_narrative": self.context.central_narrative if self.context.central_narrative else '',
                                "my_narrative": self.central_narrative if self.central_narrative else '',
                                "scene": json.dumps(self.context.current_scene, indent=2, default=datetime_handler) if self.context.current_scene else '',
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
    
    def generate_new_focus_goal(self, goal):
        """ just completed a goal, generate the next focus goal to work on"""
        pass

    def closing_acts(self, situation: str, task: Task, goal: Goal=None):

        system_prompt = """You are {{$name}}, a character at the end of an improvisational play. 
This is your chance to reflect on the past, face the future, and decide if you have anything to say to any of the other characters.
You have two goals in this.
First, reach some closure with the other characters, as part of your improvisational performance in your role as {{$name}}.
Second, give the audience a feeling of closure, even if the overall dramatic question is left unresolved.

To achieve these goals, you will be given the option below to engage in a final dialog with one or more of the other characters.
"""
        
        prefix = """The overall dramatic question has been
#Central Dramatic Question
{{$central_narrative}}
##

In the final act it narrowed down to
#Act Specific Narrative
{{$act_specific_narrative}}
##

And the current state of resolution of that question is:
#Final Outcome
{{$final_outcome}}
##

"""
        suffix = """

First, identify the other characters you want to reach closure with.
Then, for each identified character, generate at most one Say action.
The Say action should be a short statement of your feelings about the experience you had with that character, and any remaining unexpressed thoughts or intentions wrt that character that need closure.
It should be one or two sentences, no more than 20 words.

Review your character and current emotional stance when choosing what to say.. 
Emotional tone and orientation can (and should) heavily influence the boldness, mode, phrasing and style of expression.
Remember, this is your final chance to reach closure with the other characters, and to give the audience a feeling of closure, even if the overall dramatic question is left unresolved.
Your may, as examples, wish to:
- Acknowledge the other character's role in the play.
- Ask the other character a question to resolve a remaining ambiguity about the Central Dramatic Question of the Act Specific Narrative.
- Express gratitude for the other character's help.
- Express regret or anger for your own or the other character's actions.
- Express hope for the other character's future.
- Express a desire to work together in the future.

Dialog (Say) guidance:
- The specificAct must contain only the actual words to be spoken.
- Respond in the style of spoken dialog, not written text. Pay special attention to the character's emotional stance shown above in choosing tone and phrasing. 
    Use contractions and language appropriate to the character's personality, emotional tone and orientation. Speak in the first person. DO NOT repeat phrases used in recent dialog.
- If intended recipient is known  or has been spoken to before (e.g., in RecentHistory), 
    then pronoun reference is preferred to explicit naming, or can even be omitted. 
- In any case, when using pronouns, always match the pronoun gender (he, she, his, her, they, their,etc) to the sex of the referent, or use they/their if a group. 
- Avoid repeating phrases in RecentHistory derived from the task, for example: 'to help solve the mystery'.
- Avoid repeating stereotypical past dialog.
- Avoid repeating dialog from 
{{$dialog_transcripts}}.
                              
Respond using the following hash-formatted text for each , where each tag is preceded by a # and followed by a single space, followed by its content.
Each intension should be closed by a ## tag on a separate line.
Be careful to insert line breaks only where shown, separating a value from the next tag.

#mode Say
#action thoughts, words to speak. Be concise, limit your response to 30 words max for mode Say.
#target name of the actor you are speaking to.
#duration 2 minutes
##

Do not include any introductory, explanatory, or discursive text. Remember, respond with 2 or at most 3 alternative acts.
End your response with:
</end>
"""

        response = default_ask(self, system_prompt=system_prompt, prefix=prefix, suffix=suffix, 
                               addl_bindings={
                                              "final_outcome": situation,
                                              "act_specific_narrative": self.context.act_central_narrative if self.context.act_central_narrative else '',
                                              "central_narrative": self.context.central_narrative if self.context.central_narrative else '',
                                              "name": self.name,
                                              "dialog_transcripts": '\n'.join(self.actor_models.get_dialog_transcripts(20))}, 
                               tag = 'NarrativeCharacter.closing_acts',
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