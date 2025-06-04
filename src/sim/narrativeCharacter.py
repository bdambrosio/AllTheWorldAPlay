from __future__ import annotations
import asyncio
import copy
from datetime import datetime
from itertools import tee
import logging
from typing import Optional, Dict, List, Any
#from venv import create

#from attr import Out
from pyparsing import cast
import os, sys, re, traceback, requests, json

from sympy import evaluate
from trimesh import Scene
from unstructured_client import General
from sim.character_dataclasses import Act, Task, CentralNarrative, datetime_handler
from src.utils.Messages import UserMessage
from utils import hash_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from enum import Enum
from typing import TYPE_CHECKING
from utils.llm_api import LLM
from sim.prompt import ask as default_ask
from sim.agh import Character

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
        self.round_number = 0
        #self.llm.set_model('gpt-4o-mini')

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

    def validate_narrative_json(self, json_data: Dict[str, Any], require_scenes=True) -> tuple[bool, str]:
        """
        Validates the narrative JSON structure and returns (is_valid, error_message)
        """
        # Check top-level structure
        if not isinstance(json_data, dict):
            return False, "Root must be a JSON object"
    
        if "title" not in json_data or not isinstance(json_data["title"], str):
            return False, "Missing or invalid 'title' field"
        
        if "acts" not in json_data or not isinstance(json_data["acts"], list):
            return False, "Missing or invalid 'acts' array"
        
        # Validate each act
        for n, act in enumerate(json_data["acts"]):
            if not isinstance(act, dict):
                return False, f"Act {n} must be a JSON object"
            else:
                valid, json_data["acts"][n] = self.validate_narrative_act(act, require_scenes=require_scenes)
                if not valid:
                    return False, f"Act {n} is invalid: {json_data['acts'][n]}"
                
        return True, "Valid narrative structure"
 
    def validate_narrative_act(self, act: Dict[str, Any], require_scenes=True) -> tuple[bool, str]:       # Validate each act
        if not isinstance(act, dict):
            return False, "Act must be a JSON object"
                
        # Check required act fields
        if "act_number" not in act or not isinstance(act["act_number"], int):
            return False, "Missing or invalid 'act_number'"
        if "act_description" not in act or not isinstance(act["act_description"], str):
            return False, "Missing or invalid 'act_description'"
        if "act_goals" not in act or not isinstance(act["act_goals"], dict):
            return False, "Missing or invalid 'act_goals'"
        if "act_pre_state" not in act or not isinstance(act["act_pre_state"], str):
            return False, "Missing or invalid 'act_pre_state'"
        if "act_post_state" not in act or not isinstance(act["act_post_state"], str):
            return False, "Missing or invalid 'act_post_state'"
        if "tension_points" not in act or not isinstance(act["tension_points"], list):
            return False, "Missing or invalid 'tension_points'"
        for tension_point in act["tension_points"]:
            if not isinstance(tension_point, dict):
                return False, "Tension point must be a JSON object"
            if "characters" not in tension_point or not isinstance(tension_point["characters"], list):  
                return False, "Tension point must have 'characters' array"
            if "issue" not in tension_point or not isinstance(tension_point["issue"], str):
                return False, "Tension point must have 'issue' string"
            if "resolution_requirement" not in tension_point or not isinstance(tension_point["resolution_requirement"], str):
                return False, "Tension point must have 'resolution_requirement' string"
        if require_scenes:
            if act["act_number"] == 1:
                if "scenes" not in act or not isinstance(act["scenes"], list):
                    return False, "First act must have 'scenes' array"
        if "scenes" in act:
            # Validate each scene
            for scene in act["scenes"]:
                if not isinstance(scene, dict):
                    return False, "Scene must be a JSON object"
                        
                # Check required scene fields
                required_fields = {
                    "scene_number": int,
                    "scene_title": str,
                    "location": str,
                    "time": str,
                    "duration": int, # in minutes
                    "characters": dict,
                    "action_order": list,
                    "pre_narrative": str,
                    "post_narrative": str
                }
                    
                for field, field_type in required_fields.items():
                    if field not in scene or not isinstance(scene[field], field_type):
                        if field == 'pre_narrative' or field == 'post_narrative':
                            scene[field] = ''
                        elif field == 'duration':
                            scene[field] = 15
                        else:
                            return False, f"Missing or invalid '{field}' in scene"
                    
                # Validate time field
                scene_time = self.validate_scene_time(scene)
                if scene_time is None:
                    return False, f"Invalid time format in scene {scene['scene_number']}"
                else:
                    scene["time"] = scene_time # update the time field with the validated datetime object
                    
                # Validate characters structure
                for char_name, char_data in scene["characters"].items():
                    if not isinstance(char_data, dict) or "goal" not in char_data:
                        return False, f"Invalid character data for {char_name}"
                    
                # Validate action_order
                if not 1 <= len(scene["action_order"]) <= 4:
                    logger.info(f'{self.name} validate_narrative_act: scene {scene["scene_number"]} has {len(scene["action_order"])} actions')
                    if len(scene["characters"]) > 3:
                        # too many characters
                        return False, f"Scene {scene['scene_number']} must have 2-4 actions"
                    action_order = scene["action_order"]
                    characters_in_scene = []
                    new_action_order = []
                    for character in action_order:
                        if character not in characters_in_scene:
                            characters_in_scene.append(character)
                            new_action_order.append(character)
                    scene["action_order"] = new_action_order

                # Validate optional task_budget
                if "task_budget" in scene and not isinstance(scene["task_budget"], int):
                    return False, f"Invalid task_budget in scene {scene['scene_number']}"
                elif "task_budget" in scene and scene["task_budget"] > 2*len(scene["action_order"]):
                    logger.debug(f"task_budget in scene {scene['scene_number']} is too high")
                    scene["task_budget"] = int(1.75*len(scene["action_order"])+1)
                elif "task_budget" not in scene:
                    scene["task_budget"] = int(1.75*len(scene["action_order"])+1)

                # Validate narrative lengths
                if len(scene["pre_narrative"].split()) > 120:
                    return False, f"Pre-narrative too long in scene {scene['scene_number']}"
                if len(scene["post_narrative"].split()) > 120:
                    return False, f"Post-narrative too long in scene {scene['scene_number']}"
        
        return True, act

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

    async def introduce_myself(self, play_file_content, map_file_content):

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
                                 "map": map_file_content,
                                 "name": self.name,
                                 "start_time": self.context.simulation_time.isoformat(),
                                 "primary_drive": f'{self.drives[0].id}: {self.drives[0].text}; activation: {self.drives[0].activation:.2f}',
                                 "secondary_drive": f'{self.drives[1].id}: {self.drives[1].text}; activation: {self.drives[1].activation:.2f}'}, 
                                 max_tokens=200, tag='narrative')
        if introduction:
            self.context.message_queue.put({'name':self.name, 'text':introduction+'\n'})
            await asyncio.sleep(0)
    


    async def write_narrative(self, play_file_content:str, map_file_content:str, central_narrative:CentralNarrative):

        mission = """You are a skilled playwright working on an initial outline of the narrative arc for an improvisational play. 
The overall cast of characters and setting are given below.

<Scenario>
{{$scenario}}
</Scenario>

<Map>
{{$map}}
</Map>

The negotiation process has established the following central dramatic question that will drive the play:
{{$central_narrative}}

* central_narrative is a paragraph or two including the following elements:
1. Central Dramatic Question: One clear sentence framing the core conflict
2. Stakes: What happens if this question isn't resolved - consequences that matter to all
3. Character Roles: for each character, what is their role in the conflict? Protagonist/Antagonist/Catalyst/Obstacle - with 1-2 sentence role description
4. Dramatic Tension Sources: The main opposing forces that will drive the narrative
5. Success Scenario: Brief description of what "winning" looks like
6. Failure Scenario: Brief description of what "losing" looks like

After processing the above information and in light of the following information, generate a 3 act plus coda outline for the play.
Your generated narrative should be consistent with the central dramatic question and the other elements of the central narrative.

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
                            
        suffix="""
Respond only with your introduction.
Do not include any other explanatory, discursive, or formatting text.
End your response with </end>
"""

        system_prompt = """You are a seasoned writer designing a 2-3 act narrative arc for a 30 minutemovie.
Every act should push dramatic tension higher: give the protagonist a clear want, place obstacles in the way, and end each act changed by success or setback.
Keep the stakes personal and specific—loss of trust, revelation of a secret, a deadline that can’t be missed—so the audience feels the pulse of consequence.
Let conflict emerge through spoken intention and subtext, as well as through the characters' actions and reactions with the world and each other.
Characters hold real agency; they pursue goals, make trade-offs, and can fail. Survival chores are background unless they expose or escalate the core mystery.
Use vivid but economical language, vary emotional tone, and avoid repeating imagery.
By the final act, resolve—or intentionally leave poised—the protagonist’s primary drive. A very short final act can be added as a coda.
        """

        suffix = """

Imagine you are {{$name}} and your drives include:

Primary 
{{$primary_drive}}

Secondary
{{$secondary_drive}}

## 2.  TASK
Generate a single JSON document named **narrative.json** that outlines the narrative arc for yourself focused on your primary and secondary drives.
The narrative should start at the current time: {{$start_time}}
Given where you are in life, what you have achieved so far, and what you want to achieve, this should be a plan for a long enough elapse time to make interesting dramatic progress wrt a primary issue in the scenario.
Survival tasks (food, water, shelter) are assumed handled off-stage unless they advance the mystery or dramatic tension.
By the end of the narrative, the primary drive should be resolved.
### 2.1  Structure
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
2. Craft 3 acts plus a single-scene Coda - keep the momentum going, don't add acts just to add acts. By the end there should be some resolution of the dramatic tension and the characters' primary drives. An additional final act can be used as a coda.
3. Escalate tension act-to-act; expect characters to be challenged, and to be forced to reconsider their goals and perhaps change them in future.  
4. Place scenes in plausible locations drawn from `map.py` resources/terrain.  
5. Aim for <u>dialogue-forward theatre</u>: lean on conflict & objective, not big visuals, although an occasional hi-impact visual can be used to enhance the drama.  
6. Vary imagery and emotional tone; avoid repeating the same metaphor scene-to-scene.  
7. Do **NOT** invent new characters unless absolutely necessary, and never break JSON validity.  
8. Keep the JSON human-readable (indent 2 spaces).

Return **only** the JSON.  No commentary, no code fences.

"""

        narrative = default_ask(self, system_prompt=system_prompt, prefix = mission, suffix = suffix, 
                                addl_bindings={"scenario": self.context.summarize_scenario(), 
                                 "map": self.context.summarize_map(),
                                 "name": self.name,
                                 "central_narrative": central_narrative,
                                 "start_time": self.context.simulation_time.isoformat(),
                                 "primary_drive": f'{self.drives[0].id}: {self.drives[0].text}; activation: {self.drives[0].activation:.2f}',
                                 "secondary_drive": f'{self.drives[1].id}: {self.drives[1].text}; activation: {self.drives[1].activation:.2f}'}, 
                                 max_tokens=5000, tag='narrative')
        try:
            self.plan = json.loads(narrative.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            print(f'Error parsing narrative: {e}')
            self.plan = self.context.repair_json(narrative, e)
        valid, reason = self.validate_narrative_json(self.plan, require_scenes=False)
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
                                 max_tokens=240, tag='share_narrative')
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

    def update_act(self, new_act):
        valid, reason = self.validate_narrative_act(new_act, require_scenes=True)
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
 "action_order": [ "<Name>", … ], // 2-4 beats max, list only characters present 
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
      "action_order": [ "Character 1, ...,... ], // 2-4 beats max, list only characters present but a character can act more than once in a scene
      "pre_narrative": "Pre-narrative 1",
      "post_narrative": "Post-narrative 1"
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
End your response with </end>
"""
        dialogs = self.actor_models.dialogs()

        response = default_ask(self, system_prompt=system_prompt, prefix=mission, suffix=suffix,
                              addl_bindings={"name": self.name, 
                               "dialogs": '\n'.join(dialogs),
                               "plan": json.dumps(self.context.reserialize_acts_times(self.plan), default=datetime_handler),
                               "primary_drive": f'{self.drives[0].id}: {self.drives[0].text}; activation: {self.drives[0].activation:.2f}',
                               "secondary_drive": f'{self.drives[1].id}: {self.drives[1].text}; activation: {self.drives[1].activation:.2f}'},
                              max_tokens=1600, tag='update_narrative')
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

    def replan_narrative_act(self, act, previous_act, act_central_narrative:CentralNarrative):
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
The actual previous acts performed were an integration across all characters's plans, and are:

{{$previous_acts}}

The overall play Dramatic Context (central Question / Conflict driving all dramatic action is:
{{$central_narrative}}

The dramatic context agreed upon for the current act (ie, this act being replanned) is:
{{$act_central_narrative}}

"""
        suffix = """

Again, your original act plan was:
Act number: {{$act_number}}
{{$act}}

Note that the current situation, as well as any assumptions or preconditions on which your original act and plan were based, may no longer be valid.
However, the original short-term plan above should still be used to locate your new plan within the overall narrative arc of the performance.

The following act-specific guidelines supplement the general guidelines above, and override them where necessary. Again, the current act number is {{$act_number}}:

** For act 1:
    1. This act should be short and to the point..
    2. Sequence scenes to introduce characters and create unresolved tension.
    3. Establish the central dramatic question clearly: {{$central_narrative}}
    4. Act post-state must specify: what the characters now know, what they've agreed to do together, and what specific tension remains unresolved.
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


Respond with tee updated act, using the following format:

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
 "action_order": [ "<Name>", … ], // 2-4 beats max, list only characters present 
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
      "action_order": [ "Character 1, ...,... ], // 2-4 beats max, list only characters present but a character can act more than once in a scene
      "pre_narrative": "Pre-narrative 1",
      "post_narrative": "Post-narrative 1"
    },
    ...
  ]
}

Again, the act central narrative is:

{{$act_central_narrative}}

```
=== End of Example ===

End your response with </end>
"""
        acts = {"acts": [act]} # reserialize expects a narrative json object
        reserialized_act = self.context.reserialize_acts_times(acts)
        act_number = act['act_number']
        response = default_ask(self, system_prompt=system_prompt, prefix=mission, suffix=suffix,
                              addl_bindings={"name": self.name, "act": json.dumps(reserialized_act['acts'][0], indent=1, default=datetime_handler), 
                               "act_number": act['act_number'],
                               "act_central_narrative": act_central_narrative,
                               "central_narrative": self.context.central_narrative,
                               "play": json.dumps(self.context.reserialize_acts_times(self.plan), indent=1, default=datetime_handler),
                               "previous_acts": '\n'.join([json.dumps(self.context.reserialize_act_times(act), default=datetime_handler) for act in self.context.previous_acts]) if self.context.previous_acts else ''},
                              max_tokens=1400, tag='replan_narrative_act')
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
        return updated_act

    async def negotiate_central_narrative(self, round:int, play_file_content, map_file_content):
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

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Close the hash-formatted text with ##  on a separate line, as shown below.
be careful to insert line breaks only where shown, separating a value from the next tag:

#Question centeral question or conflict terse (6-10 words)
#Why dramatic potential (6-8 words)
#Reason my character's stake in this question / conflict (6-10 words)
#Others roles in this question / conflict (8-12 words)
#Risks weaknesses in this proposal(4-6 words)
##

What dramatic question do you propose?
End your response with </end>
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
                               "map_file_content": map_file_content},
                              max_tokens=100, tag='negotiate_central_narrative')
        if response:
            central_narrative = CentralNarrative.parse_from_hash(response)
            self.previous_proposal = self.current_proposal
            self.current_proposal = central_narrative
            self.context.message_queue.put({'name':self.name, 'text':f'proposes {self.current_proposal.to_string()}\n'})
            await asyncio.sleep(0.4)
        return response
    
    """
    def evaluate_commitment_significance(self, other_character:Character, commitment_task:Task):
        prompt = [UserMessage(content="valuate the significance of a tentative verbal commitment to the current dramatic context
    and determine if it is a significant commitment that should be added to the task plan.
The apparent commitment is inferred from a verbal statement to act now made by self, {{$name}}, to other, {{$target_name}}.
The apparent commitment is tentative, and may be redundant in the scene you are in.
The apparent commitment may be hypothetical, and may not be a commitment to act now.
The apparent commitment may be social chatter, e.g. simply a restatement of already established plans. These should be reported as NOISE.
The apparent commitment may simply be 'social noise' - a statement of intent or opinion, not a commitment to act now. Again, these should be reported as NOISE.
The apparent commitment may be relevant to the scene you are in or the dramatic context, and unique wrt tasks already in the task plan. These should be reported as RELEVANT.
The apparent commitment may be relevant to the scene you are in or the dramatic context, but redundant with tasks already in the task plan. These should be reported as REDUNDANT.
Alternatively, the apparent commitment may be unique and significant to the scene you are in or the central dramatic question. These should be reported as SIGNIFICANT.
The apparent commitment may be irrelevant to the scene you are in or the dramatic context. These should be reported as NOISE.

## Dramatic Context
<central_narrative>
{{$central_narrative}}
</central_narrative>

<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

<current_scene>
{{$current_scene}}
</current_scene>

##Scene Task Plan
<scene_integrated_task_plan>
{{$scene_integrated_task_plan}}
</scene_integrated_task_plan>

##Actual Scene History to date
<scene_history> 
{{$scene_history}}
</scene_history>

<transcript>    
{{$transcript}}
</transcript>
        
## Apparent Commitment
<commitment>
{{$commitment_task}}
</commitment>

Your task is to evaluate the significance of the apparent commitment.
Determine if it is 
    social NOISE that should be ignored.
    a REDUNDANT task that can be ignored. 
    a RELEVANT task that can be optionally included.
    a SIGNIFICANT commitment that should be added to the task plan,  
Respond with a single word: "NOISE", "REDUNDANT", RELEVANT", or "SIGNIFICANT".
Do not include any other text in your response.

End your response with </end>
"
        )]
        significance = 'NOISE'
        response = self.llm.ask({"name": self.name, 
                               "target_name": other_character.name,
                               "commitment_task": commitment_task.to_string(),
                               "central_narrative": self.context.central_narrative,
                               "act_specific_narrative": self.context.act_central_narrative,
                               "current_scene": json.dumps(self.context.current_scene, indent=2, default=datetime_handler) if self.context.current_scene else '',
                               "scene_integrated_task_plan": self.context.format_scene_integrated_task_plan() if self.context.scene_integrated_task_plan else json.dumps(self.goal.task_plan, indent=2, default=datetime_handler),
                               "scene_history": '\n'.join([history for history in self.context.scene_history]) if self.context.scene_history else '',
                               "transcript": self.actor_models.get_actor_model(other_character.name, create_if_missing=True).dialog.get_transcript(8)},
                                prompt, max_tokens=100, stops=['</end>'], tag='evaluate_commitment_significance')
        if response:
            response = response.lower()
            if 'significant' in response:
                return 'SIGNIFICANT'
            elif 'relevant' in response:
                return 'RELEVANT'
            elif 'redundant' in response:
                return 'REDUNDANT'
            else:
                return 'NOISE'
        return 'NOISE'
    """