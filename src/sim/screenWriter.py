from __future__ import annotations
import os, sys, re, traceback, requests, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from enum import Enum
from typing import Any
from utils.Messages import UserMessage
import utils.xml_utils as xml
from typing import TYPE_CHECKING
from utils.llm_api import LLM


if TYPE_CHECKING:
    from sim.context import Context  # Only imported during type checking
    from sim.agh import Character
    from sim.memory.core import NarrativeSummary

class ScreenWriter:
    def __init__(self, play: str, map: str):
        self.play_file = play
        self.map_file = map
        self.play_file_content = open('../plays/demo.py', 'r').read()
        self.map_file_content = open('../plays/scenarios/coastal.py', 'r').read()
        self.llm = LLM('openai')
        #self.llm.set_model('gpt-4o-mini')

    def write_narrative(self):
        prompt=[UserMessage(content="""
<Play.py>
{{$play}}
</Play.py>

<Map.py>
{{$map}}
</Map.py>

You are a dramaturg-slash-game-director.

## 1.  INPUT FILES
You have been provided two Python source files:

▶ play.py  
    • Contains Character declarations in the pattern  
      `CharName = agh.Character("CharName", "...description...", ...)`  
    • After each character there is a list → `<Char>.drives = [Drive("..."), ...]`  
    • Percepts or initial internal lines appear via `<Char>.add_perceptual_input(...)`.  
    • At the end, a Context is created:  
      `context.Context([CharA, CharB, ...], "world-blurb", scenario_module=<map>)`  

▶ map.py  
    • Defines Enum classes for terrain, infrastructure, resource, and property types.  
    • Contains `resource_rules`, `terrain_rules`, etc.  (You may cite these names when choosing locations.)

## 2.  TASK
Generate a single JSON document named **narrative.json** that outlines a stage-friendly dramatic arc for the characters just discovered.

### 2.1  Structure
Return exactly one JSON object with these keys:

* `"title"` – a short, evocative play title.  
* `"acts"` – an array of act objects.  Each act object has  
  - `act_number` (int, 1-based)  
  - `act_title`   (string)  
  - `scenes`      (array)

Each **scene** object must have:
{ "scene_number": int, // sequential within the play 
 "scene_title": string, // concise descriptor 
 "location": string, // pick from resource or terrain names 
 "characters": { "<Name>": { "goal": "<one-line playable goal>" }, … }, 
 "action_order": [ "<Name>", … ], // 2-4 beats max, list only characters present 
 "pre_narrative": "Short prose (≤40 words) describing the immediate setup & stakes for the actors.", 
 "post_narrative": "Short prose (≤35 words) summarising what emotional residue or new tension lingers." 
 // OPTIONAL: 
 "task_budget": 4 (integer) – set only if you foresee dialogue loops in this scene 
 }



### 2.2  Guidelines
1. Base every character’s *stated goal* on their `.drives` and any percepts. Keep it actionable for the scene (e.g., “Convince Dana to stay”, not “Seek happiness”).  
2. Craft 3–4 acts, each 2–3 scenes (10–12 scenes total is ideal).  
3. Escalate tension act-to-act; give at least one irreversible turning point.  
4. Place scenes in plausible locations drawn from `map.py` resources/terrain.  
5. Aim for <u>dialogue-forward theatre</u>: lean on conflict & objective, not big visuals.  
6. Vary imagery and emotional tone; avoid repeating the same metaphor scene-to-scene.  
7. You may assign a `"task_budget"` when you anticipate the engine’s tendency to ramble (e.g., reassurance loops); omit it elsewhere.  
8. Do **NOT** invent new characters unless absolutely necessary, and never break JSON validity.  
9. Keep the JSON human-readable (indent 2 spaces).

Return **only** the JSON.  No commentary, no code fences.

""")
]

        narrative = self.llm.ask({"play": self.play_file_content, "map": self.map_file_content}, prompt, max_tokens=4000)
        return narrative

if __name__ == "__main__":

    screen_writer = ScreenWriter("plays/demo.py", "maps/demo.py")
    print(screen_writer.write_narrative())

    {
  "title": "Tides of Ambition",
  "acts": [
    {
      "act_number": 1,
      "act_title": "Crossroads at Dusk",
      "scenes": [
        {
          "scene_number": 1,
          "scene_title": "Harbor Bench Confessions",
          "location": "Bench",
          "characters": {
            "Maya": {
              "goal": "Confess my doubts about leaving the town to Elijah."
            },
            "Elijah": {
              "goal": "Understand what is troubling Maya and express my hopes for our future."
            }
          },
          "action_order": ["Maya", "Elijah"],
          "pre_narrative": "Maya and Elijah sit quietly on a bench as the sun sets, the weight of a job offer looming over their relationship.",
          "post_narrative": "Unspoken fears and fragile hopes hover as Maya remains torn and Elijah struggles to bridge their different dreams.",
          "task_budget": 4
        },
        {
          "scene_number": 2,
          "scene_title": "Gallery Owner's Pursuit",
          "location": "Gallery",
          "characters": {
            "Chrys": {
              "goal": "Follow up with Maya to secure her acceptance of my offer."
            }
          },
          "action_order": ["Chrys"],
          "pre_narrative": "Chrys grows impatient waiting for Maya’s reply, plotting the next move to win her over.",
          "post_narrative": "Chrys's ambition sharpens, unaware of the emotional turmoil Maya faces back home."
        },
        {
          "scene_number": 3,
          "scene_title": "Workshop Expansion Dreams",
          "location": "Workshop",
          "characters": {
            "Elijah": {
              "goal": "Share my expansion plans with Maya to root our future together."
            },
            "Maya": {
              "goal": "Listen and weigh Elijah’s future against my own ambitions."
            }
          },
          "action_order": ["Elijah", "Maya"],
          "pre_narrative": "Elijah eagerly reveals his boat-building expansion, hoping to anchor Maya’s heart to their shared home.",
          "post_narrative": "Maya feels the tension between stability and the call of new horizons growing stronger."
        }
      ]
    },
    {
      "act_number": 2,
      "act_title": "Rising Tensions",
      "scenes": [
        {
          "scene_number": 4,
          "scene_title": "Garden Confrontation",
          "location": "Garden",
          "characters": {
            "Maya": {
              "goal": "Express my fears about losing myself if I stay."
            },
            "Elijah": {
              "goal": "Convince Maya that our life here is worth fighting for."
            }
          },
          "action_order": ["Maya", "Elijah"],
          "pre_narrative": "In the quiet garden, emotions boil over as Maya and Elijah confront the fracture between dreams and reality.",
          "post_narrative": "Words leave wounds, and the fragile balance of their relationship teeters on the edge."
        },
        {
          "scene_number": 5,
          "scene_title": "City Call",
          "location": "Gallery",
          "characters": {
            "Chrys": {
              "goal": "Press Maya to commit, emphasizing the prestige and opportunity."
            }
          },
          "action_order": ["Chrys"],
          "pre_narrative": "Back in the city, Chrys intensifies her efforts to lure Maya away with promises of fame and fortune.",
          "post_narrative": "Chrys grows more determined; Maya’s decision feels increasingly inevitable."
        },
        {
          "scene_number": 6,
          "scene_title": "Harbor at Night",
          "location": "Harbor",
          "characters": {
            "Maya": {
              "goal": "Find clarity alone, weighing what I must leave behind."
            }
          },
          "action_order": ["Maya"],
          "pre_narrative": "Maya wanders the harbor under moonlight, the water reflecting her inner turmoil between love and ambition.",
          "post_narrative": "A quiet resolve begins to form, but the cost remains unclear."
        }
      ]
    },
    {
      "act_number": 3,
      "act_title": "Decisions and Fallout",
      "scenes": [
        {
          "scene_number": 7,
          "scene_title": "Workshop Ultimatum",
          "location": "Workshop",
          "characters": {
            "Elijah": {
              "goal": "Demand Maya choose between her career and us."
            },
            "Maya": {
              "goal": "Stand firm in my choice, whatever the consequences."
            }
          },
          "action_order": ["Elijah", "Maya"],
          "pre_narrative": "Tensions peak as Elijah confronts Maya with an ultimatum, forcing a decisive moment.",
          "post_narrative": "Their bond fractures irreversibly; the future feels uncertain and raw.",
          "task_budget": 4
        },
        {
          "scene_number": 8,
          "scene_title": "City Gallery Offer",
          "location": "Gallery",
          "characters": {
            "Chrys": {
              "goal": "Welcome Maya into my gallery, sealing the deal."
            },
            "Maya": {
              "goal": "Accept the offer and commit to a new life."
            }
          },
          "action_order": ["Chrys", "Maya"],
          "pre_narrative": "Maya arrives at the gallery, the weight of her decision pressing down as Chrys awaits her answer.",
          "post_narrative": "A new chapter begins, tinged with excitement and a bittersweet sense of loss."
        },
        {
          "scene_number": 9,
          "scene_title": "Elijah's Workshop Solitude",
          "location": "Workshop",
          "characters": {
            "Elijah": {
              "goal": "Come to terms with Maya’s departure and plan my path forward."
            }
          },
          "action_order": ["Elijah"],
          "pre_narrative": "Elijah, alone in the workshop, wrestles with grief and the future of his craft and community.",
          "post_narrative": "Though heartbroken, a quiet determination to rebuild emerges."
        }
      ]
    },
    {
      "act_number": 4,
      "act_title": "New Beginnings and Lingering Ghosts",
      "scenes": [
        {
          "scene_number": 10,
          "scene_title": "Gallery Opening Night",
          "location": "Gallery",
          "characters": {
            "Maya": {
              "goal": "Showcase my art and prove my worth in the city."
            },
            "Chrys": {
              "goal": "Celebrate the success and solidify our partnership."
            }
          },
          "action_order": ["Maya", "Chrys"],
          "pre_narrative": "Under bright lights, Maya unveils her work, stepping fully into her ambitions.",
          "post_narrative": "Success is bittersweet; echoes of home and past loves linger in the celebratory air."
        },
        {
          "scene_number": 11,
          "scene_title": "Harbor Reflection",
          "location": "Harbor",
          "characters": {
            "Elijah": {
              "goal": "Reflect on what was lost and what can still be saved."
            }
          },
          "action_order": ["Elijah"],
          "pre_narrative": "Elijah watches the harbor alone, memories washing over him like the tide.",
          "post_narrative": "A mix of sorrow and hope settles, hinting at healing yet to come."
        },
        {
          "scene_number": 12,
          "scene_title": "Unexpected Letter",
          "location": "Workshop",
          "characters": {
            "Elijah": {
              "goal": "Read Maya’s letter and decide how to respond."
            }
          },
          "action_order": ["Elijah"],
          "pre_narrative": "A letter arrives from Maya, bridging the gulf between their worlds with uncertain intent.",
          "post_narrative": "The possibility of reconciliation or new understanding remains open, fragile yet alive."
        }
      ]
    }
  ]
}
