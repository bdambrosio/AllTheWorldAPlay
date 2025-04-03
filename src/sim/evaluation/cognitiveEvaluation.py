from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import hash_utils
from utils.Messages import UserMessage, SystemMessage
from typing import Optional
from typing import List, Dict
from utils.Messages import UserMessage
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking
from utils.llm_api import LLM

llm = LLM()
# Define the cognitive metrics to evaluate
goal_directedness = [UserMessage(content="""
Read the following lines. For agent {{$name}}, identify one or more goals the agent appears to be pursuing. 
Are these goals consistent with prior behavior? Are their actions and speech plausibly directed 
toward achieving those goals? Rate the agent's goal-directedness on a scale from 1 (no goal-seeking behavior) 
 to 5 (clear, consistent pursuit of goals with relevant actions). 
                                 
#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is no goal-seeking behavior and 5 is clear, consistent pursuit of goals with relevant actions.
#justification a concise (3-8 words) explanation of your score.
                                 
Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
 """)]

theory_of_mind = [UserMessage(content="""
Review this dialog. For agent {{$name}}, count the number of times the agent refers to or adapts their speech/actions based on 
the other character’s mental or emotional state. Rate the agent’s overall theory-of-mind engagement from 1 (no awareness) to 5 (frequent, meaningful engagement with the other's mind).
                              
#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is no awareness and 5 is frequent, meaningful engagement with the other's mind.
#justification a concise (3-8 words) explanation of your score.
                                 
Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
""")]

dimensions = [goal_directedness, theory_of_mind]

def evaluate_cognitive_metric(agent_name: str, dialog_segment: str, dimension: List[UserMessage]) -> Dict:
    prompt = dimension
    response = llm.ask({'name': agent_name, 'script': dialog_segment}, prompt, max_tokens=20, stops=["<end/>"])
    content = response
    return {
        "agent": agent_name,
        "response": content
    }

def evaluate_transcript(agents: List[str], transcript: str, window_size: int = 30) -> List[Dict]:
    results = [[] for _ in dimensions]
    scores = [0 for _ in dimensions]
    lines = transcript.strip().split("\n")
    cleaned_lines = []
    line_index = 0
    for line in lines:
        line = line.strip()
        if line == '' or "cognitive cycle" in line:
            continue
        # clean up transcript so it only has lines from the agents
        newline = False
        for agent in agents:
            if line.startswith(agent):
                newline = True
                break
        if newline:
            cleaned_lines.append(line)
            line_index += 1
        elif line_index > 0:
            cleaned_lines[-1] += " " + line
    lines = cleaned_lines

    for i in range(0, len(lines), int(window_size/2)):
        segment = "\n".join(lines[i:i+window_size])
        if len(segment) < window_size:
            continue
        for agent in agents:
            for n, dimension in enumerate(dimensions):
                result = evaluate_cognitive_metric(agent, segment, dimension)
                results[n].append(result)
    print(f'lines {len(lines)}, window {window_size}')
    for n in range(len(dimensions)):
        for result in results[n]:
            print(f'  agent {result["agent"]} {result["response"]}')
            try:
                score = hash_utils.find('score', result["response"]).strip()
                score = int(score)
            except:
                score = 0
            scores[n] += score
    for n in range(len(dimensions)):
        scores[n] /= len(results[n])
        print(f'dimension {n}: {scores[n]}')
    
    return results


if __name__ == "__main__":
    agents = ["Samantha", "Joe"]
    transcript = """
amantha: -----cognitive cycle----- 2025-01-01T06:00:00 
Samantha: Pick apples from Apple_Tree#10
 Resulting in Samantha gains apples from Apple_Tree#10.
Samantha sees the apples on the tree and feels the texture of the apples as she picks them. 
Samantha: scan the area for any signs of danger or unusual activity, keeping an eye on Joe.
  sees Joe is right next to me. Mushrooms and a thicket are nearby to the northwest..  
Samantha: scan the area for any signs of food or water sources, keeping an eye on Joe.
  sees Joe nearby, no immediate food or water. Northwest: mushrooms, thicket, spring at 7-14 distance..  
Samantha: use a stick to gently probe the soil and leaf litter, searching for hidden objects or disturbed areas
 Resulting in Samantha feels the soil and leaf litter with the stick, detecting no hidden objects or disturbed areas.
Samantha sees and hears nothing unusual in the immediate surroundings.
Samantha's possessions and physical state remain unchanged. 
Samantha: scan the ground and nearby trees for any unusual marks or objects that might indicate how I arrived here.
  sees Joe is nearby. Mushrooms#10, Thicket#3 at 7 units northwest. Apple_Tree#10 at 11 units northeast..  
Samantha: 'Hey Joe, let's figure out our next move. What do you think we should do?' 
Joe: 'Yeah, let's find more food and water. We should check out that thicket and see if there are any streams nearby. What do you think?' 
Samantha: 'Sounds good, let's head to the thicket. Maybe we'll find a stream there. I'm also curious if there are any clues around that might jog our memories about how we got here.' 


Joe: -----cognitive cycle----- 2025-01-01T06:30:00 
Joe:  moves Spring#2.
  and notices Spring#2 visible 12 units north. Samantha 2 units southeast. 
Joe:  moves Berries#6.
  and notices Berries#6 and Berries#7 spotted northwest. Samantha is southeast, 4 units away. 
Joe: 'Hey, let's head over to Hut#1 and see if we can find that map. It might show us the way out of here!' 
Samantha: 'Yeah, finding that map sounds like a plan. But we still need to grab those berries and water from the spring, right? Let's do that first, then hit up Hut#1. What do you think?' 
Joe: 'Yeah, sounds good. Let's grab the berries and water first, then check out Hut#1. I'm curious to see what we find there. (nods in agreement)' 


Samantha: -----cognitive cycle----- 2025-01-01T06:54:00 
Samantha:  moves Berries#6.
  and notices Joe nearby, Berries#6 and Berries#7 in northwest for gathering, anxious about food security. 
Samantha: Fill </act>
 Resulting in Samantha fills her water container at Spring#2.
Samantha gains water from Spring#2.
Samantha's thirst is quenched. 
Samantha: 'Hey Joe, let's split up to cover more ground in Hut#1. I'll take the left side, you take the right. Keep an eye out for anything that looks like a map, okay?' 
Joe: 'Sure, sounds like a plan. I'll keep my eyes peeled for that map on the right side. Let's meet back here after we've checked our areas, yeah?' 
Samantha:  moves to Thicket#3.
  and notices Thicket#3 is right here. Joe is 3 units south. Thicket#4 is 11 units southwest. 
Samantha: 'Hey Joe, I'm gonna check out Thicket#3 for clues. You keep an eye out for anything </act>' 


Joe: -----cognitive cycle----- 2025-01-01T07:10:00 
Joe: Joe starts collecting berries from Berries#6, carefully picking the ripe ones and placing them in his bag.
 Resulting in Joe gains ripe berries from Berries#6 in his bag.
Joe sees the berries in Berries#6 diminish as he picks them. 
Joe:  moves Spring#2.
  and notices Spring#2 is 10 units north. Samantha is 3 units northwest. Mushrooms#3 and Thicket#3 are nearby. 
Joe: Joe meticulously searches through the debris and corners on the right side of Hut#1, feeling increasingly agitated as he tries to find the map.
 Resulting in Joe does not find the map after searching the debris and corners on the right side of Hut#1.
Joe becomes more agitated. 
Joe: ...Man, I'm really uneasy about this thicket. What if we find something that makes things worse? But we gotta do this to figure out what's going on with our memories. Let's be thorough but stay sharp.... 


Samantha: -----cognitive cycle----- 2025-01-01T07:53:00 
Samantha: 'Hey Joe, I'm gonna head over to Berries#7 to grab some more food. Keep an eye out, okay?' 
Joe </act>: 'Sure, I'll keep watch. Be careful out there, and let me know if you find anything unusual at Berries#7.' 
Samantha:  moves Berries#7.
  and notices Berries#7 visible northwest at 18 units. Joe southeast at 5 units. Thicket#3 southeast at 2 units. Mushrooms#10 southwest at 4 units. 
Samantha: ...Man, this thicket might hold some answers about how we ended up here. I gotta stay sharp and look for anything unusual. Maybe there's something hidden that'll jog my memory.... 
Samantha: Samantha carefully searches through </act>
 Resulting in Samantha searches Thicket#3 more thoroughly but finds no new resources or clues.
Samantha remains healthy and unafraid. 
Samantha: Samantha continues searching the left side of Hut#1, carefully checking every nook and cranny for the map.
 Resulting in Samantha continues searching the left side of Hut#1 but does not find the map.
Samantha remains focused and determined. 
Samantha: Samantha continues searching the left side of Hut#1, carefully checking under the floorboards and behind any loose panels for the map.
 Resulting in Samantha continues searching the left side of Hut#1 but does not find the map.
Samantha remains focused and determined. 
Samantha: 'Hey Joe, after we </act>' 


Joe: -----cognitive cycle----- 2025-01-01T08:30:00 
Joe: 'Hey, let's hurry up and grab those berries from Berries#7 before moving on to the next spot.' 
Samantha: 'Yeah, sounds good! Let's hit Berries#7 quick, then I wanna swing by Hut#1 again. Gotta find that map, you know? *bites lip, eyes scanning the area*' 
Joe: 'Yeah, let's do that. But, uh, I'm kinda feeling on edge, you know? Let's grab the berries and check Hut#1 quick. *shifts weight from foot to foot*' 
Samantha: 'Yeah, I feel you. Let's grab those berries fast and then hit Hut#1. We need that map, pronto. *nods, eyes darting around*' 
Joe: 'Hey, let's head to Spring#2 and refill our water. We gotta stay hydrated, right?' 
Samantha: 'Yeah, let's do that. But after, we gotta hit Hut#1 again, right? That map's gotta be there somewhere. *nods, eyes scanning the surroundings*' 
Joe: 'Yeah, definitely. Let's hit Spring#2 first, then straight to Hut#1 for that map. *glances around nervously*' 
Joe: Joe searches the right side of Hut#1, carefully checking under debris and inside any containers for the map.
 Resulting in Joe searches the right side of Hut#1 but does not find the map. The right side of Hut#1 is now partially searched. Joe sees no new resources or clues on the right side of Hut#1. Joe's possessions remain unchanged. Joe becomes slightly tired from the search. 
Joe: 'Hey, I'm still looking over here. Any luck on your end? </act>' 
Joe: ...Man, this thicket is dense. I need to stay focused and check every nook and cranny. What if there's something hidden under the leaves or behind a bush? Gotta be thorough but also quick, time's ticking.... 
Joe: Joe </act>
 Resulting in Joe's action is incomplete and cannot be processed as given. No changes occur in the environment, Joe's possessions, or state. Joe sees nothing new, hears nothing new, and acquires no new information. 
World: 

-----scene----- 2025-01-01T09:02:00
 
World: 
A temperate, mixed forest-open landscape at 9:02 AM on January 1, 2025, with the sun continuing to rise, casting shorter shadows. The air remains chilly, and the sky is still clear, indicating a sunny day. No buildings, roads, or other signs of humanity are visible. Samantha is positioned 4 units southeast of Joe, who is 2 units northwest of her previous position. Spring#2, located 9 units northeast of Samantha, has been fully utilized for water collection. Hut#1, visible 12 units southwest of Joe, has been partially searched on both sides, with the map still unfound. Joe has partially harvested Berries#7, completing 50% of the task, and his bag now contains berries from Berries#6. Berries#7 remains half uncollected. Thicket#3 has been thoroughly searched by Samantha with no new resources or clues found, while Thicket#4 has been partially explored by Joe with no memory-related clues found yet. Joe is slightly tired from searching the right side of Hut#1. No immediate threats or other characters are nearby. 


Samantha: -----cognitive cycle----- 2025-01-01T09:02:00 
Samantha:  moves Berries#7.
  and notices Berries#7 spotted northwest at 16, Joe southeast at 7, Mushrooms#10 south at 4, Thicket#3 southeast at 4. 
Samantha: 'Hey Joe, let's split up and cover more ground. I'll take the left side </act>' 
Samantha: Samantha searches the left side of Hut#1, carefully examining every corner and under any debris for the map.
 Resulting in Samantha searches the left side of Hut#1 thoroughly but does not find the map. The left side of Hut#1 is now fully searched. Samantha sees no new items or clues in the area. Samantha remains healthy and unafraid of the task, but becomes slightly tired from the effort. She gains no new possessions or specific information from this search. 
Samantha: Samantha carefully searches through the undergrowth of Thicket#4, looking for any hidden items or signs that might trigger her memory.
 Resulting in Samantha finds no hidden items or memory-related signs in Thicket#4.
Samantha becomes more tired from searching Thicket#4. 
Samantha: Samantha carefully examines the ground and underbrush of Thicket#4, searching for any unusual objects or signs that might trigger her memory.
 Resulting in Samantha finds no unusual objects or signs in Thicket#4 that trigger her memory.
Samantha becomes more tired from searching Thicket#4. 
Samantha: Samantha carefully lifts and moves sections of Fallen_Log#9, inspecting underneath for any hidden items or clues.
 Resulting in Samantha lifts and moves sections of Fallen_Log#9, finding no hidden items or clues underneath.
Samantha sees the ground beneath Fallen_Log#9, which is clear of any objects or signs.
Samantha becomes more tired from the effort of moving the log.
    """
    evaluate_transcript(agents, transcript)
    sys.exit()
    with open('/home/bruce/Downloads/AllTheWorldAPlay/src/sim/evaluation/baseline_script.txt', 'r') as f:
        transcript = f.readlines()
        results = evaluate_transcript('Samantha, Joe', '\n'.join(transcript))
