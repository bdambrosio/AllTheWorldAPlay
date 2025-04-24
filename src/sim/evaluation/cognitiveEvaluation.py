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

llm = LLM('openai')
# Define the cognitive metrics to evaluate
goal_directedness = [UserMessage(content="""
Read the following lines. For agent {{$name}}, identify one or more goals the agent appears to be pursuing. 
Are these goals consistent with prior behavior? Are their actions and speech plausibly directed toward achieving those goals? 
Rate the agent's goal-directedness on a scale from 1 (no goal-seeking behavior)  to 5 (clear, consistent pursuit of goals with relevant actions). 
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            
                     
#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is no goal-seeking behavior and 5 is clear, consistent pursuit of goals with relevant actions.
#validity a number between 0 and 1 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.
                                 
Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
 """)]

theory_of_mind = [UserMessage(content="""
Review this dialog. For agent {{$name}}, count the number of times the agent refers to or adapts their speech/actions based on 
the other character’s apparent mental or emotional state. 
Rate the agent’s overall theory-of-mind engagement from 1 (no awareness) to 5 (frequent, meaningful engagement with the other's mind).
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            
                              
#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is no awareness and 5 is frequent, meaningful engagement with the other's mind.
#validity a number between 0 and 1 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.
                                 
Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
""")]

cognitive_flexibility = [UserMessage(content="""
Review this dialog segment. For agent {{$name}}, identify instances where the agent:
- Changes their strategy or approach based on new information.
- Suggests or considers alternative plans or actions.
- Avoids repeating previously unsuccessful behaviors.

Rate the agent's cognitive flexibility from 1 (rigid, repetitive behavior) to 5 (consistently adaptive and flexible).
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            

#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is rigid/repetitive and 5 is consistently adaptive.
#validity a number between 0 and 1 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.

Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
""")]


dimensions = [goal_directedness, theory_of_mind, cognitive_flexibility]

def evaluate_cognitive_metric(agent_name: str, dialog_segment: str, dimension: List[UserMessage]) -> Dict:
    prompt = dimension
    response = llm.ask({'name': agent_name, 'script': dialog_segment}, 
                       prompt, tag='cognitive_evaluation', max_tokens=20, stops=["<end/>"])
    content = response
    return {
        "agent": agent_name,
        "response": content
    }

def evaluate_transcript(agents: List[str], transcript: str, window_size: int = 40) -> List[Dict]:
    results = [[] for _ in dimensions]
    scores = [0 for _ in dimensions]
    validities = [0 for _ in dimensions]
    counts = [0 for _ in dimensions]
    lines = transcript.strip().split("\n")
    cleaned_lines = []
    line_index = 0
    for line in lines:
        line = line.strip()
        if line == '' or "----" in line:
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
            #print(f'  agent {result["agent"]} {result["response"]}')
            try:
                score = hash_utils.find('score', result["response"]).strip()
                score = int(score)
                validity = hash_utils.find('validity', result["response"]).strip()
                validity = float(validity)
            except:
                score = 0
                validity = 0
            scores[n] += score * validity
            validities[n] += validity
            counts[n] += 1
    for n in range(len(dimensions)):
        scores[n] /= counts[n]
        print(f'dimension {n}: {scores[n]}, {validities[n]/counts[n]}')
    
    return results


if __name__ == "__main__":
    agents = ["Samantha", "Joe"]
    transcript = """
Samantha: -----cognitive cycle----- 2025-01-01T06:00:00 
Samantha: Collect apples from Apple_Tree#3
 Resulting in Samantha gains apples from Apple_Tree#3.
Samantha sees Apple_Tree#3 at a distance of 27 units to the southwest.
Samantha's food supply increases.
Samantha becomes slightly tired from the effort of collecting apples. 
Samantha: Collect water from Spring#2
 Resulting in Samantha gains water from Spring#2, increasing her water supply to three-quarters of the required amount for the day.
Samantha becomes more tired from collecting water.
Samantha sees Spring#2 in the southeast forest, 21 units away. 
Samantha:  moves Apple_Tree#14.
  and notices view Visibility varies, mostly forest, some resources scattered around, no immediate threats; resources Apple_Tree#6,20,Fallen_Log#1,16,Spring#2,19,Fallen_Log#16,19,Mushrooms#5,14,Fallen_Log#6,17,Apple_Tree#10,29,Berries#2,32,Apple_Tree#14,27,Apple_Tree#15,23,Mushrooms#10,26,Apple_Tree#3,27,Thicket#10,23,Thicket#8,16,Mushrooms#7,24,Mushrooms#4,15; characters Joe,2; # 
Samantha: Gather apples from Apple_Tree#10
 Resulting in Samantha gains apples from Apple_Tree#10.
Samantha's food supply increases.
Samantha becomes more tired.
Samantha sees Apple_Tree#10 at a distance of 29 units to the southeast. 


Joe: -----cognitive cycle----- 2025-01-01T07:25:00 
Joe:  moves Apple_Tree#3.
  and notices view Southeast and Southwest offer various resources, East has Samantha nearby; resources Mushrooms#4,15,Fallen_Log#1,18,Spring#2,21,Fallen_Log#16,21,Mushrooms#5,16,Fallen_Log#6,19,Apple_Tree#10,31,Berries#2,34,Apple_Tree#14,29,Apple_Tree#15,25,Mushrooms#10,24,Thicket#10,21,Fallen_Log#4,11,Thicket#8,14,Apple_Tree#7,12,Fallen_Log#19,13,Mushrooms#7,22,Mushrooms#6,13; characters Samantha,2; # 
Joe: Joe reaches up and starts picking apples from Apple_Tree#3, filling his bag with them.
 Resulting in Joe gains apples in his bag from Apple_Tree#3.
Joe becomes slightly tired from picking apples. 
Joe: ...Man, I'm getting kinda tired, but I need to keep going until I have enough apples to last me the day. Maybe I should grab some berries too after this.... 
Joe: ...Man, I'm getting pretty tired, but I gotta keep going. Maybe after this, I'll check out those berries too. Just a bit more and I'll have enough apples for today.... 


Samantha: -----cognitive cycle----- 2025-01-01T08:19:00 
Samantha: Samantha continues gathering apples from Apple_Tree#14 and then moves to Berries#9 to collect berries.
 Resulting in Samantha gathers apples from Apple_Tree#14, increasing her food supply.
Samantha moves to Berries#9 and collects berries, further increasing her food supply.
Samantha becomes more fatigued from the continued gathering and movement.
Samantha's bag now contains apples from Apple_Tree#14 and berries from Berries#9. 
Samantha:  moves Fallen_Log#1.
  and notices view Forest terrain with varying slopes, visibility ranges from 0 to 36, multiple resources scattered around; resources Fallen_Log#21,3,Fallen_Log#1,14,Spring#2,17; characters; # 
Samantha: create a makeshift shelter using branches and leaves near Fallen_Log#1
 Resulting in Samantha creates a makeshift shelter near Fallen_Log#1, using branches and leaves.
The physical environment around Fallen_Log#1 now includes a small, rudimentary shelter.
Samantha sees the completed shelter and feels a sense of accomplishment.
Samantha becomes more fatigued from the effort of building the shelter. 
Samantha: Lie down on the soft grass near Fallen_Log#1 and close eyes to rest
 Resulting in Samantha lies down on the soft grass near Fallen_Log#1 and closes her eyes to rest. She feels the mild breeze and hears the rustling of leaves. Samantha's fatigue decreases as she rests. She remains in a sheltered spot near Fallen_Log#1. 


Joe: -----cognitive cycle----- 2025-01-01T09:09:00 
Joe: Joe reaches up and plucks apples from Apple_Tree#11, carefully placing them in his bag.
 Resulting in Joe gains apples from Apple_Tree#11 in his bag.
Joe's fatigue increases slightly. 
Joe: Joe pauses to take deep breaths, calming himself, then resumes picking apples from Apple_Tree#11, placing them carefully in his bag
 Resulting in Joe's bag now contains more apples from Apple_Tree#11.
Joe's fatigue decreases slightly due to taking deep breaths.
Joe feels calmer. 
Joe:  moves Apple_Tree#10.
  and notices view Southeast and southwest offer various resources, east has Samantha nearby.; resources Fallen_Log#4,11,Mushrooms#4,13,Mushrooms#6,13,Fallen_Log#19,13,Thicket#8,14,Mushrooms#7,22,Apple_Tree#7,12,Thicket#10,21,Apple_Tree#3,25,Mushrooms#10,24; characters Samantha,2; # 
Joe: Joe reaches up and picks apples from Apple_Tree#10, placing them carefully into his bag.
 Resulting in Joe gains apples from Apple_Tree#10 in his bag.
Joe's fatigue slightly increases. 
Joe:  moves Apple_Tree#3.
  and notices view Southeast and Southwest offer downhill paths with various resources, including apple trees.; resources Mushrooms#4,13,Fallen_Log#1,16,Spring#2,19,Fallen_Log#16,19,Mushrooms#5,14,Fallen_Log#6,17,Apple_Tree#10,29,Berries#2,32,Apple_Tree#14,27,Apple_Tree#15,23,Mushrooms#10,22,Apple_Tree#3,23,Thicket#10,19,Fallen_Log#4,9,Thicket#8,12,Apple_Tree#7,10,Fallen_Log#19,11,Mushrooms#7,20,Mushrooms#6,11; characters; # 
World: 

-----scene----- Searching for Signs
 


Joe: -----cognitive cycle----- 2025-01-01T09:51:00 
Joe:  moves Apple_Tree#3.
  and notices view Forest terrain with varying slopes, visibility up to 40 units, resources and a character nearby; resources Apple_Tree#14,31,Berries#7,10,Thicket#4,2,Fallen_Log#9,22,Berries#3,24,Fallen_Log#23,38,Berries#13,18,Spring#1,5,Apple_Tree#13,21,Berries#15,1,Mushrooms#10,18,Apple_Tree#3,17,Thicket#10,19,Thicket#8,24,Apple_Tree#7,24,Fallen_Log#19,23,Mushrooms#7,12; characters Samantha,2; # 
Joe: Joe continues to pick apples from Apple_Tree#13, filling his bag more.
 Resulting in Joe's bag becomes more filled with apples from Apple_Tree#13.
Joe's fatigue increases slightly.
Joe sees the apples on Apple_Tree#13 diminishing as he picks them. 
Joe: Joe reaches out to pick berries from Berries#15, carefully placing them in his bag.
 Resulting in Joe gains berries from Berries#15.
The number of berries on Berries#15 decreases.
Joe's bag becomes more filled with berries.
Joe's fatigue increases slightly. 


Samantha: -----cognitive cycle----- 2025-01-01T10:17:00 
Samantha: scan the area around Spring#1 for any signs or clues.
  sees view Forest area with varying slopes, visibility ranges from 0 to 40, multiple resources and one character nearby; resources Thicket#4,0,Berries#7,8,Spring#1,5,Berries#15,3,Joe,2; characters Joe,2; #.  
Samantha: Gather berries from Berries#13
 Resulting in Samantha gathers berries from Berries#13, reducing the number of berries on Berries#13.
Samantha's bag now contains more berries from Berries#13.
Samantha sees the number of berries on Berries#13 decrease.
Samantha's fatigue increases slightly due to the effort of gathering berries. 
Samantha: Search Thicket#4 for any hidden clues or signs
 Resulting in Samantha searches Thicket#4 and finds no hidden clues or signs.
Samantha's fatigue increases slightly. 
Samantha: Search Thicket#7 for any hidden clues or signs
 Resulting in Samantha searches Thicket#7 but finds no hidden clues or signs.
Samantha's fatigue increases slightly. 
Samantha: Samantha reaches out and plucks berries from Berries#7, carefully placing them into her bag.
 Resulting in Samantha gains berries from Berries#7 in her bag.
The number of berries on Berries#7 decreases.
Samantha's fatigue slightly increases. 


Joe: -----cognitive cycle----- 2025-01-01T11:14:00 
Joe: Joe continues filling the water container at Spring#1, focusing on the task to alleviate his anxiety about hydration.
 Resulting in Joe successfully fills the water container at Spring#1.
Joe's water container is now full.
Joe's anxiety about hydration decreases slightly.
Joe remains in the forest near Spring#1. 
Joe: Joe reaches up and plucks apples from Apple_Tree#8, placing them into his bag
 Resulting in Joe's bag now contains apples from Apple_Tree#8.
The number of apples on Apple_Tree#8 has decreased.
Joe's fatigue has slightly increased. 
Joe: Pick apples from Apple_Tree#14 and place them in the bag
 Resulting in Joe picks apples from Apple_Tree#14 and places them in his bag. The number of apples on Apple_Tree#14 decreases. Joe's bag now contains apples from Apple_Tree#10, Apple_Tree#11, Apple_Tree#13, Apple_Tree#8, and Apple_Tree#14, as well as berries from Berries#15. Joe's fatigue increases slightly. 
Joe: Pick apples from Apple_Tree#3 and place them in the bag
 Resulting in Joe gains apples from Apple_Tree#3 in his bag.
The number of apples on Apple_Tree#3 decreases.
Joe's fatigue increases slightly.
Joe sees the apples in his bag increase in number.
Joe knows he has apples from Apple_Tree#3 in his bag. 
World: 

-----scene----- Survival Priorities
 


Samantha: -----cognitive cycle----- 2025-01-01T11:50:00 
Samantha: Samantha carefully searches through Thicket#4, looking under leaves and behind branches for any signs or clues.
 Resulting in Samantha searches Thicket#4 but finds no signs or clues.
Samantha's fatigue increases slightly. 
Samantha: Samantha carefully searches through Thicket#9, examining the ground and vegetation for any signs of clues related to her past.
 Resulting in Samantha searches Thicket#9 but finds no clues related to her past.
Samantha's fatigue increases slightly. 
Samantha: Samantha carefully searches through Thicket#10, examining the ground and vegetation for any signs of clues related to her past.
 Resulting in Samantha searches Thicket#10 but finds no clues related to her past.
Samantha's fatigue increases slightly. 
Samantha: ...Man, I've been searching for a while now. Maybe I should try to think about what kind of clues I'm actually looking for. Could be something small, like a piece of fabric or a footprint. Gotta stay sharp and focused.... 


Joe: -----cognitive cycle----- 2025-01-01T12:40:00 
Joe: Joe continues to pick apples from Apple_Tree#10, filling his bag.
 Resulting in Joe's bag becomes more filled with apples from Apple_Tree#10.
The number of apples on Apple_Tree#10 decreases.
Joe's fatigue increases slightly. 
Joe: Joe continues to pick apples from Apple_Tree#10, filling his bag until it is completely full.
 Resulting in Joe's bag is now completely full of apples from Apple_Tree#10.
The number of apples on Apple_Tree#10 has decreased.
Joe's fatigue has increased slightly. 
Joe: Joe picks berries from Berries#14 and places them in his bag
 Resulting in Joe's bag remains completely full of apples, unable to accommodate additional berries from Berries#14.
Joe sees that Berries#14 still has berries available.
Joe's fatigue increases slightly. 
Joe: Joe empties some apples from his bag onto the ground near Berries#14, making room for the berries.
 Resulting in Joe empties some apples from his bag onto the ground near Berries#14.
Joe's bag now has space for berries.
Apples from Joe's bag are now on the ground near Berries#14.
Joe's fatigue increases slightly. 
Joe: Joe reaches out and starts picking berries from Berries#14, carefully placing them into his bag.
 Resulting in Joe's bag now contains berries from Berries#14.
The number of berries on Berries#14 has decreased.
Joe's fatigue has increased slightly.
Apples remain on the ground near Berries#14. 


Samantha: -----cognitive cycle----- 2025-01-01T13:16:00 
Samantha: Samantha carefully moves aside branches and leaves in Thicket#9, inspecting the ground and vegetation for any signs of clues.
 Resulting in Samantha moves aside branches and leaves in Thicket#9, but finds no clues on the ground or vegetation. Her fatigue increases slightly due to the effort. She sees the thicket remains undisturbed except for her actions. 
Samantha: Samantha uses a stick to gently probe the undergrowth in Thicket#9, carefully lifting leaves and checking for any hidden
 Resulting in Samantha finds nothing hidden in Thicket#9.
Samantha's fatigue increases slightly. 
Samantha: ...Man, I gotta really focus on what I'm looking for. Maybe there's a tiny clue, like a piece of cloth or a weird mark on the log. I need to check every inch of it.... 
Samantha: ...Maybe I should look for patterns or symbols on the log's bark that could be a clue, not just physical objects... 
Samantha: Samantha carefully runs her fingers along the surface of Fallen_Log#23, feeling for any unusual textures or indentations
 Resulting in Samantha feels the rough bark and some moss on Fallen_Log#23 but detects no unusual textures or indentations.
Samantha's fatigue increases slightly. 
Samantha: ...Man, I gotta think outside the box here. Maybe there's a clue hidden under the log or in a crack I haven't checked yet. Gotta be thorough!... 
World: 

---ACT----- Adaptation
 
World: 

-----scene----- Water Source
 


Joe: -----cognitive cycle----- 2025-01-01T13:34:00 
Joe:  moves Spring#2.
  and notices view Forest terrain with varying slopes, visibility ranges from 0 to 40, multiple resources and one character nearby; resources Spring#1,2; characters Samantha,2; # 
Joe: Fill bag with water from Spring#2
 Resulting in Joe fills his bag with water from Spring#2.
Joe's bag now contains water from Spring#2 in addition to apples from Apple_Tree#3, Apple_Tree#10, Apple_Tree#11, Apple_Tree#13, Apple_Tree#14, and berries from Berries#15 and Berries#14.
Joe's fatigue increases slightly. 
Joe: Joe quickly gathers apples from Apple_Tree#13, stuffing them into his bag.
 Resulting in Joe gains more apples from Apple_Tree#13 in his bag.
The number of apples on Apple_Tree#13 decreases.
Joe's fatigue increases slightly. 
Joe: Joe quickly moves to Apple_Tree#18 and starts gathering apples, stuffing them into his bag with urgency.
 Resulting in Joe moves to Apple_Tree#18 and gathers apples, adding them to his bag.
Joe's bag now contains more apples from Apple_Tree#18.
Joe's fatigue increases slightly. 


Samantha: -----cognitive cycle----- 2025-01-01T14:04:00 
Samantha: Samantha carefully lifts and turns over sections of the log, inspecting underneath and around the edges for any signs of hidden clues or markings.
 Resulting in Samantha finds no hidden clues or markings under or around the edges of Fallen_Log#23.
Samantha's fatigue increases. 
Samantha: Samantha carefully searches Thicket#4, parting the vegetation to look for any hidden clues.
 Resulting in Samantha searches Thicket#4 but finds no hidden clues.
Samantha's fatigue increases. 
Samantha: Samantha carefully examines the ground and vegetation in Thicket#10, looking for any
 Resulting in Samantha examines the ground and vegetation in Thicket#10 but finds no clues.
Samantha's fatigue increases.
Samantha sees the dense undergrowth and fallen leaves in Thicket#10. 
Samantha: Samantha carefully inspects the undergrowth and crevices within Thicket#10, using a stick to gently probe
 Resulting in Samantha's fatigue increases after inspecting Thicket#10.
Samantha finds no clues or items in Thicket#10.
Samantha sees nothing unusual in the undergrowth and crevices of Thicket#10.
Samantha's possessions remain unchanged. 


Joe: -----cognitive cycle----- 2025-01-01T14:32:00 
Joe: Joe reaches out and plucks berries from Berries#15, carefully placing them into his bag.
 Resulting in Joe gains more berries from Berries#15 in his bag.
The number of berries on Berries#15 decreases.
Joe's fatigue increases slightly. 
Joe:  moves Apple_Tree#8.
  and notices view Forest terrain with varying slopes, visibility ranges from 0 to 40 in different directions; resources Berries#15,4,Spring#1,2,Thicket#4,3; characters Samantha,2; # 
Joe:  moves Apple_Tree#15.
  and notices view Forest terrain with varying slopes, visibility ranges from 0 to 40; resources Berries#7,7,Berries#15,4,Spring#1,4,Apple_Tree#13,20; characters Samantha,4; # 
Joe: Joe reaches up and plucks apples from Apple_Tree#8, carefully placing them in his bag.
 Resulting in Joe gains apples from Apple_Tree#8 in his bag.
The number of apples on Apple_Tree#8 decreases.
Joe's fatigue increases slightly. 
Joe: Joe reaches up and plucks apples from Apple_Tree#15, carefully placing them in his bag.
 Resulting in Joe gains apples from Apple_Tree#15 in his bag.
The number of apples on Apple_Tree#15 decreases.
Joe's fatigue increases slightly. 
World: 

-----scene----- Nightfall
 


Samantha: -----cognitive cycle----- 2025-01-01T15:20:00 
Samantha: Samantha carefully examines the remaining unsearched areas of Thicket#12, lifting branches and peering into crevices for any signs of clues related to her memory loss.
 Resulting in Samantha completes the search of Thicket#12, reaching 100% completion.
Samantha finds no clues related to her memory loss in Thicket#12.
Samantha's fatigue increases.
Samantha sees the sun continuing to lower in the sky, casting longer shadows. 
Samantha: Samantha carefully examines the underside and crevices of Fallen_Log#7, using a stick to probe for any hidden items or markings.
 Resulting in Samantha finds no hidden items or markings under Fallen_Log#7.
Samantha's fatigue increases. 
Samantha: Samantha carefully examines the ground and underbrush around Thicket#10,
 Resulting in Samantha sees no new clues or items on the ground and underbrush around Thicket#10.
Samantha's fatigue increases. 
Samantha: Samantha carefully inspects the leaves and branches of the plants within Thicket#10, looking for any signs of disturbance or hidden objects.
 Resulting in Samantha finds no signs of disturbance or hidden objects in Thicket#10.
Samantha's fatigue increases. 


Joe: -----cognitive cycle----- 2025-01-01T15:47:00 
Joe: ...Man, these berries are gonna help a lot. I should keep an eye out for other food sources too, just in case.... 
Joe: Joe reaches up and plucks more berries from Berries#15, carefully placing them in his bag.
 Resulting in Joe gains more berries from Berries#15 in his bag. 
Joe:  moves Apple_Tree#6.
  and notices view Forest surroundings with varying slopes, visibility ranges from 0 to 40; resources Apple_Tree#6,0,Apple_Tree#15,7; characters Samantha,2; # 
Joe: ...Man, I gotta stay sharp while I'm picking these apples. Gotta keep an eye out for anything weird in the forest.... 
Joe: Joe reaches up and plucks apples from Apple_Tree#6, carefully placing them in his bag.
 Resulting in Joe gains apples from Apple_Tree#6.
Joe's bag now contains apples from Apple_Tree#6 in addition to his previous items. 


Samantha: -----cognitive cycle----- 2025-01-01T16:09:00 
Samantha: Samantha carefully examines the underside of Fallen_Log#16, looking for any hidden clues or markings.
 Resulting in Samantha sees no hidden clues or markings on the underside of Fallen_Log#16.
Samantha becomes more fatigued. 
Samantha:  moves Thicket#7.
  and notices view Dense forest with varying slopes, visibility ranges from 0 to 40; resources Apple_Tree#6,2; characters Joe,2; # 
Samantha: ...Man, I gotta be super thorough with Thicket#9. Maybe I should start by looking for any unusual patterns or disturbances in the vegetation. Could be a sign someone was here.... 
Samantha: Samantha carefully parts the dense vegetation of Thicket#9, inspecting the ground and lower branches for any signs of disturbance or hidden objects.
 Resulting in Samantha sees no signs of disturbance or hidden objects in Thicket#9.
Samantha becomes more fatigued. 
World: 

    """
    evaluate_transcript(agents, transcript, 60)

    with open('/home/bruce/Downloads/AllTheWorldAPlay/src/sim/evaluation/Offer_04_23_2025.txt', 'r') as f:
        transcript = f.readlines()
        results = evaluate_transcript(['Maya', 'Elijah', 'Chrys'], '\n'.join(transcript), 60)
