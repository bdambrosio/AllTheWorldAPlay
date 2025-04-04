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

cognitive_flexibility = [UserMessage(content="""
Review this dialog segment. For agent {{$name}}, identify instances where the agent:
- Changes their strategy or approach based on new information.
- Suggests or considers alternative plans or actions.
- Avoids repeating previously unsuccessful behaviors.

Rate the agent's cognitive flexibility from 1 (rigid, repetitive behavior) to 5 (consistently adaptive and flexible).

#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is rigid/repetitive and 5 is consistently adaptive.
#justification a concise (3-8 words) explanation of your score.

Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
""")]


dimensions = [goal_directedness, theory_of_mind, cognitive_flexibility]

def evaluate_cognitive_metric(agent_name: str, dialog_segment: str, dimension: List[UserMessage]) -> Dict:
    prompt = dimension
    response = llm.ask({'name': agent_name, 'script': dialog_segment}, prompt, max_tokens=20, stops=["<end/>"])
    content = response
    return {
        "agent": agent_name,
        "response": content
    }

def evaluate_transcript(agents: List[str], transcript: str, window_size: int = 40) -> List[Dict]:
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
Samantha: -----cognitive cycle----- 2025-01-01T06:00:00 
Samantha: look around for any signs of human activity or artifacts near Fallen_Log#10 and Mushrooms#4.
  sees Mushrooms#4 at 6 units and Fallen_Log#10 at 11 units northeast. Joe is nearby. Exploring for civilization clues..  
Samantha: Examine the area around Fallen_Log#10 for any carvings or markings that might indicate human presence
 Resulting in Samantha sees no carvings or markings on Fallen_Log#10 that indicate human presence.
Samantha remains in the same physical and emotional state. 
Samantha: Observe Joe's body language and facial expressions for signs of trust or deceit.
  sees Joe is nearby; observing his body language and facial expressions for signs of trust or deceit..  
Samantha: 'Hey Joe, what's your story? How'd you end up here?' 
Joe: 'I wish I knew, Sam. Woke up here, no clue how or why. You? *scratches head, looking around* Maybe we should check out those berries you found, see if they're safe to eat.' 
Samantha: 'Same here, Joe. Woke up with no memories. Let's check those berries, but carefully. We don't wanna end up sick in the middle of nowhere. *nods towards the berries*' 
Samantha: Samantha carefully picks a few berries from Berries#2 and rubs them on her skin to test for any allergic reactions.
 Resulting in Samantha rubs a few berries from Berries#2 on her skin.
Samantha feels the texture of the berries on her skin.
Samantha observes her skin for any signs of allergic reactions.
Samantha's possessions remain unchanged.
Samantha's state remains unchanged. 
Samantha: 'Hey Joe, I just ate a few berries to see if they're safe. Keep an eye on me for a bit, okay?' 
Joe: 'Yeah, I'm on it. Let me know if you feel anything weird, okay? *leans in, watching closely*' 
Samantha: 'Samantha says: So far, so good. No weird vibes from the berries. *gives a thumbs up* Maybe we should gather more and start looking for a way out of here?' 
Joe: 'Yeah, let's grab more berries and then scout around. Maybe we'll find a clue on how to get out of here. *nods, scanning the surroundings*' 
Samantha: scan the area to the northeast for any unusual markings or objects near Fallen_Log#10.
  sees Fallen_Log#10 at 11m northeast, near Mushrooms#4 at 6m. Joe is nearby..  
Samantha: Samantha begins dragging Fallen_Log#10 towards a clear area, preparing to construct the shelter's frame.
 Resulting in Samantha begins dragging Fallen_Log#10, moving it from its original position 11 meters northeast towards a clear area. 
Samantha feels the weight of the log and starts to exert effort, becoming slightly fatigued.
Samantha hears the sound of the log scraping against the forest floor.
Samantha sees the log's progress as she drags it, noting the clear area getting closer. 
Samantha: Samantha continues to arrange Fallen_Log#10 and begins to position Fallen_Log#9 to form the base structure of the shelter.
 Resulting in Fallen_Log#10 is positioned as part of the shelter's base structure.
Fallen_Log#9 is moved closer to the shelter construction site.
Samantha becomes more fatigued from arranging the logs.
Samantha sees the logs forming the initial structure of the shelter. 
Samantha:  moves Berries#2.
  and notices Joe is nearby. Berries are 4 units north. Other resources like fallen logs and mushrooms are farther away. 
Samantha: scan the area for any unusual markings or paths that might indicate an exit.
  sees Joe is nearby to the south. Berries and mushrooms within 10 units in various directions. No visible paths or markings indicating an exit..  
Samantha: 'Hey Joe, can you help me secure this log? We're almost done with the frame.' 
Joe: 'Yeah, I'm on it. Let's finish this up and then figure out our next move. *moves to help secure the log*' 
Samantha: Secure Fallen_Log#9 to the shelter frame, aligning it properly with the existing structure
 Resulting in Fallen_Log#9 is now secured to the shelter frame, aligning with the existing structure.
Samantha sees the shelter structure becoming more complete.
Samantha's fatigue level increases due to the effort of securing Fallen_Log#9.
Samantha gains a better understanding of the shelter's structural integrity. 
Samantha: Samantha gathers more berries from Berries#2, carefully selecting only the safe ones she tested earlier.
 Resulting in Samantha gathers more berries from Berries#2, increasing her collection of safe berries.
Samantha's fatigue level increases slightly due to the effort of gathering more berries. 
Samantha: Samantha continues gathering berries from Berries#2, ensuring she only picks the ones previously identified as safe.
 Resulting in Samantha's collection of safe berries from Berries#2 increases.
Samantha's fatigue level increases slightly. 
Samantha: Secure Fallen_Log#17 to the shelter frame, tightening the structure
 Resulting in Fallen_Log#17 is now secured to the shelter frame, making the structure tighter and more stable.
Samantha sees the shelter frame becoming more robust with the addition of Fallen_Log#17.
Samantha hears the sound of the log being secured and the slight creaking of the shelter frame as it tightens.
Samantha becomes more fatigued from the effort of securing the log. 
Samantha: scan the surroundings for any unusual markings or paths that might lead to an exit.
  sees Joe is nearby. Berries and mushrooms within 10 units. Fallen logs and uphill paths visible, potential clues or exit routes..  
Samantha: ...Hmm, if we're looking for an exit, maybe we should start by checking the highest points around here. From up there, we might spot a path or something unusual that could lead us out.... 
Samantha: 'Hey Joe, while I'm gathering these berries, can you keep an eye out for any signs of a path or clues that might help us get out of here?' 
Joe: 'Yeah, I'll look around. Maybe we'll find something useful while we're at it. *scans the area, squinting against the morning light*' 
Samantha: 'Sounds good, Joe. Let's keep an eye out for anything unusual while we work on the shelter. Maybe we'll spot something from a higher point. *glances around, then back at Joe with a hopeful smile*' 
Samantha: scan the horizon for any signs of paths or unusual landmarks.
  sees Northeast uphill offers high visibility; Joe is nearby to the south. Berries and mushrooms within 10 units..  
Samantha:  moves Northeast.
  and notices Northeast visibility 18, uphill, Fallen_Log#9 at 11. Joe nearby at 3. 
Samantha:  moves Northeast.
  and notices Northeast visibility 15, uphill forest, no nearby resources or characters. 
Samantha: 'Hey Joe, keep an eye out while I grab these last berries, okay? We gotta stay sharp.' 
Joe: 'Yeah, I'm on it. I'll keep scanning for any signs of a way out. Maybe we'll spot something from a higher vantage point soon.' 
Samantha: 'Hey Joe, let's finish securing this log and then we can take a break, yeah?' 
Joe: 'Yeah, sounds good. Let's get this done quick, then we can chill for a bit. I'll keep an eye out for any signs of a way out while we work. *nods and glances around*' 
Samantha: ...Hmm, if we climb northeast, we might spot a path or something that could lead us out. It's worth a shot, and it'll give us a better view of the area.... 
Samantha: 'Hey Joe, I'm gonna head northeast to scout for high ground. Keep an eye out here, yeah?' 
Joe: 'Yeah, I'll keep watch. Be careful up there and let me know if you see anything that could be our way out. *nods, scanning the surroundings*' 
Samantha: scan the horizon for any signs of paths or unusual landmarks.
  sees Northeast uphill, visibility 15, no nearby resources or characters. Scanning for paths or landmarks..  
Samantha: Pick the remaining berries from Berries#2
 Resulting in Samantha gains the remaining 50% of berries from Berries#2.
Samantha now possesses all the berries from Berries#2.
Berries#2 is now depleted of berries.
Samantha sees that Berries#2 is now empty. 
Samantha: Secure Fallen_Log#18 to the shelter frame, tightening the knots to ensure stability.
 Resulting in Fallen_Log#18 is now secured to the shelter frame, increasing its stability.
Samantha hears the sound of tightening knots.
Samantha's hands feel the rough texture of the log and rope.
The shelter construction is now 60% complete.
Samantha becomes slightly tired from the effort. 
Samantha: 'Hey Joe, let's grab Fallen_Log#24 next. We're almost done, and I wanna make sure this shelter's solid.' 
Joe: 'Yeah, let's do it. I'll grab Fallen_Log#24 while you rest a bit. We're close to finishing this, and then we can focus on finding a way out. *nods, scanning the surroundings*' 
Samantha: 'Hey Joe, let's get this log in place quick so we can finally chill, yeah?' 
Joe: 'Yeah, let's do it fast. I'll grab the log, you rest up. We're close to finishing, then we can scout for a way out.' 
Samantha: Samantha drinks water from her canteen and sits down to rest, taking deep breaths to regain energy.
 Resulting in Samantha's energy level increases as she drinks water and rests. She remains near the shelter, 1 unit north of Joe. The physical environment and her possessions remain unchanged. Samantha feels more relaxed. 
Samantha:  moves Northeast.
  and notices Northeast visibility 12, uphill forest. No nearby characters or resources within 10 units. Focused on reaching high ground for exit search. 
Samantha: scan the horizon for paths or landmarks.
  sees Northeast visibility 12, uphill forest. No nearby characters or resources within 10 units..  
Samantha:  moves Northeast.
  and notices Northeast visibility 10, uphill forest, no resources or characters nearby. 
Samantha: scan the horizon for paths or landmarks.
  sees Visibility varies; uphill to Northeast, level Southeast, downhill Southwest. Joe nearby at 9 units. Fallen_Log#10 at 4 units Southeast..  
Samantha: Samantha descends from the high ground, carefully navigating the terrain to return to Joe's location.
 Resulting in Samantha moves to Joe's location, 9 units southwest of her previous position.
Samantha sees Joe and the surrounding forest terrain.
Samantha feels more relaxed and energized. 
Samantha: 'Hey Joe, I checked out the high ground to the northeast. Didn't see any clear paths out, but there's a weird rock formation that might be worth checking out later.' 
Joe: 'Yeah, that rock formation sounds interesting. Let's finish the shelter first, then we can check it out. Maybe it'll lead us to a way out.' 
Samantha:  moves Northeast.
  and notices Northeast visibility 9, uphill forest. Joe 11 units southwest. No nearby resources. 
Samantha:  moves Northeast.
  and notices Northeast visibility 7, uphill forest, no resources. Joe 13 units southwest, near berries and fallen log. 
Samantha: Samantha carefully examines the rock formation, searching for any unusual markings or hidden compartments that might contain clues.
 Resulting in Samantha sees no unusual markings or hidden compartments on the rock formation.
Samantha remains energized and relaxed. 
Samantha:  moves Mushrooms#4.
  and notices Joe is nearby at 11 units. Mushrooms#4 are close at 5 units. Berries#2 are at 6 units. 
Samantha:  moves northeast.
  and notices Northeast visibility 7, uphill forest. Joe 13 units southwest. No nearby resources or characters within 10 units. 
Samantha: scan the northeast horizon for any signs of paths or unusual landmarks.
  sees Northeast visibility 7, uphill forest, no paths or landmarks visible. Joe at 13 distance southwest..  
Samantha: Samantha carefully examines the rock formation, running her hands over the surfaces and peering into any cracks or hollows, searching for any signs
 Resulting in Samantha finds no signs or notable features in the rock formation.
Samantha remains uninjured and not tired. 
Samantha: Pick mushrooms#4
 Resulting in Samantha gains 7 units of Mushrooms#4.
Samantha sees the mushrooms being picked from the ground. 
Samantha: ...Man, we're halfway there already. Gotta keep pushing to the top to get that full view. What if there's a hidden path or something we can't see from here? I need to stay sharp and keep an eye out for anything unusual.... 
Samantha: Continue climbing northeast towards the high ground, keeping an eye out for any signs of a path or unusual landmarks.
 Resulting in Samantha continues climbing northeast and reaches a higher point on the high ground.
Samantha sees more of the forest terrain uphill to the northeast.
Samantha does not find any signs of a path or unusual landmarks.
Samantha's position is now further up the northeast high ground.
Samantha remains healthy and unafraid, but becomes slightly more tired from the climb. 
Samantha: Samantha carefully examines the rock formation, looking for any unusual markings or hidden compartments that might contain clues.
 Resulting in Samantha sees no unusual markings or hidden compartments on the rock formation.
Samantha acquires no new clues from the rock formation.
Samantha remains slightly tired from the climb. 
Samantha:  moves Berries#2.
  and notices Berries#2 spotted 6 units southwest, closest resource. Joe is 11 units away in the same direction. 
Samantha: Collect the remaining berries from Berries#2
 Resulting in Samantha collects the remaining 4 units of Berries#2.
Samantha gains 4 units of berries.
Berries#2 are now depleted.
Samantha becomes slightly more tired. 
Samantha: Samantha helps Joe position Fallen_Log#24 onto the shelter frame, securing it in place.
 Resulting in Fallen_Log#24 is now part of the shelter frame, increasing its completion to 75%.
Samantha and Joe are now 10 units apart.
Samantha becomes more tired. 
Samantha:  moves Northeast.
  and notices Northeast visibility 7, uphill forest. No nearby characters or resources within 10 units. 
Samantha: Samantha surveys the northeast high ground, looking for any visible exit paths or notable features that could lead them out of the forest.
 Resulting in Samantha sees uphill forest terrain to the northeast with no visible exit paths or notable features within 7 units.
Samantha remains tired. 
Samantha: 'Hey Joe, let's split up to cover more ground. You take the left side of the rock formation, and I'll check the right. Keep an eye out for anything weird, okay?' 
Samantha: 'Hey Joe, let's finish this up quick so we can rest and then check out that rock formation.' 
Joe: 'Yeah, sounds good. Let's wrap up the shelter and then hit the rock formation. I'm curious what we might find there.' 
Samantha: Samantha places Fallen_Log#10 onto the shelter frame, securing it tightly to increase the shelter's stability.
 Resulting in Fallen_Log#10 is removed from the southeast at a distance of 4 units.
The shelter's stability increases due to the addition of Fallen_Log#10.
Samantha's berry inventory remains at 4 units.
Samantha remains tired. 
Samantha: Samantha and Joe work together to secure the final pieces of the shelter, ensuring it is sturdy and complete.
 Resulting in The shelter is now 100% complete and sturdy.
Samantha sees the finished shelter and feels a sense of accomplishment.
Samantha remains tired.
Joe is now 10 units away from Samantha. 
Samantha: Samantha drinks water from her water bottle and lies down on the soft forest floor to rest.
 Resulting in Samantha's water bottle contains less water after she drinks from it. She lies down on the soft forest floor, feeling more rested. The physical environment remains unchanged. 
Samantha: 'Hey Joe, let's head over to that rock formation and see if it holds any secrets to getting out of here!' 
Joe: 'Yeah, let's check it out! I'm curious what secrets it might hold. Maybe we'll find a way out of here.' 
Samantha: Samantha and Joe begin to carefully examine the rock formation, looking for any unusual markings or hidden paths.
 Resulting in Samantha and Joe find no unusual markings or hidden paths on the rock formation.
Samantha remains tired and needs to hydrate further. 
Samantha: Take a few sips from the water bottle and then lie down to rest for a bit.
 Resulting in Samantha's water bottle contains less water.
Samantha is now lying down on the forest floor.
Samantha feels more rested but still needs to hydrate further.
Samantha remains tired. 
Samantha: Samantha carefully examines the crevices and surfaces of the rock formation, searching for any unusual markings or hidden paths that
 Resulting in Samantha finds no unusual markings or hidden paths on the rock formation.
Samantha remains tired and needs to hydrate further. 
Samantha: 'Hey Joe, what do you think about checking the other side of this rock formation? Maybe we missed something over there.' 
Joe: 'Yeah, let's do it. I'm curious what we might find on the other side. Maybe there's a clue we overlooked.' 
Samantha: Pick berries from Berries#4 bush
 Resulting in Samantha gains berries from Berries#4 bush.
Samantha sees the berries on Berries#4 bush are now fewer.
Samantha becomes slightly more tired. 
Samantha: Drink water from the water bottle and then lie down to rest on the forest floor.
 Resulting in Samantha drinks water from her water bottle, reducing the amount of water in it.
Samantha lies down on the forest floor, becoming more rested.
Samantha's energy level increases slightly due to resting. 
Samantha: Take a few sips from the water bottle and then sit down to rest on a nearby log.
 Resulting in Samantha takes a few sips from her water bottle, reducing the water level in the bottle.
Samantha sits down on a nearby log, which is Fallen_Log#10 located 4 units to the southeast.
Samantha's energy level increases slightly due to resting.
Samantha's hydration level increases slightly due to drinking water. 
Samantha:  moves Berries#4.
  and notices Berries#4 visible at 24 units southwest; Joe nearby at 11 units southwest. No immediate resources for gathering. 
Samantha: Samantha and Joe split up to explore different sections of the rock formation, carefully examining crevices and overhang
 Resulting in Samantha and Joe are now exploring different sections of the rock formation.
Samantha examines crevices and overhangs but finds no new clues or resources.
Samantha's energy level decreases slightly due to the physical activity.
Samantha remains wary but focused on the task. 
Samantha: Samantha and Joe use sticks to probe deeper into the rock formation's crevices, searching for hidden passages or clues.
 Resulting in Samantha and Joe find no hidden passages or clues in the rock formation's crevices.
Samantha's energy level decreases slightly due to physical activity.
Samantha feels more fatigued. 
Samantha: Samantha carefully examines the crevices and surfaces of the rock formation, searching for any hidden clues or potential exits.
 Resulting in Samantha finds no new clues or potential exits in the rock formation.
Samantha's energy level decreases slightly due to the physical activity.
Samantha feels more fatigued.'
    """
    evaluate_transcript(agents, transcript)
    sys.exit()
    with open('/home/bruce/Downloads/AllTheWorldAPlay/src/sim/evaluation/baseline_script.txt', 'r') as f:
        transcript = f.readlines()
        results = evaluate_transcript('Samantha, Joe', '\n'.join(transcript))
