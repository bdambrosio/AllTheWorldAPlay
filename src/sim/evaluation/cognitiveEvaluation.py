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
amantha: -----cognitive cycle----- 2025-01-01T06:00:00 
Samantha: look around for landmarks or signs of civilization, like roads or buildings.
  sees Joe nearby, Apple_Tree#13 and Thicket#7 in Northwest, Apple_Tree#11 in North.  
Samantha: 'Hey, I'm Samantha, what's your name and how did you end up here?' 
Joe: 'I'm Joe, nice to meet you, but honestly, I have no idea how I got here, my memory is kinda fuzzy right now.' 
Samantha: 'Yeah, me too, super weird! Let's look around, see if we can find any clues or something. I'm getting a little hungry, maybe we can find some food or water too.' 
Joe: 'Yeah, let's check that apple tree over there, might be some food, and maybe it'll spark something in my memory, you know?' 
Samantha: 'That sounds like a great idea, let's go check it out! I could totally go for an apple right now.' 
Samantha:  moves Apple_Tree#11.
  and notices Apple_Tree#11 nearby, Joe 1 unit south 
Samantha: 'Hey Joe, let's check out that apple tree, maybe we can find some shelter or something' 
Joe: 'Yeah, and maybe some apples too, I'm starving! Let's go take a look.' 
Samantha:  moves Apple_Tree#11.
  and notices Apple_Tree#11 nearby, Joe 2 units south 


Joe: -----cognitive cycle----- 2025-01-01T06:40:00 
Joe:  moves Apple_Tree#11.
  and notices Apple_Tree#11 and Samantha are nearby to the north 
Joe: 'Hey, Samantha, let's check if we can use this tree for shelter or find some apples to eat' 
Samantha: 'Yeah, definitely, let's take a closer look around the tree and see what we can find, maybe there's a clearing or something we can use for shelter, and hopefully some apples are still good to eat.' 
Joe: 'Awesome, let's check around the tree and see what we can find, I'll climb up and take a look.' 
Samantha: 'Okay, be careful up there, I'll check around the base of the tree for any apples that might have fallen, maybe we can find some other stuff too.' 
Joe: 'Alright, I'll check the branches, yell if you find anything cool down there!' 
Samantha: 'Okay, I'll keep an eye out for any apples or other useful stuff down here, and I'll yell up to you if I find anything, oh and Joe, be super careful up there, we don't know how sturdy those branches are!' 
Joe: 'Don't worry, I've got this, I'll test each branch before I put my weight on it, and I'll holler if I find any apples up here!' 
Joe: scan the surrounding area for any signs of potential threats, such as strange noises or movements.
  sees Samantha and Apple_Tree#11 nearby, uphill forest to the north.  
Joe: 'Hey Samantha, we need to find some water, let's check around the apple trees and see if there's a stream or something nearby' 
Samantha: 'Yeah, that's a great idea, let's check downhill, streams usually flow downward, right? And we can look for some animal tracks too, they might lead us to water.' 
Joe: 'Yeah, let's get moving then, I'm getting thirsty. We should also keep an eye out for any other food sources, like berries or nuts.' 
Joe: scan the surrounding area for any signs of a stream or river.
  sees Nearby: Samantha, Apple_Tree#11, Apple_Tree#13, Thicket#7, Fallen_Log#14, Mushrooms#8.  


Samantha: -----cognitive cycle----- 2025-01-01T09:02:00 
Samantha: check the base of Apple_Tree#11 for any fallen apples.
  sees Apple_Tree#11 nearby, Joe 1 unit south.  
Samantha: scan the surrounding area for any potential threats, like wild animals or steep slopes.
  sees Joe nearby, Apple_Tree#13 and Apple_Tree#11 in sight.  
Samantha: 'Hey Joe, let's head downhill together and see if we can find a stream, I'm getting pretty thirsty' 
Joe: 'Yeah, sounds good, let's go, I could use a drink too' 
Samantha: scan the terrain downhill for signs of a stream or water source.
  sees Joe nearby, water source downhill to the south.  
Samantha: 'Hey Joe, let's split up and search for berries or nuts, we can cover more ground that way' 
Joe: 'I don't know, Samantha, I kinda like stickin' together, it's safer that way, plus we can help each other out if we find somethin' cool.' 
Samantha: 'Yeah, that makes sense, I guess we can scouting together then, and if we find something really cool, we can always come back for it later, or mark it so we don't forget where it is.' 
Samantha: scan the surrounding area for berry bushes or nut trees.
  sees Apple_Tree#13, Joe, and Thicket#7 are nearby.  


Joe: -----cognitive cycle----- 2025-01-01T11:11:00 
Joe: 'Hey Samantha, let's head downhill together and follow any animal tracks we can find, maybe they'll lead us to a stream' 
Samantha: 'Yeah, that sounds like a plan, I could really use a drink of fresh water, let's go check it out' 
Joe: 'Hey, let's follow that animal track over there, it might lead us to a stream' 
Samantha: 'Yeah, that sounds like a great idea, I'm totally down for finding a stream, let's go check it out!' 
Joe: scan the ground for animal tracks.
  sees Animal tracks likely near Samantha and Apple_Tree#11 to the North.  
Joe: scan the ground for animal tracks near Apple_Tree#11.
  sees See Samantha, Apple_Tree#11, and Thicket#7 nearby.  
Joe: scan the area around Apple_Tree#11 and Apple_Tree#13 for any signs of water.
  sees Apple_Tree#11, Apple_Tree#13, and Samantha are nearby.  
Joe: 'Hey Samantha, I don't see any water around here, let's check the other side of the apple trees' 
Samantha: 'Yeah, let's go check, I'm getting pretty thirsty anyway' 
Joe: 'Hey Samantha, let's split up and search the thicket areas, we can cover more ground that way' 
Samantha: 'I don't know, Joe, I'm not sure that's a good idea, we should stick together, it's safer that way' 
Joe: 'But we can cover more ground if we split up, and we can meet back here at the apple tree in, like, 20 minutes, that way we can make sure we don't miss anything, and I'll be able to check out that thicket over there, it looks like it might have a stream or something, and I'll be careful, I promise.' 
Samantha: 'I'm still not sure, what if something happens to you?' 
Joe: 'Don't worry, I'll be fine, I promise! We can keep an eye on each other from a distance, and like I said, we'll meet back here in 20 minutes, okay?' 
Joe:  moves Thicket#7.
  and notices Thicket#7 nearby, Samantha 1 unit east 


Samantha: -----cognitive cycle----- 2025-01-01T11:44:00 
Samantha: 'Hey Joe, I think I see some animal tracks over here, let's follow them and see if they lead to a stream' 
Joe: 'Yeah, that's a great idea, let's check it out! I'll keep searching this thicket and meet you back at the apple tree if I don't find anything.' 
Samantha: look downhill for any signs of a stream or water source.
  sees Joe nearby, water downhill to East, South, and Southeast.  
Samantha: look around Thicket#7 for stream signs.
  sees Joe and Apple_Tree#13 nearby, Thicket#7 at 6 distance.  
Samantha:  moves Apple_Tree#13.
  and notices Joe and Apple_Tree#13 are nearby 
Samantha:  moves Apple_Tree#13.
  and notices Joe and Apple_Tree#13 are nearby 
Samantha: look upstream for any signs of shelter or unusual land formations.
  sees Joe, Apple_Tree#13, and Thicket#7 nearby.  


Joe: -----cognitive cycle----- 2025-01-01T12:30:00 
Joe: examine Thicket#7 more closely for stream signs.
  sees Thicket#7 nearby, Samantha present, Apple_Tree#13 and Apple_Tree#11 in view.  
Joe: 'Hey Samantha, found anything on your end, should we meet up and compare what we've found so far' 
Samantha: 'Yeah, I found some animal tracks, but no stream yet. Let's meet up and see if we can figure out where to go from here. I'm getting a little thirsty, so hopefully we can find something soon!' 
Joe: 'Hey Samantha, how's it going with those animal tracks, are we getting closer to finding that stream?' 
Samantha: 'Yeah, I'm still on the tracks, but no stream yet. I'm thinking maybe we should follow them a bit longer, see if they lead us to something. I'm getting pretty thirsty, so hopefully we're close!' 
Joe: 'Yeah, let's keep following, I'm getting thirsty too, and I don't wanna give up yet, maybe we can find some other clues or something.' 
Samantha: 'Yeah, me too, let's check that thicket again, maybe we missed something, and I'll keep following these tracks, see where they go.' 
Joe: 'Alright, I'll take a closer look, maybe we can find some signs of water or something, I'm really getting thirsty and a bit anxious about finding that stream, you think we're close?' 
Samantha: 'Yeah, I think we're getting close, the animal tracks are leading me to a spot just past that big tree, want to meet up there and check it out?' 
Joe: 'Yeah, sounds good, I'll meet you there, hopefully we find something soon, I'm getting pretty parched.' 
Joe: 'Hey, I'm gonna check out that big tree, want to meet up there and see if we can find any signs of water?' 
Samantha: 'Yeah, sounds like a plan, I'll meet you there and we can check it out together, maybe those animal tracks I've been following will lead us to something.' 
Joe:  moves Apple_Tree#13.
  and notices Samantha and Apple_Tree#13 are nearby 
Joe:  moves Apple_Tree#13.
  and notices Apple_Tree#13, Samantha nearby 
Joe:  moves Apple_Tree#13.
  and notices Apple_Tree#13, Samantha nearby 


Samantha: -----cognitive cycle----- 2025-01-01T13:27:00 
Samantha: 'Hey Joe, I'm gonna keep following these tracks, want to meet up near that big tree and see if we find anything?' 
Joe: 'Yeah, I'm already near the tree, I'll wait for you and see what we can find, maybe those tracks lead to a stream or something.' 
Samantha: 'Hey Joe, I'm getting close to the tree, want to check out those tracks together and see if we can find that stream?' 
Joe: 'Yeah, I'm ready, let's take a look, I've been checking around the tree and I think I saw something interesting over there.' 
Samantha: 'What is it, show me, I'm curious about those tracks too.' 
Joe: 'Okay, let me show you, it looks like the tracks lead right past the tree and into that thicket over there, I think we should check it out together.' 
Samantha: 'Yeah, that sounds like a plan, lead the way and let's check it out, I'm excited to see where those tracks go.' 
Samantha: 'Hey Joe, what do you think we'll find around this tree, maybe some food or a clue to where we are?' 
Joe: 'I'm hoping we find that stream, I'm getting pretty thirsty, and maybe some food would be awesome too, let's check around the tree and see what we can find.' 
Samantha: 'Hey Joe, let's check the tree and the tracks together, see if we can find any clues or food' 
Joe: 'Yeah, let's take a closer look, I think I saw some tracks leading into that thicket over there, maybe we can find something' 
Samantha: 'That sounds awesome, lead the way! I'm super curious about those tracks now, and hopefully we'll find that stream soon, I'm getting pretty thirsty. Let's go check it out together!' 
Joe: 'Alright, let's head into the thicket, I'll show you where the tracks lead, and hopefully we'll find that stream and get some water, I'm pretty parched too.' 
Samantha: 'Yeah, let's go, I'm so down to find that stream and get some water, my mouth is totally dry, and I'm curious about where those tracks lead, maybe we'll find some other clues too!' 
Samantha: 'Hey Joe, I didn't find any water yet, but I think we should check that thicket over there, what do you think?' 
Joe: 'Yeah, that's a great idea, let's check it out together, I'm pretty sure those tracks lead into it anyway.' 
Samantha: 'Awesome, let's go check it out! I'm so curious about those tracks now, and hopefully we'll find that stream soon.' 
Joe: 'Lead the way, I'm right behind you, let's see where those tracks go and hopefully find that stream, I'm getting pretty thirsty too.' 
Samantha: 'Okay, I'll head towards the tree and check out those tracks, you can see them from here, right? Let's move quietly, maybe we'll spot something, and hopefully, that stream is just beyond the tree.' 
Joe: 'Yeah, I see them, let's move in, I'll check the tree and you check the tracks, see if we can find any signs of water.' 
Samantha: 'Alright, I'll keep following the tracks, see where they lead, maybe we'll get lucky and find that stream just past the tree!' 
Joe: 'Yeah, I'll check around the tree, see if I can find any signs of water or anything else useful, maybe we can find some food too!' 
Samantha: 'That sounds like a plan, I'll keep following these tracks and meet you back here in a minute, hopefully we'll both find something useful!' 
Samantha: 'Hey Joe, I've been thinking, we should compare what we've found so far and figure out our next move to find that stream' 
Joe: 'Yeah, that's a great idea, let's take a look at what we've got so far and see if we can piece together where that stream might be, I've been checking around this tree and I think I saw some signs of water nearby.' 
Samantha: 'That's awesome, let's check it out together! I'm super curious about those signs of water, maybe we can finally find that stream and get some rest, I'm getting pretty thirsty. What kind of signs did you see?' 
Joe: 'Let's take a closer look, I saw some wet leaves and a slight depression in the ground, could be a trickle of water, and those animal tracks seem to be heading in the same direction, I'm thinking it might lead us to the stream.' 
Samantha: 'That sounds promising, let's follow the tracks together and see where they lead, maybe we'll find that stream just past the tree!' 
Joe: 'Yeah, let's move quietly and see, I'll take the lead and you keep an eye on those tracks, stay close behind me.' 
Samantha: 'Okay, I've got the tracks, let's go, I'll stay right behind you and keep an eye out for any changes in the trail or signs of water.'
    """
    evaluate_transcript(agents, transcript)
    sys.exit()
    with open('/home/bruce/Downloads/AllTheWorldAPlay/src/sim/evaluation/baseline_script.txt', 'r') as f:
        transcript = f.readlines()
        results = evaluate_transcript('Samantha, Joe', '\n'.join(transcript))
