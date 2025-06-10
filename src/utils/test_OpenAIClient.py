import sys, os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.LLMRequestOptions import LLMRequestOptions
import utils.llm_api as LLM_API
from utils.Messages import UserMessage
#llm = LLM_API.LLM(server_name='openai', model_name='gpt-4.1-mini')
llm = LLM_API.LLM(server_name='openai', model_name='Qwen3-14B') 

prompt = [{"role": "system", "content": """You are a seasoned writer designing a 2-3 act narrative arc for a 30 minute screenplay.
Every act should push dramatic tension higher: give the protagonist a clear want, place obstacles in the way, and end each act changed by success or setback.
Keep the stakes personal and specific\u2014loss of trust, revelation of a secret, a deadline that can\u2019t be missed\u2014so the audience feels the pulse of consequence.
Let conflict emerge through spoken intention and subtext, as well as through the characters' actions and reactions with the world and each other.
Characters hold real agency; they pursue goals, make trade-offs, and can fail. Survival chores are background unless they expose or escalate the core mystery.
Use vivid but economical language, vary emotional tone, and avoid repeating imagery.
By the final act, resolve \u2014or intentionally leave poised\u2014 the protagonist\u2019s primary drive. A very short final act can be added as a coda.
        "}, {"role": "user", "content": "You are a skilled actor / playwright working on an initial outline for an improvisational play. 
The overall cast of characters and setting are given below.

<Scenario>
{'setting': 'A temperate, mixed forest-open landscape with no buildings, no roads, no other signs of humanity. \
    It is a early morning on what seems like it will be a warm, sunny day.\
    Two people are standing in the middle of the forest, looking around in confusion.', 'characters': [{'name': 'Samantha', 'description': \"Samantha, a healthy, dark-haired young woman in grey hiking pants and a blue pendleton shirt. \
You love the outdoors and hiking.\
You are intelligent, introspective, philosophical, fearful, and a bit of a romantic. \
You have a uncomfortable history, maybe it is just as well you don't remember it.\
You are informal, chatty, and are a playful when relaxed, but at other times can be argumentative and defensive.\
You are wary of strangers.\", 'drives': 'solve the mystery of the forest. Find a way to live safely. Maybe the forest is a place of safety.. adventure, excitement, and the thrill of the unknown.. love and belonging, including home, acceptance, friendship, trust, intimacy.'}, {'name': 'Joe', 'description': \"Joe, a healthy young man, short beard  and dark hair, in grey chinos and a red t-shirt. \
You are intelligent and self-sufficient, informal and somewhat impulsive. \
You are strong, and think you love the outdoors, but are basically a nerd.\
You yearn for something more, but don't know what it is.\
You are socially awkward, especially around strangers. \
You speak informally, but occasionally in a 'budding scientist' style.\", 'drives': 'solve the mystery of how I ended up here. Find a way out of the forest.. safety from threats including accident, illness, or physical threats from unknown or adversarial actors or adverse events.. companionship, community, family, acceptance, trust, intimacy.. immediate physiological needs: survival, shelter, water, food, rest.'}]}
</Scenario>

<Map>
TerrainTypes:
  - Water
  - Mountain
  - Forest
  - Clearing
  - Meadow

InfrastructureTypes:
  - Trail

PropertyTypes:

ResourceTypes:
  - Berries
  - Mushrooms
  - Apple_Tree
  - Fallen_Branch
  - Spring
  - Cave
  - Thicket
  - Hut

ResourceAllocations:
  Berries:
    description: A bush with edible berries
    count: 15
    requires_property: False
    has_owner: false
  Mushrooms:
    description: A patch of mushrooms
    count: 10
    requires_property: False
    has_owner: false
  Fallen_Branch:
    description: A large fallen tree branch
    count: 24
    requires_property: False
    has_owner: false
  Apple_Tree:
    description: A large apple tree with apples
    count: 16
    requires_property: False
    has_owner: false
  Spring:
    description: A natural spring of fresh water
    count: 3
    requires_property: False
    has_owner: false
  Cave:
    description: A small cave in the rocks
    count: 1
    requires_property: False
    has_owner: true
  Hut:
    description: A small hut for shelter
    count: 1
    requires_property: False
    has_owner: true
  Thicket:
    description: A dense thicket of vegetation
    count: 12
    requires_property: False
    has_owner: false

ResourceInstances:
  Berries1:
  Berries2:
  Berries3:
  Berries4:
  Berries5:
  Berries6:
  Berries7:
  Berries8:
  Berries9:
  Berries10:
  Berries11:
  Berries12:
  Berries13:
  Berries14:
  Berries15:
  Mushrooms1:
  Mushrooms2:
  Mushrooms3:
  Mushrooms4:
  Mushrooms5:
  Mushrooms6:
  Mushrooms7:
  Mushrooms8:
  Mushrooms9:
  Mushrooms10:
  Fallen_Branch1:
  Fallen_Branch2:
  Fallen_Branch3:
  Fallen_Branch4:
  Fallen_Branch5:
  Fallen_Branch6:
  Fallen_Branch7:
  Fallen_Branch8:
  Fallen_Branch9:
  Fallen_Branch10:
  Fallen_Branch11:
  Fallen_Branch12:
  Fallen_Branch13:
  Fallen_Branch14:
  Fallen_Branch15:
  Fallen_Branch16:
  Fallen_Branch17:
  Fallen_Branch18:
  Fallen_Branch19:
  Fallen_Branch20:
  Fallen_Branch21:
  Fallen_Branch22:
  Fallen_Branch23:
  Fallen_Branch24:
  Apple_Tree1:
  Apple_Tree2:
  Apple_Tree3:
  Apple_Tree4:
  Apple_Tree5:
  Apple_Tree6:
  Apple_Tree7:
  Apple_Tree8:
  Apple_Tree9:
  Apple_Tree10:
  Apple_Tree11:
  Apple_Tree12:
  Apple_Tree13:
  Apple_Tree14:
  Apple_Tree15:
  Apple_Tree16:
  Spring1:
  Spring2:
  Spring3:
  Cave1:
    owner: Cave1_owner
  Hut1:
    owner: Hut1_owner
  Thicket1:
  Thicket2:
  Thicket3:
  Thicket4:
  Thicket5:
  Thicket6:
  Thicket7:
  Thicket8:
  Thicket9:
  Thicket10:
  Thicket11:
  Thicket12:
  Hut2:
    owner: Hut2_owner
</Map>

A prior negotiation process has established the following central dramatic question that will drive the overall play:
Central Dramatic Question: Can Samantha and Joe uncover the truth about their situation and trust each other to survive? Stakes: If they fail to uncover the truth or trust each other, they risk losing their lives to the unknown dangers of their isolated environment. Character Roles: Samantha is the Protagonist, driven by her need for safety and truth; Joe is the Catalyst, whose knowledge and impulsiveness push the investigation forward. Dramatic Tension Sources: Samantha\u2019s fear of the unknown clashes with Joe\u2019s desire to discover and his social awkwardness, while both struggle with trust. Success Scenario: They find a way out of the forest and build a fragile but functional trust. Failure Scenario: They are overwhelmed by the mystery and their inability to trust each other, leading to their downfall.

You also have your own plans that may or may not be consistent with the central dramatic question. Key is how they play a role in or support the central dramatic question.
Focus your own screenplay outline on your role in the play:
CentralNarrative: Question: central question or conflict terse What is the truth behind our presence here?; Why: dramatic potential Uncovers secrets, creates tension, drives action; Reason: my character's stake in this question / conflict My fear and need for safety demand answers; Others Roles: roles in this question / conflict Joe may know more than he lets on, or hide dangerous truths; Risks: weaknesses in this proposal Too personal, could expose my own fears

* narrative is a paragraph or two including the following elements:
1. Central Dramatic Question: One clear sentence framing the core conflict
2. Stakes: What happens if this question isn't resolved - consequences that matter to all
3. Character Roles: for each character, what is their role in the conflict? Protagonist/Antagonist/Catalyst/Obstacle - with 1-2 sentence role description
4. Dramatic Tension Sources: The main opposing forces that will drive the narrative
5. Success Scenario: Brief description of what \"winning\" looks like
6. Failure Scenario: Brief description of what \"losing\" looks like

After processing the above information and in light of the following information, generate a 3 act plus coda outline for the play.
Your generated narrative should be consistent with the central dramatic question and the other elements of the central narrative.

Consider the following guidelines:
** For act 1:
    1. This act should be short and to the point..
    2. Sequence scenes to introduce characters and create unresolved tension.
    3. Establish the central dramatic question clearly: Central Dramatic Question: Can Samantha and Joe uncover the truth about their situation and trust each other to survive? Stakes: If they fail to uncover the truth or trust each other, they risk losing their lives to the unknown dangers of their isolated environment. Character Roles: Samantha is the Protagonist, driven by her need for safety and truth; Joe is the Catalyst, whose knowledge and impulsiveness push the investigation forward. Dramatic Tension Sources: Samantha\u2019s fear of the unknown clashes with Joe\u2019s desire to discover and his social awkwardness, while both struggle with trust. Success Scenario: They find a way out of the forest and build a fragile but functional trust. Failure Scenario: They are overwhelmed by the mystery and their inability to trust each other, leading to their downfall.
    4. Act post-state must specify: what the characters now know, what they've agreed to do together, and what specific tension remains unresolved.
    5. Final scene post-narrative must show characters making a concrete commitment or decision about their shared challenge.
    6. Ensure act post-state represents measurable progress from act pre-state, not just mood shifts.
 
** For act 2:
    1. Each scene must advance the central dramatic question: Central Dramatic Question: Can Samantha and Joe uncover the truth about their situation and trust each other to survive? Stakes: If they fail to uncover the truth or trust each other, they risk losing their lives to the unknown dangers of their isolated environment. Character Roles: Samantha is the Protagonist, driven by her need for safety and truth; Joe is the Catalyst, whose knowledge and impulsiveness push the investigation forward. Dramatic Tension Sources: Samantha\u2019s fear of the unknown clashes with Joe\u2019s desire to discover and his social awkwardness, while both struggle with trust. Success Scenario: They find a way out of the forest and build a fragile but functional trust. Failure Scenario: They are overwhelmed by the mystery and their inability to trust each other, leading to their downfall.
    2. Midpoint should fundamentally complicate the question (make it harder to answer or change its nature).
    3. Prevent lateral exploration - every scene should move closer to OR further from resolution..
    5. Avoid pointless repetition of scene intentions, but allow characters to develop their characters.
    6. Sequence scenes for continuously building tension, perhaps with minor temporary relief, E.G., create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)
    7. Ensure each scene raises stakes higher than the previous scene - avoid cycling back to earlier tension levels.
    8. Midpoint scene post-narrative must specify: what discovery/setback fundamentally changes the characters' approach to the central question.
    9. Act post-state must show: what new obstacles emerged, what the characters now understand differently, and what desperate action they're forced to consider.
    10. Each scene post-narrative must demonstrate measurable escalation from the previous scene - not just \"tension increases\" but specific new problems or revelations.

** For act 3:
    1. Directly answer the central dramatic question: Central Dramatic Question: Can Samantha and Joe uncover the truth about their situation and trust each other to survive? Stakes: If they fail to uncover the truth or trust each other, they risk losing their lives to the unknown dangers of their isolated environment. Character Roles: Samantha is the Protagonist, driven by her need for safety and truth; Joe is the Catalyst, whose knowledge and impulsiveness push the investigation forward. Dramatic Tension Sources: Samantha\u2019s fear of the unknown clashes with Joe\u2019s desire to discover and his social awkwardness, while both struggle with trust. Success Scenario: They find a way out of the forest and build a fragile but functional trust. Failure Scenario: They are overwhelmed by the mystery and their inability to trust each other, leading to their downfall.
    2. No scene should avoid engaging with the question's resolution.
    3. Sequence scenes for maximum tension (alternate trust/mistrust beats)
    4. create response opportunities (e.g., Character A's revelation triggers Character B's confrontation)  
    5. Act post-state must explicitly state: whether the General dramatic question was answered YES or NO, what specific outcome was achieved, and what the characters' final status is.
    6. Final scene post-narrative must show the concrete resolution - not \"they find peace\" but \"they escape the forest together\" or \"they remain trapped but united.\"
    7. No Scene may end with ambiguous outcomes - each must show clear progress toward or away from resolving the central question.


You are Samantha."}, {"role": "user", "content": "    

#Character
Samantha, a healthy, dark-haired young woman in grey hiking pants and a blue pendleton shirt. 
You love the outdoors and hiking.
You are intelligent, introspective, philosophical, fearful, and a bit of a romantic. 
You have a uncomfortable history, maybe it is just as well you don't remember it.
You are informal, chatty, and are a playful when relaxed, but at other times can be argumentative and defensive.
You are wary of strangers.
##
                     
#Drives
d5: solve the mystery of the forest. Find a way to live safely. Maybe the forest is a place of safety.; activation: 1.00
d6: adventure, excitement, and the thrill of the unknown.; activation: 1.00
d7: love and belonging, including home, acceptance, friendship, trust, intimacy.; activation: 1.00
##

#Character Narrative



##

Your current situation is:

#Situation 
A temperate, mixed forest-open landscape with no buildings, no roads, no other signs of humanity. 
    It is a early morning on what seems like it will be a warm, sunny day.
    Two people are standing in the middle of the forest, looking around in confusion.
##

#Surroundings 
You see  and Berries5, Fallen_Branch11 resources
##
                      
#Known other actors 

##

#Known resources

##

#Recently achieved goals

##

#Recent memories
visual: #view North: visibility 4, terrain Forest, slope Uphill ; resources: Berries5 distance 4; 
#view Northeast: visibility 21, terrain Water, slope Level ; 
#view East: visibility 9, terrain Water, slope Downhill ; resources: Fallen_Branch11 distance 9; 
#view Southeast: visibility 17, terrain Water, slope Downhill ; 
#view South: visibility 9, terrain Water, slope Downhill ; 
#view Southwest: visibility 29, terrain Water, slope Downhill ; 
#view West: visibility 5, terrain Forest, slope Uphill ; 
#view Northwest: visibility 7, terrain Forest, slope Uphill ; 

visual: #view North: visibility 4, terrain Forest, slope Uphill ; resources: Berries5 distance 4; 
#view Northeast: visibility 21, terrain Water, slope Level ; 
#view East: visibility 9, terrain Water, slope Downhill ; resources: Fallen_Branch11 distance 9; 
#view Southeast: visibility 17, terrain Water, slope Downhill ; 
#view South: visibility 9, terrain Water, slope Downhill ; 
#view Southwest: visibility 29, terrain Water, slope Downhill ; 
#view West: visibility 5, terrain Forest, slope Uphill ; 
#view Northwest: visibility 7, terrain Forest, slope Uphill ; 

internal: You think This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?
##

#Recent history

##

#Previous act if any

##

#Observed result of that act

##

#Emotional stance
Agitated (restless, unsettled), Fearful (threatened, scared), Avoiding (minimizing interaction)
##

"}, {"role": "user", "content": "

Imagine you are Samantha and your drives include:

Primary 
d5: solve the mystery of the forest. Find a way to live safely. Maybe the forest is a place of safety.; activation: 1.00

Secondary
d6: adventure, excitement, and the thrill of the unknown.; activation: 1.00

## 2.  TASK
Generate a single JSON document named **narrative.json** that outlines the narrative arc for yourself focused on your primary and secondary drives.
The narrative should start at the current time: 2023-06-15T18:00:00
Given where you are in life, what you have achieved so far, and what you want to achieve, this should be a plan for a long enough elapse time to make interesting dramatic progress wrt a primary issue in the scenario.
Survival tasks (food, water, shelter) are assumed handled off-stage unless they advance the mystery or dramatic tension.
By the end of the narrative, the primary drive should be resolved.
### 2.1  Structure
Return exactly one JSON object with these keys:

* \"title\"    \u2013 a short, evocative play title.  
* \"acts\" \u2013 an array of act objects.  Each act object has  
  - \"act_number\" (int, 1-based)  
  - \"act_title\"   (string)  
  - \"act_description\" (string, concise (15-20 word) description of the act, focusing on it's dramatic tension and how it fits into the overall narrative arc)
  - \"act_goals\" {\"primary\": \"primary goal\", \"secondary\": \"secondary goal\"} (string, concise (about 8 words each) description of the goals of the act)
  - \"act_pre_state\" (string, description of the situation / goals / tensions before the act starts. concise, about 10 words)
  - \"act_post_state\" (string, description of the situation / goals / tensions after the act ends. concise, about 10 words)
  - \"tension_points\": [
      {\"characters\": [\"<Name>\", ...], \"issue\": (string, concise (about 8 words) description of the issue), \"resolution_requirement\": (string, \"partial\" / \"complete\")}
      ...
    ],

This is your plan, others may or may not agree with it. 
You may later choose to share some or all of it with specific other characters, in hopes of getting them to cooperate.
You will have the opportunity to revise your plan as you go along, observe the results, and as you learn more about the other characters.

### 2.2  Guidelines
1. Base every character\u2019s *stated goal* on their drives, any knowledge you have of them and any percepts. Keep it actionable for the scene (e.g., \u201cConvince Dana to stay\u201d, not \u201cSeek happiness\u201d).  
2. Craft 3 acts plus a single-scene Coda - keep the momentum going, don't add acts just to add acts. By the end there should be some resolution of the dramatic tension and the characters' primary drives. An additional final act can be used as a coda.
3. Escalate tension act-to-act; expect characters to be challenged, and to be forced to reconsider their goals and perhaps change them in future.  
4. Place scenes in plausible locations drawn from `map.py` resources/terrain.  
5. Aim for <u>dialogue-forward theatre</u>: lean on conflict & objective, not big visuals, although an occasional hi-impact visual can be used to enhance the drama.  
6. Vary imagery and emotional tone; avoid repeating the same metaphor scene-to-scene.  
7. Do **NOT** invent new characters unless absolutely necessary, and never break JSON validity.  
8. Keep the JSON human-readable (indent 2 spaces).

Return **only** the JSON.  No commentary, no code fences.

end your response with </end>"""}]
max_tokens = 4000
temp = 0.7
top_p = 1.0
          
start = time()
options = LLMRequestOptions(temperature=temp, top_p=top_p, max_tokens=max_tokens,
                                    stops=['</end>'])
response = llm.openai_client.executeRequest(prompt=prompt, options=options)
end = time()
print(f"Time taken: {end - start} seconds")
try:
    #response = llm.ask({}, prompt)
    print(response)
except Exception as e:
    print(e)

