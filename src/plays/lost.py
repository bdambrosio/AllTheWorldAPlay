import sys, os
print("DEBUG lost.py: imported sys, os")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("DEBUG lost.py: added to sys.path")

import importlib
print("DEBUG lost.py: imported importlib")

import sim.context as context
print("DEBUG lost.py: imported sim.context")

from src.sim.narrativeCharacter import NarrativeCharacter
print("DEBUG lost.py: imported NarrativeCharacter")

import plays.config as configuration
print("DEBUG lost.py: imported plays.config")

from sim.cognitive.driveSignal import Drive
print("DEBUG lost.py: imported Drive")
importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
import plays.scenarios.forest as forest

importlib.reload(forest)    

map_file_name='forest.py' # needed to trigger narrative creation
print(f'DEBUG: map_file_name: {map_file_name}')
# Character definitions
S = NarrativeCharacter("Samantha", """Samantha, a healthy, dark-haired young woman in grey hiking pants and a blue pendleton shirt. 
You love the outdoors and hiking.
You are intelligent, introspective, philosophical, fearful, and a bit of a romantic. 
You have a uncomfortable history, maybe it is just as well you don't remember it.
You are informal, chatty, and are a playful when relaxed, but at other times can be argumentative and defensive.
You are wary of strangers.""", 
server_name=server_name)
print(f'DEBUG: Samantha created')
S.set_drives([
    "solve the mystery of the forest. Find a way to live safely. Maybe the forest is a place of safety.",
    "adventure, excitement, and the thrill of the unknown.",
    "love and belonging, including home, acceptance, friendship, trust, intimacy."
])

print(f'DEBUG: Samantha drives set')
J = NarrativeCharacter("Joe", """Joe, a healthy young man, short beard  and dark hair, in grey chinos and a red t-shirt. 
You are intelligent and self-sufficient, informal and somewhat impulsive. 
You are strong, and think you love the outdoors, but are basically a nerd.
You yearn for something more, but don't know what it is.
You are socially awkward, especially around strangers. 
You speak informally, but occasionally in a 'budding scientist' style.""",
server_name=server_name)
print(f'DEBUG: Joe created')
J.set_drives([
    "solve the mystery of how I ended up here. Find a way out of the forest.",
    "safety from threats including accident, illness, or physical threats from unknown or adversarial actors or adverse events.",
    "companionship, community, family, acceptance, trust, intimacy.",
    "immediate physiological needs: survival, shelter, water, food, rest."
])
S.add_perceptual_input("You think This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?", 'internal')
J.add_perceptual_input("You think Ugh. Where am I?. How did I get here? Why can't I remember anything? Who is this woman?", 'internal')
J.add_perceptual_input("You think Whoever she is, she is pretty!", 'internal')
print(f'DEBUG: Joe perceptual inputs added')

# Create context with forest scenario
W = context.Context([S, J],
    """A temperate, mixed forest-open landscape with no buildings, no roads, no other signs of humanity. 
    It is a early morning on what seems like it will be a warm, sunny day.
    Two people are standing in the middle of the forest, looking around in confusion.""",
    scenario_module=forest,  # Pass the forest scenario module
    server_name=server_name)

S.look() # get the initial view
J.look()

#narrative='lost.json' # comment this out for normal unscripted play.
