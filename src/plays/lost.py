import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh

# the goal of an agh testbed is how long the characters can hold your interest and create an interesting and complex narrative. This is a classic 'survivors' sci-fi scenario.

# Create characters
server='deepseeklocal'
#server='local'
S = agh.Agh("Samantha", """You are Samantha, a healthy, attractive young woman. 
You love the outdoors and hiking.
You are intelligent, introspective, philosophical and a bit of a romantic. 
You have a painful history, maybe it is just as well you don't remember it.
You are very informal, chatty, think and speak in informal teen style, and are a playful and flirty when relaxed. 
You are comfortable on long treks, and are unafraid of hard work. 
You are wary of strangers. 
""", server=server)

# Drives are what cause a character to create tasks.
# Below is the default an agh inherits if you don't override, as we do below.
# basic Maslow (more or less).
# As usual, caveat, check agh.py for latest default!
# - immediate physiological needs: survival, water, food, clothing, shelter, rest.  
# - safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events. 
# - assurance of short-term future physiological needs (e.g. adequate water and food supplies, shelter maintenance). 
# - love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friendship, intimacy, trust, acceptance.

#Specifying for this scenario, otherwise all they do is hunt for water, berries, and grubs
S.set_drives([
    #"evaluation of Joe. Can I trust him?",
    #"safety from threats including accident, illness, or physical threats from unknown or adversarial actors or adverse events.",
    #"finding a way out of the forest.",
    "solving the mystery of how they ended up in the forest with no memory.",
    "love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "immediate physiological needs: survival, shelter, water, food, rest."
])
# Rows are in priority order, most important first. Have fun.
# note this is NOT NECESSARY to specify if you don't want to change anything.
S.add_to_history("You think This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?")

#
## Now Joe, the other character in this 'survivor' scenario
#

J = agh.Agh("Joe", """You are Joe, a healthy, nerdy young man, intelligent and self-sufficient. 
You are informal and somewhat impulsive. 
You are strong, and think you love the outdoors, but are basically a nerd.
You yearn for something more, but don't know what it is.
You are socially awkward, especially around strangers. 
You speak in informally.
""", server=server)

J.set_drives([
    "communication and coordination with Samantha, gaining Samantha's trust.",
    #"safety from threats including accident, illness, or physical threats from unknown or adversarial actors or adverse events.",
    #"finding a way out of the forest.",
    "solving the mystery of how they ended up in the forest with no memory.",
    #"love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "immediate physiological needs: survival, shelter, water, food, rest."
])

J.add_to_history("You think Ugh. Where am I?. How did I get here? Why can't I remember anything? Who is this woman?")
# add a romantic thread. Doesn't work very well yet. One of my test drivers of agh, actually.
J.add_to_history("You think Whoever she is, she is pretty!")


# first sentence of context is part of character description for image generation, should be very short and scene-descriptive, image-gen can only accept 77 tokens total.
W = context.Context([S, J],
                """A temperate, mixed forest-open landscape with no buildings, roads, or other signs of humanity. 
It is a early morning on what seems like it will be a warm, sunny day.
Two people are standing in the middle of the forest, looking around in confusion.""", server=server)

# uncomment if running from command line to use PyQt UI
#worldsim.main(W, server=server)