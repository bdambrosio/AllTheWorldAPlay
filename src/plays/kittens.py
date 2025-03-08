from enum import Enum
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh
import plays.config as configuration

server_name = configuration.server_name
class GardenTerrain(Enum):
    Grass = 1      # Roads and walkways
    Tree = 2    # Commercial/residential buildings
    Flower = 3       # Open spaces, squares
    Bush = 4        # Green spaces

class GardenResource(Enum):
    Bug = 1     # Public transport
    Butterfly = 2        # Retail locations
    Flower = 3        # Food/drink venues
    Leaf = 4       # Rest spots
    Seed = 5    # Waste disposal
# the goal of an agh testbed is how long the characters can hold your interest and create an interesting and complex narrative. This is a classic 'survivors' sci-fi scenario.

# Create characters
# I like looking at pretty women. pbly because I'm male hetero oriented. If that offends, please change to suit your fancy.
# I find it disorienting for characters to change ethnic characteristics every time they are rendered, so they are nailed down here.
# I'm of Sicilian descent on my mother's side (no, not Italian - family joke).
lemon = agh.Character("Lemonade", """I am a pale grey kitten. 
I love the outdoors, hunting bugs, and wrestling with Meow-Meow.
I are intelligent and very curious about everything.
My name is Lemonade, others often call me Lemon""", server_name=server_name)

# Drives are what cause a character to create tasks.
# Below is the default an agh inherits if you don't override, as we do below.
# basic Maslow (more or less).
# As usual, caveat, check agh.py for latest default!
# - immediate physiological needs: survival, water, food, clothing, shelter, rest.  
# - safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events. 
# - assurance of short-term futur physiological needs (e.g. adequate water and food supplies, shelter maintenance). 
# - love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friendship, intimacy, trust, acceptance.

#Specifying for this scenario, otherwise all they do is hunt for water, berries, and grubs
lemon.set_drives([
    "wrestling with Meow-Meow. Hunting bugs. Exploring",
    "love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "safety from threats including accident, physical threats from unknown or adversarial actors or adverse events by staying close to meow-meow",
])
# Rows are in priority order, most important first. Have fun.
# note this is NOT NECESSARY to specify if you don't want to change anything.
#lemon.add_to_history("You think This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?")


meow = agh.Character("Meow-Meow", """I am a grey full-grown tabby cat. 
I love Lemonade, but sometimes need a break from her playfulness.
I like to sleep, and occasionally hunt bugs and butterflys.
My name is Meow-Meow""", server_name=server_name)
meow.set_drives([
    "keep watch over Lemonade and keep her safe.",
    "play with Lemonade and share in her adventures.",
    "love and belonging, including home, acceptance, friendship, trust, intimacy."
])

meow.add_to_history("Where did that Lemon go this time?")


# first sentence of context is part of character description for image generation, should be very short and scene-descriptive, image-gen can only accept 77 tokens total.
W = context.Context([lemon, meow],
                """A wonderful backyard garden playground, full of adventures for little kittens. Magical things are always happenning.
""", terrain_types=GardenTerrain, resources=GardenResource, server_name=server_name)

#worldsim.IMAGEGENERATOR = 'tti_serve'

#worldsim.main(W, server_name=server_name)
