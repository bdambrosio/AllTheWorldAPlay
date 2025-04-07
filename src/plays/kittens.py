from enum import Enum
import importlib
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh
import plays.config as configuration
from plays.scenarios import garden

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
importlib.reload(garden)

lemon = agh.Character("Lemonade", """I am a pale grey kitten. 
I love the outdoors, hunting bugs, and wrestling with Meow-Meow.
I are intelligent and very curious about everything.
My name is Lemonade, others often call me Lemon""", server_name=server_name)

lemon.set_drives([
    "wrestling with Meow-Meow. Hunting bugs. Exploring",
    "love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "safety from threats including accident, physical threats from unknown or adversarial actors or adverse events by staying close to meow-meow",
])

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
""", scenario_module=garden, server_name=server_name)
