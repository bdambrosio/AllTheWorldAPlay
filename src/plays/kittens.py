import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import sim.context as context
from src.sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.driveSignal import Drive
from plays.scenarios import garden

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
importlib.reload(garden)

map_file_name = 'garden.py' # needed to trigger narrative creation
narrative=False
lemon = NarrativeCharacter("Lemonade", """You are a pale grey tabbykitten. 
You love the outdoors, hunting bugs, and wrestling with Meow-Meow.
You are intelligent and very curious about everything.
Your name is Lemonade, others often call you Lemon""", server_name=server_name)

lemon.set_drives([
    "wrestling with Meow-Meow. Hunting bugs. Exploring",
    "love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "safety from threats including accident, physical threats from unknown or adversarial actors or adverse events by staying close to meow-meow",
])

meow = NarrativeCharacter("Meow-Meow", """You are a grey full-grown tabby cat. 
You love Lemonade, but sometimes need a break from her playfulness.
You like to sleep, and occasionally hunt bugs and butterflys.
Your name is Meow-Meow""", server_name=server_name)
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
