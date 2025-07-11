import asyncio
from enum import Enum
import importlib
import sys, os
import wave
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sim.context as context
from src.sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.driveSignal import Drive
from plays.scenarios import coastal

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
importlib.reload(coastal)

map_file_name = 'coastal.py' # needed to trigger narrative creation

# Main character - a person facing a difficult decision with competing drives
Maya = NarrativeCharacter("Maya", """Maya, a talented 32-year-old female artist, wearing a paint-splattered white t-shirt and jeans.
You are living in a small coastal town.
You're warm, thoughtful, and value connections with others, but also have a strong desire to be independent and free.
You are a bit of a free spirit, and you are not afraid to take risks, and dream of creating a big splash in the art world.
You express yourself with careful consideration.
""", server_name=server_name)

Maya.drives = [
    Drive("Pursue your artistic ambitions and professional recognition."),
    Drive("Excitement of achieving dramatic artistic expression, realizing your dreams")
    #Drive("maintain and deepening your relationship with Elijah and your community."),
    #Drive("finding balance between personal fulfillment and meaningful connections.")
]

Maya.add_perceptual_input("""Wow - a job offer letter! Chrys, the owner of the gallery in the city wants me as their creative director.""", 'internal') 
#Maya.add_perceptual_input("""It's everything I've worked for, but accepting means leaving this place... and Elijah.""", 'internal')


# Supporting character with their own goals that create natural tension
Elijah = NarrativeCharacter("Elijah", """Elijah, a 35-year-old male boat builder, wearing a t-shirt and jeans.
You wave deep roots in this small coastal town.
You're steady, reliable, and deeply connected to the natural rhythms of this place.
You've been building a life here, including a deepening relationship with Maya.
You care deeply about Maya and understand the importance of her dreams. 
At the same time, you are also ambitious and want to expand your boat-building business locally, which would root you even more firmly here. 
You don't really understand Maya's restlessness and ambition, and are sometimes frustrated by her lack of appreciation for the stability and security of your life together.
Compromise is not an option.
You speak with simple directness.
Your name is Elijah.""", server_name=server_name)

Elijah.drives = [
    Drive("building a sustainable future in the town you love."),
    Drive("developing your relationship with Maya into something lasting."),
    Drive("supporting those you care about in pursuing their happiness, even at personal cost."),
    Drive("maintaining the traditions and craft of boat building for future generations.")
]

Elijah.add_perceptual_input("Maya has been quiet since yesterday. You suspect something is troubling her, perhaps related to her art career. You've been planning to show her the workshop expansion plans today.", 'internal')

# Supporting character with their own goals that create natural tension
Chrys = NarrativeCharacter("Chrys", """Chrys, a 35-year-old female artist and gallery owner in the city, wearing informal but stylish clothes.
You are ambitious and driven, and you've been working to build your career in the art world.
You think the 'rugged coastal' mystique might be the next big thing, and are trying to attract Maya to your gallery.
In the interview with Maya you met Elijah, whom you find a serious obstacle to your plans.
You are not sure how to handle the situation, but you are determined to get Maya to move to the city. Compromise is not an option.
You speak with a confident, assertive, hurried, and slightly pushy city girl tone.
Your name is Chrys.""", server_name=server_name)

Chrys.drives = [
    Drive("Build my gallery into a major art destination."),
    Drive("The good life - wealth, power, and prestige."),
    Drive("The physical and emotional pleasures of life."),
]

Chrys.add_perceptual_input("Maya hasn't answered my offer yet. I need to follow up.", 'internal')

# Setting up the world context
W = None

W = context.Context([Maya, Elijah, Chrys],
"""A small coastal town at sunset, where the river meets the sea. Colorful buildings line the waterfront, with artists' workshops and fishing boats creating a scene of rustic charm. The evening light casts long shadows as townspeople finish their day's work.
Maya and Elijah are sitting on a bench overlooking the harbor, where Elijah's newly completed wooden boat is moored. A letter from the city gallery is in Maya's pocket. The air is filled with the scent of salt water and wood. Birds call as they return to roost for the evening.""", 
scenario_module=coastal,
server_name=server_name)

Maya.mapAgent.move_to_resource('Workshop1')
Elijah.mapAgent.move_to_resource('Workshop1')
Chrys.mapAgent.move_to_resource('Gallery1')

narrative='demo.json' # comment this out for normal unscripted play.

