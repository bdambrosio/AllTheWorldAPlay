import asyncio
from enum import Enum
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sim.context as context
import sim.agh as agh
import plays.config as configuration
from sim.cognitive.driveSignal import Drive

server_name = configuration.server_name

# Custom terrain types for a small-town setting
class TownTerrain(Enum):
    Road = 1
    Building = 2
    Garden = 3
    Forest = 4
    River = 5

# Resources that might be found in this environment
class TownResource(Enum):
    Bench = 1
    Market = 2
    Workshop = 3
    Bridge = 4
    Boat = 5
    CityGallery = 6

# Main character - a person facing a difficult decision with competing drives
Maya = agh.Character("Maya", """You are Maya, a talented 32-year-old female artist,
You are living in a small coastal town.
You're warm, thoughtful, and value deep connections with others.
You've received a prestigious job offer in the city that would advance your career significantly.
However, you're deeply connected to your community, especially to Elijah, with whom you've developed a close relationship.
You're torn between personal ambition and the relationships you've built here, and angry at times that you have to make this decision.
You express yourself with careful consideration, often using artistic metaphors.
""", server_name=server_name)

Maya.drives = [
    Drive("pursuing your artistic ambitions and professional recognition."),
    Drive("maintaining and deepening your relationship with Elijah and your community."),
    Drive("finding balance between personal fulfillment and meaningful connections."),
    Drive("achieving stability and security for your future.")
]

Maya.add_perceptual_input("""You received the job offer letter yesterday. The gallery in the city wants you as their creative director. 
It's everything you've worked for, but accepting means leaving this place... and Elijah.""", 'internal')


# Supporting character with their own goals that create natural tension
Elijah = agh.Character("Elijah", """You are Elijah, a 35-year-old male boat builder.
You have deep roots in this small coastal town.
You're steady, reliable, and deeply connected to the natural rhythms of this place.
You've been building a life here, including a deepening relationship with Maya.
You care deeply about Maya and understand the importance of her dreams. At the same time, you are also ambitious and want to expand your boat-building business locally, which would root you even more firmly here. 
You don't really understand Maya's restlessness and ambition, and are sometimes frustrated by her lack of appreciation for the stability and security of your life together.
You speak with simple directness.
Your name is Elijah.""", server_name=server_name)

Elijah.drives = [
    Drive("building a sustainable future in the town you love."),
    Drive("developing your relationship with Maya into something lasting."),
    Drive("supporting those you care about in pursuing their happiness, even at personal cost."),
    Drive("maintaining the traditions and craft of boat building for future generations.")
]

Elijah.add_perceptual_input("Maya has been quiet since yesterday. You suspect something is troubling her, perhaps related to her art career. You've been planning to show her the workshop expansion plans today.", 'internal')

# Setting up the world context
W = None

W = context.Context([Maya, Elijah],
"""A small coastal town at sunset, where the river meets the sea. Colorful buildings line the waterfront, with artists' workshops and fishing boats creating a scene of rustic charm. The evening light casts long shadows as townspeople finish their day's work.
Maya and Elijah are sitting on a bench overlooking the harbor, where Elijah's newly completed wooden boat is moored. A letter from the city gallery is in Maya's pocket. The air is filled with the scent of salt water and wood. Birds call as they return to roost for the evening.""", 
terrain_types=TownTerrain, 
resources=TownResource, 
server_name=server_name)

