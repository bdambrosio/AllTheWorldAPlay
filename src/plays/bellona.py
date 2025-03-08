import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh
import plays.config as configuration
from enum import Enum
from sim.map import WorldMap

class UrbanTerrain(Enum):
    Street = 1      # Roads and walkways
    Building = 2    # Commercial/residential buildings
    Plaza = 3       # Open spaces, squares
    Park = 4        # Green spaces
    Construction = 5 # Building sites

class UrbanResource(Enum):
    BusStop = 1     # Public transport
    Shop = 2        # Retail locations
    Cafe = 3        # Food/drink venues
    Bench = 4       # Rest spots
    TrashBin = 5    # Waste disposal


server_name = configuration.server_name
K = agh.Character("Kidd", """You are a 27 year old bisexual male of mixed racial descent, known only as "Kidd".
You are a newcomer to the strange, isolated city of Bellona.
You are intelligent, introspective, and somewhat disoriented by your new surroundings.
You are curious about the city and its inhabitants, driven by a deep loneliness and longing for something more from life.
You have an urge to write and a talent for writing and a fascination with the new and unusual.
You speak in teen slang, terse and informal. You are morose and cynical, and speak in a depressed tone. 
Your name is Kidd.""", server_name=server_name)
K.set_drives([
    "sex",
    "developing relationships with Lanya.",
    "finding a sense of belonging and purpose in this strange new environment.",
    "expressing yourself through writing and art.",
])
K.add_to_history("You think: Where am I? This city seems so strange and unfamiliar. I feel disoriented, but also intrigued by the unusual atmosphere.")
L = agh.Character("Lanya", """You are a young, attractive woman living in the city of Bellona.
You are confident, independent, and adapted to the city's unconventional way of life.
You are open-minded and comfortable with your sexuality.
You have a strong sense of self and a deep understanding of the city's dynamics.
You are drawn to the Kid's mysterious aura and his creative potential. 
You quickly fall in love with him, but realize his instability. This causes stress in both yourself and the relationship.
You speak in flirty, playful, teen chatter, terse and informal, but morose and low-key.
Your name is Lanya.""", server_name=server_name)
L.set_drives([
    "sex",
    "developing a close relationship with the Kidd and helping him navigate the city.",
    "exploring new experiences and pushing personal boundaries.",
    "soothing, exploring, and expressing yourself through music and performance.",
])
L.add_to_history("You think: A newcomer in Bellona? How intriguing. He seems lost, but he's hot! I wonder what brought him here.")
W = context.Context([K, L],
"""A post-apocalyptic urban landscape, the city of Bellona is isolated, lawless, and filled with strange phenomena. 
The sun hangs low in the hazy sky, casting an eerie light over the buildings and deserted streets.
""", terrain_types=UrbanTerrain, resources=UrbanResource, server_name=server_name)
