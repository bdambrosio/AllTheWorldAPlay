import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import sim.context as context
from src.sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.driveSignal import Drive
from plays.scenarios import apocalypse

importlib.reload(configuration)# force reload in case cached version
map_file_name = 'apocalypse.py' # needed to trigger narrative creation

server_name = configuration.server_name

K = NarrativeCharacter("Kidd", """Kidd,a 27 year old bisexual male of mixed racial descent, known only as "Kidd".
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
L = NarrativeCharacter("Lanya", """Lanya,a young, dark-haired woman dressed in scavenged clothes, living in the city of Bellona.
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
""", scenario_module=apocalypse, server_name=server_name)
