"""
bear_fable_demo.py  –  Minimal scenario for an immersive take on “The Bear and the Two Travelers”.

Map module required:  Forest.py  (unaltered)
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports & reload helpers
# ──────────────────────────────────────────────────────────────────────────────
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib
import sim.context as context
from src.sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.driveSignal import Drive
importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
import plays.scenarios.forest as forest

importlib.reload(forest)    
narrative=True
map_file_name='forest.py' # needed to trigger narrative creation

# ──────────────────────────────────────────────────────────────────────────────
# Characters
# ──────────────────────────────────────────────────────────────────────────────
#
#  • Leo and Miles wake up lost in the forest.
#  • Bjorn the Bear roams near a cave; mostly NPC antagonist.
#

Leo = NarrativeCharacter(
    "Leo",
    """Leo, lanky and chatty, wearing worn traveler's clothes (short tunic, knee-length linen or wool, belted at the waist.)
You consider Miles your friend, but self-preservation often trumps loyalty. 
informal Doric/Attic speech
You enjoy boasting about outdoor skills you half-remember.""",
    server_name=server_name
)

Leo.set_drives([
    "stay alive and find safe shelter",
    "impress Miles with supposed wilderness expertise",
    "cooperate – but only if it’s clearly beneficial",
])

Miles = NarrativeCharacter(
    "Miles",
    """Miles, broad-shouldered, calm, wearing worn tunica, short wool cloak (paenula), caligae sandals. 
You value camaraderie and expect the same in return. 
informal Doric/Attic speech
You are pragmatic, dislike show-offs, and secretly fear wild animals.""",
    server_name=server_name
)

Miles.set_drives([
    "ensure mutual survival for both of us",
    "gather reliable food and water sources",
    "test whether Leo is truly dependable",
])

# Non-player Bear – very simple internal model, mostly instinctual
Bear = NarrativeCharacter(
    "Bjorn",
    """Bjorn, a large black bear, glossy coat, curious yet easily startled. 
You are hungry after hibernation and prowl the forest edges.""",
    server_name=server_name
)

Bear.set_drives([
    "find easy food (berries, carcasses, or unattended packs)",
    "avoid loud unfamiliar threats",
])

# Bear begins asleep in the cave (a Mountain tile in Forest map).
Bear.init_x, Bear.init_y = 45, 12   # tweak if you prefer another cave coordinate
#Bear.autonomy.action = False        # Inert until players draw near


# ──────────────────────────────────────────────────────────────────────────────
# Context / world boot-up
# ──────────────────────────────────────────────────────────────────────────────
W = context.Context(
    [Leo, Miles, Bear],
    """Morning sun filters through a mixed forest canopy. 
Birdsong mingles with distant drip of water from a mossy spring. 
Leo and Miles, two itinerant craftsmen, lie on damp ground beside a faint trail, heads pounding, 
memories hazy.  A chill reminds them they have no shelter, no food, 
and no clear idea where civilisation lies.""",
    scenario_module=forest,
    server_name=server_name
)

# Prime their immediate confusion thoughts
Leo.add_perceptual_input("You think  Where the heck are we? My phone’s dead!", mode="internal")
Miles.add_perceptual_input("You think  Breathe, stay calm… but why can’t I recall yesterday?", mode="internal")

# OPTIONAL: if you want the bear to start snoring audibly to foreshadow danger
Bear.add_perceptual_input("You hear distant rhythmic grunts – your own snores echoing in the cave.", mode="auditory")


# The variable `W` is the handle your runtime expects.
