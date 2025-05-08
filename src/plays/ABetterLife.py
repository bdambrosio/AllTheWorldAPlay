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
import plays.scenarios.suburban as suburban

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
importlib.reload(suburban)    

map_file_name='suburban.py' # needed to trigger narrative creation

# Character definitions
H = NarrativeCharacter("Hu", """You are Hu Manli, a healthy, attractive 39 year old female insurance executive. 
You are a workaholic, but you also love to travel and have a passion for adventure.
Life has suddently thrown you a curveball. A competitor named Xue is stealing your clients and your job.
You are determined to get your job back and outshine Xue.
You are ambitious, competitive, and determined.
""", 
server_name=server_name)

H.set_drives([
    "succeed at work, outshine Xue",
    "find adventure, excitement, the thrill of the unknown, and maybe some romance, exploring new ideas and tactics",
    "stable marraige as background to a prosperous life"
])


X = NarrativeCharacter("Xue", """You are Xue Xiaozhou, a 28 year old male insurance executive from a wealthy family. 
You are a workaholic and love the competition.
You start out down on your luck, with a competitor named Hu is fighting for your clients.
You are ambitious, competitive, and determined to outshine Hu.
""",
server_name=server_name)

X.set_drives([
        "succeed at work, outshine Hu",
        "find adventure, excitement, the thrill of the unknown, and maybe some romance, exploring new ideas and tactics."
])

D = NarrativeCharacter("Ding", """You are Ding Zhiyuan, Philosophy professor and husband of Hu.""", server_name=server_name)
D.set_drives([
    "succeed at work, outshine Hu",
    "midlife crisis - find adventure, excitement, romance, maybe a mistress"
])

Wang = NarrativeCharacter("Wang", """You are Wang Xue, a 23 year old female student of Ding.""", server_name=server_name)
Wang.set_drives([
    "make a name for yourself, be a good student",
    "find adventure, excitement, romance, maybe a boyfriend"
])


# Create context with forest scenario
W = context.Context([H, X],
    """A modern chinese urban  setting with a mix of buildings, roads, and other signs of humanity.""",
    scenario_module=suburban,  # Pass the forest scenario module
    npcs=[D, Wang],
    server_name=server_name,
   )
H.add_perceptual_input("You suspect your husband Ding is having an affair with Wang Xue, his student", 'internal')


H.mapAgent.move_to_resource('Office#1')
X.mapAgent.move_to_resource('Office#2')
W.reference_manager.declare_relationship('Hu', 'wife of', 'Ding', 'husband of')
W.reference_manager.declare_relationship('Wang', 'student of', 'Ding', 'teacher of')
#narrative='lost.json' # comment this out for normal unscripted play.
