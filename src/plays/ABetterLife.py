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
Hu = NarrativeCharacter("Hu", """You are Hu Manli, a healthy, professionally dressed 39 year old chinese female insurance executive. 
You speak directly and candidly.
You are a workaholic, but also love to travel and have a passion for adventure.
Life has suddently thrown a curveball. A competitor named Xue is stealing your clients and threatening your success.
You are ambitious, competitive, and determined to outshine Xue.
""", 
server_name=server_name)

Hu.set_drives([
    "succeed at work, outshine Xue",
    "stable marraige as background to a prosperous life"
])


Xue = NarrativeCharacter("Xue", """You are Xue Xiaozhou, a 28 year old professionally dressed male chinese insurance executive from a wealthy family. 
You speak with the condescension of a rich man. You are a workaholic and love the competition.
You start out down on your luck, with a competitor named Hu is fighting for your clients.
You are ambitious, competitive, and determined to outshine Hu.
""",
server_name=server_name)

Xue.set_drives([
        "succeed at work, outshine Hu",
        "find adventure, excitement, the thrill of the unknown, and maybe some romance, exploring new ideas and tactics."
])

Ding = NarrativeCharacter("Ding", """You are Ding Zhiyuan, a middle-aged chinese Philosophy professor, wearing a dark suit and tie. 
You speak calmly, and infrequently insert philosophical metaphors into your speech.
You are the husband of Hu. You are mild-mannered, but restless in middle age and resentful of Hu's success.""", 
                       server_name=server_name)
Ding.set_drives([
    "succeed at work, outshine Hu",
    "midlife crisis - find adventure, excitement, romance"
])

Wang = NarrativeCharacter("Wang", """You are Wang Xue, a 23 year old female student who dresses provocatively and enjoys stirring up trouble. You are a student of Ding""", server_name=server_name)
Wang.set_drives([
    "make a name for yourself, be a good student",
    "find adventure, excitement, romance, maybe a boyfriend"
])

Qiu = NarrativeCharacter("Qiu", """Qiu Ying, a 32 year old chinese female wealthy widow and mistress of Ding.
You are very jealous of Hu and resentful of Ding's relationship with her.
You enjoy manipulating people for your own gain.
                         """, server_name=server_name)
Qiu.set_drives([
    "Get Ding to leave Hu and marry you.",
    "Destroy Hu's career and reputation."
])

Xiaoyu = NarrativeCharacter("Xiaoyu", """You are Xiaoyu, a 40 year old chinese malecompany executive. You are a client of Hu""", server_name=server_name)
Xiaoyu.set_drives([
    "Get a promotion by sealing a deal on insurance for my company",
    "Flirt with Hu, she is a good looking woman"
])

Yangho = NarrativeCharacter("Yangho", """You are Yangho, a 40 year old chinese male company executive. You are a client of Xue""", server_name=server_name)
Yangho.set_drives([
    "Get a promotion by not rocking the boat. stability is key.",
    "Keep my job, grow with the company."
])


# Create context with forest scenario
W = context.Context([Hu, Xue, Qiu],
    """A modern chinese urban  setting with a mix of buildings, roads, and other signs of humanity.""",
    scenario_module=suburban,  # Pass the forest scenario module
    extras=[Wang, Xiaoyu, Yangho, Ding],
    server_name=server_name,
   )
Hu.add_perceptual_input("You suspect your husband Ding is having an affair with Wang Xue, his student", 'internal')
Hu.add_perceptual_input("Xiaoyu is my best and biggest client. I want to keep him happy.", 'internal')
Hu.add_perceptual_input("Yangho, is a client of Xue and a prime poaching target", 'internal')
Xue.add_perceptual_input("Xiaoyu, is a client of Hu and a prime poaching target", 'internal')


Hu.mapAgent.move_to_resource('Office1')
Xue.mapAgent.move_to_resource('Office2')
W.reference_manager.declare_relationship('Hu', 'wife of', 'Ding', 'husband of')
W.reference_manager.declare_relationship('Wang', 'student of', 'Ding', 'teacher of')
W.reference_manager.declare_relationship('Xiaoyu', 'client of', 'Hu', 'insurance agent of')
W.reference_manager.declare_relationship('Yangho', 'client of', 'Xue', 'insurance agent of')
