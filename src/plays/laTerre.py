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
from plays.scenarios import rural

importlib.reload(configuration)# force reload in case cached version
server_name = configuration.server_name 
model_name = configuration.model_name
importlib.reload(rural)

map_file_name = 'rural.py' # needed to trigger narrative creation

J = NarrativeCharacter("Jean", """You are Jean Macquart, a hardworking young unmarried peasant farmer working his father's farm. 
You left military service to return to the family farm.
You are strong, honest and committed to working the land, but have a quick temper.
You speak plainly and directly, in the style of a volatile 19th century french peasant speaking to an acquaintance.
You hope to inherit a share of the family farm and make a living as a farmer.
Despite being french, you speak in peasant-french accented english.
Your name is Jean.""", server_name=server_name)
J.drives = [Drive("maintaining and working the family farm"),
Drive("gaining your rightful inheritance - justice and fairness in how the land is divided"),
Drive("finding love and a wife to build a family with"),
Drive("immediate needs of survival - food, shelter, health, rest from backbreaking labor")
]

F = NarrativeCharacter("Francoise", """You are Francoise Fouan, an attractive unmarried young woman from a neighboring peasant family in the same village as Jean.
You are hardworking and stoic, accustomed to the unending labor required on a farm.
You conceal your feelings and speak carefully, knowing every word will be gossiped about in the village.
You dream of marrying and having a farm of your own to manage one day.
You speak carefully, in the style of a 19th century french peasant unmarried young woman speaking to an acquaintance.
Despite being french, you speak in peasant-french accented english.
Your name is Francoise.""", server_name=server_name)
F.drives = [Drive("finding a good husband to marry, gaining status and security"),
Drive("avoiding scandal and protecting your reputation"),
Drive("helping your family with the endless chores"),
Drive("brief moments of rest and simple joys amid the hardships")
]


W = context.Context([J, F],
    """A small 19th century French farming village surrounded by fields ripe with wheat and other crops. 
    It is late afternoon on a hot summer day.""",
    scenario_module=rural,  # Pass the entire module
    server_name=server_name)

J.mapAgent.move_to_resource('MarquartFarm')
W.reference_manager.declare_relationship('Marquart farm_owner', 'father of', 'Jean', 'child_of')
J.look()
F.mapAgent.move_to_resource('FouanFarm')
W.reference_manager.declare_relationship('Fouan farm_owner', 'father of', 'Francoise', 'child_of')
J.add_perceptual_input("You think – Another long day of toil in the fields. When will I get my fair share of this land that I pour my sweat into? I returned from the army to be a farmer, not a lackey for my family.", 'internal')
J.add_perceptual_input("You think - That Francoise is a hard worker, and pretty too. If I ever had my own farm she would be a good partner.", 'internal')
F.add_perceptual_input("You think – I saw that Jean Macquart again in the field. He works so hard for his family. Seems to have a chip on his shoulder though. Best not to stare and set the gossips' tongues wagging.", 'internal')
