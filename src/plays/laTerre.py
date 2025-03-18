from enum import Enum
from sim.cognitive.driveSignal import Drive
import sim.worldsim as worldsim
import sim.agh as agh
from sim.context import Context
import plays.config as configuration

server_name = configuration.server_name
class RuralTerrain(Enum):
    Field = 1    # Farmland (flat, open)
    Woods = 2    # Forest/trees
    Meadow = 3   # Natural grassland

class RuralInfrastructure(Enum):
    Road = 1         # All roads/paths combined
    MarketSquare = 2 # Central gathering point

class RuralBuilding(Enum):
    Farmhouse = 1    # Living/working buildings
    Barn = 2         # Storage/work buildings

class RuralResource(Enum):
    Produce = 1      # Farm goods
    Wood = 2         # Forest materials


J = agh.Character("Jean", """You are Jean Macquart, a hardworking young unmarried peasant farmer working his father's farm. 
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
J.add_perceptual_input("You think – Another long day of toil in the fields. When will I get my fair share of this land that I pour my sweat into? I returned from the army to be a farmer, not a lackey for my family.", 'internal')
J.add_perceptual_input("You think - That Francoise is a hard worker, and pretty too. If I ever had my own farm she would be a good partner.", 'internal')

F = agh.Character("Francoise", """You are Francoise Fouan, an attractive unmarried young woman from a neighboring peasant family in the same village as Jean.
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
F.add_perceptual_input("You think – I saw that Jean Macquart again in the field. He works so hard for his family. Seems to have a chip on his shoulder though. Best not to stare and set the gossips' tongues wagging.", 'internal')


W =Context([J, F],
"""A small 19th century French farming village surrounded by fields ripe with wheat and other crops. 
It is late afternoon on a hot summer day.""", terrain_types=RuralTerrain, resources=RuralResource, server_name=server_name)
#worldsim.IMAGEGENERATOR = 'tti_serve'
#worldsim.main(W)

# Simple rules based on elevation
terrain_rules = {
    'Field': {'elevation_range': (0.2, 0.6)},  # Mid elevations
    'Woods': {'elevation_range': (0.6, 1.0)},   # Higher ground
    'Meadow': {'elevation_range': (0.0, 0.3)}   # Near water/low ground
}

# Place market in center-ish of map
# Connect with simple A* pathfinding for roads
infrastructure_rules = {
    'MarketSquare': {
        'prefer_terrain': 'Field',
        'min_distance_water': 3
    },
    'Road': {
        'max_slope': 0.2,
        'avoid_terrain': 'Woods'
    }
}

property_rules = {
    'min_size': 50,
    'require_road_access': True,
    'max_distance_road': 5
}

placement_rules = {
    'Farmhouse': {
        'require_terrain': 'Field',
        'max_distance_road': 3
    },
    'Barn': {
        'near_farmhouse': True,
        'max_distance': 2
    },
    'Produce': {
        'require_terrain': 'Field',
        'near_building': True
    },
    'Wood': {
        'require_terrain': 'Woods'
    }
}
