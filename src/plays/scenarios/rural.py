from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

# Import the new dynamic resource system
from sim.map import ResourceTypeRegistry

# All enums for rural scenario
class RuralTerrain(Enum):
    Water = 1
    Mountain = 2
    Forest = 3
    Grassland = 4
    Field = 5

class RuralInfrastructure(Enum):
    Road = auto()

class RuralResources(Enum):
    Well = auto()
    Mill = auto()
    Blacksmith = auto()
    Farm = auto()
    Market = auto()

class RuralProperty(Enum):
    Farm = auto()
    Village = auto()

# Rules for terrain generation
terrain_rules = {
    'elevation_noise_scale': 50.0,
    'water_level': 0.2,
    'mountain_level': 0.8,
    'terrain_by_elevation': {
        'water': {'max': 0.2, 'type': 'Water'},
        'mountain': {'min': 0.8, 'type': 'Mountain'}
    },
    'lowland_distribution': {
        'Forest': 0.4,
        'Grassland': 0.3,
        'Field': 0.3
    }
}

infrastructure_rules = {
    'road_density': 0.1,
    'path_type': 'Road',
    'slope_factor': 2.0,
    'terrain_costs': {
        'Water': float('inf'),
        'Mountain': float('inf'),
        'Forest': 2.0,
        'Grassland': 1.0,
        'Field': 1.0
    }
}

property_rules = {
    'min_size': 50,
    'max_size': 150,
    'valid_terrain': ['Field', 'Grassland']
}

# Standard interface names
terrain_types = RuralTerrain
infrastructure_types = RuralInfrastructure
property_types = RuralProperty
resource_types = ResourceTypeRegistry(RuralResources)  # Use dynamic registry

# Add at top with other interface names
required_resource = resource_types.Mill  # or Market, etc.
required_resource_name = "Mill"  # or "Market", etc.

resource_rules = {

    'names': {
        'Farm': ['MarquartFarm', 'FouanFarm'],

    },
    'allocations': [
        {
            'resource_type': resource_types.Well,
            'description': 'A deep well for water',
            'count': 5,
            'has_npc': True,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Grassland: 1.0,
                terrain_types.Field: 1.0
            }
        },
        {
            'resource_type': resource_types.Mill,
            'has_npc': True,
            'description': 'A grain mill',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Field: 2.0,
                terrain_types.Grassland: 1.0
            }
        },
        {
            'resource_type': resource_types.Blacksmith,
            'has_npc': True,
            'description': 'A blacksmith workshop',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Field: 1.0,
                terrain_types.Grassland: 1.0
            }
        },
        {
            'resource_type': resource_types.Farm,
            'has_npc': True,
            'description': 'A farmhouse with barn',
            'count': 8,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Field: 2.0,
                terrain_types.Grassland: 1.0
            }
        },
        {
            'resource_type': resource_types.Market,
            'description': 'The village market square',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Field: 1.0,
                terrain_types.Grassland: 1.0
            }
        }
    ]
}



# What is at stake in this world (planning and act prompts; sim/survival.py stakes_text).
stakes = ("In this peasant village, land, money, harvest and inheritance are what is at stake: who holds which "
          "field, who works it, what the wheat fetches at market, who pays whom, and who will inherit. Heat, hard "
          "labour and thirst are real. A farm belongs to whoever its owner has signed the deed over to: the owner holds "
          "the deed and signs it over only in person, with the heir standing there. Use these to make concrete, "
          "checkable goals and obstacles tied to the farms, the mill, the market and the family.")

# Stage business (sim/survival.py): field work, wheat, flour and bread, money, land claims.
affordances = {
    'needs': {   # late afternoon on a hot summer day
        'thirst': {'initial': 0.4, 'per_hour': 0.06, 'neglect_harm': 0.03,
                   'words': ('not thirsty', 'thirsty', 'very thirsty', 'parched'),
                   'percepts': {0.5: "The heat has dried your mouth; you want water from the well.",
                                0.8: "Your head swims in the heat; you need water."}},
        'hunger': {'initial': 0.3, 'per_hour': 0.03,
                   'words': ('fed', 'hungry', 'very hungry', 'famished'),
                   'percepts': {0.5: "You are hungry; supper is a long way off.",
                                0.8: "Hunger gnaws at you; you need bread."}},
        'fatigue': {'initial': 0.35, 'per_hour': 0.03, 'shelter_factor': 0.5,
                    'words': ('fresh', 'tired', 'worn out', 'spent'),
                    'percepts': {0.5: "Your back aches from the day's work.",
                                 0.8: "You are worn out; you can barely lift your arms."}},
    },
    'harm': {'name': 'injury', 'words': ('unhurt', 'bruised', 'hurt', 'badly hurt')},
    'travel_need': 'fatigue',
    'currency': 'francs', 'initial_money': 5,
    'tile_minutes': 4,
    'type_nouns': {'Well': ['well'], 'Mill': ['mill', 'miller'], 'Blacksmith': ['blacksmith', 'forge', 'smithy'],
                   'Farm': ['farm', 'field', 'fields', 'wheat', 'land', 'plot'], 'Market': ['market', 'stall']},
    'actions': [
        {'name': 'draw water', 'verbs': ['drink', 'draw water', 'drink water'], 'named_verbs': ['draw', 'haul', 'lower the bucket'],
         'resource_types': ['Well'], 'effects': {'thirst': -0.7}, 'minutes': 10, 'text': 'draws water at {resource} and drinks'},
        # hiring out is on another family's land and is said as such: cutting wheat is 'harvest
        # family wheat' (it stood first with the same verbs, so Jean harvesting became day labour)
        {'name': 'work for wages', 'category': 'work', 'verbs': ['hire out', 'hire myself', 'labour for', 'work for wages', 'day labour', 'day-wage'],
         'named_verbs': ['hire'], 'resource_types': ['Farm'], 'requires_property': {'owner': '!{household}'},
         'effects': {'fatigue': 0.25, 'thirst': 0.15}, 'earn': 2, 'minutes': 180,
         'text': 'labours in the fields at {resource} for a day-wage'},
        {'name': 'take possession', 'category': 'work', 'verbs': ['claim', 'take possession', 'stake a claim', 'pace out'],
         'resource_types': ['Farm'], 'requires_property': {'owner': None}, 'sets_property': {'owner': '{actor}'},
         'minutes': 30, 'text': 'takes possession of {resource}, pacing out its boundaries'},
        # '{household}': the farm is the actor's own, or a parent's or spouse's (see laTerre.py)
        {'name': 'harvest family wheat', 'category': 'work', 'verbs': ['harvest', 'reap', 'scythe', 'bring in the wheat'],
         'named_verbs': ['work', 'cut', 'gather'], 'resource_types': ['Farm'], 'requires_property': {'owner': '{household}'},
         'effects': {'fatigue': 0.3, 'thirst': 0.15}, 'gain': 'sack of wheat', 'minutes': 180,
         'text': "harvests wheat on the family's land at {resource}"},
        {'name': 'grind wheat', 'category': 'work', 'verbs': ['grind', 'mill'], 'resource_types': ['Mill'],
         'requires': 'sack of wheat', 'consumes': 'sack of wheat', 'gain': 'sack of flour', 'cost': 1, 'minutes': 60,
         'text': "has the wheat ground at {resource}, paying the miller's fee"},
        {'name': 'sell wheat', 'category': 'work', 'verbs': ['sell', 'trade'], 'resource_types': ['Market'],
         'requires': 'sack of wheat', 'consumes': 'sack of wheat', 'earn': 6, 'minutes': 40,
         'text': 'sells a sack of wheat at {resource}'},
        {'name': 'sell flour', 'category': 'work', 'verbs': ['sell', 'trade'], 'resource_types': ['Market'],
         'requires': 'sack of flour', 'consumes': 'sack of flour', 'earn': 9, 'minutes': 40,
         'text': 'sells a sack of flour at {resource}'},
        {'name': 'buy bread', 'verbs': ['buy', 'purchase'], 'resource_types': ['Market'], 'cost': 1, 'gain': 'loaf of bread',
         'minutes': 10, 'text': 'buys a loaf of bread at {resource}'},
        {'name': 'eat bread', 'verbs': ['eat', 'chew', 'tear off a piece'], 'resource_types': None,
         'requires': 'loaf of bread', 'consumes': 'loaf of bread', 'effects': {'hunger': -0.5}, 'minutes': 15,
         'text': 'eats bread'},
        {'name': 'have tools mended', 'category': 'work', 'verbs': ['mend', 'repair', 'sharpen', 'shoe'], 'resource_types': ['Blacksmith'],
         'cost': 2, 'sets': 'tools mended', 'minutes': 60, 'text': 'has tools mended at {resource}'},
        {'name': 'go indoors', 'verbs': ['go inside', 'go indoors', 'shelter', 'take shelter', 'take cover', 'step inside'],
         'resource_types': ['Farm', 'Mill', 'Blacksmith'], 'effects': {'fatigue': -0.1}, 'minutes': 10, 'sets': 'sheltered',
         'text': 'goes indoors at {resource}'},
        {'name': 'rest in the shade', 'verbs': ['rest', 'sit', 'lie down', 'take a break'], 'resource_types': None,
         'effects': {'fatigue': -0.15}, 'minutes': 30, 'text': 'rests in the shade'},
        {'name': 'pay', 'category': 'social', 'target': 'character', 'verbs': ['pay', 'hand over', 'settle'],
         'transfer_money': 2, 'minutes': 2, 'text': 'pays {target} two francs'},
        {'name': 'give bread', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'offer', 'share'],
         'transfer': 'loaf of bread', 'minutes': 2, 'text': 'gives {target} a loaf of bread'},
        # a farm passes when its owner, holding the deed, signs it over with the heir standing there
        {'name': 'sign over the farm', 'category': 'social', 'target': 'character',
         'verbs': ['sign over', 'sign the deed', 'signs the deed', 'put my mark', 'makes his mark', 'make my mark', 'deed the farm'],
         'transfer': '*deed', 'conveys': ' deed', 'minutes': 10, 'text': 'signs the {item} over to {target}'},
        {'name': 'give wheat', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'pass', 'deliver'],
         'transfer': 'sack of wheat', 'minutes': 5, 'text': 'hands {target} a sack of wheat'},
    ],
    'hazards': [
        {'name': 'hailstorm', 'description': 'a black summer hailstorm rolling in over the fields', 'when': 'day',
         'terrain': ['Field', 'Grassland'], 'chance_per_hour': 0.04, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 30, 'harm': 0.1, 'fail_text': 'the hail catches them in the open',
         'retreat_text': 'The storm veers away over the hills; the wheat stands.',
         'strike_text': 'Hail flattens the wheat and batters anyone caught in the open.',
         'responses': {'take shelter': {'verbs': ['take shelter', 'shelter', 'run for cover', 'get inside', 'take cover'], 'success': 0.7},
                       'cover the sheaves': {'verbs': ['cover', 'stack', 'bring in'], 'success': 0.4}}},
    ],
}
