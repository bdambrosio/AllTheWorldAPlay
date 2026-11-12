from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

# Import the new dynamic resource system
from sim.map import ResourceTypeRegistry

# All enums for apocalypse scenario
class ApocalypseTerrain(Enum):
    Water = 1       # Required by map.py
    Street = 2      # Roads and walkways
    Building = 3    # Commercial/residential buildings
    Plaza = 4       # Open spaces, squares
    Park = 5        # Green spaces
    Construction = 6 # Building sites

class ApocalypseInfrastructure(Enum):
    Road = auto()   # Main thoroughfares
    Path = auto()   # Smaller walkways

class ApocalypseResources(Enum):
    BusStop = auto()     # Public transport
    Shop = auto()        # Retail locations
    Cafe = auto()        # Food/drink venues
    Bench = auto()       # Rest spots
    TrashBin = auto()    # Waste disposal
    Park = auto()      # Required by map.py

class ApocalypseProperty(Enum):
    Building = auto()    # Commercial/residential
    Shop = auto()        # Retail space
    Cafe = auto()        # Food venue
    Plaza = auto()       # Public space

# Rules for terrain generation
terrain_rules = {
    'elevation_noise_scale': 20.0,  # Larger scale for urban environment
    'water_level': 0.1,    # Minimal water features
    'mountain_level': 0.9,  # No mountains
    'terrain_by_elevation': {
        'water': {'max': 0.1, 'type': 'Water'},  # Required water type
    },
    'lowland_distribution': {
        'Building': 0.3,    # Buildings
        'Street': 0.2,      # Roads
        'Plaza': 0.2,       # Open spaces
        'Park': 0.2,        # Green spaces
        'Construction': 0.1  # Building sites
    }
}

infrastructure_rules = {
    'road_density': 0.4,      # High density for urban setting
    'path_type': 'Road',      # Primary infrastructure type
    'slope_factor': 1.0,      # Minimal slope impact
    'terrain_costs': {
        'Water': float('inf'),
        'Building': float('inf'),
        'Street': 1.0,
        'Plaza': 1.2,
        'Park': 1.5,
        'Construction': 2.0
    }
}

property_rules = {
    'min_size': 10,    # Smaller property sizes for urban lots
    'max_size': 30,
    'valid_terrain': ['Building', 'Plaza']  # What can be owned
}

# Standard interface names
terrain_types = ApocalypseTerrain
infrastructure_types = ApocalypseInfrastructure
property_types = ApocalypseProperty
resource_types = ResourceTypeRegistry(ApocalypseResources)  # Use dynamic registry

# Add at top with other interface names
required_resource = resource_types.Park  # or Market, etc.
required_resource_name = "Park"  # or "Market", etc.

resource_rules = {
    'allocations': [
        {
            'resource_type': resource_types.BusStop,
            'description': 'A weathered bus stop shelter',
            'count': 2,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Street: 1.0
            }
        },
        {
            'resource_type': resource_types.Shop,
            'description': 'A small retail shop',
            'count': 2,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Building: 1.0
            }
        },
        {
            'resource_type': resource_types.Cafe,
            'description': 'A cozy cafe',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Building: 1.0
            }
        },
        {
            'resource_type': resource_types.Bench,
            'description': 'A weathered bench',
            'count': 3,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Park: 1.0,
                terrain_types.Plaza: 1.0
            }
        },
        {
            'resource_type': resource_types.TrashBin,
            'description': 'A rusty trash bin',
            'count': 2,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Street: 1.0
            }
        },
        {
            'resource_type': resource_types.Park,  # Required by map.py
            'description': 'A park',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Plaza: 2.0  # Higher weight to ensure placement
            }
        }
    ]
} 

# What is at stake in this world (planning and act prompts; sim/survival.py stakes_text).
stakes = ("In Bellona, food, clean water and a safe place to sleep must be scavenged or bargained for, and the city "
          "is unpredictable: gangs roam the streets at night and strange phenomena break out without warning. Use "
          "scavenging, barter, trust and betrayal over scarce goods as concrete, checkable goals and obstacles tied "
          "to places, alongside the characters' personal mysteries.")

# Stage business (sim/survival.py): scavenged goods, barter, shelter, gangs and phenomena.
affordances = {
    'needs': {
        'hunger': {'initial': 0.45, 'per_hour': 0.04, 'neglect_harm': 0.05,
                   'words': ('fed', 'hungry', 'very hungry', 'starving'),
                   'percepts': {0.5: "Your stomach knots; you have not eaten properly in a long time.",
                                0.8: "Hunger makes your hands shake and your thoughts slow."}},
        'thirst': {'initial': 0.35, 'per_hour': 0.06, 'neglect_harm': 0.05,
                   'words': ('not thirsty', 'thirsty', 'very thirsty', 'parched'),
                   'percepts': {0.5: "Your throat is dry and gritty with the city's dust.",
                                0.8: "Thirst pounds behind your eyes; you need water now."}},
        'fatigue': {'initial': 0.3, 'per_hour': 0.04, 'shelter_factor': 0.3,
                    'words': ('alert', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "Weariness drags at you; the streets seem longer.",
                                 0.8: "You are exhausted; you need somewhere safe to sleep."}},
    },
    'harm': {'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')},
    'travel_need': 'fatigue',
    'tile_minutes': 3,
    'type_nouns': {'Shop': ['shop', 'store'], 'TrashBin': ['trash', 'bin', 'garbage', 'dumpster'],
                   'Cafe': ['cafe', 'café'], 'BusStop': ['bus stop', 'shelter'], 'Bench': ['bench'],
                   'Park': ['park', 'fountain']},
    'actions': [
        {'name': 'scavenge', 'category': 'explore', 'verbs': ['scavenge', 'rummage', 'loot', 'dig through', 'pick through'],
         'named_verbs': ['search', 'check', 'explore'], 'resource_types': ['Shop', 'TrashBin', 'Cafe'],
         'finds': [{'item': 'canned food', 'chance': 0.3}, {'item': 'bottled water', 'chance': 0.3},
                   {'item': 'batteries', 'chance': 0.1}, {'item': 'knife', 'chance': 0.1}],
         'minutes': 30, 'deplete': 3, 'text': 'scavenges through {resource}'},
        {'name': 'eat canned food', 'verbs': ['eat', 'open the can', 'wolf down'], 'resource_types': None,
         'requires': 'canned food', 'consumes': 'canned food', 'effects': {'hunger': -0.5}, 'minutes': 10,
         'text': 'opens a can and eats'},
        {'name': 'drink bottled water', 'verbs': ['drink', 'sip', 'gulp'], 'resource_types': None,
         'requires': 'bottled water', 'consumes': 'bottled water', 'effects': {'thirst': -0.6}, 'minutes': 2,
         'text': 'drinks from a bottle of water'},
        {'name': 'drink from the fountain', 'verbs': ['drink', 'sip', 'scoop'], 'resource_types': ['Park'],
         'effects': {'thirst': -0.5}, 'minutes': 5, 'text': 'drinks from the fountain in {resource}',
         'risk': {'chance': 0.2, 'effects': {'injury': 0.2}, 'text': 'The water tastes of metal; cramps follow.'}},
        {'name': 'rest', 'verbs': ['rest', 'sit', 'catch breath'], 'resource_types': ['Bench', 'BusStop'],
         'effects': {'fatigue': -0.2}, 'minutes': 30, 'text': 'rests at {resource}'},
        {'name': 'take shelter', 'verbs': ['shelter', 'take cover', 'hole up', 'barricade', 'hide inside', 'go inside'],
         'resource_types': ['Shop', 'Cafe', 'BusStop'], 'effects': {'fatigue': -0.2}, 'minutes': 20, 'sets': 'sheltered',
         'text': 'takes shelter in {resource}'},
        {'name': 'sleep', 'verbs': ['sleep', 'doze', 'nap'], 'resource_types': ['Shop', 'Cafe'],
         'effects': {'fatigue': -1.0}, 'minutes': 300, 'sets': 'sheltered', 'text': 'sleeps hidden in {resource}'},
        {'name': 'give canned food', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'offer', 'share', 'trade', 'barter'],
         'transfer': 'canned food', 'minutes': 2, 'text': 'gives {target} a can of food'},
        {'name': 'give bottled water', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'offer', 'share', 'trade', 'barter'],
         'transfer': 'bottled water', 'minutes': 2, 'text': 'gives {target} a bottle of water'},
        # sharing what one has: both eat or drink, and it is used up (a can split onto cardboard
        # was taken as handing over the whole can, and the one who split it then had none to eat)
        {'name': 'share canned food', 'category': 'social', 'target': 'character', 'requires': 'canned food',
         'consumes': 'canned food', 'effects': {'hunger': -0.3}, 'target_effects': {'hunger': -0.3}, 'minutes': 10,
         'text': 'opens a can and shares it with {target}; they eat'},
        {'name': 'share bottled water', 'category': 'social', 'target': 'character', 'requires': 'bottled water',
         'consumes': 'bottled water', 'effects': {'thirst': -0.35}, 'target_effects': {'thirst': -0.35}, 'minutes': 3,
         'text': 'shares a bottle of water with {target}; they drink'},
        {'name': 'give batteries', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'offer', 'trade', 'barter'],
         'transfer': 'batteries', 'minutes': 2, 'text': 'gives {target} the batteries'},
        {'name': 'tend wound', 'category': 'social', 'target': 'character', 'verbs': ['tend', 'bandage', 'dress', 'treat', 'clean'],
         'target_effects': {'injury': -0.2}, 'minutes': 15, 'text': "tends {target}'s wounds"},
    ],
    'hazards': [
        {'name': 'gang', 'description': 'a knot of scorpion-chained gang members', 'when': 'night',
         'terrain': ['Street', 'Plaza', 'Construction'], 'chance_per_hour': 0.2, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 30, 'harm': 0.35, 'fail_text': 'the gang closes in, laughing',
         'retreat_text': 'The gang loses interest and drifts off into the smoke.',
         'strike_text': 'The gang surrounds them, takes a shove or a swing, and moves on.',
         'responses': {'bargain': {'verbs': ['bargain', 'negotiate', 'offer', 'talk', 'trade', 'bribe'], 'success': 0.5,
                                   'bonus': {'canned food': 0.2, 'bottled water': 0.2, 'batteries': 0.2}},
                       'defend': {'verbs': ['defend', 'fight', 'strike', 'threaten', 'brandish'], 'success': 0.3, 'bonus': {'knife': 0.3}},
                       'flee': {'verbs': ['flee', 'run', 'retreat', 'escape', 'back away'], 'success': 0.5},
                       'hide': {'verbs': ['hide', 'duck', 'freeze', 'stay still'], 'success': 0.4}}},
        {'name': 'phenomenon', 'description': 'a strange light that splits the hazy sky', 'when': 'any',
         'terrain': ['Street', 'Plaza', 'Park', 'Construction'], 'chance_per_hour': 0.04, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 20, 'harm': 0.15, 'fail_text': 'the light swells and the ground lurches',
         'retreat_text': 'The light fades as suddenly as it came; the city is quiet again.',
         'strike_text': 'The ground heaves and glass rains from the buildings.',
         'responses': {'take cover': {'verbs': ['take cover', 'shelter', 'duck', 'hide'], 'success': 0.6},
                       'flee': {'verbs': ['flee', 'run', 'escape'], 'success': 0.5}}},
    ],
}
