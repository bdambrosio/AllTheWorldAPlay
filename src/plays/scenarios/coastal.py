from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

# Import the new dynamic resource system
from sim.map import ResourceTypeRegistry

# All enums for coastal scenario
class CoastalTerrain(Enum):
    Water = 1         # Required terrain type
    River = 2         # River terrain
    Workshop = 3      # Boat building workshop
    Downtown = 4      # Commercial area
    Garden = 5        # Public gardens
    Road = 6         # Main roads
    Harbor = 7       # Harbor area

class CoastalInfrastructure(Enum):
    Road = auto()     # Main roads
    Bridge = auto()   # River crossings
    Dock = auto()     # Water access

class CoastalResources(Enum):
    Bench = auto()
    Boat = auto()
    Tools = auto()
    Gallery = auto()  # Required by map.py
    Workshop = auto()
    Bridge = auto()
    Cafe = auto()
    Studio = auto()

class CoastalProperty(Enum):
    Workshop = auto()     # Boat building
    Gallery = auto()      # Art gallery
    Garden = auto()       # Public gardens
    CoffeeShop = auto()   # Meeting place

# Rules for terrain generation
terrain_rules = {
    'elevation_noise_scale': 15.0,  # Moderate scale for coastal environment
    'water_level': 0.2,    # More water features
    'mountain_level': 0.9,  # No mountains
    'terrain_by_elevation': {
        'water': {'max': 0.2, 'type': 'Water'},  # Required water type
        'river': {'min': 0.1, 'max': 0.2, 'type': 'River'}
    },
    'lowland_distribution': {
        'Workshop': 0.2,    # Boat building
        'Downtown': 0.2,    # Commercial area
        'Garden': 0.2,      # Public spaces
        'Road': 0.2,        # Main roads
        'Harbor': 0.2       # Harbor area
    }
}

infrastructure_rules = {
    'road_density': 0.3,      # Moderate density for coastal town
    'path_type': 'Road',      # Primary infrastructure type
    'slope_factor': 1.0,      # Minimal slope impact
    'terrain_costs': {
        'Water': float('inf'),
        'River': float('inf'),
        'Workshop': float('inf'),
        'Downtown': 1.5,
        'Garden': 1.2,
        'Road': 1.0,
        'Harbor': 1.3
    }
}

property_rules = {
    'min_size': 10,    # Smaller property sizes for coastal town
    'max_size': 30,
    'valid_terrain': ['Workshop', 'Downtown', 'Garden']  # What can be owned
}

# Standard interface names
terrain_types = CoastalTerrain
infrastructure_types = CoastalInfrastructure
property_types = CoastalProperty
resource_types = ResourceTypeRegistry(CoastalResources)  # Use dynamic registry

# Add at top with other interface names
required_resource = resource_types.Workshop  # or Market, etc.
required_resource_name = "Workshop"  # or "Market", etc.

resource_rules = {
    'allocations': [
        {
            'resource_type': resource_types.Bench,
            'description': 'A wooden bench overlooking the harbor',
            'count': 2,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Garden: 1.0,
                terrain_types.Harbor: 1.0
            }
        },
        {
            'resource_type': resource_types.Boat,
            'description': 'A wooden boat moored at the dock',
            'count': 1,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Harbor: 1.0
            }
        },
        {
            'resource_type': resource_types.Tools,
            'description': 'Boat building tools',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Workshop: 1.0
            }
        },
        {
            'resource_type': resource_types.Workshop,
            'description': 'A boat building workshop',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Harbor: 1.0
            }
        },
        {
            'resource_type': resource_types.Bridge,
            'description': 'A bridge over the river',
            'count': 1,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.River: 1.0
            }
        },
        {
            'resource_type': resource_types.Gallery,  # Required by map.py
            'description': 'The city art gallery',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Downtown: 2.0  # Higher weight to ensure placement
            }
        },
        {
            'resource_type': resource_types.Cafe,  # Required by map.py
            'description': 'The local cafe',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Harbor: 2.0  # Higher weight to ensure placement
            }
        },
        {
            'resource_type': resource_types.Studio,  # Required by map.py
            'description': 'The artist studio',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Harbor: 2.0  # Higher weight to ensure placement
            }
        }
    ]
} 

# What is at stake in this world (planning and act prompts; sim/survival.py stakes_text).
stakes = ("In this harbour town, livelihoods, art, money and relationships are at stake: rent and commissions, a "
          "gallery sale, a boat that needs mending, favours and promises between neighbours. Use concrete, checkable "
          "goals and obstacles tied to the studio, the workshop, the gallery, the cafe and the harbour.")

# Stage business (sim/survival.py): making and selling work, the cafe, the boat, a squall.
affordances = {
    'needs': {
        'hunger': {'initial': 0.3, 'per_hour': 0.04,
                   'words': ('not hungry', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.5: "You are getting hungry.", 0.8: "You are very hungry; it is hard to focus."}},
        'fatigue': {'initial': 0.3, 'per_hour': 0.03, 'shelter_factor': 0.5,
                    'words': ('fresh', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "You are tiring.", 0.8: "You are exhausted."}},
    },
    'harm': {'name': 'injury', 'words': ('fine', 'bruised', 'hurt', 'badly hurt')},
    'travel_need': 'fatigue',
    'currency': 'dollars', 'initial_money': 30,
    'tile_minutes': 2,
    'type_nouns': {'Cafe': ['cafe', 'café', 'coffee'], 'Studio': ['studio', 'easel', 'canvas'], 'Workshop': ['workshop', 'bench'],
                   'Gallery': ['gallery'], 'Boat': ['boat', 'hull', 'dinghy'], 'Tools': ['tools', 'toolbox'],
                   'Bench': ['bench'], 'Bridge': ['bridge']},
    'actions': [
        {'name': 'buy coffee', 'verbs': ['buy coffee', 'order coffee', 'drink coffee'], 'named_verbs': ['buy', 'order', 'drink'],
         'resource_types': ['Cafe'], 'cost': 4, 'effects': {'fatigue': -0.2}, 'minutes': 15, 'text': 'buys and drinks a coffee at {resource}'},
        {'name': 'eat a meal', 'verbs': ['eat', 'order lunch', 'order food'], 'resource_types': ['Cafe'], 'cost': 12,
         'effects': {'hunger': -0.6}, 'minutes': 30, 'text': 'eats a meal at {resource}'},
        {'name': 'sit and watch the harbour', 'verbs': ['sit', 'rest', 'watch the harbour', 'sketch'], 'resource_types': ['Bench'],
         'effects': {'fatigue': -0.15}, 'minutes': 20, 'text': 'sits at {resource}, watching the harbour'},
        {'name': 'paint', 'category': 'work', 'verbs': ['paint', 'finish the canvas', 'work on the painting'], 'named_verbs': ['work', 'create'],
         'resource_types': ['Studio'], 'effects': {'fatigue': 0.2}, 'gain': 'finished painting', 'minutes': 180,
         'text': 'paints at {resource} until a canvas is finished'},
        {'name': 'make a carving', 'category': 'work', 'verbs': ['carve', 'build', 'make'], 'named_verbs': ['work'],
         'resource_types': ['Workshop'], 'effects': {'fatigue': 0.2}, 'gain': 'carving', 'minutes': 180,
         'text': 'works at {resource} until a carving is finished'},
        {'name': 'sell a painting', 'category': 'work', 'verbs': ['sell', 'hang', 'consign'], 'resource_types': ['Gallery'],
         'requires': 'finished painting', 'consumes': 'finished painting', 'earn': 150, 'minutes': 30,
         'text': 'sells a painting through {resource}'},
        {'name': 'sell a carving', 'category': 'work', 'verbs': ['sell', 'consign'], 'resource_types': ['Gallery'],
         'requires': 'carving', 'consumes': 'carving', 'earn': 90, 'minutes': 30, 'text': 'sells a carving through {resource}'},
        {'name': 'borrow tools', 'verbs': ['borrow', 'take', 'pick up', 'grab'], 'named_verbs': [], 'resource_types': ['Tools'],
         'gain': 'tools', 'minutes': 5, 'deplete': 2, 'text': 'borrows tools from {resource}'},
        {'name': 'mend the boat', 'category': 'work', 'verbs': ['mend', 'repair', 'caulk', 'patch', 'fix'], 'resource_types': ['Boat'],
         'requires': 'tools', 'requires_property': {'mended': None}, 'sets_property': {'mended': True},
         'effects': {'fatigue': 0.25}, 'minutes': 120, 'text': 'mends {resource}'},
        {'name': 'go sailing', 'verbs': ['sail', 'go sailing', 'row out', 'take the boat out'], 'resource_types': ['Boat'],
         'requires_property': {'mended': True}, 'effects': {'fatigue': -0.1}, 'minutes': 90, 'text': 'takes {resource} out on the water'},
        {'name': 'shelter', 'verbs': ['shelter', 'take cover', 'go inside', 'duck inside'], 'resource_types': ['Cafe', 'Gallery', 'Studio', 'Workshop'],
         'minutes': 10, 'sets': 'sheltered', 'text': 'goes inside {resource}'},
        {'name': 'give a painting', 'category': 'social', 'target': 'character', 'verbs': ['give', 'gift', 'present', 'offer'],
         'transfer': 'finished painting', 'minutes': 5, 'text': 'gives {target} a painting'},
        {'name': 'lend tools', 'category': 'social', 'target': 'character', 'verbs': ['lend', 'give', 'hand', 'pass'],
         'transfer': 'tools', 'minutes': 2, 'text': 'lends {target} the tools'},
        {'name': 'pay', 'category': 'social', 'target': 'character', 'verbs': ['pay', 'lend money', 'hand cash', 'settle up'],
         'transfer_money': 20, 'minutes': 2, 'text': 'pays {target} twenty dollars'},
    ],
    'hazards': [
        {'name': 'squall', 'description': 'a sudden squall racing in off the sea', 'when': 'any',
         'terrain': ['Harbor', 'Road', 'Garden'], 'chance_per_hour': 0.03, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 20, 'harm': 0.1, 'fail_text': 'the squall soaks them and slams a shutter',
         'retreat_text': 'The squall blows itself out; the harbour glitters again.',
         'strike_text': 'The squall hits: rain sheets down and loose gear flies across the quay.',
         'responses': {'take cover': {'verbs': ['take cover', 'shelter', 'duck inside', 'run inside'], 'success': 0.7},
                       'secure the boat': {'verbs': ['secure', 'tie down', 'lash', 'make fast'], 'success': 0.5}}},
    ],
}
