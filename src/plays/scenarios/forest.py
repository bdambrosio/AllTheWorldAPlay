from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

# Import the new dynamic resource system
from sim.map import ResourceTypeRegistry

# All enums for forest scenario
class ForestTerrain(Enum):
    Water = 1
    Mountain = 2
    Forest = 4
    Clearing = 5
    Meadow = 6

class ForestInfrastructure(Enum):
    Trail = auto()    

class ForestResources(Enum):
    Berries = auto()
    Mushrooms = auto()
    Apple_Tree = auto()
    Fallen_Branch = auto()  # Potential shelter
    Spring = auto()     # Water source
    Cave = auto()       # Potential shelter
    Thicket = auto()    # Dense vegetation, potential shelter
    Hut = auto()         # Added for the new required_resource

class ForestProperty(Enum):
    pass  # Keeping property system but wilderness has no ownership

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
        'Forest': 0.7,        # Much more forest
        'Clearing': 0.2,      # Some clearings
        'Meadow': 0.1         # Few meadows
    }
}

infrastructure_rules = {
    'road_density': 0.05,     # Fewer roads in forest
    'path_type': 'Trail',     # Specify the path type
    'slope_factor': 2.5,      # Slightly higher penalty for slopes in forests
    'terrain_costs': {
        'Water': float('inf'),
        'Mountain': float('inf'),
        'Forest': 2.5,        # Forests are harder to traverse
        'Clearing': 1.2,      # Clearings are easy but not as easy as fields
        'Meadow': 1.0         # Meadows are easy to traverse
    }
}

property_rules = {
    'min_size': 0,    # No properties in wilderness
    'max_size': 0,
    'valid_terrain': ['Clearing', 'Meadow']  # Add this even if no properties are used
}

# Standard interface names
terrain_types = ForestTerrain
infrastructure_types = ForestInfrastructure
property_types = ForestProperty
resource_types = ResourceTypeRegistry(ForestResources)  # Use dynamic registry

# Add at top with other interface names
required_resource = resource_types.Hut  # or Market, etc.
required_resource_name = "Hut"  # or "Market", etc.

resource_rules = {
    'allocations': [
        {
            'resource_type': resource_types.Berries,
            'description': 'A bush with edible berries',
            'count': 15,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Clearing: 2.0,
                terrain_types.Meadow: 1.0,
                terrain_types.Forest: 0.5
            }
        },
        {
            'resource_type': resource_types.Mushrooms,
            'description': 'A patch of mushrooms',
            'count': 10,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Forest: 2.0
            }
        },
        {
            'resource_type': resource_types.Fallen_Branch,
            'description': 'A large fallen tree branch',
            'count': 24,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Forest: 1.0,
                terrain_types.Clearing: 0.5
            }
        },
        {
            'resource_type': resource_types.Apple_Tree,
            'description': 'A large apple tree with apples',
            'count': 16,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Forest: 0.6,
                terrain_types.Clearing: 1.0
            }
        },
         {
            'resource_type': resource_types.Spring,
            'description': 'A natural spring of fresh water',
            'count': 3,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Forest: 1.0
            }
        },
        {
            'resource_type': resource_types.Cave,
            'description': 'A small cave in the rocks',
            'has_npc': True,
            'count': 1,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Mountain: 2.0
            }
        },
        {
            'resource_type': resource_types.Hut,
            'description': 'A small hut for shelter',
            'has_npc': True,
            'count': 1,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Mountain: 2.0
            }
        },
        {
            'resource_type': resource_types.Thicket,
            'description': 'A dense thicket of vegetation',
            'count': 12,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Forest: 2.0,
                terrain_types.Clearing: 1.0
            }
        }
    ]
} 


# Survival stage business (sim/survival.py): needs rise with simulation time, resources have
# affordances, and natural hazards appear. Effects are mechanical; the LLM only narrates them.
survival = {
    'initial': {'hunger': 0.3, 'thirst': 0.45, 'fatigue': 0.25},   # woke at midday, disoriented
    'rates_per_hour': {'hunger': 0.04, 'thirst': 0.08, 'fatigue': 0.05},
    'tile_minutes': 2,
    'actions': [   # order matters: the first action whose verbs match the Do text applies
        {'name': 'sleep', 'verbs': ['sleep', 'nap', 'doze'], 'resource_types': ['Cave', 'Hut', 'Thicket'],
         'effects': {'fatigue': -1.0}, 'minutes': 240, 'sets': 'sheltered', 'text': 'sleeps in {resource}'},
        {'name': 'take shelter', 'verbs': ['shelter', 'take cover', 'go inside', 'enter', 'hide inside', 'rest inside', 'crawl into', 'crawl under',
                   'curl into', 'curl up', 'huddle', 'settle into', 'duck into', 'burrow'],
         'resource_types': ['Cave', 'Hut', 'Thicket'], 'effects': {'fatigue': -0.3}, 'minutes': 30, 'sets': 'sheltered',
         'text': 'takes shelter in {resource}'},
        {'name': 'drink', 'verbs': ['drink', 'sip', 'gulp', 'slurp'], 'named_verbs': ['cup', 'scoop', 'dip', 'lap', 'fill'],
         'resource_types': ['Spring'],
         'effects': {'thirst': -0.8}, 'minutes': 5, 'text': 'drinks cold water from {resource}'},
        {'name': 'eat mushrooms', 'verbs': ['eat', 'taste', 'forage', 'pick', 'pluck', 'chew', 'bite'], 'resource_types': ['Mushrooms'],
         'effects': {'hunger': -0.3}, 'minutes': 10, 'deplete': 2, 'text': 'eats mushrooms from {resource}',
         'risk': {'chance': 0.3, 'effects': {'injury': 0.3}, 'text': 'Soon after, cramps and nausea hit hard.'}},
        {'name': 'eat', 'verbs': ['eat', 'pick', 'pluck', 'forage', 'taste', 'gather food', 'harvest', 'chew', 'munch', 'bite', 'devour'],
         'resource_types': ['Berries', 'Apple_Tree'],
         'effects': {'hunger': -0.4}, 'minutes': 10, 'deplete': 3, 'text': 'eats from {resource}'},
        {'name': 'take a branch', 'verbs': [], 'named_verbs': ['take', 'pick up', 'grab', 'gather', 'collect', 'break off', 'lift', 'snap', 'break', 'wrench', 'seize'],
         'resource_types': ['Fallen_Branch'], 'gain': 'branch', 'minutes': 3, 'deplete': 4,
         'text': 'takes a sturdy branch from {resource}'},
        {'name': 'build a fire', 'verbs': ['fire', 'kindle', 'campfire', 'light a fire'], 'resource_types': None,
         'requires': 'branch', 'consumes': 'branch', 'minutes': 20, 'sets': 'fire', 'text': 'builds a small fire'},
        {'name': 'rest', 'verbs': ['rest', 'sit down', 'catch breath', 'lie down', 'recover'], 'resource_types': None,
         'effects': {'fatigue': -0.15}, 'minutes': 20, 'text': 'rests for a while'},
    ],
    'hazards': [
        {'name': 'wolf', 'description': 'a gaunt grey wolf', 'when': 'night', 'terrain': ['Forest', 'Clearing', 'Meadow'],
         'chance_per_hour': 0.35, 'blocked_by': ['sheltered', 'fire'], 'minutes_to_strike': 40, 'harm': 0.4,
         'retreat_text': 'The wolf slinks back into the dark trees.',
         'strike_text': 'The wolf darts in and bites before vanishing into the dark.',
         'responses': {'defend': {'verbs': ['defend', 'fight', 'club', 'strike', 'hit', 'swing', 'throw', 'shout', 'yell', 'wave', 'scare'],
                                  'success': 0.45, 'bonus': {'branch': 0.3, 'fire': 0.4}},
                       'flee': {'verbs': ['flee', 'run', 'retreat', 'escape', 'back away'], 'success': 0.5},
                       'hide': {'verbs': ['hide', 'climb', 'freeze', 'stay still'], 'success': 0.4}}},
        {'name': 'boar', 'description': 'a bristling wild boar', 'when': 'day', 'terrain': ['Forest'],
         'chance_per_hour': 0.06, 'blocked_by': ['sheltered'], 'minutes_to_strike': 20, 'harm': 0.3,
         'retreat_text': 'The boar snorts and crashes away through the undergrowth.',
         'strike_text': 'The boar charges, its tusks catching a leg before it crashes away.',
         'responses': {'defend': {'verbs': ['defend', 'fight', 'club', 'strike', 'hit', 'throw', 'shout', 'yell', 'scare'],
                                  'success': 0.4, 'bonus': {'branch': 0.3}},
                       'flee': {'verbs': ['flee', 'run', 'retreat', 'escape', 'back away'], 'success': 0.6},
                       'hide': {'verbs': ['hide', 'climb', 'freeze', 'stay still'], 'success': 0.6}}},
    ],
}
