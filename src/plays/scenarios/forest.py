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
    Fallen_Branch = auto()  # wood: a branch to carry, a fire
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

# How the places of this world are, and how they sit: laid out in metres, anew for each
# performance (sim/land.py). Counts and descriptions are those of resource_rules below.
#   radius: metres. enclosed: walls (open: standing open, as a cave's mouth). shelter: inside it is
#   under shelter. blocks_view: not seen across. hides: what is inside is not seen from outside.
kinds = {
    'Berries': {'radius': 2}, 'Mushrooms': {'radius': 3}, 'Fallen_Branch': {'radius': 3}, 'Apple_Tree': {'radius': 5},
    'Spring': {'radius': 4},
    'Thicket': {'radius': 6, 'shelter': True, 'blocks_view': True, 'hides': True},
    'Cave': {'radius': 5, 'enclosed': True, 'open': True, 'shelter': True},
    'Hut': {'radius': 4, 'enclosed': True, 'shelter': True},
}
layout = {
    # the forest holds everything; pace is metres a minute on foot; cover is what is overhead
    'regions': [
        {'kind': 'Forest', 'whole': True, 'radius': 1200, 'pace': 50, 'cover': 'trees'},
        {'kind': 'Clearing', 'count': 6, 'radius': (40, 80), 'pace': 75, 'cover': 'open'},
        {'kind': 'Meadow', 'count': 2, 'radius': (100, 150), 'pace': 75, 'cover': 'open'},
        # a rocky rise some way off: high ground, and nothing is seen across it
        {'kind': 'Rise', 'count': 1, 'radius': 150, 'pace': 35, 'cover': 'open', 'from_start': (500, 900), 'blocks_view': True, 'overlooks': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 700, 'apart': 40},
    # 'on': the ground a kind is found on, by weight. from_start: metres from where they start.
    'places': [
        {'kind': 'Berries', 'on': {'Clearing': 2.0, 'Meadow': 1.0, 'Forest': 0.5}},
        {'kind': 'Mushrooms', 'on': {'Forest': 1.0}},
        {'kind': 'Fallen_Branch', 'on': {'Forest': 1.0, 'Clearing': 0.5}},
        {'kind': 'Apple_Tree', 'on': {'Clearing': 1.0, 'Forest': 0.6}},
        {'kind': 'Spring', 'on': {'Forest': 1.0}, 'from_start': (150, 700)},
        {'kind': 'Thicket', 'on': {'Forest': 2.0, 'Clearing': 1.0}},
        {'kind': 'Cave', 'on': {'Rise': 1.0}},
        {'kind': 'Hut', 'foot_of': 'Rise'},     # at the foot of the rise, on the side they start from
    ],
}

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


# What is at stake in this world (planning and act prompts; sim/survival.py stakes_text).
stakes = ("Physical needs (water, food, rest, shelter) and natural hazards (wolves at night, wild boar) are real "
          "stakes in this wilderness: use them to create concrete, checkable goals and obstacles tied to places and "
          "things, and keep any mystery grounded in them.")

# Stage business (sim/survival.py): needs rise with simulation time, resources have affordances,
# and natural hazards appear. Effects are mechanical; the LLM only narrates them.
affordances = {
    'needs': {   # where they start: thirsty from the night
        'thirst': {'initial': 0.45, 'per_hour': 0.08, 'neglect_harm': 0.05,
                   'words': ('not thirsty', 'thirsty', 'very thirsty', 'desperately thirsty'),
                   'percepts': {0.5: "Your mouth is dry; you have not had water for a while.",
                                0.8: "Thirst is becoming hard to ignore; your head aches and your lips are cracked."}},
        'hunger': {'initial': 0.3, 'per_hour': 0.04, 'neglect_harm': 0.05,
                   'words': ('not hungry', 'hungry', 'very hungry', 'starving'),
                   'percepts': {0.5: "Your stomach is empty and growling.",
                                0.8: "Hunger makes you light-headed and weak."}},
        'fatigue': {'initial': 0.25, 'per_hour': 0.05, 'shelter_factor': 0.3,
                    'words': ('rested', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "Your legs are heavy; you are getting tired.",
                                 0.8: "You are exhausted; every step is an effort and you need rest."}},
    },
    'harm': {'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')},
    'travel_need': 'fatigue',
    'tile_minutes': 2,
    'type_nouns': {'Spring': ['water', 'stream', 'pool', 'spring'], 'Berries': ['berry', 'berries'],
                   'Apple_Tree': ['apple', 'apples'], 'Mushrooms': ['mushroom', 'mushrooms', 'fungus', 'fungi'],
                   'Fallen_Branch': ['branch', 'branches', 'stick', 'wood', 'log'],
                   'Thicket': ['thicket', 'bush', 'undergrowth'], 'Cave': ['cave', 'cavern'], 'Hut': ['hut', 'cabin', 'shack']},
    'actions': [   # a deed's action is chosen by the model from these labels (survival.classify_deed); verbs serve spoken claims and threat responses
        {'name': 'sleep', 'category': 'survival', 'verbs': ['sleep', 'nap', 'doze'], 'resource_types': ['Cave', 'Hut', 'Thicket'],
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
        # (No 'build a fire' here. How a fire is made, fed, banked and put out is not written ahead: the
        # keeper models it when someone first sets about one, sim/keeper.py. The wolf is still 'blocked_by'
        # fire below: that is the name a model's effect is matched to.)
        {'name': 'rest', 'verbs': ['rest', 'sit down', 'catch breath', 'lie down', 'recover'], 'resource_types': None,
         'effects': {'fatigue': -0.15}, 'minutes': 20, 'text': 'rests for a while'},
        {'name': 'gather berries', 'category': 'survival', 'verbs': ['stash', 'pocket'], 'named_verbs': ['gather', 'collect', 'save', 'pluck', 'pick', 'harvest', 'strip'],
         'resource_types': ['Berries'], 'gain': 'berries', 'minutes': 10, 'deplete': 3,
         'text': 'gathers a handful of berries from {resource} to carry'},
        {'name': 'eat carried berries', 'verbs': ['eat', 'chew', 'munch'], 'named_verbs': [], 'resource_types': None,
         'requires': 'berries', 'consumes': 'berries', 'effects': {'hunger': -0.3}, 'minutes': 5,
         'text': 'eats the berries they were carrying'},
        {'name': 'search', 'category': 'explore', 'verbs': ['search', 'rummage', 'investigate'], 'named_verbs': ['explore', 'examine', 'inspect'],
         'resource_types': ['Hut', 'Cave'], 'finds': [{'item': 'knife', 'chance': 0.25}, {'item': 'rope', 'chance': 0.25}],
         'minutes': 20, 'deplete': 2, 'text': 'searches {resource} carefully'},
        {'name': 'give berries', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'offer', 'share'],
         'transfer': 'berries', 'minutes': 1, 'text': 'gives {target} the berries'},
        # an apple can be picked, carried, eaten later and handed over, like berries (a play
        # planned 'she picks an apple and holds it out to him': the tree could only be eaten at)
        {'name': 'pick an apple', 'category': 'survival', 'verbs': ['stash', 'pocket'], 'named_verbs': ['pick', 'pluck', 'gather', 'collect', 'take', 'harvest'],
         'resource_types': ['Apple_Tree'], 'gain': 'apple', 'minutes': 5, 'deplete': 1,
         'text': 'picks an apple from {resource} to carry'},
        {'name': 'eat carried apple', 'verbs': ['eat', 'chew', 'munch', 'bite'], 'named_verbs': [], 'resource_types': None,
         'requires': 'apple', 'consumes': 'apple', 'effects': {'hunger': -0.3}, 'minutes': 5,
         'text': 'eats the apple they were carrying'},
        {'name': 'give apple', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'offer', 'share'],
         'transfer': 'apple', 'minutes': 1, 'text': 'gives {target} the apple'},
        {'name': 'give branch', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'offer', 'pass', 'shove'],
         'transfer': 'branch', 'minutes': 1, 'text': 'hands {target} the branch'},
        {'name': 'tend wound', 'category': 'social', 'target': 'character', 'verbs': ['tend', 'bandage', 'dress', 'treat', 'clean'],
         'target_effects': {'injury': -0.2}, 'minutes': 15, 'text': "tends {target}'s wounds"},
    ],
    'hazards': [
        {'name': 'wolf', 'description': 'a gaunt grey wolf', 'when': 'night', 'terrain': ['Forest', 'Clearing', 'Meadow'],
         'chance_per_hour': 0.35, 'blocked_by': ['sheltered', 'fire'], 'minutes_to_strike': 40, 'harm': 0.4,
         'retreat_text': 'The wolf slinks back into the dark trees.',
         'strike_text': 'The wolf darts in and bites before vanishing into the dark.', 'fail_text': 'the wolf is not deterred and lunges',
         'responses': {'defend': {'verbs': ['defend', 'fight', 'club', 'strike', 'hit', 'swing', 'throw', 'shout', 'yell', 'wave', 'scare'],
                                  'success': 0.45, 'bonus': {'branch': 0.3, 'fire': 0.4}},
                       'flee': {'verbs': ['flee', 'run', 'retreat', 'escape', 'back away'], 'success': 0.5},
                       'hide': {'verbs': ['hide', 'climb', 'freeze', 'stay still'], 'success': 0.4}}},
        {'name': 'boar', 'description': 'a bristling wild boar', 'when': 'day', 'terrain': ['Forest'],
         'chance_per_hour': 0.06, 'blocked_by': ['sheltered'], 'minutes_to_strike': 20, 'harm': 0.3,
         'retreat_text': 'The boar snorts and crashes away through the undergrowth.',
         'strike_text': 'The boar charges, its tusks catching a leg before it crashes away.', 'fail_text': 'the boar is not deterred and charges',
         'responses': {'defend': {'verbs': ['defend', 'fight', 'club', 'strike', 'hit', 'throw', 'shout', 'yell', 'scare'],
                                  'success': 0.4, 'bonus': {'branch': 0.3}},
                       'flee': {'verbs': ['flee', 'run', 'retreat', 'escape', 'back away'], 'success': 0.6},
                       'hide': {'verbs': ['hide', 'climb', 'freeze', 'stay still'], 'success': 0.6}}},
    ],
}
