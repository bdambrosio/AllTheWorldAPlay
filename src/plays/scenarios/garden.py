from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

# Import the new dynamic resource system
from sim.map import ResourceTypeRegistry

# All enums for forest scenario
class GardenTerrain(Enum):
    Water = 1
    Grass = 2
    Shrubs = 3
    Trees = 4
    Flowers = 5
    Rocks = 6

class GardenInfrastructure(Enum):
    Path = auto()    
    Trail = auto()    

class GardenResources(Enum):
    Tree = auto()
    Bug = auto()
    Butterfly = auto()
    Leaf = auto()
    Seed = auto()
    Ball = auto()  # Potential shelter


class GardenProperty(Enum):
    pass  # Keeping property system but no ownership

# Rules for terrain generation
terrain_rules = {
    'elevation_noise_scale': 50.0,
    'water_level': 0.2,
    'mountain_level': 0.8,
    'terrain_by_elevation': {
        'water': {'max': 0.2, 'type': 'Water'},
        'trees': {'min': 0.8, 'type': 'Trees'}
    },
    'lowland_distribution': {
        'Grass': 0.4,        
        'Shrubs': 0.3,      
        'Flowers': 0.2,
        'Rocks': 0.1
    }
}

infrastructure_rules = {
    'road_density': 0.05,     
    'path_type': 'Trail',    
    'slope_factor': 2.5,      
    'terrain_costs': {
        'Water': float('inf'),
        'Grass': 1.0,
        'Shrubs': 1.0,
        'Trees': 1.0,
        'Flowers': 1.0,
        'Rocks': 2.0
    }
}

property_rules = {
    'min_size': 0,    # No properties in wilderness
    'max_size': 0,
    'valid_terrain': ['Shrubs', 'Trees', 'Flowers', 'Rocks']  # Add this even if no properties are used
}

# Standard interface names
terrain_types = GardenTerrain
infrastructure_types = GardenInfrastructure
property_types = GardenProperty
resource_types = ResourceTypeRegistry(GardenResources)  # Use dynamic registry

# Add at top with other interface names
required_resource = resource_types.Tree  # or Market, etc.
required_resource_name = "Tree"  # or "Market", etc.

resource_rules = {
    'allocations': [
        {
            'resource_type': resource_types.Bug,
            'description': 'yummy bug',
            'count': 15,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Grass: 2.0,
                terrain_types.Shrubs: 1.0,
                terrain_types.Flowers: 0.5
            }
        },
        {
            'resource_type': resource_types.Butterfly,
            'description': 'A bright butterfly flitting among the flowers',
            'count': 10,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Grass: 2.0,
                terrain_types.Shrubs: 1.0,
                terrain_types.Flowers: 0.5
            }
        },
        {
            'resource_type': resource_types.Leaf,
            'description': 'A deep pile of crunchy fallen leaves',
            'count': 24,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Grass: 1.0,
                terrain_types.Shrubs: 0.5
            }
        },
        {
            'resource_type': resource_types.Seed,
            'description': 'A scatter of seeds under the bird feeder',
            'count': 16,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Grass: 0.6,
                terrain_types.Shrubs: 1.0
            }
        },
         {
            'resource_type': resource_types.Ball,
            'description': 'A bright red ball',
            'count': 3,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Grass: 1.0,
                terrain_types.Shrubs: 0.5
            }
        },

    ]
} 

# What is at stake in this world (planning and act prompts; sim/survival.py stakes_text).
stakes = ("For the kittens, the garden is a world of play, food, naps and small dangers: a bug to catch, a "
          "butterfly to chase, a ball to win, a warm nap, the neighbour's dog. Use these concrete, playful goals and "
          "obstacles tied to things in the garden; keep the tone gentle and the magic light.")

# Stage business (sim/survival.py): hunting bugs, chasing, play, naps, the neighbour's dog.
affordances = {
    'needs': {
        'hunger': {'initial': 0.35, 'per_hour': 0.08,
                   'words': ('full', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.5: "Your tummy rumbles; a bug would be nice.", 0.8: "You are SO hungry."}},
        'sleepiness': {'initial': 0.2, 'per_hour': 0.1, 'shelter_factor': 0.5,
                       'words': ('bouncy', 'drowsy', 'sleepy', 'can barely keep eyes open'),
                       'percepts': {0.5: "Your eyes are getting heavy; a sunny spot looks very nice.",
                                    0.8: "You are so sleepy you keep yawning."}},
    },
    'harm': {'name': 'scrapes', 'words': ('unruffled', 'ruffled', 'scraped', 'hurting')},
    'travel_need': 'sleepiness',
    'tile_minutes': 1,
    'type_nouns': {'Bug': ['bug', 'beetle', 'cricket'], 'Butterfly': ['butterfly'], 'Leaf': ['leaf', 'leaves'],
                   'Seed': ['seed', 'seeds'], 'Ball': ['ball']},
    'actions': [
        {'name': 'catch a bug', 'verbs': ['pounce', 'catch', 'eat', 'snap up', 'gobble'], 'resource_types': ['Bug'],
         'effects': {'hunger': -0.35, 'sleepiness': 0.05}, 'minutes': 5, 'deplete': 3, 'text': 'pounces and gobbles a bug at {resource}'},
        {'name': 'chase the butterfly', 'category': 'explore', 'verbs': ['chase', 'leap at', 'swat', 'stalk'],
         'resource_types': ['Butterfly'], 'effects': {'sleepiness': 0.1}, 'minutes': 10,
         'finds': [{'item': 'a shimmering wing-scale', 'chance': 0.2}], 'text': 'chases the butterfly round and round {resource}',
         'risk': {'chance': 0.15, 'effects': {'scrapes': 0.1}, 'text': 'A tumble into the rose bush!'}},
        {'name': 'pick up the ball', 'verbs': ['grab', 'carry', 'pick up', 'take'], 'named_verbs': ['bat', 'nose'],
         'resource_types': ['Ball'], 'gain': 'ball', 'minutes': 2, 'deplete': 1, 'text': 'grabs the ball from {resource}'},
        {'name': 'play with the ball', 'category': 'explore', 'verbs': ['bat the ball', 'dribble', 'play with the ball'],
         'named_verbs': ['play', 'bat', 'pounce'], 'resource_types': None, 'requires': 'ball', 'effects': {'sleepiness': 0.1},
         'minutes': 10, 'text': 'bats the ball about'},
        {'name': 'burrow into the leaves', 'verbs': ['burrow', 'dive into', 'hide in', 'snuggle into'], 'resource_types': ['Leaf'],
         'minutes': 5, 'sets': 'sheltered', 'text': 'burrows deep into the leaves at {resource}'},
        {'name': 'nap', 'verbs': ['nap', 'sleep', 'doze', 'curl up', 'snooze'], 'resource_types': None,
         'effects': {'sleepiness': -0.8}, 'minutes': 60, 'text': 'curls up for a nap'},
        {'name': 'nibble seeds', 'category': 'explore', 'verbs': ['nibble', 'sniff', 'paw at'], 'resource_types': ['Seed'],
         'effects': {'hunger': -0.1}, 'minutes': 5, 'text': 'sniffs and nibbles at the seeds by {resource}'},
        {'name': 'give the ball', 'category': 'social', 'target': 'character', 'verbs': ['give', 'share', 'drop', 'nudge', 'pass'],
         'transfer': 'ball', 'minutes': 1, 'text': 'nudges the ball over to {target}'},
        {'name': 'groom', 'category': 'social', 'target': 'character', 'verbs': ['groom', 'lick', 'nuzzle', 'snuggle'],
         'target_effects': {'scrapes': -0.15}, 'minutes': 10, 'text': 'grooms {target} gently'},
    ],
    'hazards': [
        {'name': 'dog', 'description': "the neighbour's big bouncy dog bounding over the fence", 'when': 'day',
         'terrain': ['Grass', 'Flowers'], 'chance_per_hour': 0.06, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 15, 'harm': 0.15, 'fail_text': 'the dog bounds closer, barking happily',
         'retreat_text': 'The dog is called home and lollops away.',
         'strike_text': 'The dog bowls everyone over in a slobbery tumble before it is called home.',
         'responses': {'hiss': {'verbs': ['hiss', 'puff up', 'arch', 'swat', 'growl'], 'success': 0.5},
                       'run away': {'verbs': ['run', 'flee', 'scamper', 'dash'], 'success': 0.6},
                       'hide': {'verbs': ['hide', 'burrow', 'freeze', 'crouch'], 'success': 0.5}}},
    ],
}
