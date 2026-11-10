from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

# Import the new dynamic resource system
from sim.map import ResourceTypeRegistry

# All enums for suburban scenario
class SuburbanTerrain(Enum):
    Water = 1         # Required terrain type
    House = 2         # Indoor living space
    Yard = 3          # Outdoor private space
    Street = 4        # Vehicle paths
    Sidewalk = 5      # Pedestrian paths
    Park = 6          # Public outdoor space
    City = 7        # Indoor work space

class SuburbanInfrastructure(Enum):
    Sidewalk = auto()  # Pedestrian infrastructure

class SuburbanResources(Enum):
    Refrigerator = auto()
    Shower = auto()
    Closet = auto()
    Car = auto()
    Bus_Stop = auto()
    Mailbox = auto()
    Coffee_Maker = auto()
    Bed = auto()
    Office = auto()     # Required by map.py

class SuburbanProperty(Enum):
    House = auto()     # Private residential
    City = auto()    # Commercial
    Park = auto()    # Parks, streets, etc.

# Rules for terrain generation
terrain_rules = {
    'elevation_noise_scale': 10.0,  # Smaller scale for suburban environment
    'water_level': 0.1,    # Minimal water features
    'mountain_level': 0.9,  # No mountains in suburbs
    'terrain_by_elevation': {
        'water': {'max': 0.1, 'type': 'Water'},  # Required water type
    },
    'lowland_distribution': {
        'House': 0.3,     # Buildings
        'Yard': 0.2,      # Private outdoor space
        'Street': 0.2,    # Roads
        'Sidewalk': 0.1,  # Pedestrian paths
        'Park': 0.1,      # Public spaces
        'City': 0.1     # Commercial buildings
    }
}

infrastructure_rules = {
    'road_density': 0.2,      # Higher density of paths in suburban area
    'path_type': 'Sidewalk',  # Primary infrastructure type
    'slope_factor': 1.0,      # Minimal slope impact in suburban setting
    'terrain_costs': {
        'Water': float('inf'),
        'House': float('inf'),  # Can't walk through houses
        'City': float('inf'), # Can't walk through offices
        'Yard': 2.0,           # Crossing yards is possible but discouraged
        'Street': 1.5,         # Streets are crossable but not preferred
        'Sidewalk': 1.0,       # Preferred walking path
        'Park': 1.2           # Parks are walkable but may be indirect
    }
}

property_rules = {
    'min_size': 10,    # Smaller property sizes for suburban lots
    'max_size': 30,
    'valid_terrain': ['House', 'Yard', 'City']  # What can be owned
}

# Standard interface names
terrain_types = SuburbanTerrain
infrastructure_types = SuburbanInfrastructure
property_types = SuburbanProperty
resource_types = ResourceTypeRegistry(SuburbanResources)  # Use dynamic registry

# Add at top with other interface names
required_resource = resource_types.Office  # or Market, etc.
required_resource_name = "Office"  # or "Market", etc.

resource_rules = {
    'allocations': [
        {
            'resource_type': resource_types.Bed,
            'description': 'A comfortable bed',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.House: 1.0
            }
        },
        {
            'resource_type': resource_types.Shower,
            'description': 'A bathroom with shower',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.House: 1.0
            }
        },
        {
            'resource_type': resource_types.Closet,
            'description': 'A closet with clothes',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.House: 1.0
            }
        },
        {
            'resource_type': resource_types.Coffee_Maker,
            'description': 'An automatic coffee maker',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.House: 1.0
            }
        },
        {
            'resource_type': resource_types.Car,
            'description': 'A parked car',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Yard: 1.0
            }
        },
        {
            'resource_type': resource_types.Mailbox,
            'description': 'A mailbox',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Yard: 1.0
            }
        },
        {
            'resource_type': resource_types.Bus_Stop,
            'description': 'A bus stop',
            'count': 2,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Sidewalk: 1.0
            }
        },
        {
            'resource_type': resource_types.Office,  # This will be our office building
            'description': 'A downtown office building',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.City: 2.0  # Higher weight to ensure placement
            }
        }
    ]
} 

# What is at stake in this world (planning and act prompts; sim/survival.py stakes_text).
stakes = ("In this everyday suburban world the stakes are time, money, work and relationships: being late, a "
          "missed commitment, a bill, a tense conversation, a decision that cannot wait. Bodily needs are ordinary "
          "(coffee, food, sleep, getting ready), not survival. Use concrete, checkable goals and obstacles tied to the "
          "house, the commute and the office.")

# Stage business (sim/survival.py): morning routine, mail, work and money. No hazards.
affordances = {
    'needs': {   # early morning
        'hunger': {'initial': 0.4, 'per_hour': 0.05,
                   'words': ('not hungry', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.5: "You are hungry; you have not eaten yet today.",
                                0.8: "Hunger is making it hard to concentrate."}},
        'fatigue': {'initial': 0.45, 'per_hour': 0.04,
                    'words': ('alert', 'groggy', 'tired', 'exhausted'),
                    'percepts': {0.5: "You feel groggy; coffee would help.",
                                 0.8: "You are exhausted and your thoughts keep drifting."}},
    },
    'harm': {'name': 'injury', 'words': ('fine', 'sore', 'hurt', 'badly hurt')},
    'currency': 'dollars', 'initial_money': 40,
    'tile_minutes': 1,
    'type_nouns': {'Coffee_Maker': ['coffee', 'coffee maker'], 'Refrigerator': ['fridge', 'refrigerator', 'breakfast'],
                   'Shower': ['shower', 'bathroom'], 'Closet': ['closet', 'clothes', 'suit'], 'Bed': ['bed'],
                   'Mailbox': ['mailbox', 'mail', 'letters'], 'Office': ['office', 'desk', 'work'],
                   'Car': ['car'], 'Bus_Stop': ['bus stop', 'bus']},
    'actions': [
        {'name': 'make coffee', 'verbs': ['brew', 'pour coffee', 'make coffee', 'drink coffee'], 'named_verbs': ['pour', 'drink', 'sip'],
         'resource_types': ['Coffee_Maker'], 'effects': {'fatigue': -0.2}, 'minutes': 5, 'text': 'pours and drinks coffee from {resource}'},
        {'name': 'eat breakfast', 'verbs': ['eat', 'make breakfast', 'grab a bite'], 'resource_types': ['Refrigerator'],
         'effects': {'hunger': -0.5}, 'minutes': 15, 'text': 'eats breakfast from {resource}'},
        {'name': 'shower', 'verbs': ['shower', 'wash'], 'resource_types': ['Shower'], 'effects': {'fatigue': -0.1},
         'minutes': 15, 'sets': 'showered', 'text': 'showers'},
        {'name': 'get dressed', 'verbs': ['dress', 'get dressed', 'put on', 'change into'], 'resource_types': ['Closet'],
         'minutes': 10, 'sets': 'dressed', 'text': 'gets dressed for the day'},
        {'name': 'sleep', 'verbs': ['sleep', 'nap', 'go back to bed', 'lie down'], 'resource_types': ['Bed'],
         'effects': {'fatigue': -0.8}, 'minutes': 240, 'text': 'sleeps'},
        {'name': 'check the mail', 'category': 'explore', 'verbs': ['check the mail', 'open the mailbox', 'get the mail'],
         'named_verbs': ['check', 'open'], 'resource_types': ['Mailbox'],
         'finds': [{'item': 'an overdue bill', 'chance': 0.35}, {'item': 'a personal letter', 'chance': 0.25}],
         'minutes': 3, 'deplete': 1, 'text': 'checks the mail'},
        {'name': 'work', 'category': 'work', 'verbs': ['answer email', 'take the meeting', 'finish the report'], 'named_verbs': ['work'],
         'resource_types': ['Office'], 'requires_flag': 'dressed', 'effects': {'fatigue': 0.25}, 'earn': 120,
         'minutes': 240, 'text': 'puts in a block of work at {resource}'},
        {'name': 'lend money', 'category': 'social', 'target': 'character', 'verbs': ['lend', 'pay', 'give money', 'hand cash'],
         'transfer_money': 20, 'minutes': 2, 'text': 'gives {target} twenty dollars'},
    ],
    'hazards': [],
}
