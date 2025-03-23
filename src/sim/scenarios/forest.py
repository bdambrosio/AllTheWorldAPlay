from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

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
    Market = auto()
    Berries = auto()
    Mushrooms = auto()
    Apple_Tree = auto()
    Fallen_Log = auto()  # Potential shelter
    Spring = auto()     # Water source
    Cave = auto()       # Potential shelter
    Thicket = auto()    # Dense vegetation, potential shelter

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
resource_types = ForestResources

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
            'resource_type': resource_types.Fallen_Log,
            'description': 'A large fallen tree',
            'count': 24,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Forest: 1.0,
                terrain_types.Clearing: 0.5
            }
        },
        {
            'resource_type': resource_types.Apple_Tree,
            'description': 'A large fallen tree',
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
            'count': 2,
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