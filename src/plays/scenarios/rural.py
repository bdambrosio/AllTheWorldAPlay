from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

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
resource_types = RuralResources

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

