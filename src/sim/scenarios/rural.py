from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

# All enums for rural scenario
class RuralTerrain(Enum):
    WATER = auto()
    MOUNTAIN = auto()
    HILL = auto()
    FOREST = auto()
    GRASSLAND = auto()
    FIELD = auto()

class RuralInfrastructure(Enum):
    ROAD = auto()
    MARKET = auto()

class RuralResources(Enum):
    MARKET = auto()
    WELL = auto()
    BLACKSMITH = auto()
    MILL = auto()
    FARM = auto()
    FARMHOUSE = auto()
    BARN = auto()
    QUARRY = auto()


class RuralProperty(Enum):
    RESIDENTIAL = auto()
    AGRICULTURAL = auto()
    COMMERCIAL = auto()
    # ... other property types ...

# Extract rules from RuralScenario to module level
terrain_rules = {
    'elevation_noise_scale': 50.0,
    'water_level': 0.2,
    'mountain_level': 0.8
}

infrastructure_rules = {
    'road_density': 0.1
}

property_rules = {
    'min_size': 50,
    'max_size': 150
}

# Add standard interface names
terrain_types = RuralTerrain
infrastructure_types = RuralInfrastructure
property_types = RuralProperty
resource_types = RuralResources 

resource_rules = {
    'names': {
        'FARMHOUSE': ['Marquadt Farmhouse', 'Fouan Farmhouse']
    },
    'allocations': [
        {
            'resource_type': resource_types.WELL,
            'description': 'A well for water',
            'count': 5,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.GRASSLAND: 1.0,
                terrain_types.FIELD: 1.0
            }
        },
        {
            'resource_type': resource_types.BLACKSMITH,
            'description': 'A blacksmith shop',
            'count': 6,
            'requires_property': True,
            'has_npc': True,
            'terrain_weights': {
                terrain_types.GRASSLAND: 1.0,
                terrain_types.FIELD: 1.0
            }
        },
        {
            'resource_type': resource_types.MILL,
            'description': 'A mill for grinding grain',
            'count': 3,
            'has_npc': True,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.FIELD: 1.0,
                terrain_types.GRASSLAND: 0.5
            }
        },
        {
            'resource_type': resource_types.FARM,
            'description': 'A farm for growing crops',
            'count': 9,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.FIELD: 2.0,
                terrain_types.GRASSLAND: 1.0
            }
        },
         {
            'resource_type': resource_types.FARMHOUSE,
            'description': 'A farmhouse for living in',
            'count': 7,
            'requires_property': True,
            'has_npc': True,
            'terrain_weights': {
                terrain_types.FIELD: 2.0,
                terrain_types.GRASSLAND: 1.0
            }
        },
         {
            'resource_type': resource_types.BARN,
            'description': 'A barn for storing hay and housing livestock',
            'count': 8,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.FIELD: 2.0,
                terrain_types.GRASSLAND: 1.0
            }
        },
        {
            'resource_type': resource_types.QUARRY,
            'description': 'A quarry for stone',
            'count': 2,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.HILL: 2.0,
                terrain_types.MOUNTAIN: 1.0
            }
        }
    ]
}

