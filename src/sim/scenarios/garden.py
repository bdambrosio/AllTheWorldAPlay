from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

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
    Market = auto()
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
resource_types = GardenResources

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
            'description': 'A patch of mushrooms',
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
            'description': 'A large fallen tree',
            'count': 24,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Grass: 1.0,
                terrain_types.Shrubs: 0.5
            }
        },
        {
            'resource_type': resource_types.Seed,
            'description': 'A large fallen tree',
            'count': 16,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Grass: 0.6,
                terrain_types.Shrubs: 1.0
            }
        },
         {
            'resource_type': resource_types.Ball,
            'description': 'A natural spring of fresh water',
            'count': 3,
            'requires_property': False,
            'terrain_weights': {
                terrain_types.Grass: 1.0,
                terrain_types.Shrubs: 0.5
            }
        },

    ]
} 