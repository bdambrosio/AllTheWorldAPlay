from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

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
    Market = auto()      # Required by map.py

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
resource_types = ApocalypseResources

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
            'resource_type': resource_types.Market,  # Required by map.py
            'description': 'A bustling marketplace',
            'count': 1,
            'requires_property': True,
            'terrain_weights': {
                terrain_types.Plaza: 2.0  # Higher weight to ensure placement
            }
        }
    ]
} 