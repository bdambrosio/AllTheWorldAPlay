from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

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
resource_types = CoastalResources

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
                terrain_types.Workshop: 1.0
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
        }
    ]
} 