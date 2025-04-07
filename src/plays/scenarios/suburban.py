from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List

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
resource_types = SuburbanResources

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