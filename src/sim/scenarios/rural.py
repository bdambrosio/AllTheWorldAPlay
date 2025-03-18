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
    QUARRY = auto()

class RuralBuilding(Enum):
    FARMHOUSE = auto()
    BARN = auto()

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

resource_rules = {
    'allocations': []  # Fill in from RuralScenario
}


# Add standard interface names
terrain_types = RuralTerrain
infrastructure_types = RuralInfrastructure
property_types = RuralProperty
resource_types = RuralResources 