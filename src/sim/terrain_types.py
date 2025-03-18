from enum import Enum, auto

class RuralTerrain(Enum):
    WATER = auto()
    FIELD = auto()
    GRASSLAND = auto()
    FOREST = auto()
    HILL = auto()
    MOUNTAIN = auto()

class RuralInfrastructure(Enum):
    ROAD = auto()
    MARKET = auto()

class RuralResources(Enum):
    WELL = auto()
    BLACKSMITH = auto()
    MILL = auto()
    FARM = auto()
    QUARRY = auto() 