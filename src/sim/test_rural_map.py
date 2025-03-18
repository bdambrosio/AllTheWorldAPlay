from enum import Enum, auto

class RuralResources(Enum):
    WELL = auto()
    BLACKSMITH = auto()
    MILL = auto()
    FARM = auto()
    QUARRY = auto()


def test_rural_terrain_generation():
    # Define resource rules specific to rural scenario
    resource_rules = {
        'allocations': [
            ResourceAllocation(
                resource_type=RuralResources.WELL,
                count=3,
                requires_property=True,
                terrain_weights={
                    RuralTerrain.FIELD: 2,
                    RuralTerrain.GRASSLAND: 1
                }
            ),
            ResourceAllocation(
                resource_type=RuralResources.BLACKSMITH,
                count=1,
                requires_property=True,
                terrain_weights={
                    RuralTerrain.FIELD: 1,
                    RuralTerrain.GRASSLAND: 1
                }
            ),
            ResourceAllocation(
                resource_type=RuralResources.QUARRY,
                count=2,
                requires_property=False,  # Natural resource
                terrain_weights={
                    RuralTerrain.HILL: 3,
                    RuralTerrain.MOUNTAIN: 1
                }
            )
        ]
    } 

