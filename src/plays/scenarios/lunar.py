"""
lunar.py  –  Procedural map definition for the Selene‑3 Moon‑base scenario
used by VacuumLine.py.

This mirrors the structure of existing scenario modules (e.g. apocalypse.py)
but swaps in lunar‑specific terrain, infrastructure and resources while still
conforming to the small requirements imposed by map.py (namely the presence of
a Terrain.Water entry and a resource_types.Park placeholder).

NOTE ▸ If the runtime supports explicit resource names, add
        "name": "AirlockA" / "GantryA" to the relevant allocations below.
      ▸ Otherwise Jade should spawn at "Gantry1" and Cmdr. Alvarez at
        "Airlock1" once the map is built, since the engine auto‑numbers from 1.
"""

from enum import Enum, auto

# Import the new dynamic resource system
from sim.map import ResourceTypeRegistry

# ──────────────────────────────────────────────────────────────────────────────
# Enum declarations – terrain, infrastructure, resources, property
# ──────────────────────────────────────────────────────────────────────────────
class LunarTerrain(Enum):
    Water = 1          # Mandatory sentinel – here: shadowed ice pockets
    Regolith = 2       # Flat open dust
    Crater = 3         # Depressions / rough ground
    Habitat = 4        # Pressurised modules & decks
    Gantry = 5         # Exterior metal walkways / service trusses
    Solararray = 6     # Panel fields (non‑walkable)

class LunarInfrastructure(Enum):
    Walkway = auto()   # EVA‑rated paths on the surface
    Tube = auto()      # Pressurised corridors between modules

class LunarResources(Enum):
    Airlock = auto()       # Primary base ingress/egress
    Gantry = auto()        # Exterior platform segment
    Evastation = auto()    # Suit storage & recharge nook
    Repairpanel = auto()   # Electrical / fluid interface box
    Habitmodule = auto() # Living or working compartment
    Oxygentank = auto()    # Exposed backup O₂ bundle
    Park = auto()          # Hydroponic greenhouse (placeholder to satisfy map.py)

class LunarProperty(Enum):
    Habitat = auto()       # Pressurised hab/lab lots
    Lab = auto()
    Storage = auto()
    Solararray = auto()

# ──────────────────────────────────────────────────────────────────────────────
# Terrain generation rules
# ──────────────────────────────────────────────────────────────────────────────
terrain_rules = {
    "elevation_noise_scale": 10.0,      # Small‑area base, gentle undulation
    "water_level": 0.05,               # Only deepest craters flagged as ice
    "mountain_level": 0.85,
    "terrain_by_elevation": {
        "water": {"max": 0.05, "type": "Water"},  # Permanently shadowed rim
    },
    "lowland_distribution": {
        "Regolith": 0.45,
        "Crater": 0.25,
        "Habitat": 0.12,
        "Gantry": 0.10,
        "Solararray": 0.08,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Infrastructure / path‑finding rules
# ──────────────────────────────────────────────────────────────────────────────
infrastructure_rules = {
    "road_density": 0.25,         # Sparse EVA tracks
    "path_type": "Walkway",      # Default connector
    "slope_factor": 0.7,         # Traversing shallow crater walls is harder
    "terrain_costs": {
        "Water": float("inf"),
        "Solararray": float("inf"),
        "Crater": 3.0,
        "Regolith": 1.0,
        "Habitat": 0.5,      # Inside modules is easy walking
        "Gantry": 0.8,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Property parcel rules
# ──────────────────────────────────────────────────────────────────────────────
property_rules = {
    "min_size": 6,
    "max_size": 18,
    "valid_terrain": ["Habitat"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Resource placement rules
# ──────────────────────────────────────────────────────────────────────────────
terrain_types = LunarTerrain
infrastructure_types = LunarInfrastructure
property_types = LunarProperty
resource_types = ResourceTypeRegistry(LunarResources)  # Use dynamic registry

required_resource = resource_types.Park       # Mandatory per map.py
required_resource_name = "HydroponicGarden"  # Friendly label

resource_rules = {
    "allocations": [
        {
            "resource_type": resource_types.Airlock,
            "description": "Primary EVA airlock into the habitat ring",
            "count": 1,
            "requires_property": True,
            "terrain_weights": {terrain_types.Habitat: 2.0},
        },
        {
            "resource_type": resource_types.Gantry,
            "description": "Metal service gantry overlooking the vacuum‑line weld",
            "count": 1,
            "requires_property": False,
            "terrain_weights": {terrain_types.Gantry: 2.0},
        },
        {
            "resource_type": resource_types.Evastation,
            "description": "Suit stow & recharge alcove",
            "count": 1,
            "requires_property": False,
            "terrain_weights": {terrain_types.Gantry: 1.0, terrain_types.Habitat: 0.5},
        },
        {
            "resource_type": resource_types.Repairpanel,
            "description": "External repair junction box for vacuum line",
            "count": 1,
            "requires_property": False,
            "terrain_weights": {terrain_types.Gantry: 1.0},
        },
        {
            "resource_type": resource_types.Habitmodule,
            "description": "Pressurised living/work module",
            "count": 3,
            "requires_property": True,
            "terrain_weights": {terrain_types.Habitat: 2.0},
        },
        {
            "resource_type": resource_types.Oxygentank,
            "description": "Exposed backup O₂ bundle",
            "count": 1,
            "requires_property": False,
            "terrain_weights": {terrain_types.Habitat: 1.0},
        },
        {
            "resource_type": resource_types.Park,   # Hydroponic greenhouse (placeholder to satisfy map.py)
            "description": "Hydroponic greenhouse dome (Park placeholder)",
            "count": 1,
            "requires_property": True,
            "terrain_weights": {terrain_types.Habitat: 2.0},
        },
    ]
}

# Expose canonical interface names
terrain_types = LunarTerrain
infrastructure_types = LunarInfrastructure
property_types = LunarProperty
resource_types = ResourceTypeRegistry(LunarResources)  # Use dynamic registry
