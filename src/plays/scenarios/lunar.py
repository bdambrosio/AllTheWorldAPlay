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


# What is at stake in this world (planning and act prompts; sim/survival.py stakes_text).
stakes = ("On this lunar base, air, suit integrity and the failing coolant line are the stakes: oxygen runs down "
          "outside the habitat, repairs need parts, time and steady hands, and vacuum punishes mistakes. Use these, "
          "and the crew's trust and procedures under pressure, as concrete, checkable goals and obstacles tied to the "
          "airlock, the gantry, the repair panel and the oxygen supply.")

# Stage business (sim/survival.py): suit oxygen outside (none used inside), repairs, airlock, EVA hazards.
affordances = {
    'needs': {
        # 'oxygen' is suit oxygen used: it rises outside and stops inside the pressurised habitat (sheltered)
        'oxygen': {'initial': 0.35, 'per_hour': 0.25, 'shelter_factor': 0.0, 'neglect_harm': 0.3,
                   'words': ('suit air plentiful', 'suit air past half', 'suit air low', 'suit air critical'),
                   'percepts': {0.5: "Your suit display flashes: oxygen past half.",
                                0.8: "Low-oxygen alarm: your suit is chiming, and every breath feels thin."}},
        'fatigue': {'initial': 0.3, 'per_hour': 0.06, 'shelter_factor': 0.3,
                    'words': ('steady', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "Working in the suit is wearing you down; your arms ache.",
                                 0.8: "You are exhausted; your hands are clumsy in the gloves."}},
    },
    'harm': {'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')},
    'travel_need': 'fatigue',
    'tile_minutes': 2,
    'type_nouns': {'Airlock': ['airlock', 'lock', 'hatch'], 'Gantry': ['gantry', 'coolant line', 'weld', 'line'],
                   'Evastation': ['eva station', 'suit station', 'stow', 'recharge'], 'Repairpanel': ['repair panel', 'panel'],
                   'Habitmodule': ['habitat', 'hab', 'module'], 'Oxygentank': ['oxygen tank', 'o2', 'oxygen', 'tank']},
    'actions': [
        {'name': 'refill oxygen', 'verbs': ['refill', 'recharge', 'top up', 'swap tank', 'plug in'], 'named_verbs': ['connect', 'hook up'],
         'resource_types': ['Oxygentank', 'Evastation'], 'effects': {'oxygen': -1.0}, 'minutes': 10,
         'text': 'refills suit oxygen at {resource}'},
        {'name': 'cycle the airlock', 'verbs': ['cycle', 'enter the airlock', 'go inside', 'pressurise', 'pressurize'],
         'named_verbs': ['enter', 'open'], 'resource_types': ['Airlock'], 'minutes': 10, 'sets': 'sheltered',
         'text': 'cycles through {resource} into the pressurised habitat'},
        {'name': 'fetch sealant kit', 'category': 'work', 'verbs': ['fetch', 'grab', 'take', 'collect', 'get'],
         'named_verbs': [], 'resource_types': ['Evastation', 'Repairpanel'], 'gain': 'sealant kit', 'minutes': 5, 'deplete': 3,
         'text': 'takes a sealant kit from {resource}'},
        {'name': 'seal the coolant line', 'category': 'work', 'verbs': ['seal', 'weld', 'patch', 'clamp', 'repair', 'fix'],
         'resource_types': ['Gantry', 'Repairpanel'], 'requires': 'sealant kit', 'consumes': 'sealant kit',
         'requires_property': {'repaired': None}, 'sets_property': {'repaired': True},
         'effects': {'fatigue': 0.15}, 'minutes': 30, 'text': 'seals the cracked coolant line at {resource}',
         'risk': {'chance': 0.15, 'effects': {'injury': 0.15}, 'text': 'A jet of coolant vapour slams against the visor.'}},
        {'name': 'run diagnostics', 'category': 'explore', 'verbs': ['diagnose', 'run diagnostics', 'scan', 'check readings', 'inspect'],
         'resource_types': ['Repairpanel'], 'finds': [{'item': 'pressure-loss log', 'chance': 0.5}], 'minutes': 15, 'deplete': 2,
         'text': 'runs diagnostics at {resource}'},
        {'name': 'rest', 'verbs': ['rest', 'sit', 'sleep', 'lie down'], 'resource_types': ['Habitmodule'],
         'effects': {'fatigue': -0.4}, 'minutes': 60, 'sets': 'sheltered', 'text': 'rests in {resource}'},
        {'name': 'buddy-breathe', 'category': 'social', 'target': 'character', 'verbs': ['share air', 'buddy-breathe', 'buddy breathe', 'cross-connect'],
         'effects': {'oxygen': 0.25}, 'target_effects': {'oxygen': -0.25}, 'minutes': 5,
         'text': "cross-connects suit lines and shares air with {target}"},
        {'name': 'give sealant kit', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'pass'],
         'transfer': 'sealant kit', 'minutes': 1, 'text': 'hands {target} a sealant kit'},
        {'name': 'tend injury', 'category': 'social', 'target': 'character', 'verbs': ['tend', 'treat', 'patch up', 'check on'],
         'target_effects': {'injury': -0.2}, 'minutes': 15, 'text': "treats {target}'s injury"},
    ],
    'hazards': [
        {'name': 'micrometeorites', 'description': 'a shower of micrometeorites pinging off the regolith', 'when': 'any',
         'terrain': ['Regolith', 'Crater', 'Gantry'], 'chance_per_hour': 0.06, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 15, 'harm': 0.3, 'fail_text': 'a grain scores the suit and a leak alarm shrieks',
         'retreat_text': 'The shower passes; the surface is still again.',
         'strike_text': 'Micrometeorites rake the surface; suits are scored and a seal weeps air.',
         'responses': {'take cover': {'verbs': ['take cover', 'shelter', 'duck behind', 'get behind'], 'success': 0.6},
                       'retreat inside': {'verbs': ['retreat', 'head for the airlock', 'go inside'], 'success': 0.5}}},
        {'name': 'line rupture', 'description': 'the cracked coolant line venting a plume of frozen vapour', 'when': 'any',
         'terrain': ['Gantry'], 'chance_per_hour': 0.15, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 20, 'harm': 0.3, 'fail_text': 'the vapour plume whips across the gantry',
         'retreat_text': 'The venting slows to a trickle; the line holds for now.',
         'strike_text': 'The line ruptures; a blast of frozen coolant knocks bodies against the rail.',
         'responses': {'clamp the line': {'verbs': ['clamp', 'seal', 'patch', 'shut the valve', 'isolate'], 'success': 0.4,
                                          'bonus': {'sealant kit': 0.35}},
                       'back away': {'verbs': ['back away', 'retreat', 'move clear'], 'success': 0.6}}},
    ],
}
