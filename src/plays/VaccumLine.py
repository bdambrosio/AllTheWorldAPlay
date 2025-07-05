"""
VacuumLine.py  –  Lunar‑base scenario for the high‑stakes teaser "Vacuum Line".

▸ Map module required:  lunar.py  (to be created next)
▸ Part of WebWorld plays collection.

This file declares the main characters, seeds their internal drives, and sets up
the immediate crisis outside the Selene‑3 research habitat.  Designed for
strong dialogue with physical peril simmering in the backdrop.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports & reload helpers
# ──────────────────────────────────────────────────────────────────────────────
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib

import sim.context as context
from src.sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.driveSignal import Drive

# Ensure the latest config is loaded (useful inside notebooks / REPL reloads)
importlib.reload(configuration)
server_name = configuration.server_name
model_name   = configuration.model_name

# Scenario / map placeholder – the lunar map will be authored separately.
from plays.scenarios import lunar  # pylint: disable=import-error
importlib.reload(lunar)            # safe‑reload for dev cycles

# Required by the engine: tells the runtime which map file to load.
map_file_name = "lunar.py"

# Flag marking this as a narrative‑driven (vs. sandbox) play.
narrative = True

# ──────────────────────────────────────────────────────────────────────────────
# Character definitions
# ──────────────────────────────────────────────────────────────────────────────

Jade = NarrativeCharacter(
    "Jade",
    """Jade Arora, a 29‑year‑old structural engineer on her first off‑world rotation. 
You wear a red‑striped EVA suit; a tiny fissure spiders across your visor. 
You’re brilliant under pressure yet question your own authority. 
When anxious, you rattle off tech jargon laced with gallows humour. 
""",
    server_name=server_name,
)

Jade.set_drives([
    "keep the crew (and yourself) alive by maintaining the physical integrity of the habitat",
    "prove your competence to command and Earth‑side sponsors",
    "protect the base’s reputation to ensure continued funding",
])

# Immediate internal thought to kick off tension
Jade.add_perceptual_input(
    "You spot a hairline crack snaking across the external coolant line weld. You are not sure how soon it will become a problem, but the line could fail in 15 minutes or less.",
    "internal",
)

Cmdr = NarrativeCharacter(
    "Commander",
    """Commander Luis Alvarez, 47‑year‑old veteran mission leader. 
You stand inside the maintenance airlock, a smear of blood on your cheek from a prior micro‑fracture mishap. 
Balancing crew safety against political directives from Earth is second nature. 
You speak in measured, authoritative tones that mask exhaustion.""",
    server_name=server_name,
)

Cmdr.set_drives([
    "maintain crew safety and mission success at all costs",
    "keep the scheduled broadcast to secure parliament funding back home",
    "mentor younger crew to become autonomous leaders",
])

Cmdr.add_perceptual_input(
    "You hear PR back‑channel audio: ‘Go live in fifteen minutes or we lose the vote.’",
    "internal",
)
Novak = NarrativeCharacter(
    "Novak",
    """Novak, 55‑year‑old veteran astronaut, is the Earthside Flight Director for the mission. 
You speak in measured, authoritative tones that mask tension and pressure. 
Balancing crew safety against political directives from Earth is second nature, but political pressure to look good on camera is high.
""",
    server_name=server_name,
)

Novak.set_drives([
    "keep live feed on schedule",
    "avoid headline-worthy failure",
])

Novak.add_perceptual_input(
    "You hear PR back‑channel audio: ‘Remember, we need to keep the live feed on schedule: 15 minutes to go.",
    "internal",
)

# ──────────────────────────────────────────────────────────────────────────────
# World / Context setup
# ──────────────────────────────────────────────────────────────────────────────

W = context.Context(
    [Jade, Cmdr, Novak],
    """Exterior maintenance gantry, Selene‑3 lunar research base.  Stark sunlight glares off regolith as Earth hangs low on the horizon. 
Jade is tethered outside, welding a high‑pressure coolant line in the vacuum of the lunar surface; a hairline crack begins to spider. 
Inside the pressurised hatch, Cmdr Alvarez monitors Jade's suit vitals while fielding urgent calls from Earth‑side PR.  Time is oxygen; reputation is funding. 
One bad choice and everything vents into the void.""",
    scenario_module=lunar,
    server_name=server_name,
)

# Position characters on the (to‑be‑defined) map
try:
    Jade.mapAgent.move_to_resource("Gantry1")    # exterior catwalk
    Cmdr.mapAgent.move_to_resource("Airlock1")    # interior hatch zone
except Exception:
    # Map not yet authored; safe to ignore during initial authoring.
    pass

# Give both characters an initial ‘look’ to seed perception queues.
try:
    Jade.look()
    Cmdr.look()
except Exception:
    pass

# Uncomment to drive a pre‑baked narrative prototype once the play is ready.
# narrative = "vacuum_demo.json"
