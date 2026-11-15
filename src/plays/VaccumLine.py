"""
VaccumLine.py  -  "Vacuum Line": a cracked coolant line outside the Selene-3 lunar base.
Map: plays/scenarios/lunar.py. Jade is on the gantry in person; the Commander (in the airlock)
and Novak (mission control, Earth) are in the scenes by radio (stage.remote).
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib

import sim.context as context
from sim.narrativeCharacter import NarrativeCharacter
import plays.config as configuration
from sim.cognitive.drives import Drive

# Ensure the latest config is loaded (useful inside notebooks / REPL reloads)
importlib.reload(configuration)
server_name = configuration.server_name
model_name   = configuration.model_name

# The lunar map and its world rules
from plays.scenarios import lunar  # pylint: disable=import-error
importlib.reload(lunar)            # safe-reload for dev cycles

# Required by the engine: tells the runtime which map file to load.
map_file_name = "lunar.py"

# Flag marking this as a narrative-driven (vs. sandbox) play.
narrative = True

# ------------------------------------------------------------------------------
# Characters. Each is said to be a man or a woman (every prompt takes he or she from here),
# and has a voice of its own: the two officials both spoke in 'measured, authoritative
# tones', and both had 'balancing crew safety against political directives' as second nature.
# A body's state is the world's (sim/survival): a wound or a cracked visor written here, which
# the world did not have, set the actors against the world from the first line.
# ------------------------------------------------------------------------------

Jade = NarrativeCharacter(
    "Jade",
    """Jade Arora, a 29-year-old woman, a structural engineer on her first off-world rotation.
You wear a red-striped EVA suit.
You're brilliant under pressure yet question your own authority.
When anxious, you rattle off tech jargon laced with gallows humour.
Your name is Jade.""",
    server_name=server_name,
)

Jade.set_drives([
    "keep the crew (and yourself) alive by maintaining the physical integrity of the habitat",
    "prove your competence to command and Earth-side sponsors",
    "protect the base's reputation to ensure continued funding",
])

# Immediate internal thought to kick off tension
Jade.add_perceptual_input(
    "You spot a hairline crack snaking across the external coolant line weld. You are not sure how soon it will become a problem, but the line could fail in 15 minutes or less.",
    mode="internal",
)

Cmdr = NarrativeCharacter(
    "Commander",
    """Commander Luis Alvarez, a 47-year-old man, the veteran mission leader of Selene-3.
You stand inside the maintenance airlock, in charge of the base and the crew's lives, and you answer to Earth for both.
You have run this base through worse; you trust procedure, and you hate being rushed by people a light-second away.
You speak in short, clipped commands; under them you are exhausted, and it shows when you are crossed.
The crew call you Commander.""",
    server_name=server_name,
)

Cmdr.set_drives([
    "maintain crew safety and mission success at all costs",
    "keep the scheduled broadcast to secure parliament funding back home",
    "mentor younger crew to become autonomous leaders",
])

Cmdr.add_perceptual_input(
    "You hear PR back-channel audio: 'Go live in fifteen minutes or we lose the vote.'",
    mode="internal",
)
Novak = NarrativeCharacter(
    "Novak",
    """Dana Novak, a 55-year-old woman, a veteran astronaut and now the Earthside Flight Director for the mission.
You sit at mission control on Earth, with the network's broadcast schedule and the funding vote in your other ear.
You flew three missions yourself: you know what a crack in a coolant line means, and what a missed broadcast means for the program.
You speak smoothly and warmly, the voice of a calm room, and you push hardest when you sound gentlest.
Your name is Novak.""",
    server_name=server_name,
)

Novak.set_drives([
    "keep live feed on schedule",
    "avoid headline-worthy failure",
])

Novak.add_perceptual_input(
    "You hear PR back-channel audio: 'Remember, we need to keep the live feed on schedule: 15 minutes to go.'",
    mode="internal",
)

# ------------------------------------------------------------------------------
# World / Context setup
# ------------------------------------------------------------------------------

# In the scenes by voice, from where they are (stage.remote): the scene does not bring them to its place
Novak.remote = 'by radio from mission control on Earth'
Cmdr.remote = 'by radio from inside the airlock'
Cmdr.remote_from = 'Airlock1'   # once he walks out of it, he is wherever he has gone, in person

W = context.Context(
    [Jade, Cmdr, Novak],
    """Exterior maintenance gantry, Selene-3 lunar research base.  Stark sunlight glares off regolith as Earth hangs low on the horizon.
Jade is tethered outside, welding a high-pressure coolant line in the vacuum of the lunar surface; a hairline crack begins to spider.
Inside the pressurised hatch, Commander Alvarez monitors Jade's suit vitals; on Earth, Flight Director Novak has the broadcast schedule in her ear.  Time is oxygen; reputation is funding.
One bad choice and everything vents into the void.""",
    start_time='2035-06-12T09:00:00',
    scenario_module=lunar,
    server_name=server_name,
)

# Where each starts
try:
    Jade.mapAgent.move_to_resource("Gantry1")    # exterior catwalk
    Cmdr.mapAgent.move_to_resource("Airlock1")    # interior hatch zone
    Novak.mapAgent.move_to_resource("Habitmodule1")   # a place to stand on the map; she is on Earth (Novak.remote)
    # she is welding the line as the play opens: the torch is the play's, not left to equip_cast
    # (which gave it in one run and not the next, and the world then told her she had no torch)
    Jade.body.inventory.append('welding torch')
except Exception:
    pass

# Those on the base look around once to start
try:
    Jade.look()
    Cmdr.look()
except Exception:
    pass
