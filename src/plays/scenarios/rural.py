"""A small 19th century French farming village, as the sketch gives it: its farms (the Marquarts' and the Fouans'), the strips
of wheat land the play lays beside them, and the fields ripe with wheat around it (plays/laTerre.py). Nothing more is stated
here: what else the village holds, and how it works, is not this file's to say.

How the places of this world are, and how they sit. They are laid out in metres, anew for each
performance (sim/land.py); other places come into it as the play needs them (sim/world.py).
  kinds: for each kind of place, what it is, and in metres its radius. enclosed: it has walls
    (open: they stand open, as a cave's mouth or a shop with its front broken in). shelter: inside it
    is under shelter. blocks_view: nothing is seen across it. hides: what is inside is not seen
    from outside. owner: it has an owner's part, which someone may be cast in. vehicle: it is gone
    by, at `pace` metres a minute, and `by` is the word for that.
  layout: the ground (regions: large places that hold the rest; pace is metres a minute on foot,
    cover is what is overhead) and how many of each kind there are and where: on which ground, by
    weight; inside or near (metres from its edge) another kind; from_start, metres from where the
    play's people first stand; names, for those the author has named.
"""

kinds = {
    'Farm': {'description': 'A family farm', 'radius': 20},
    'Strip': {'description': 'A strip of wheat land', 'radius': 60},
}

layout = {
    'regions': [
        {'kind': 'Field', 'radius': 3000, 'pace': 60, 'cover': 'open', 'whole': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 1800, 'apart': 150},
    'places': [
        {'kind': 'Farm', 'count': 2, 'on': {'Field': 1.0}, 'from_start': (300, 1800), 'apart': 250, 'names': ['MarquartFarm', 'FouanFarm']},
        {'kind': 'Strip', 'count': 0, 'names': ['MarquartUpperStrip', 'MarquartLowerStrip', 'FouanUpperStrip', 'FouanLowerStrip']},
    ],
}

# Color: what helps the calls that write the play (director, planners, stage manager, actors); never a rule, and never read
# by a ruling or a check (survival.color). Approved by the author, 2026-11-28.
color = 'Land, harvest, money and inheritance; heat and hard labour.'

# The engine's world layer is on for a world that has this; it states no need, danger or rule of the world.
affordances = {
    'tile_minutes': 4,
}
