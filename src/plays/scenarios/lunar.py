"""Selene-3 lunar research base, as the sketch gives it: the exterior maintenance gantry on the regolith, the airlock,
and the vacuum of the lunar surface outside (plays/VaccumLine.py); a module of the base is kept as the place Novak's
body stands (she is on Earth). Nothing more is stated here: what else the base holds, and how it works, is not this file's to say.

How the places of this world are, and how they sit. They are laid out in metres, anew for each
performance (sim/land.py); other places come into it as the play needs them (sim/world.py).
  kinds: for each kind of place, what it is, and in metres its radius. enclosed: it has walls
    (open: they stand open, as a cave's mouth or a shop with its front broken in). shelter: inside it
    is under shelter. blocks_view: nothing is seen across it. hides: what is inside is not seen
    from outside. owner: it has an owner's part, which someone may be cast in. vehicle: it is gone
    by, at `pace` metres a minute, and `by` is the word for that. search: what is left there is found
    only by someone who searches the place.
  layout: the ground (regions: large places that hold the rest; pace is metres a minute on foot,
    cover is what is overhead) and how many of each kind there are and where: on which ground, by
    weight; inside or near (metres from its edge) another kind; from_start, metres from where the
    play's people first stand; names, for those the author has named.
"""

kinds = {
    'Airlock': {'description': 'The maintenance airlock', 'radius': 3, 'enclosed': True},
    'Gantry': {'description': 'Exterior maintenance gantry', 'radius': 6},
    'Habitmodule': {'description': 'A module of Selene-3 lunar research base', 'radius': 5},
}

layout = {
    'regions': [
        {'kind': 'Regolith', 'radius': 400, 'pace': 40, 'cover': 'open', 'whole': True},
        {'kind': 'Habitat', 'radius': 40, 'pace': 60, 'cover': 'open', 'count': 1, 'from_start': (0, 0)},
        {'kind': 'Gantry', 'radius': 25, 'pace': 30, 'cover': 'open', 'count': 1, 'from_start': (100, 160)},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 300, 'apart': 6},
    'places': [
        {'kind': 'Habitmodule', 'count': 1, 'on': {'Habitat': 1.0}, 'apart': 12},
        {'kind': 'Airlock', 'count': 1, 'on': {'Habitat': 1.0}, 'apart': 9},
        {'kind': 'Gantry', 'count': 1, 'on': {'Gantry': 1.0}},
    ],
}

# Color: what helps the calls that write the play (director, planners, stage manager, actors); never a rule, and never read
# by a ruling or a check (survival.color). Approved by the author, 2026-11-28.
color = 'Air, suit integrity and procedure matter here; vacuum does not forgive mistakes.'

# The engine's world layer is on for a world that has this; it states no need, danger or rule of the world.
affordances = {
    'tile_minutes': 2,
}

# the sky of a picture (stage.light), in the scene's words: "Stark sunlight glares off regolith as Earth
# hangs low on the horizon"; the play's clock runs on Earth hours
_SUNLIT = 'stark sunlight glaring off regolith, Earth low on the horizon'
image_light = {part: _SUNLIT for part in ('dawn', 'morning', 'midday', 'afternoon', 'dusk', 'night')}
