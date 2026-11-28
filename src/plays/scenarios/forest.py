"""A temperate, mixed forest-open landscape, as the sketches give it: no buildings, no roads, a mossy spring,
and the bear's cave (plays/lost.py, plays/aesopBear.py). Nothing more is stated here: what else the forest holds,
and how it works, is not this file's to say.

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
    'Spring': {'description': 'A mossy spring', 'radius': 4},
    'Cave': {'description': 'A cave', 'radius': 5},
}

layout = {
    'regions': [
        {'kind': 'Forest', 'radius': 1200, 'pace': 50, 'cover': 'trees', 'whole': True},
        {'kind': 'Clearing', 'radius': (40, 80), 'pace': 75, 'cover': 'open', 'count': 6},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 700, 'apart': 40},
    'places': [
        {'kind': 'Spring', 'count': 1, 'on': {'Forest': 1.0}, 'from_start': (150, 700)},
        {'kind': 'Cave', 'count': 1, 'on': {'Forest': 1.0}},
    ],
}

# Color: what helps the calls that write the play (director, planners, stage manager, actors); never a rule, and never read
# by a ruling or a check (survival.color). Approved by the author, 2026-11-28.
color = 'Wilderness, where water, food, rest and shelter matter, and wild animals are about.'

# The engine's world layer is on for a world that has this; it states no need, danger or rule of the world.
affordances = {
    'tile_minutes': 2,
}
