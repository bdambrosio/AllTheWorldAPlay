"""A suburban house, as the sketch gives it: the bed, the closet, the bathroom with shower, the automatic coffee maker, the car
in the driveway, the quiet suburban street, and the Office downtown about 30 minutes away by car (plays/alex.py; plays/sage.py
states a cafe, which is not laid out here). Nothing more is stated here: what else the world holds, and how it works, is not
this file's to say.

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
    'House': {'description': 'The suburban house Alex and Susan live in', 'radius': 8, 'enclosed': True},
    'Bed': {'description': 'A bed with rumpled sheets', 'radius': 1.5},
    'Shower': {'description': 'A bathroom with shower', 'radius': 1.5},
    'Closet': {'description': 'A closet that contains professional clothing', 'radius': 1.5},
    'Coffee_Maker': {'description': 'An automatic coffee maker', 'radius': 1.5},
    'Car': {'description': 'A car in the driveway', 'radius': 3, 'vehicle': True, 'pace': 500, 'by': 'drive'},
    'Office': {'description': 'The Office, downtown', 'radius': 15},
}

layout = {
    'regions': [
        {'kind': 'Street', 'radius': 16000, 'pace': 75, 'cover': 'open', 'whole': True},
        {'kind': 'City', 'radius': 300, 'pace': 75, 'cover': 'open', 'count': 1, 'from_start': (13500, 14500)},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 400, 'apart': 30},
    'places': [
        {'kind': 'House', 'count': 1, 'from_start': (0, 0)},
        {'kind': 'Bed', 'count': 1, 'inside': 'House', 'apart': 3},
        {'kind': 'Shower', 'count': 1, 'inside': 'House', 'apart': 3},
        {'kind': 'Closet', 'count': 1, 'inside': 'House', 'apart': 3},
        {'kind': 'Coffee_Maker', 'count': 1, 'inside': 'House', 'apart': 3},
        {'kind': 'Car', 'count': 1, 'near': ('House', 3, 10)},
        {'kind': 'Office', 'count': 1, 'on': {'City': 1.0}},
    ],
}

# Color: what helps the calls that write the play (director, planners, stage manager, actors); never a rule, and never read
# by a ruling or a check (survival.color). Approved by the author, 2026-11-28.
color = 'Time, money, work and relationships: ordinary days and decisions that cannot wait.'

# The engine's world layer is on for a world that has this; it states no need, danger or rule of the world.
affordances = {
    'tile_minutes': 1,
}
