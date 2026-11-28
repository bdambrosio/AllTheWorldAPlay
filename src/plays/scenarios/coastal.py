"""A small coastal town where the river meets the sea, as the sketches give it: the harbor with Elijah's boat workshop, his
moored wooden boat, Maya's art studio and Harbor Light Cafe overlooking the water, down the street; and the gallery in the city
(plays/dilemma.py, plays/cafe.py). Nothing more is stated here: what else the town holds, and how it works, is not this file's to say.

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
    'Boat': {'description': 'A wooden boat moored at the harbor', 'radius': 4},
    'Workshop': {'description': 'A boat workshop', 'radius': 8},
    'Gallery': {'description': 'The gallery in the city', 'radius': 12},
    'Cafe': {'description': 'Harbor Light Cafe, overlooking the water', 'radius': 7},
    'Studio': {'description': 'An art studio', 'radius': 6},
}

layout = {
    'regions': [
        {'kind': 'Street', 'radius': 900, 'pace': 75, 'cover': 'open', 'whole': True},
        {'kind': 'Harbor', 'radius': 150, 'pace': 75, 'cover': 'open', 'count': 1, 'from_start': (0, 0)},
        {'kind': 'River', 'radius': 60, 'pace': 60, 'cover': 'open', 'count': 1, 'from_start': (220, 400)},
        {'kind': 'City', 'radius': 200, 'pace': 75, 'cover': 'open', 'count': 1, 'from_start': (400, 700)},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 700, 'apart': 25},
    'places': [
        {'kind': 'Workshop', 'count': 1, 'on': {'Harbor': 1.0}},
        {'kind': 'Boat', 'count': 1, 'on': {'Harbor': 1.0}},
        {'kind': 'Cafe', 'count': 1, 'on': {'Harbor': 1.0}},
        {'kind': 'Studio', 'count': 1, 'on': {'Harbor': 1.0}},
        {'kind': 'Gallery', 'count': 1, 'on': {'City': 1.0}},
    ],
}


# The engine's world layer is on for a world that has this; it states no need, danger or rule of the world.
affordances = {
    'tile_minutes': 2,
}
