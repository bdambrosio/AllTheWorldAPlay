"""The city of Bellona, as the sketch gives it: deserted streets, buildings, and the park with its fountain
(plays/bellona.py). Nothing more is stated here: what else the city holds, and how it works, is not this file's to say.

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
    'Park': {'description': 'A park with a fountain', 'radius': 8},
}

layout = {
    'regions': [
        {'kind': 'Street', 'radius': 600, 'pace': 70, 'cover': 'open', 'whole': True},
        {'kind': 'Park', 'radius': 80, 'pace': 55, 'cover': 'open', 'count': 1, 'from_start': (80, 300)},
        {'kind': 'Building', 'radius': (30, 50), 'pace': 70, 'cover': 'open', 'count': 8, 'from_start': (60, 400), 'blocks_view': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 350, 'apart': 25},
    'places': [
        {'kind': 'Park', 'count': 1, 'on': {'Park': 1.0}},
    ],
}

# The engine's world layer is on for a world that has this; it states no need, danger or rule of the world.
affordances = {
    'tile_minutes': 3,
}
