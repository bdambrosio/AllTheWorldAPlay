"""A harbour town: the harbour and its workshop, a garden, a river and its bridge, downtown.

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
    'Bench': {'description': 'A wooden bench overlooking the harbor', 'radius': 2},
    'Boat': {'description': 'A wooden boat moored at the dock', 'radius': 4},
    'Tools': {'description': 'Boat building tools', 'radius': 1.5},
    'Workshop': {'description': 'A boat building workshop', 'radius': 8, 'enclosed': True, 'shelter': True},
    'Bridge': {'description': 'A bridge over the river', 'radius': 8},
    'Gallery': {'description': 'The city art gallery', 'radius': 12, 'enclosed': True, 'shelter': True},
    'Cafe': {'description': 'The local cafe', 'radius': 7, 'enclosed': True, 'shelter': True},
    'Studio': {'description': 'The artist studio', 'radius': 6, 'enclosed': True, 'shelter': True},
}

layout = {
    'regions': [
        {'kind': 'Road', 'radius': 900, 'pace': 75, 'cover': 'open', 'whole': True},
        {'kind': 'Harbor', 'radius': 150, 'pace': 75, 'cover': 'open', 'count': 1, 'from_start': (0, 0)},
        {'kind': 'Garden', 'radius': 60, 'pace': 70, 'cover': 'open', 'count': 1, 'from_start': (160, 300)},
        {'kind': 'River', 'radius': 60, 'pace': 60, 'cover': 'open', 'count': 1, 'from_start': (220, 400)},
        {'kind': 'Downtown', 'radius': 200, 'pace': 75, 'cover': 'open', 'count': 1, 'from_start': (400, 700)},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 700, 'apart': 25},
    'places': [
        {'kind': 'Workshop', 'count': 1, 'on': {'Harbor': 1.0}},
        {'kind': 'Tools', 'count': 1, 'inside': 'Workshop', 'apart': 2},
        {'kind': 'Boat', 'count': 1, 'on': {'Harbor': 1.0}},
        {'kind': 'Cafe', 'count': 1, 'on': {'Harbor': 1.0}},
        {'kind': 'Studio', 'count': 1, 'on': {'Harbor': 1.0}},
        {'kind': 'Bench', 'count': 2, 'on': {'Garden': 1.0, 'Harbor': 1.0}},
        {'kind': 'Bridge', 'count': 1, 'on': {'River': 1.0}},
        {'kind': 'Gallery', 'count': 1, 'on': {'Downtown': 1.0}},
    ],
}


stakes = ("In this harbour town, livelihoods, art, money and relationships are at stake: rent and commissions, a "
          "gallery sale, a boat that needs mending, favours and promises between neighbours. Use concrete, checkable "
          "goals and obstacles tied to the studio, the workshop, the gallery, the cafe and the harbour.")

# How this world works, in sentences (docs/world-facts-design.md). Nothing here is matched by code.
facts = [
    'A coffee at the cafe costs about 4 dollars and takes the edge off tiredness; a meal there costs about 12 and eases hunger well.',
    'A painting takes an afternoon in the studio, about three hours, and a carving the same in the workshop; both are tiring.',
    'The gallery sells a finished painting for about 150 dollars and a carving for about 90; a sale takes about half an hour.',
    "Tools may be borrowed from the tool store. Mending the boat needs tools and about two hours, and is tiring. An hour and a half's sail is restful.",
    'Sitting on a bench watching the harbour eases tiredness a little.',
]
# What threatens people here, in sentences (docs/dangers-design.md): the play brings one on, and the world says what comes of it
dangers = [
    'A sudden squall may race in off the sea over the harbour, the road and the gardens, twenty minutes from first sight. Under cover a person is safe from it, and most who run for cover make it; a boat can be made fast in time about half the time. It soaks and batters whoever is caught out and sends loose gear across the quay, but does little real harm.',
]
# What no deed or word gets past
held = [
    'A boat that has not been mended does not sail.',
    'Money comes into the play only as a sale these facts name.',
]

# Stage business (sim/survival.py): making and selling work, the cafe, the boat, a squall.
affordances = {
    'needs': {
        'hunger': {'initial': 0.3, 'per_hour': 0.04,
                   'words': ('not hungry', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.5: "You are getting hungry.", 0.8: "You are very hungry; it is hard to focus."}},
        'fatigue': {                    'initial': 0.3, 'per_hour': 0.03, 'shelter_factor': 0.5,
                    'words': ('fresh', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "You are tiring.", 0.8: "You are exhausted."}},
    },
    'harm': {'name': 'injury', 'words': ('fine', 'bruised', 'hurt', 'badly hurt')},
    'travel_need': 'fatigue',
    'currency': 'dollars', 'initial_money': 30,
    'tile_minutes': 2,
}
