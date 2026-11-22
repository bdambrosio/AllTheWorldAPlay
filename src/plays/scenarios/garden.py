"""One garden, as kittens have it: grass, shrubs, flower beds, a tree.

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
    'Bug': {'description': 'yummy bug', 'radius': 0.3},
    'Butterfly': {'description': 'A bright butterfly flitting among the flowers', 'radius': 0.3},
    'Leaf': {'description': 'A deep pile of crunchy fallen leaves', 'radius': 0.6},
    'Seed': {'description': 'A scatter of seeds under the bird feeder', 'radius': 0.4},
    'Ball': {'description': 'A bright red ball', 'radius': 0.3},
    'Tree': {'description': 'The garden tree', 'radius': 1},
}

layout = {
    'regions': [
        {'kind': 'Grass', 'radius': 30, 'pace': 40, 'cover': 'open', 'whole': True},
        {'kind': 'Shrubs', 'radius': (3, 5), 'pace': 25, 'cover': 'trees', 'count': 4, 'from_start': (4, 25), 'hides': True},
        {'kind': 'Trees', 'radius': 4, 'pace': 35, 'cover': 'trees', 'count': 2, 'from_start': (6, 25)},
        {'kind': 'Flowers', 'radius': (2, 3), 'pace': 35, 'cover': 'open', 'count': 3, 'from_start': (4, 25)},
        {'kind': 'Rocks', 'radius': 2, 'pace': 30, 'cover': 'open', 'count': 1, 'from_start': (6, 25)},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 25, 'apart': 1.5},
    'places': [
        {'kind': 'Bug', 'count': 15, 'on': {'Grass': 2.0, 'Shrubs': 1.0, 'Flowers': 0.5}},
        {'kind': 'Butterfly', 'count': 10, 'on': {'Grass': 2.0, 'Shrubs': 1.0, 'Flowers': 0.5}},
        {'kind': 'Leaf', 'count': 24, 'on': {'Grass': 1.0, 'Shrubs': 0.5}},
        {'kind': 'Seed', 'count': 16, 'on': {'Grass': 0.6, 'Shrubs': 1.0}},
        {'kind': 'Ball', 'count': 3, 'on': {'Grass': 1.0, 'Shrubs': 0.5}},
        {'kind': 'Tree', 'count': 1, 'on': {'Trees': 1.0}},
    ],
}


stakes = ("For the kittens, the garden is a world of play, food, naps and small dangers: a bug to catch, a "
          "butterfly to chase, a ball to win, a warm nap, the neighbour's dog. Use these concrete, playful goals and "
          "obstacles tied to things in the garden; keep the tone gentle and the magic light.")

# How this world works, in sentences (docs/world-facts-design.md). Nothing here is matched by code.
facts = [
    'A bug caught and eaten eases hunger well; there are about three to be had at a spot. Seeds nibbled ease it a little.',
    "An hour's nap eases sleepiness fully.",
    'Being groomed eases scrapes a little.',
    'Chasing the butterfly takes about ten minutes and leaves a kitten sleepier. About one time in five a shimmering wing-scale is left behind, and about one time in six the chase ends in a tumble into the rose bush.',
]
# What threatens people here, in sentences (docs/dangers-design.md): the play brings one on, and the world says what comes of it
dangers = [
    "By day the neighbour's big bouncy dog may come bounding over the fence onto the grass and the flower beds. A kitten that hisses at it, runs or hides gets clear of it about half the time; one that is under cover it does not find. It means no harm: it bowls everyone over in a slobbery tumble, a few scrapes at worst, and is called home within a quarter of an hour.",
]
# What no deed or word gets past
held = [
]

# Stage business (sim/survival.py): hunting bugs, chasing, play, naps, the neighbour's dog.
affordances = {
    'needs': {
        'hunger': {                   'initial': 0.35, 'per_hour': 0.08,
                   'words': ('full', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.5: "Your tummy rumbles; a bug would be nice.", 0.8: "You are SO hungry."}},
        'sleepiness': {                       'initial': 0.2, 'per_hour': 0.1, 'shelter_factor': 0.5,
                       'words': ('bouncy', 'drowsy', 'sleepy', 'can barely keep eyes open'),
                       'percepts': {0.5: "Your eyes are getting heavy; a sunny spot looks very nice.",
                                    0.8: "You are so sleepy you keep yawning."}},
    },
    'harm': {             'name': 'scrapes', 'words': ('unruffled', 'ruffled', 'scraped', 'hurting')},
    'travel_need': 'sleepiness',
    'tile_minutes': 1,
}
