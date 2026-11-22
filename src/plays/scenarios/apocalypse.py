"""A ruined city: streets, a plaza, a park with a fountain that still runs.

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
    'BusStop': {'description': 'A weathered bus stop shelter', 'radius': 3, 'shelter': True},
    'Shop': {'description': 'A small retail shop', 'radius': 7, 'enclosed': True, 'open': True, 'shelter': True},
    'Cafe': {'description': 'A cozy cafe', 'radius': 7, 'enclosed': True, 'open': True, 'shelter': True},
    'Bench': {'description': 'A weathered bench', 'radius': 2},
    'TrashBin': {'description': 'A rusty trash bin', 'radius': 2},
    'Park': {'description': 'A park with a fountain that still runs', 'radius': 8},
}

layout = {
    'regions': [
        {'kind': 'Street', 'radius': 600, 'pace': 70, 'cover': 'open', 'whole': True},
        {'kind': 'Plaza', 'radius': 50, 'pace': 65, 'cover': 'open', 'count': 1, 'from_start': (60, 250)},
        {'kind': 'Park', 'radius': 80, 'pace': 55, 'cover': 'open', 'count': 1, 'from_start': (80, 300)},
        {'kind': 'Construction', 'radius': 60, 'pace': 40, 'cover': 'open', 'count': 1, 'from_start': (150, 350)},
        {'kind': 'Building', 'radius': (30, 50), 'pace': 70, 'cover': 'open', 'count': 8, 'from_start': (60, 400), 'blocks_view': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 350, 'apart': 25},
    'places': [
        {'kind': 'Park', 'count': 1, 'on': {'Park': 1.0}},
        {'kind': 'Bench', 'count': 3, 'on': {'Park': 2.0, 'Plaza': 1.0}, 'apart': 10},
        {'kind': 'Cafe', 'count': 1, 'on': {'Plaza': 1.0}},
        {'kind': 'Shop', 'count': 2, 'on': {'Street': 1.0}},
        {'kind': 'BusStop', 'count': 2, 'on': {'Street': 1.0}},
        {'kind': 'TrashBin', 'count': 2, 'on': {'Street': 1.0, 'Plaza': 0.5}},
    ],
}


stakes = ("In Bellona, food, clean water and a safe place to sleep must be scavenged or bargained for, and the city "
          "is unpredictable: gangs roam the streets at night and strange phenomena break out without warning. Use "
          "scavenging, barter, trust and betrayal over scarce goods as concrete, checkable goals and obstacles tied "
          "to places, alongside the characters' personal mysteries.")

# How this world works, in sentences (docs/world-facts-design.md). Nothing here is matched by code.
facts = [
    'Food and clean water are hard come by: there is none to hand unless someone holds it or a search turns it up.',
    'A can of food is a meal for one, or half a meal each for two who share it. A bottle of water quenches one, or takes the edge off the thirst of two who share it.',
    'The fountain in the park runs: a drink there eases thirst well, and about one time in five the water tastes of metal and cramps follow.',
    'Scavenging a shop, a cafe or a trash bin takes about half an hour and may turn up a can of food or a bottle of water (each about one time in three), now and then batteries or a knife. A place gives up about three finds before it is picked clean.',
    'Sleep eases tiredness fully and takes about five hours. Half an hour on a bench or under a bus shelter, or twenty minutes indoors, eases it a little.',
    'A wound cleaned and bound is eased a little; it does not heal in a day.',
]
# What threatens people here, in sentences (docs/dangers-design.md): the play brings one on, and the world says what comes of it
dangers = [
    'At night gangs in scorpion chains roam the streets, the plazas and the building sites; they leave alone anyone indoors or under cover. Within half an hour of sighting someone they close in. They can be bought off about half the time, more surely with food, water or batteries to give; fought off seldom, a little more often with a knife; run from about half the time; hidden from a little less often. Whoever they catch is shoved and beaten and left.',
    'At any hour a strange light may split the hazy sky over the streets and the park. Within twenty minutes the ground heaves and glass rains from the buildings, unless the light fades as it came. Anyone indoors is safe from it; taking cover saves a person more often than not, and running about half the time. Those caught are cut and bruised, not badly.',
]
# What no deed or word gets past
held = [
]

# Stage business (sim/survival.py): scavenged goods, barter, shelter, gangs and phenomena.
affordances = {
    'needs': {
        'hunger': {                   'initial': 0.45, 'per_hour': 0.04, 'neglect_harm': 0.05,
                   'words': ('fed', 'hungry', 'very hungry', 'starving'),
                   'percepts': {0.5: "Your stomach knots; you have not eaten properly in a long time.",
                                0.8: "Hunger makes your hands shake and your thoughts slow."}},
        'thirst': {                   'initial': 0.35, 'per_hour': 0.06, 'neglect_harm': 0.05,
                   'words': ('not thirsty', 'thirsty', 'very thirsty', 'parched'),
                   'percepts': {0.5: "Your throat is dry and gritty with the city's dust.",
                                0.8: "Thirst pounds behind your eyes; you need water now."}},
        'fatigue': {                    'initial': 0.3, 'per_hour': 0.04, 'shelter_factor': 0.3,
                    'words': ('alert', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "Weariness drags at you; the streets seem longer.",
                                 0.8: "You are exhausted; you need somewhere safe to sleep."}},
    },
    'harm': {             'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')},
    'travel_need': 'fatigue',
    'tile_minutes': 3,
}
