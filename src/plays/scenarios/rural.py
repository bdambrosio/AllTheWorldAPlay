"""A peasant village and its farms: a market square, a mill, a smithy, wells, fields.

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
    'Market': {'description': 'The village market square', 'radius': 25},
    'Mill': {'description': 'A grain mill', 'radius': 10, 'enclosed': True, 'shelter': True, 'owner': True},
    'Blacksmith': {'description': 'A blacksmith workshop', 'radius': 8, 'enclosed': True, 'open': True, 'shelter': True, 'owner': True},
    'Well': {'description': 'A deep well for water', 'radius': 2, 'owner': True},
    'Farm': {'description': 'A farmhouse with barn', 'radius': 20, 'enclosed': True, 'shelter': True, 'owner': True},
    'Strip': {'description': 'A strip of wheat land', 'radius': 60},
}

layout = {
    'regions': [
        {'kind': 'Grassland', 'radius': 3000, 'pace': 70, 'cover': 'open', 'whole': True},
        {'kind': 'Field', 'radius': (200, 350), 'pace': 60, 'cover': 'open', 'count': 6, 'from_start': (200, 1700)},
        {'kind': 'Forest', 'radius': 250, 'pace': 45, 'cover': 'trees', 'count': 2, 'from_start': (800, 2200)},
        {'kind': 'Hill', 'radius': 300, 'pace': 35, 'cover': 'open', 'count': 1, 'from_start': (1500, 2500), 'blocks_view': True, 'overlooks': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 1800, 'apart': 150},
    'places': [
        {'kind': 'Market', 'count': 1, 'from_start': (0, 0)},
        {'kind': 'Mill', 'count': 1, 'near': ('Market', 200, 600)},
        {'kind': 'Blacksmith', 'count': 1, 'near': ('Market', 20, 150), 'apart': 20},
        {'kind': 'Farm', 'count': 8, 'on': {'Field': 1.0}, 'from_start': (300, 1800), 'apart': 250, 'names': ['MarquartFarm', 'FouanFarm']},
        {'kind': 'Well', 'count': 5, 'on': {'Field': 1.0, 'Grassland': 1.0}, 'from_start': (50, 1500), 'apart': 60},
        {'kind': 'Strip', 'count': 0, 'names': ['MarquartUpperStrip', 'MarquartLowerStrip', 'FouanUpperStrip', 'FouanLowerStrip']},
    ],
}


stakes = ("In this peasant village, land, money, harvest and inheritance are what is at stake: who holds which "
          "farm and which strip of land, who works it, what the wheat fetches at market, who pays whom, and who will inherit. Heat, hard "
          "labour and thirst are real. Use these to make concrete, "
          "checkable goals and obstacles tied to the farms, the mill, the market and the family.")

# How this world works, in sentences (docs/world-facts-design.md). Nothing here is matched by code.
facts = [
    "Land is a farm, or a strip of wheat land beside it; each has its own deed and passes whole. It belongs to whoever "
    "its owner has signed the deed over to: the owner holds the deed and signs it over only in person, with the heir standing there.",
    "A farm with no owner may be taken into possession by pacing out its boundaries; it takes about half an hour.",
    "A family harvests the wheat on its own land: an afternoon's reaping, about three hours, brings in a sack of wheat "
    "and leaves the reaper tired and thirsty.",
    "A day's labour on another family's land, about three hours of an afternoon, is paid about 2 francs, and is as tiring.",
    "The miller grinds a sack of wheat into a sack of flour for about 1 franc; it takes about an hour.",
    "At the market a sack of wheat sells for about 6 francs and a sack of flour for about 9; a loaf of bread costs about 1 franc.",
    "The blacksmith mends tools for about 2 francs; it takes about an hour.",
    "Water drawn at a well eases thirst well. A loaf of bread is a meal. Rest indoors or in the shade eases tiredness a little.",
]
# What threatens people here, in sentences (docs/dangers-design.md): the play brings one on, and the world says what comes of it
dangers = [
    'On a summer day a black hailstorm may roll in over the fields and the grassland, half an hour from first sight. Anyone under a roof is safe from it, and someone who runs for shelter usually makes it; sheaves can be covered in time less than half the time. Hail flattens standing wheat and bruises whoever is caught in the open.',
]
held = [
    "Land passes only by its deed, signed over by its owner in person with the heir standing there, and passes whole.",
    "Money comes into the play only as a wage or a sale these facts name.",
]

# Bodies (sim/survival.py): the needs this world keeps count of and how they rise; money; its one danger.
affordances = {
    'needs': {   # late afternoon on a hot summer day
        'thirst': {'initial': 0.4, 'per_hour': 0.06, 'neglect_harm': 0.03,
                   'words': ('not thirsty', 'thirsty', 'very thirsty', 'parched'),
                   'percepts': {0.5: "The heat has dried your mouth; you want water from the well.",
                                0.8: "Your head swims in the heat; you need water."}},
        'hunger': {'initial': 0.3, 'per_hour': 0.03,
                   'words': ('fed', 'hungry', 'very hungry', 'famished'),
                   'percepts': {0.5: "You are hungry; supper is a long way off.",
                                0.8: "Hunger gnaws at you; you need bread."}},
        'fatigue': {'initial': 0.35, 'per_hour': 0.03, 'shelter_factor': 0.5,
                    'words': ('fresh', 'tired', 'worn out', 'spent'),
                    'percepts': {0.5: "Your back aches from the day's work.",
                                 0.8: "You are worn out; you can barely lift your arms."}},
    },
    'harm': {'name': 'injury', 'words': ('unhurt', 'bruised', 'hurt', 'badly hurt')},
    'travel_need': 'fatigue',
    'currency': 'francs', 'initial_money': 5,
    'tile_minutes': 4,
}
