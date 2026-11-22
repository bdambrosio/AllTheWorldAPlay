"""A temperate forest with no buildings but a hut, no roads.

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
    'Berries': {'description': 'A bush with edible berries', 'radius': 2},
    'Mushrooms': {'description': 'A patch of mushrooms', 'radius': 3},
    'Fallen_Branch': {'description': 'A large fallen tree branch', 'radius': 3},
    'Apple_Tree': {'description': 'A large apple tree with apples', 'radius': 5},
    'Spring': {'description': 'A natural spring of fresh water', 'radius': 4},
    'Thicket': {'description': 'A dense thicket of vegetation', 'radius': 6, 'shelter': True, 'blocks_view': True, 'hides': True},
    'Cave': {'description': 'A small cave in the rocks', 'radius': 5, 'enclosed': True, 'open': True, 'shelter': True, 'owner': True},
    'Hut': {'description': 'A small hut for shelter', 'radius': 4, 'enclosed': True, 'shelter': True, 'owner': True},
}

layout = {
    'regions': [
        {'kind': 'Forest', 'radius': 1200, 'pace': 50, 'cover': 'trees', 'whole': True},
        {'kind': 'Clearing', 'radius': (40, 80), 'pace': 75, 'cover': 'open', 'count': 6},
        {'kind': 'Meadow', 'radius': (100, 150), 'pace': 75, 'cover': 'open', 'count': 2},
        {'kind': 'Rise', 'radius': 150, 'pace': 35, 'cover': 'open', 'count': 1, 'from_start': (500, 900), 'blocks_view': True, 'overlooks': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 700, 'apart': 40},
    'places': [
        {'kind': 'Berries', 'count': 15, 'on': {'Clearing': 2.0, 'Meadow': 1.0, 'Forest': 0.5}},
        {'kind': 'Mushrooms', 'count': 10, 'on': {'Forest': 1.0}},
        {'kind': 'Fallen_Branch', 'count': 24, 'on': {'Forest': 1.0, 'Clearing': 0.5}},
        {'kind': 'Apple_Tree', 'count': 16, 'on': {'Clearing': 1.0, 'Forest': 0.6}},
        {'kind': 'Spring', 'count': 3, 'on': {'Forest': 1.0}, 'from_start': (150, 700)},
        {'kind': 'Thicket', 'count': 12, 'on': {'Forest': 2.0, 'Clearing': 1.0}},
        {'kind': 'Cave', 'count': 1, 'on': {'Rise': 1.0}},
        {'kind': 'Hut', 'count': 1, 'foot_of': 'Rise'},
    ],
}


stakes = ("Physical needs (water, food, rest, shelter) and natural hazards (wolves at night, wild boar) are real "
          "stakes in this wilderness: use them to create concrete, checkable goals and obstacles tied to places and "
          "things, and keep any mystery grounded in them.")

# How this world works, in sentences (docs/world-facts-design.md). Nothing here is matched by code.
facts = [
    "A spring gives clean water: a drink there eases thirst fully.",
    "A berry bush or an apple tree gives about three helpings before it is bare. A helping eaten where it grows eases "
    "hunger well; berries or an apple carried off and eaten later ease it a little.",
    "Mushrooms grow in patches of about two helpings. A helping eases hunger a little, and they are sometimes bad: "
    "about one time in three, cramps and nausea follow within the hour.",
    "Sleep eases tiredness fully and takes about four hours. Half an hour's rest under shelter (a cave, a hut, a thicket) "
    "eases it well; twenty minutes' rest in the open, a little.",
    "A wound cleaned and bound is eased a little; it does not heal in a day.",
    "The hut and the cave have been used by someone before: a careful search of either takes about twenty minutes and now "
    "and then turns up a knife or a length of rope.",
]
# What threatens people here, in sentences (docs/dangers-design.md): the play brings one on, and the world says what comes of it
dangers = [
    'Wolves hunt this forest at night, under the trees and across the clearings and meadows. A wolf does not come at people who are under shelter or beside a burning fire. It circles for the better part of an hour before it comes in. Someone who stands and fights or shouts it down drives it off a little less than half the time with bare hands, most times with a stout branch, and nearly always with fire; running or hiding works about half the time. A wolf that gets in bites deep and is gone.',
    "By day a wild boar may come at someone in the thick of the forest, within a quarter of an hour of being come upon; it leaves alone anyone under shelter. Standing against it works less than half the time, more often with a branch; running or getting out of its way works more often than not. Its tusks leave a wound, a lesser one than a wolf's bite.",
]
held = []

# Bodies (sim/survival.py): the needs this world keeps count of and how they rise; its dangers.
affordances = {
    'needs': {   # where they start: thirsty from the night
        'thirst': {'initial': 0.45, 'per_hour': 0.08, 'neglect_harm': 0.05,
                   'words': ('not thirsty', 'thirsty', 'very thirsty', 'desperately thirsty'),
                   'percepts': {0.5: "Your mouth is dry; you have not had water for a while.",
                                0.8: "Thirst is becoming hard to ignore; your head aches and your lips are cracked."}},
        'hunger': {'initial': 0.3, 'per_hour': 0.04, 'neglect_harm': 0.05,
                   'words': ('not hungry', 'hungry', 'very hungry', 'starving'),
                   'percepts': {0.5: "Your stomach is empty and growling.",
                                0.8: "Hunger makes you light-headed and weak."}},
        'fatigue': {'initial': 0.25, 'per_hour': 0.05, 'shelter_factor': 0.3,
                    'words': ('rested', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "Your legs are heavy; you are getting tired.",
                                 0.8: "You are exhausted; every step is an effort and you need rest."}},
    },
    'harm': {'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')},
    'travel_need': 'fatigue',
    'tile_minutes': 2,
}
