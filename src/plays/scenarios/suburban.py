"""A suburb: a house and what is in it, a car, the street, and an office downtown.

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
    'House': {'description': 'The house Alex and Susan live in', 'radius': 8, 'enclosed': True, 'shelter': True},
    'Bed': {'description': 'A comfortable bed', 'radius': 1.5},
    'Shower': {'description': 'A bathroom with shower', 'radius': 1.5},
    'Closet': {'description': 'A closet with clothes', 'radius': 1.5},
    'Coffee_Maker': {'description': 'An automatic coffee maker', 'radius': 1.5},
    'Refrigerator': {'description': 'A refrigerator with breakfast things', 'radius': 1.5},
    'Car': {'description': 'A parked car', 'radius': 3, 'vehicle': True, 'pace': 500, 'by': 'drive'},
    'Mailbox': {'description': 'A mailbox', 'radius': 1.5},
    'Bus_Stop': {'description': 'A bus stop', 'radius': 3},
    'Office': {'description': 'A downtown office building', 'radius': 15, 'enclosed': True, 'shelter': True},
}

layout = {
    'regions': [
        {'kind': 'Street', 'radius': 16000, 'pace': 75, 'cover': 'open', 'whole': True},
        {'kind': 'Yard', 'radius': 25, 'pace': 70, 'cover': 'open', 'count': 1, 'from_start': (0, 0)},
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
        {'kind': 'Refrigerator', 'count': 1, 'inside': 'House', 'apart': 3},
        {'kind': 'Car', 'count': 1, 'near': ('House', 3, 10)},
        {'kind': 'Mailbox', 'count': 1, 'near': ('House', 8, 14)},
        {'kind': 'Bus_Stop', 'count': 2, 'from_start': (150, 400)},
        {'kind': 'Office', 'count': 1, 'on': {'City': 1.0}},
    ],
}


stakes = ("In this everyday suburban world the stakes are time, money, work and relationships: being late, a "
          "missed commitment, a bill, a tense conversation, a decision that cannot wait. Bodily needs are ordinary "
          "(coffee, food, sleep, getting ready), not survival. Use concrete, checkable goals and obstacles tied to the "
          "house, the commute and the office.")

# How this world works, in sentences (docs/world-facts-design.md). Nothing here is matched by code.
facts = [
    'Breakfast from the refrigerator takes about fifteen minutes and eases hunger well.',
    "A cup of coffee takes the edge off tiredness for the one who drinks it; a shower does a little less; four hours' sleep in bed eases it well.",
    'Getting dressed at the closet takes about ten minutes, a shower about fifteen. A person is dressed for the day once they have put on clothes from the closet.',
    'The mail comes to the mailbox: checking it takes a few minutes, and there is sometimes an overdue bill in it (about one day in three) or a personal letter (about one in four).',
    "A half-day's work at the office, about four hours, pays about 120 dollars and is tiring.",
    'An interview at the office takes about three quarters of an hour.',
]
# What no deed or word gets past
held = [
    'Nobody works at the office, or sits an interview there, who is not dressed for the day.',
    'Money comes into the play only as a wage these facts name.',
]

# Stage business (sim/survival.py): morning routine, mail, work and money. No hazards.
affordances = {
    'needs': {   # early morning
        'hunger': {                   'initial': 0.4, 'per_hour': 0.05,
                   'words': ('not hungry', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.5: "You are hungry; you have not eaten yet today.",
                                0.8: "Hunger is making it hard to concentrate."}},
        'fatigue': {                    'initial': 0.45, 'per_hour': 0.04,
                    'words': ('alert', 'groggy', 'tired', 'exhausted'),
                    'percepts': {0.5: "You feel groggy; coffee would help.",
                                 0.8: "You are exhausted and your thoughts keep drifting."}},
    },
    'harm': {'name': 'injury', 'words': ('fine', 'sore', 'hurt', 'badly hurt')},
    'currency': 'dollars', 'initial_money': 40,
    'tile_minutes': 1,
    'hazards': [],
}
