"""A city of offices and apartments, a cafe and a hotel bar: the places are named.

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
    'Office': {'description': 'An office with a desk, client files and contracts', 'radius': 6, 'enclosed': True, 'shelter': True},
    'University_Office': {'description': "A professor's office at the university", 'radius': 6, 'enclosed': True, 'shelter': True},
    'Apartment': {'description': 'A city apartment', 'radius': 7, 'enclosed': True, 'shelter': True},
    'Cafe': {'description': 'A cafe: neutral ground for a meeting', 'radius': 7, 'enclosed': True, 'shelter': True},
    'Restaurant': {'description': 'A hotel bar and restaurant, busy in the evening', 'radius': 10, 'enclosed': True, 'shelter': True},
}

layout = {
    'regions': [
        {'kind': 'Street', 'radius': 2500, 'pace': 75, 'cover': 'open', 'whole': True},
        {'kind': 'Building', 'radius': (40, 70), 'pace': 75, 'cover': 'open', 'count': 12, 'from_start': (150, 2000), 'blocks_view': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 800, 'apart': 100},
    'places': [
        {'kind': 'Office', 'count': 4, 'on': {'Street': 1.0}, 'names': ['HuOffice', 'XueOffice', 'XiaoyuOffice', 'YanghoOffice']},
        {'kind': 'University_Office', 'count': 1, 'on': {'Street': 1.0}, 'names': ['UniversityOffice']},
        {'kind': 'Apartment', 'count': 3, 'on': {'Street': 1.0}, 'names': ['HuDingApartment', 'XueApartment', 'QiuApartment']},
        {'kind': 'Cafe', 'count': 1, 'on': {'Street': 1.0}, 'names': ['Cafe']},
        {'kind': 'Restaurant', 'count': 1, 'on': {'Street': 1.0}, 'names': ['HotelBar']},
    ],
}


# how a place the author has named is said on the page
said = {'HuDingApartment': "Hu and Ding's apartment", 'XueApartment': "Xue's apartment", 'QiuApartment': "Qiu's apartment",
        'HuOffice': "Hu's office", 'XueOffice': "Xue's office", 'XiaoyuOffice': "Xiaoyu's office", 'YanghoOffice': "Yangho's office",
        'UniversityOffice': 'the university office', 'Cafe': 'the cafe', 'HotelBar': 'the hotel bar'}


stakes = ("In this city the stakes are clients, careers, marriages and reputations. What is settled is what is on "
          "paper or on a phone: a contract drafted, handed over or signed; a message sent and later found. Bodily needs "
          "are ordinary (a meal, a drink, rest). Use concrete, checkable goals and obstacles tied to the offices, the "
          "apartments, the cafe and the hotel bar.")

# How this world works, in sentences: what the world rules by, and what the cast and the planner are shown
# (docs/world-facts-design.md). Nothing here is matched by code.
facts = [
    "An insurance contract runs for a year, and a company is insured by one agent at a time. In its last week it is up "
    "for renewal: the client may sign the renewal with the agent who holds it, or a new contract with another agent, and "
    "whichever is signed first settles it for the coming year. A contract not up for renewal cannot be taken.",
    "An agent drafts a contract at their own office; it takes about twenty minutes. A contract is signed on the paper, "
    "by the client, in person.",
    "Going through a client's file at one's own office takes about twenty minutes and sometimes turns up a gap in a rival's terms.",
    "Everyone carries a phone. A message reaches its person wherever they are and stays on their phone. What is on a "
    "phone is seen only by its owner, or by someone who picks it up and looks.",
    "A drink at the cafe or the hotel bar costs about 30 yuan, a meal about 60, a small gift about 100. Whoever orders pays.",
    "A meal eases hunger well. A drink or a short rest takes the edge off tiredness; an evening's rest at home eases it well.",
    "A morning's teaching at the university pays about 200 yuan.",
]
# What no deed or word gets past
held = [
    "A company is insured by one agent at a time.",
    "A client signs once in a renewal period, and that settles the year.",
    "A contract not in its last week cannot be taken.",
    "Money comes into the play only as a wage or a payment these facts name.",
]

# Bodies (sim/survival.py): the needs this world keeps count of and how they rise; money. No hazards.
affordances = {
    'needs': {
        'hunger': {'initial': 0.3, 'per_hour': 0.04,
                   'words': ('not hungry', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.6: "You have not eaten for hours.", 0.85: "Hunger is making it hard to concentrate."}},
        'fatigue': {'initial': 0.2, 'per_hour': 0.03,
                    'words': ('alert', 'a little tired', 'tired', 'exhausted'),
                    'percepts': {0.6: "It has been a long day and you are tired.",
                                 0.85: "You are exhausted; your judgment is slipping."}},
    },
    'harm': {'name': 'injury', 'words': ('fine', 'sore', 'hurt', 'badly hurt')},
    'currency': 'yuan', 'initial_money': 2000,
    'private_items': ['message from'],   # what is on a phone is seen only by looking through it
    'tile_minutes': 1,
    'hazards': [],
}
