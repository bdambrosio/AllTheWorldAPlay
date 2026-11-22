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
    'Cafe': {'description': 'A cafe: neutral ground for a meeting', 'radius': 7, 'enclosed': True, 'open': True, 'shelter': True},
    'Restaurant': {'description': 'A hotel bar and restaurant, busy in the evening', 'radius': 10, 'enclosed': True, 'shelter': True},
}

layout = {
    'regions': [
        {'kind': 'Street', 'radius': 2500, 'pace': 75, 'cover': 'open', 'whole': True},
        {'kind': 'Building', 'radius': (40, 70), 'pace': 75, 'cover': 'open', 'count': 12, 'from_start': (150, 2000), 'blocks_view': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 2000, 'apart': 200},
    'places': [
        {'kind': 'Office', 'count': 4, 'on': {'Street': 1.0}, 'names': ['HuOffice', 'XueOffice', 'XiaoyuOffice', 'YanghoOffice']},
        {'kind': 'University_Office', 'count': 1, 'on': {'Street': 1.0}, 'names': ['UniversityOffice']},
        {'kind': 'Apartment', 'count': 3, 'on': {'Street': 1.0}, 'names': ['HuDingApartment', 'XueApartment', 'QiuApartment']},
        {'kind': 'Cafe', 'count': 1, 'on': {'Street': 1.0}, 'names': ['Cafe']},
        {'kind': 'Restaurant', 'count': 1, 'on': {'Street': 1.0}, 'names': ['HotelBar']},
    ],
}


stakes = ("In this city the stakes are clients, careers, marriages and reputations. What is settled is what is on "
          "paper or on a phone: a contract drafted, handed over or signed; a message sent and later found. A client "
          "belongs to whoever holds their signed contract. Bodily needs are ordinary (a meal, a drink, rest). "
          "Use concrete, checkable goals and obstacles tied to the offices, the apartments, the cafe and the hotel bar.")

# Stage business (sim/survival.py): contracts, messages, money, meals. No hazards.
affordances = {
    'needs': {
        'hunger': {'initial': 0.3, 'per_hour': 0.04,
                   'words': ('not hungry', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.6: "You have not eaten for hours.", 0.85: "Hunger is making it hard to concentrate."}},
        'fatigue': {'eased': [{'by': 'resting at home', 'at': ['Apartment'], 'amount': 0.5, 'minutes': 60}],
                    'initial': 0.2, 'per_hour': 0.03,
                    'words': ('alert', 'a little tired', 'tired', 'exhausted'),
                    'percepts': {0.6: "It has been a long day and you are tired.",
                                 0.85: "You are exhausted; your judgment is slipping."}},
    },
    'harm': {'name': 'injury', 'words': ('fine', 'sore', 'hurt', 'badly hurt')},
    'currency': 'yuan', 'initial_money': 2000,
    'private_items': ['message from'],   # what is on a phone is seen only by looking through it
    'tile_minutes': 1,
    'type_nouns': {'Office': ['office', 'desk', 'firm'], 'University_Office': ['university', 'campus', 'department'],
                   'Apartment': ['apartment', 'home', 'flat'], 'Cafe': ['cafe', 'café', 'coffee shop'],
                   'Restaurant': ['restaurant', 'hotel bar', 'bar'], 'Taxi_Stand': ['taxi stand', 'taxi']},
    # what this world rules on: exchanges and odds (sim/survival.py rules). Everything else a person does is told in
    # words and the resolver says what comes of it (sim/resolver.py).
    'actions': [
        # contracts: drafted at an office, handed to a client, signed by whoever holds one
        {'name': 'draft a contract', 'category': 'work', 'verbs': ['draft', 'draw up', 'write up', 'prepare the contract'],
         'named_verbs': ['prepare', 'print'], 'resource_types': ['Office'], 'gain': "{actor}'s contract",
         'effects': {'fatigue': 0.05}, 'minutes': 20, 'text': 'drafts a contract at {resource}'},
        {'name': 'sign the contract', 'category': 'work', 'verbs': ['sign', 'put my name', 'put his name', 'put her name'],
         'requires': '*contract', 'consumes': '*contract', 'gain': '{item}, signed', 'minutes': 2,
         'text': 'signs {item}'},
        {'name': 'review the client file', 'category': 'explore', 'verbs': ['review the file', 'go through the file', 'study the file'],
         'named_verbs': ['review', 'read', 'study', 'check'], 'resource_types': ['Office'],
         'finds': [{'item': "a gap in the rival's terms", 'chance': 0.5}], 'effects': {'fatigue': 0.05},
         'minutes': 20, 'deplete': 2, 'text': 'goes through the client file at {resource}'},
        # phones: a message reaches anyone, and stays on their phone to be found
        {'name': 'check their phone', 'category': 'explore', 'target': 'character',
         'verbs': ['check his phone', 'check her phone', 'look at his phone', 'look at her phone', 'go through his phone',
                   'go through her phone', 'read his messages', 'read her messages', 'pick up his phone', 'pick up her phone',
                   'scroll through'],
         'reveals': 'message from', 'minutes': 2, 'text': "looks through {target}'s phone"},
        {'name': 'send a message', 'category': 'social', 'target': 'character', 'remote': True,
         'verbs': ['text', 'send a message', 'send a text', 'message him', 'message her'],
         'target_gain': 'message from {actor}: {said}',
         'minutes': 1, 'text': 'sends {target} a message'},
        # money, gifts, meals
        {'name': 'pay', 'category': 'social', 'target': 'character', 'verbs': ['pay', 'give money', 'hand cash', 'transfer money'],
         'transfer_money': 500, 'minutes': 2, 'text': 'pays {target} five hundred yuan'},
        {'name': 'buy a gift', 'verbs': ['buy a gift', 'pick out a gift', 'buy a present'], 'resource_types': ['Cafe', 'Restaurant'],
         'cost': 100, 'gain': 'gift', 'minutes': 5, 'text': 'buys a small gift at {resource}'},
        {'name': 'buy a meal', 'verbs': ['eat', 'order food', 'order lunch', 'order dinner', 'have lunch', 'have dinner'],
         'named_verbs': ['order'], 'resource_types': ['Cafe', 'Restaurant'], 'cost': 60,
         'effects': {'hunger': -0.6, 'fatigue': -0.05}, 'minutes': 30, 'text': 'has a meal at {resource}'},
        {'name': 'buy a drink', 'verbs': ['order a drink', 'order coffee', 'order tea', 'have a drink', 'have a coffee'],
         'named_verbs': ['drink', 'sip'], 'resource_types': ['Cafe', 'Restaurant'], 'cost': 30,
         'effects': {'fatigue': -0.1}, 'minutes': 10, 'text': 'has a drink at {resource}'},
        {'name': 'stand a round', 'category': 'social', 'target': 'character',
         'verbs': ['treat', 'buy him a drink', 'buy her a drink', 'stand a round', 'pick up the bill'], 'cost': 60,
         'effects': {'fatigue': -0.1}, 'target_effects': {'fatigue': -0.1}, 'minutes': 10,
         'text': 'buys {target} a drink'},
        # home and work
        {'name': 'teach', 'category': 'work', 'verbs': ['teach', 'lecture', 'hold office hours', 'grade papers'],
         'resource_types': ['University_Office'], 'effects': {'fatigue': 0.2}, 'earn': 200, 'minutes': 120,
         'text': 'teaches and sees students at {resource}'},
    ],
    'hazards': [],
}
