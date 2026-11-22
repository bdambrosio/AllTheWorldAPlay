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

# Stage business (sim/survival.py): making and selling work, the cafe, the boat, a squall.
affordances = {
    'needs': {
        'hunger': {'initial': 0.3, 'per_hour': 0.04,
                   'words': ('not hungry', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.5: "You are getting hungry.", 0.8: "You are very hungry; it is hard to focus."}},
        'fatigue': {'initial': 0.3, 'per_hour': 0.03, 'shelter_factor': 0.5,
                    'words': ('fresh', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "You are tiring.", 0.8: "You are exhausted."}},
    },
    'harm': {'name': 'injury', 'words': ('fine', 'bruised', 'hurt', 'badly hurt')},
    'travel_need': 'fatigue',
    'currency': 'dollars', 'initial_money': 30,
    'tile_minutes': 2,
    'type_nouns': {'Cafe': ['cafe', 'café', 'coffee'], 'Studio': ['studio', 'easel', 'canvas'], 'Workshop': ['workshop', 'bench'],
                   'Gallery': ['gallery'], 'Boat': ['boat', 'hull', 'dinghy'], 'Tools': ['tools', 'toolbox'],
                   'Bench': ['bench'], 'Bridge': ['bridge']},
    'actions': [
        {'name': 'buy coffee', 'verbs': ['buy coffee', 'order coffee', 'drink coffee'], 'named_verbs': ['buy', 'order', 'drink'],
         'resource_types': ['Cafe'], 'cost': 4, 'effects': {'fatigue': -0.2}, 'minutes': 15, 'text': 'buys and drinks a coffee at {resource}'},
        {'name': 'eat a meal', 'verbs': ['eat', 'order lunch', 'order food'], 'resource_types': ['Cafe'], 'cost': 12,
         'effects': {'hunger': -0.6}, 'minutes': 30, 'text': 'eats a meal at {resource}'},
        {'name': 'sit and watch the harbour', 'verbs': ['sit', 'rest', 'watch the harbour', 'sketch'], 'resource_types': ['Bench'],
         'effects': {'fatigue': -0.15}, 'minutes': 20, 'text': 'sits at {resource}, watching the harbour'},
        {'name': 'paint', 'category': 'work', 'verbs': ['paint', 'finish the canvas', 'work on the painting'], 'named_verbs': ['work', 'create'],
         'resource_types': ['Studio'], 'effects': {'fatigue': 0.2}, 'gain': 'finished painting', 'minutes': 180,
         'text': 'paints at {resource} until a canvas is finished'},
        {'name': 'make a carving', 'category': 'work', 'verbs': ['carve', 'build', 'make'], 'named_verbs': ['work'],
         'resource_types': ['Workshop'], 'effects': {'fatigue': 0.2}, 'gain': 'carving', 'minutes': 180,
         'text': 'works at {resource} until a carving is finished'},
        {'name': 'sell a painting', 'category': 'work', 'verbs': ['sell', 'hang', 'consign'], 'resource_types': ['Gallery'],
         'requires': 'finished painting', 'consumes': 'finished painting', 'earn': 150, 'minutes': 30,
         'text': 'sells a painting through {resource}'},
        {'name': 'sell a carving', 'category': 'work', 'verbs': ['sell', 'consign'], 'resource_types': ['Gallery'],
         'requires': 'carving', 'consumes': 'carving', 'earn': 90, 'minutes': 30, 'text': 'sells a carving through {resource}'},
        {'name': 'borrow tools', 'verbs': ['borrow', 'take', 'pick up', 'grab'], 'named_verbs': [], 'resource_types': ['Tools'],
         'gain': 'tools', 'minutes': 5, 'deplete': 2, 'text': 'borrows tools from {resource}'},
        {'name': 'mend the boat', 'category': 'work', 'verbs': ['mend', 'repair', 'caulk', 'patch', 'fix'], 'resource_types': ['Boat'],
         'requires': 'tools', 'requires_property': {'mended': None}, 'sets_property': {'mended': True},
         'effects': {'fatigue': 0.25}, 'minutes': 120, 'text': 'mends {resource}'},
        {'name': 'go sailing', 'verbs': ['sail', 'go sailing', 'row out', 'take the boat out'], 'resource_types': ['Boat'],
         'requires_property': {'mended': True}, 'effects': {'fatigue': -0.1}, 'minutes': 90, 'text': 'takes {resource} out on the water'},
        {'name': 'shelter', 'verbs': ['shelter', 'take cover', 'go inside', 'duck inside'], 'resource_types': ['Cafe', 'Gallery', 'Studio', 'Workshop'],
         'minutes': 10, 'sets': 'sheltered', 'text': 'goes inside {resource}'},
        {'name': 'give a painting', 'category': 'social', 'target': 'character', 'verbs': ['give', 'gift', 'present', 'offer'],
         'transfer': 'finished painting', 'minutes': 5, 'text': 'gives {target} a painting'},
        {'name': 'lend tools', 'category': 'social', 'target': 'character', 'verbs': ['lend', 'give', 'hand', 'pass'],
         'transfer': 'tools', 'minutes': 2, 'text': 'lends {target} the tools'},
        {'name': 'pay', 'category': 'social', 'target': 'character', 'verbs': ['pay', 'lend money', 'hand cash', 'settle up'],
         'transfer_money': 20, 'minutes': 2, 'text': 'pays {target} twenty dollars'},
    ],
    'hazards': [
        {'name': 'squall', 'description': 'a sudden squall racing in off the sea', 'when': 'any',
         'terrain': ['Harbor', 'Road', 'Garden'], 'chance_per_hour': 0.03, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 20, 'harm': 0.1, 'fail_text': 'the squall soaks them and slams a shutter',
         'retreat_text': 'The squall blows itself out; the harbour glitters again.',
         'strike_text': 'The squall hits: rain sheets down and loose gear flies across the quay.',
         'responses': {'take cover': {'verbs': ['take cover', 'shelter', 'duck inside', 'run inside'], 'success': 0.7},
                       'secure the boat': {'verbs': ['secure', 'tie down', 'lash', 'make fast'], 'success': 0.5}}},
    ],
}
