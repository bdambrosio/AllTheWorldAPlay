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

# Stage business (sim/survival.py): needs rise with simulation time, resources have affordances,
# and natural hazards appear. Effects are mechanical; the LLM only narrates them.
affordances = {
    'needs': {   # where they start: thirsty from the night
        'thirst': {'eased': [{'by': 'drinking at a spring', 'at': ['Spring'], 'amount': 0.8, 'minutes': 5}],
                   'initial': 0.45, 'per_hour': 0.08, 'neglect_harm': 0.05,
                   'words': ('not thirsty', 'thirsty', 'very thirsty', 'desperately thirsty'),
                   'percepts': {0.5: "Your mouth is dry; you have not had water for a while.",
                                0.8: "Thirst is becoming hard to ignore; your head aches and your lips are cracked."}},
        'hunger': {'eased': [{'by': 'eating from a berry bush or an apple tree', 'at': ['Berries', 'Apple_Tree'], 'amount': 0.4, 'minutes': 10, 'stock': 3},
                             {'by': 'eating mushrooms where they grow', 'at': ['Mushrooms'], 'amount': 0.3, 'minutes': 10, 'stock': 2,
                              'risk': {'chance': 0.3, 'effects': {'injury': 0.3}, 'text': 'Soon after, cramps and nausea hit hard.'}},
                             {'by': 'eating berries or an apple that is carried', 'uses': ['berries', 'apple'], 'amount': 0.3, 'minutes': 5}],
                   'initial': 0.3, 'per_hour': 0.04, 'neglect_harm': 0.05,
                   'words': ('not hungry', 'hungry', 'very hungry', 'starving'),
                   'percepts': {0.5: "Your stomach is empty and growling.",
                                0.8: "Hunger makes you light-headed and weak."}},
        'fatigue': {'eased': [{'by': 'sleeping', 'amount': 1.0, 'minutes': 240},
                              {'by': 'resting under shelter', 'at': ['Cave', 'Hut', 'Thicket'], 'amount': 0.3, 'minutes': 30},
                              {'by': 'resting a while', 'amount': 0.15, 'minutes': 20}],
                    'initial': 0.25, 'per_hour': 0.05, 'shelter_factor': 0.3,
                    'words': ('rested', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "Your legs are heavy; you are getting tired.",
                                 0.8: "You are exhausted; every step is an effort and you need rest."}},
    },
    'harm': {'eased': [{'by': 'tending the wound', 'amount': 0.2, 'minutes': 15, 'floor': 0.3}],
             'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')},
    'travel_need': 'fatigue',
    'tile_minutes': 2,
    'type_nouns': {'Spring': ['water', 'stream', 'pool', 'spring'], 'Berries': ['berry', 'berries'],
                   'Apple_Tree': ['apple', 'apples'], 'Mushrooms': ['mushroom', 'mushrooms', 'fungus', 'fungi'],
                   'Fallen_Branch': ['branch', 'branches', 'stick', 'wood', 'log'],
                   'Thicket': ['thicket', 'bush', 'undergrowth'], 'Cave': ['cave', 'cavern'], 'Hut': ['hut', 'cabin', 'shack']},
    # what this world rules on: exchanges and odds (sim/survival.py rules). Everything else a person does is told in
    # words and the resolver says what comes of it (sim/resolver.py).
    'actions': [   # a deed's action is chosen by the model from these labels (survival.classify_deed); verbs serve spoken claims and threat responses
        # (No 'build a fire' here. How a fire is made, fed, banked and put out is not written ahead: the
        # keeper models it when someone first sets about one, sim/keeper.py. The wolf is still 'blocked_by'
        # fire below: that is the name a model's effect is matched to.)
        {'name': 'search', 'category': 'explore', 'verbs': ['search', 'rummage', 'investigate'], 'named_verbs': ['explore', 'examine', 'inspect'],
         'resource_types': ['Hut', 'Cave'], 'finds': [{'item': 'knife', 'chance': 0.25}, {'item': 'rope', 'chance': 0.25}],
         'minutes': 20, 'deplete': 2, 'text': 'searches {resource} carefully'},
        # an apple can be picked, carried, eaten later and handed over, like berries (a play
        # planned 'she picks an apple and holds it out to him': the tree could only be eaten at)
    ],
    'hazards': [
        {'name': 'wolf', 'description': 'a gaunt grey wolf', 'when': 'night', 'terrain': ['Forest', 'Clearing', 'Meadow'],
         'chance_per_hour': 0.35, 'blocked_by': ['sheltered', 'fire'], 'minutes_to_strike': 40, 'harm': 0.4,
         'retreat_text': 'The wolf slinks back into the dark trees.',
         'strike_text': 'The wolf darts in and bites before vanishing into the dark.', 'fail_text': 'the wolf is not deterred and lunges',
         'responses': {'defend': {'verbs': ['defend', 'fight', 'club', 'strike', 'hit', 'swing', 'throw', 'shout', 'yell', 'wave', 'scare'],
                                  'success': 0.45, 'bonus': {'branch': 0.3, 'fire': 0.4}},
                       'flee': {'verbs': ['flee', 'run', 'retreat', 'escape', 'back away'], 'success': 0.5},
                       'hide': {'verbs': ['hide', 'climb', 'freeze', 'stay still'], 'success': 0.4}}},
        {'name': 'boar', 'description': 'a bristling wild boar', 'when': 'day', 'terrain': ['Forest'],
         'chance_per_hour': 0.06, 'blocked_by': ['sheltered'], 'minutes_to_strike': 20, 'harm': 0.3,
         'retreat_text': 'The boar snorts and crashes away through the undergrowth.',
         'strike_text': 'The boar charges, its tusks catching a leg before it crashes away.', 'fail_text': 'the boar is not deterred and charges',
         'responses': {'defend': {'verbs': ['defend', 'fight', 'club', 'strike', 'hit', 'throw', 'shout', 'yell', 'scare'],
                                  'success': 0.4, 'bonus': {'branch': 0.3}},
                       'flee': {'verbs': ['flee', 'run', 'retreat', 'escape', 'back away'], 'success': 0.6},
                       'hide': {'verbs': ['hide', 'climb', 'freeze', 'stay still'], 'success': 0.6}}},
    ],
}
