"""A small lunar base: the habitat ring, the service gantry over the vacuum line, a crater, the solar array.

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
    'Airlock': {'description': 'Primary EVA airlock into the habitat ring', 'radius': 3, 'enclosed': True, 'shelter': True},
    'Gantry': {'description': 'Metal service gantry overlooking the vacuum‑line weld', 'radius': 6},
    'Evastation': {'description': 'Suit stow & recharge alcove', 'radius': 2},
    'Repairpanel': {'description': 'External repair junction box for vacuum line', 'radius': 1.5},
    'Habitmodule': {'description': 'Pressurised living/work module', 'radius': 5, 'enclosed': True, 'shelter': True},
    'Oxygentank': {'description': 'Exposed backup O₂ bundle', 'radius': 2},
    'Park': {'description': 'Hydroponic greenhouse dome', 'radius': 8, 'enclosed': True, 'shelter': True},
}

layout = {
    'regions': [
        {'kind': 'Regolith', 'radius': 400, 'pace': 40, 'cover': 'open', 'whole': True},
        {'kind': 'Habitat', 'radius': 40, 'pace': 60, 'cover': 'open', 'count': 1, 'from_start': (0, 0)},
        {'kind': 'Gantry', 'radius': 25, 'pace': 30, 'cover': 'open', 'count': 1, 'from_start': (100, 160)},
        {'kind': 'Solararray', 'radius': 50, 'pace': 40, 'cover': 'open', 'count': 1, 'from_start': (140, 240)},
        {'kind': 'Crater', 'radius': 80, 'pace': 25, 'cover': 'open', 'count': 1, 'from_start': (190, 340)},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 300, 'apart': 6},
    'places': [
        {'kind': 'Habitmodule', 'count': 3, 'on': {'Habitat': 1.0}, 'apart': 12},
        {'kind': 'Park', 'count': 1, 'on': {'Habitat': 1.0}, 'apart': 12},
        {'kind': 'Airlock', 'count': 1, 'on': {'Habitat': 1.0}, 'apart': 9},
        {'kind': 'Oxygentank', 'count': 1, 'on': {'Habitat': 1.0}},
        {'kind': 'Gantry', 'count': 1, 'on': {'Gantry': 1.0}},
        {'kind': 'Evastation', 'count': 1, 'on': {'Gantry': 1.0}},
        {'kind': 'Repairpanel', 'count': 1, 'on': {'Gantry': 1.0}},
    ],
}


stakes = ("On this lunar base, air, suit integrity and the failing coolant line are the stakes: oxygen runs down "
          "outside the habitat, repairs need parts, time and steady hands, and vacuum punishes mistakes. Use these, "
          "and the crew's trust and procedures under pressure, as concrete, checkable goals and obstacles tied to the "
          "airlock, the gantry, the repair panel and the oxygen supply.")

# Stage business (sim/survival.py): suit oxygen outside (none used inside), repairs, airlock, EVA hazards.
affordances = {
    'needs': {
        # 'oxygen' is suit oxygen used: it rises outside and stops inside the pressurised habitat (sheltered)
        'oxygen': {'initial': 0.35, 'per_hour': 0.25, 'shelter_factor': 0.0, 'neglect_harm': 0.3,
                   'words': ('suit air plentiful', 'suit air past half', 'suit air low', 'suit air critical'),
                   'percepts': {0.5: "Your suit display flashes: oxygen past half.",
                                0.8: "Low-oxygen alarm: your suit is chiming, and every breath feels thin."}},
        'fatigue': {'initial': 0.3, 'per_hour': 0.06, 'shelter_factor': 0.3,
                    'words': ('steady', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "Working in the suit is wearing you down; your arms ache.",
                                 0.8: "You are exhausted; your hands are clumsy in the gloves."}},
    },
    'harm': {'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')},
    'travel_need': 'fatigue',
    'tile_minutes': 2,
    'type_nouns': {'Airlock': ['airlock', 'lock', 'hatch'], 'Gantry': ['gantry', 'coolant line', 'weld', 'line'],
                   'Evastation': ['eva station', 'suit station', 'stow', 'recharge'], 'Repairpanel': ['repair panel', 'panel'],
                   'Habitmodule': ['habitat', 'hab', 'module'], 'Oxygentank': ['oxygen tank', 'o2', 'oxygen', 'tank']},
    'actions': [
        {'name': 'refill oxygen', 'verbs': ['refill', 'recharge', 'top up', 'swap tank', 'plug in'], 'named_verbs': ['connect', 'hook up'],
         'resource_types': ['Oxygentank', 'Evastation'], 'effects': {'oxygen': -1.0}, 'minutes': 10,
         'text': 'refills suit oxygen at {resource}'},
        {'name': 'cycle the airlock', 'verbs': ['cycle', 'enter the airlock', 'go inside', 'pressurise', 'pressurize'],
         'named_verbs': ['enter', 'open'], 'resource_types': ['Airlock'], 'minutes': 10, 'sets': 'sheltered',
         'text': 'cycles through {resource} into the pressurised habitat'},
        {'name': 'fetch sealant kit', 'category': 'work', 'verbs': ['fetch', 'grab', 'take', 'collect', 'get'],
         'named_verbs': [], 'resource_types': ['Evastation', 'Repairpanel'], 'gain': 'sealant kit', 'minutes': 5, 'deplete': 3,
         'text': 'takes a sealant kit from {resource}'},
        {'name': 'seal the coolant line', 'category': 'work', 'verbs': ['seal', 'patch', 'apply sealant', 'repair', 'fix'],
         'resource_types': ['Gantry'], 'requires': 'sealant kit', 'consumes': 'sealant kit',
         'requires_property': {'repaired': None}, 'sets_property': {'repaired': True},
         'effects': {'fatigue': 0.15}, 'minutes': 30, 'text': 'seals the cracked coolant line at {resource}',
         'risk': {'chance': 0.15, 'effects': {'injury': 0.15}, 'text': 'A jet of coolant vapour slams against the visor.'}},
        # the torch she carries: done where she stands, but a bead on a live crack may not hold
        # (the kit is a sure seal, half an hour's walk away)
        {'name': 'weld the coolant line', 'category': 'work', 'verbs': ['weld', 'run a bead', 'braze', 'strike the arc'],
         'resource_types': ['Gantry'], 'requires': '*torch',
         'requires_property': {'repaired': None}, 'sets_property': {'repaired': True},
         'effects': {'fatigue': 0.15}, 'minutes': 15, 'text': 'welds the cracked coolant line shut at {resource}',
         'success': {'chance': 0.5, 'text': 'The bead cracks as it cools; the line still vents.'},
         'risk': {'chance': 0.2, 'effects': {'injury': 0.15}, 'text': 'A jet of coolant vapour slams against the visor.'}},
        {'name': 'run diagnostics', 'category': 'explore', 'verbs': ['diagnose', 'run diagnostics', 'scan', 'check readings', 'inspect'],
         'resource_types': ['Repairpanel'], 'finds': [{'item': 'pressure-loss log', 'chance': 0.5}], 'minutes': 15, 'deplete': 2,
         'text': 'runs diagnostics at {resource}'},
        {'name': 'rest', 'verbs': ['rest', 'sit', 'sleep', 'lie down'], 'resource_types': ['Habitmodule'],
         'effects': {'fatigue': -0.4}, 'minutes': 60, 'sets': 'sheltered', 'text': 'rests in {resource}'},
        {'name': 'buddy-breathe', 'category': 'social', 'target': 'character', 'verbs': ['share air', 'buddy-breathe', 'buddy breathe', 'cross-connect'],
         'effects': {'oxygen': 0.25}, 'target_effects': {'oxygen': -0.25}, 'minutes': 5,
         'text': "cross-connects suit lines and shares air with {target}"},
        {'name': 'give sealant kit', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'pass'],
         'transfer': 'sealant kit', 'minutes': 1, 'text': 'hands {target} a sealant kit'},
        {'name': 'tend injury', 'category': 'social', 'target': 'character', 'verbs': ['tend', 'treat', 'patch up', 'check on'],
         'target_effects': {'injury': -0.2}, 'minutes': 15, 'text': "treats {target}'s injury"},
    ],
    'hazards': [
        {'name': 'micrometeorites', 'description': 'a shower of micrometeorites pinging off the regolith', 'when': 'any',
         'terrain': ['Regolith', 'Crater', 'Gantry'], 'chance_per_hour': 0.06, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 15, 'harm': 0.3, 'fail_text': 'a grain scores the suit and a leak alarm shrieks',
         'retreat_text': 'The shower passes; the surface is still again.',
         'strike_text': 'Micrometeorites rake the surface; suits are scored and a seal weeps air.',
         'responses': {'take cover': {'verbs': ['take cover', 'shelter', 'duck behind', 'get behind'], 'success': 0.6},
                       'retreat inside': {'verbs': ['retreat', 'head for the airlock', 'go inside'], 'success': 0.5}}},
        {'name': 'line rupture', 'description': 'the cracked coolant line venting a plume of frozen vapour', 'when': 'any',
         'chance_per_hour': 0.15, 'blocked_by': ['sheltered'],
         # the crack is in the line at the gantry: no rupture once that line is sealed
         'source': {'resource_types': ['Gantry'], 'requires_property': {'repaired': None}, 'within': 6},
         'resolved_text': 'The sealed line holds; the venting stops.',
         'minutes_to_strike': 20, 'harm': 0.3, 'fail_text': 'the vapour plume whips across the gantry',
         'retreat_text': 'The venting slows to a trickle; the line holds for now.',
         'strike_text': 'The line ruptures; a blast of frozen coolant knocks bodies against the rail.',
         # a clamp only holds for now; 'seal the coolant line' (with a sealant kit) or 'weld the
         # coolant line' (with a torch, which may not hold) is the repair
         'responses': {'clamp the line': {'verbs': ['clamp', 'shut the valve', 'close the valve', 'isolate'], 'success': 0.5},
                       'back away': {'verbs': ['back away', 'retreat', 'move clear'], 'success': 0.6}}},
    ],
}

# the sky of a picture (stage.light): no air, so a black sky and hard sunlight at any hour the
# sun is up; the play's clock runs on Earth hours
_SUNLIT = 'harsh unfiltered sunlight, black sky, Earth low on the horizon, hard black shadows'
image_light = {'dawn': _SUNLIT, 'morning': _SUNLIT, 'midday': _SUNLIT, 'afternoon': _SUNLIT, 'dusk': _SUNLIT,
               'night': 'lunar night, black starry sky, Earth low on the horizon, the base lit by floodlights'}
