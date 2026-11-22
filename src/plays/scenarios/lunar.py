"""A small lunar base: the habitat ring, the service gantry over the vacuum line, a crater, the solar array.

How the places of this world are, and how they sit. They are laid out in metres, anew for each
performance (sim/land.py); other places come into it as the play needs them (sim/world.py).
  kinds: for each kind of place, what it is, and in metres its radius. enclosed: it has walls
    (open: they stand open, as a cave's mouth or a shop with its front broken in). shelter: inside it
    is under shelter. blocks_view: nothing is seen across it. hides: what is inside is not seen
    from outside. owner: it has an owner's part, which someone may be cast in. vehicle: it is gone
    by, at `pace` metres a minute, and `by` is the word for that. search: what is left there is found
    only by someone who searches the place.
  layout: the ground (regions: large places that hold the rest; pace is metres a minute on foot,
    cover is what is overhead) and how many of each kind there are and where: on which ground, by
    weight; inside or near (metres from its edge) another kind; from_start, metres from where the
    play's people first stand; names, for those the author has named.
"""

kinds = {
    'Airlock': {'description': 'Primary EVA airlock into the habitat ring', 'radius': 3, 'enclosed': True, 'shelter': True},
    'Gantry': {'description': 'Metal service gantry overlooking the vacuum‑line weld', 'radius': 6},
    'Evastation': {'description': 'Suit stow & recharge alcove', 'radius': 2},
    'Repairpanel': {'description': 'External repair junction box for vacuum line', 'radius': 1.5, 'search': True},
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

# How this world works, in sentences (docs/world-facts-design.md). Nothing here is matched by code.
facts = [
    "Outside the habitat a suit's oxygen runs down, from full to empty in about four hours. Refilling at the oxygen bundle or the suit station takes about ten minutes and fills it; breathing from another person's suit eases it a little, and takes as much from theirs.",
    "An hour's rest in a habitat module eases tiredness well.",
    'Sealant kits are kept at the suit station and the repair panel, about three at each.',
    'The coolant line at the gantry is cracked and venting. Sealing it takes a sealant kit and about half an hour, and holds. Welding it shut with a torch takes about a quarter of an hour and fails about half the time. Either way, about one time in six a jet of coolant vapour catches the one at work on it.',
    'Running diagnostics at the repair panel takes about a quarter of an hour and about half the time turns up a pressure-loss log.',
    'An injury tended is eased a little.',
]
# What threatens people here, in sentences (docs/dangers-design.md): the play brings one on, and the world says what comes of it
dangers = [
    'A shower of micrometeorites may sweep the open surface and the gantry at any hour, a quarter of an hour from the first pings. Inside the habitat a person is safe from it; taking cover outside works more often than not, and getting back inside in time about half the time. A strike scores a suit and can start a seal weeping air.',
    'While the coolant line at the gantry is cracked it may rupture when someone is near it, within twenty minutes of the venting worsening. Clamping it holds about half the time; backing away gets a person clear more often than not. A rupture throws bodies against the rail. Once the line is sealed or welded shut it does not rupture.',
]
# What no deed or word gets past
held = [
    'A cracked line is not shut without a sealant kit or a torch.',
]

# Stage business (sim/survival.py): suit oxygen outside (none used inside), repairs, airlock, EVA hazards.
affordances = {
    'needs': {
        # 'oxygen' is suit oxygen used: it rises outside and stops inside the pressurised habitat (sheltered)
        'oxygen': {                   'initial': 0.35, 'per_hour': 0.25, 'shelter_factor': 0.0, 'neglect_harm': 0.3,
                   'words': ('suit air plentiful', 'suit air past half', 'suit air low', 'suit air critical'),
                   'percepts': {0.5: "Your suit display flashes: oxygen past half.",
                                0.8: "Low-oxygen alarm: your suit is chiming, and every breath feels thin."}},
        'fatigue': {                    'initial': 0.3, 'per_hour': 0.06, 'shelter_factor': 0.3,
                    'words': ('steady', 'tired', 'very tired', 'exhausted'),
                    'percepts': {0.5: "Working in the suit is wearing you down; your arms ache.",
                                 0.8: "You are exhausted; your hands are clumsy in the gloves."}},
    },
    'harm': {             'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')},
    'travel_need': 'fatigue',
    'tile_minutes': 2,
}

# the sky of a picture (stage.light): no air, so a black sky and hard sunlight at any hour the
# sun is up; the play's clock runs on Earth hours
_SUNLIT = 'harsh unfiltered sunlight, black sky, Earth low on the horizon, hard black shadows'
image_light = {'dawn': _SUNLIT, 'morning': _SUNLIT, 'midday': _SUNLIT, 'afternoon': _SUNLIT, 'dusk': _SUNLIT,
               'night': 'lunar night, black starry sky, Earth low on the horizon, the base lit by floodlights'}
