"""Stage actions every world has: what a body can do anywhere, with anyone present.

A world's own actions (its `affordances['actions']`) are what its places and things allow: drink
at a spring, sign over a farm. These are the rest of what people on a stage do with their
bodies, and they are the same in a forest and an office. sim/survival.py offers them beside the
world's own (shared_offers), and a deed that is neither is not done (classify_deed).

A world leaves all of them out with `affordances['shared'] = False`, or some of them with
`affordances['without'] = ['embrace', ...]`. A world action of the same name stands in place of
the one here.

    kind        gesture   nothing is different afterwards
                posture   how the doer's body is, until it moves or the scene ends
                position  where the doer stands with respect to someone, likewise
                contact   the doer touching someone, until either lets go, moves, or the scene ends
    target      'character': done to or toward someone in reach (label 'name@Who')
    item        True: done with a thing the doer holds (label 'name <thing>@Who', or 'name <thing>'
                when it has no target)
    fact        what the record keeps of it ({actor}, {item}); without one a gesture leaves nothing
    sets        posture: the word the body then carries ('sitting'); position and contact: the
                words between the two ('close to', 'has the arm of'); None with 'clears': undoes
    gloss       what it is, for the actor and for whoever judges a deed; {other} is the target

Not here, on purpose: force on a person (a grab, a shove: who prevails is not settled by a
gesture), and anything that changes a thing, a garment or the place (a door, a tap, a fire):
those are a world's to give, where it tracks what they change.
"""

actions = [
    {'name': 'gesture', 'kind': 'gesture',
     'gloss': "a movement of one's own that is for nobody and changes nothing: looks at own hands, shrugs, hums, breathes out, sounds a note on an instrument already in hand"},
    {'name': 'examine', 'kind': 'gesture',
     'gloss': "looks closely at a thing held, a thing lying here, the place or a person, without moving or changing it; shows only what the eye can see"},
    {'name': 'gesture to', 'kind': 'gesture', 'target': 'character',
     'gloss': "a movement of hand, head or face meant for {other}: holds out an open hand, nods, points, beckons, meets their eyes, waits facing them; touches nobody, moves nothing"},
    {'name': 'show', 'kind': 'gesture', 'target': 'character', 'item': True,
     'gloss': "holds a thing already in hand where {other} can see it; it stays in hand"},

    {'name': 'stow', 'kind': 'gesture', 'item': True, 'fact': '{actor} puts the {item} away and still has it.',
     'gloss': "puts a thing already in hand into a pocket, a belt or under an arm; it is still theirs and still carried"},

    {'name': 'sit', 'kind': 'posture', 'sets': 'sitting', 'gloss': 'sits down where they are'},
    {'name': 'kneel', 'kind': 'posture', 'sets': 'kneeling', 'gloss': 'kneels or crouches where they are'},
    {'name': 'lie down', 'kind': 'posture', 'sets': 'lying down', 'gloss': 'lies down where they are (awake: sleeping is another action)'},
    {'name': 'stand up', 'kind': 'posture', 'sets': None, 'clears': True, 'gloss': 'gets to their feet'},

    {'name': 'step closer to', 'kind': 'position', 'target': 'character', 'sets': 'close to',
     'gloss': 'moves to within arm\'s reach of {other}, or falls in beside them'},
    {'name': 'step back from', 'kind': 'position', 'target': 'character', 'sets': 'keeping back from',
     'gloss': 'puts a few paces between themselves and {other}, staying in the place'},
    {'name': 'stand in the way of', 'kind': 'position', 'target': 'character', 'sets': 'standing in the way of',
     'gloss': "plants themselves where {other} would have to go round; it stops nobody by force"},

    {'name': 'take the hand of', 'kind': 'contact', 'target': 'character', 'sets': 'holding the hand of',
     'gloss': "takes {other}'s hand; {other} may let go"},
    {'name': 'take the arm of', 'kind': 'contact', 'target': 'character', 'sets': 'holding the arm of',
     'gloss': "takes {other}'s arm or lays a hand on their arm or shoulder; {other} may let go"},
    {'name': 'lean on', 'kind': 'contact', 'target': 'character', 'sets': 'leaning on',
     'gloss': "rests their weight or head against {other}; {other} may let go"},
    {'name': 'embrace', 'kind': 'contact', 'target': 'character', 'sets': 'holding',
     'gloss': "puts their arms round {other}; {other} may let go"},
    {'name': 'stand back to back with', 'kind': 'contact', 'target': 'character', 'sets': 'back to back with',
     'gloss': "sets their back against {other}'s; {other} may let go"},
    {'name': 'let go of', 'kind': 'contact', 'target': 'character', 'sets': None, 'clears': True,
     'gloss': "ends any touch between themselves and {other}, whoever began it"},
]
