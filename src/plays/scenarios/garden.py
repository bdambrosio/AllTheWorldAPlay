"""One garden, as kittens have it: grass, shrubs, flower beds, a tree.

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
    'Bug': {'description': 'yummy bug', 'radius': 0.3},
    'Butterfly': {'description': 'A bright butterfly flitting among the flowers', 'radius': 0.3},
    'Leaf': {'description': 'A deep pile of crunchy fallen leaves', 'radius': 0.6},
    'Seed': {'description': 'A scatter of seeds under the bird feeder', 'radius': 0.4},
    'Ball': {'description': 'A bright red ball', 'radius': 0.3},
    'Tree': {'description': 'The garden tree', 'radius': 1},
}

layout = {
    'regions': [
        {'kind': 'Grass', 'radius': 30, 'pace': 40, 'cover': 'open', 'whole': True},
        {'kind': 'Shrubs', 'radius': (3, 5), 'pace': 25, 'cover': 'trees', 'count': 4, 'from_start': (4, 25), 'hides': True},
        {'kind': 'Trees', 'radius': 4, 'pace': 35, 'cover': 'trees', 'count': 2, 'from_start': (6, 25)},
        {'kind': 'Flowers', 'radius': (2, 3), 'pace': 35, 'cover': 'open', 'count': 3, 'from_start': (4, 25)},
        {'kind': 'Rocks', 'radius': 2, 'pace': 30, 'cover': 'open', 'count': 1, 'from_start': (6, 25)},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 25, 'apart': 1.5},
    'places': [
        {'kind': 'Bug', 'count': 15, 'on': {'Grass': 2.0, 'Shrubs': 1.0, 'Flowers': 0.5}},
        {'kind': 'Butterfly', 'count': 10, 'on': {'Grass': 2.0, 'Shrubs': 1.0, 'Flowers': 0.5}},
        {'kind': 'Leaf', 'count': 24, 'on': {'Grass': 1.0, 'Shrubs': 0.5}},
        {'kind': 'Seed', 'count': 16, 'on': {'Grass': 0.6, 'Shrubs': 1.0}},
        {'kind': 'Ball', 'count': 3, 'on': {'Grass': 1.0, 'Shrubs': 0.5}},
        {'kind': 'Tree', 'count': 1, 'on': {'Trees': 1.0}},
    ],
}


stakes = ("For the kittens, the garden is a world of play, food, naps and small dangers: a bug to catch, a "
          "butterfly to chase, a ball to win, a warm nap, the neighbour's dog. Use these concrete, playful goals and "
          "obstacles tied to things in the garden; keep the tone gentle and the magic light.")

# Stage business (sim/survival.py): hunting bugs, chasing, play, naps, the neighbour's dog.
affordances = {
    'needs': {
        'hunger': {'initial': 0.35, 'per_hour': 0.08,
                   'words': ('full', 'peckish', 'hungry', 'very hungry'),
                   'percepts': {0.5: "Your tummy rumbles; a bug would be nice.", 0.8: "You are SO hungry."}},
        'sleepiness': {'initial': 0.2, 'per_hour': 0.1, 'shelter_factor': 0.5,
                       'words': ('bouncy', 'drowsy', 'sleepy', 'can barely keep eyes open'),
                       'percepts': {0.5: "Your eyes are getting heavy; a sunny spot looks very nice.",
                                    0.8: "You are so sleepy you keep yawning."}},
    },
    'harm': {'name': 'scrapes', 'words': ('unruffled', 'ruffled', 'scraped', 'hurting')},
    'travel_need': 'sleepiness',
    'tile_minutes': 1,
    'type_nouns': {'Bug': ['bug', 'beetle', 'cricket'], 'Butterfly': ['butterfly'], 'Leaf': ['leaf', 'leaves'],
                   'Seed': ['seed', 'seeds'], 'Ball': ['ball']},
    'actions': [
        {'name': 'catch a bug', 'verbs': ['pounce', 'catch', 'eat', 'snap up', 'gobble'], 'resource_types': ['Bug'],
         'effects': {'hunger': -0.35, 'sleepiness': 0.05}, 'minutes': 5, 'deplete': 3, 'text': 'pounces and gobbles a bug at {resource}'},
        {'name': 'chase the butterfly', 'category': 'explore', 'verbs': ['chase', 'leap at', 'swat', 'stalk'],
         'resource_types': ['Butterfly'], 'effects': {'sleepiness': 0.1}, 'minutes': 10,
         'finds': [{'item': 'a shimmering wing-scale', 'chance': 0.2}], 'text': 'chases the butterfly round and round {resource}',
         'risk': {'chance': 0.15, 'effects': {'scrapes': 0.1}, 'text': 'A tumble into the rose bush!'}},
        {'name': 'pick up the ball', 'verbs': ['grab', 'carry', 'pick up', 'take'], 'named_verbs': ['bat', 'nose'],
         'resource_types': ['Ball'], 'gain': 'ball', 'minutes': 2, 'deplete': 1, 'text': 'grabs the ball from {resource}'},
        {'name': 'play with the ball', 'category': 'explore', 'verbs': ['bat the ball', 'dribble', 'play with the ball'],
         'named_verbs': ['play', 'bat', 'pounce'], 'resource_types': None, 'requires': 'ball', 'effects': {'sleepiness': 0.1},
         'minutes': 10, 'text': 'bats the ball about'},
        {'name': 'burrow into the leaves', 'verbs': ['burrow', 'dive into', 'hide in', 'snuggle into'], 'resource_types': ['Leaf'],
         'minutes': 5, 'sets': 'sheltered', 'text': 'burrows deep into the leaves at {resource}'},
        {'name': 'nap', 'verbs': ['nap', 'sleep', 'doze', 'curl up', 'snooze'], 'resource_types': None,
         'effects': {'sleepiness': -0.8}, 'minutes': 60, 'text': 'curls up for a nap'},
        {'name': 'nibble seeds', 'category': 'explore', 'verbs': ['nibble', 'sniff', 'paw at'], 'resource_types': ['Seed'],
         'effects': {'hunger': -0.1}, 'minutes': 5, 'text': 'sniffs and nibbles at the seeds by {resource}'},
        {'name': 'give the ball', 'category': 'social', 'target': 'character', 'verbs': ['give', 'share', 'drop', 'nudge', 'pass'],
         'transfer': 'ball', 'minutes': 1, 'text': 'nudges the ball over to {target}'},
        {'name': 'groom', 'category': 'social', 'target': 'character', 'verbs': ['groom', 'lick', 'nuzzle', 'snuggle'],
         'target_effects': {'scrapes': -0.15}, 'minutes': 10, 'text': 'grooms {target} gently'},
    ],
    'hazards': [
        {'name': 'dog', 'description': "the neighbour's big bouncy dog bounding over the fence", 'when': 'day',
         'terrain': ['Grass', 'Flowers'], 'chance_per_hour': 0.06, 'blocked_by': ['sheltered'],
         'minutes_to_strike': 15, 'harm': 0.15, 'fail_text': 'the dog bounds closer, barking happily',
         'retreat_text': 'The dog is called home and lollops away.',
         'strike_text': 'The dog bowls everyone over in a slobbery tumble before it is called home.',
         'responses': {'hiss': {'verbs': ['hiss', 'puff up', 'arch', 'swat', 'growl'], 'success': 0.5},
                       'run away': {'verbs': ['run', 'flee', 'scamper', 'dash'], 'success': 0.6},
                       'hide': {'verbs': ['hide', 'burrow', 'freeze', 'crouch'], 'success': 0.5}}},
    ],
}
