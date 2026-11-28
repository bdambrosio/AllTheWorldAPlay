"""A modern Chinese city, as the sketch gives it: offices, apartments, a cafe and a hotel bar, and the roads between them
(plays/ABetterLife.py). Nothing more is stated here: what else the city holds, and how it works, is not this file's to say.

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
    'Office': {'description': 'An office', 'radius': 6},
    'University_Office': {'description': 'An office at the university', 'radius': 6},
    'Apartment': {'description': 'An apartment', 'radius': 7},
    'Cafe': {'description': 'A cafe', 'radius': 7},
    'Hotel_Bar': {'description': 'A hotel bar', 'radius': 10},
}

layout = {
    'regions': [
        {'kind': 'Road', 'radius': 2500, 'pace': 75, 'cover': 'open', 'whole': True},
    ],
    # places lie within this of where the play's people start, more of them nearer, and this far apart
    'scatter': {'within': 800, 'apart': 100},
    'places': [
        {'kind': 'Office', 'count': 4, 'on': {'Road': 1.0}, 'names': ['HuOffice', 'XueOffice', 'XiaoyuOffice', 'YanghoOffice']},
        {'kind': 'University_Office', 'count': 1, 'on': {'Road': 1.0}, 'names': ['UniversityOffice']},
        {'kind': 'Apartment', 'count': 3, 'on': {'Road': 1.0}, 'names': ['HuDingApartment', 'XueApartment', 'QiuApartment']},
        {'kind': 'Cafe', 'count': 1, 'on': {'Road': 1.0}, 'names': ['Cafe']},
        {'kind': 'Hotel_Bar', 'count': 1, 'on': {'Road': 1.0}, 'names': ['HotelBar']},
    ],
}


# how a place the author has named is said on the page
said = {'HuDingApartment': "Hu and Ding's apartment", 'XueApartment': "Xue's apartment", 'QiuApartment': "Qiu's apartment",
        'HuOffice': "Hu's office", 'XueOffice': "Xue's office", 'XiaoyuOffice': "Xiaoyu's office", 'YanghoOffice': "Yangho's office",
        'UniversityOffice': 'the university office', 'Cafe': 'the cafe', 'HotelBar': 'the hotel bar'}


# The engine's world layer is on for a world that has this; it states no need, danger or rule of the world.
affordances = {
    'tile_minutes': 1,
}
