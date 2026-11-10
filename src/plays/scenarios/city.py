"""A city of offices, apartments and places to meet: the world of a white-collar drama
(ABetterLife.py). The hard facts are contracts in hand and messages on phones; there are no
hazards, and bodily needs only colour a long day."""
from enum import Enum, auto

from sim.map import ResourceTypeRegistry


class CityTerrain(Enum):
    Water = 1         # Required terrain type
    Building = 2      # Offices, apartments, the university
    Street = 3
    Sidewalk = 4
    Park = 5


class CityInfrastructure(Enum):
    Sidewalk = auto()


class CityResources(Enum):
    Office = auto()
    University_Office = auto()
    Apartment = auto()
    Cafe = auto()
    Restaurant = auto()
    Taxi_Stand = auto()    # Required by map.py (the map's meeting point)


class CityProperty(Enum):
    Building = auto()
    Park = auto()


terrain_rules = {
    'elevation_noise_scale': 10.0,
    'water_level': 0.05,
    'mountain_level': 0.95,
    'terrain_by_elevation': {
        'water': {'max': 0.05, 'type': 'Water'},
    },
    'lowland_distribution': {
        'Building': 0.45,
        'Street': 0.2,
        'Sidewalk': 0.25,
        'Park': 0.1,
    }
}

infrastructure_rules = {
    'road_density': 0.2,
    'path_type': 'Sidewalk',
    'slope_factor': 1.0,
    'terrain_costs': {
        'Water': float('inf'),
        'Building': 1.0,
        'Street': 1.5,
        'Sidewalk': 1.0,
        'Park': 1.2,
    }
}

property_rules = {
    'min_size': 10,
    'max_size': 30,
    'valid_terrain': ['Building']
}

terrain_types = CityTerrain
infrastructure_types = CityInfrastructure
property_types = CityProperty
resource_types = ResourceTypeRegistry(CityResources)

required_resource = resource_types.Taxi_Stand
required_resource_name = "Taxi_Stand"


def _place(resource_type, description, count, terrain):
    return {'resource_type': resource_type, 'description': description, 'count': count,
            'requires_property': False, 'terrain_weights': {terrain: 1.0}}


resource_rules = {
    # the places a scene can be set, by name
    'names': {
        'Office': ['HuOffice', 'XueOffice', 'XiaoyuOffice', 'YanghoOffice'],
        'University_Office': ['UniversityOffice'],
        'Apartment': ['HuDingApartment', 'XueApartment', 'QiuApartment'],
        'Cafe': ['Cafe'],
        'Restaurant': ['HotelBar'],
    },
    'allocations': [
        _place(resource_types.Office, 'An office with a desk, client files and contracts', 4, terrain_types.Building),
        _place(resource_types.University_Office, "A professor's office at the university", 1, terrain_types.Building),
        _place(resource_types.Apartment, 'A city apartment', 3, terrain_types.Building),
        _place(resource_types.Cafe, 'A cafe: neutral ground for a meeting', 1, terrain_types.Sidewalk),
        _place(resource_types.Restaurant, 'A hotel bar and restaurant, busy in the evening', 1, terrain_types.Building),
    ]
}

# What is at stake in this world (planning and act prompts; sim/survival.py stakes_text).
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
        'fatigue': {'initial': 0.2, 'per_hour': 0.03,
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
    'actions': [
        # contracts: drafted at an office, handed to a client, signed by whoever holds one
        {'name': 'draft a contract', 'category': 'work', 'verbs': ['draft', 'draw up', 'write up', 'prepare the contract'],
         'named_verbs': ['prepare', 'print'], 'resource_types': ['Office'], 'gain': "{actor}'s contract",
         'effects': {'fatigue': 0.05}, 'minutes': 20, 'text': 'drafts a contract at {resource}'},
        {'name': 'hand over the contract', 'category': 'social', 'target': 'character',
         'verbs': ['hand', 'give', 'pass', 'slide', 'present', 'push'], 'transfer': '*contract', 'minutes': 1,
         'text': 'hands {target} {item}'},
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
        {'name': 'give a gift', 'category': 'social', 'target': 'character', 'verbs': ['give', 'hand', 'present', 'offer'],
         'transfer': 'gift', 'minutes': 1, 'text': 'gives {target} a gift'},
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
        {'name': 'rest at home', 'verbs': ['rest', 'lie down', 'sleep', 'nap', 'put my feet up'], 'resource_types': ['Apartment'],
         'effects': {'fatigue': -0.5}, 'minutes': 60, 'text': 'rests at {resource}'},
        {'name': 'teach', 'category': 'work', 'verbs': ['teach', 'lecture', 'hold office hours', 'grade papers'],
         'resource_types': ['University_Office'], 'effects': {'fatigue': 0.2}, 'earn': 200, 'minutes': 120,
         'text': 'teaches and sees students at {resource}'},
    ],
    'hazards': [],
}
