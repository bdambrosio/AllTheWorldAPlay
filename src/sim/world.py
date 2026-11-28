"""The world's space: its places, where each person stands, how far and which way one thing is
from another, and what is in view. Everything above asks here; nothing else reads a position,
the table of places or the ground.

The places are laid out in metres from the world file's `kinds` and `layout` (sim/land.py, where
the rules of nearness, sight and shelter are stated once), anew for each performance, and others
come into the world as the play needs them (`make_place`: asked only by the stage manager's entries and the
resolver, sim/resolver.py).

A position is whatever `at` gives. Callers hand it back to this module (span, bearing, where,
put) and do not look inside it. Every question that takes the context is answered by the
context's own land."""
import logging
import os
import re

from sim.land import Land, Place, Standing, bearing   # noqa: F401  (Place and bearing are part of what is asked here)

logger = logging.getLogger('simulation_core')

SAME_PLACE = ('', 'same', 'same place', 'same location', 'same as before', 'here', 'unchanged')


def _land(context):
    return getattr(context, 'land', None)


def build(context, scenario_module):
    """Lay out the world of a performance from its world file."""
    from sim.trace import trace
    context.land = Land(context, scenario_module, seed=int(os.environ['ATWAP_WORLD_SEED']) if os.environ.get('ATWAP_WORLD_SEED') else None)
    # the places as laid out, so that what a prompt was shown of them can be made again from a trace
    trace.event('world_built', seed=context.land.seed, places=[
        {'id': p.id, 'of_kind': p.kind, 'at': list(p.xy), 'radius': p.radius, 'region': p.region, 'description': p.description}
        for p in context.land.table.values()])


def built(context):
    """Has this performance a world laid out?"""
    return _land(context) is not None


def scenario(context):
    """The world file this performance is played in (plays/scenarios/*.py), or None."""
    land = _land(context)
    return land.scenario if land is not None else None


def makes_places(context):
    """Can a place be added to this world while it is played?"""
    return built(context)


def make_place(context, name, kind, at, inside=False, metres=None, enclosed=False, description=''):
    """Bring a place into the world that the play has come to need: called `name`, of a kind,
    inside or beside the listed place `at` (its id). Returns it, or None when it cannot be made."""
    from sim import ledger
    from sim.trace import trace
    p = _land(context).make_place(name, kind, at, inside, metres, enclosed, description) if built(context) else None
    if p is not None:
        ledger.of(context).wrote('places')
        trace.event('place_added', id=p.id, name=name, of_kind=p.kind, at=at if isinstance(at, str) else getattr(at, 'id', ''), inside=bool(inside),
                    xy=list(p.xy), enclosed=p.enclosed, description=p.description)
    return p


# ---------------------------------------------------------------------------- people
def on(who):
    """Is this person somewhere in the world (a voice with no body in it is not)?"""
    return getattr(who, 'mapAgent', None) is not None


def at(who):
    """Where a person stands."""
    return (who.mapAgent.x, who.mapAgent.y)


def put(who, xy):
    """Stand a person at a position. (The ledger is told by the caller's event: sim/ledger.py.)"""
    who.mapAgent.x, who.mapAgent.y = float(xy[0]), float(xy[1])


def enter(context, who, xy=None):
    """Give a person a position in this world: where the play's people first stand, unless told."""
    who.mapAgent = Standing(*(xy if xy is not None else start(context)))


def put_at(who, rid):
    """Stand a person at a named place (a play file, before the play begins). False when this
    world has no such place: the person stays where it was."""
    p = called(who.context, rid)
    if p is None:
        logger.warning(f"{who.name} was to stand at '{rid}', which is no place in this world")
        return False
    put(who, p.xy)
    return True


def beside(who, other):
    """Stand a person where another stands."""
    put(who, at(other))


def token(name):
    """A name as one token ('Storm Chaser' -> 'Storm_Chaser')."""
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(name or '').strip())
    return '_'.join(word.capitalize() for word in re.sub(r'[^a-zA-Z0-9\s]', ' ', name).split())


COMPASS = {'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest'}


def compass(text):
    """Is this the name of a compass direction, and nothing else ('north', 'North-west')?"""
    return re.sub(r'[\s-]+', '', str(text or '').strip().lower().removeprefix('to the ').removeprefix('the ')) in COMPASS


# ---------------------------------------------------------------------------- what is in view
def surroundings(who):
    """What a person sees from where it stands: (the view in words, the ids of the places seen,
    the names of the people seen, [], one line of summary)."""
    return _land(who.context).surroundings(who)


def view_report(who):
    """The same view, as a narrator is shown it."""
    return _land(who.context).view_report(who)


def picture(context):
    """The world drawn for whoever is running the play (the server's own window)."""
    import matplotlib.pyplot as plt
    land = _land(context)
    fig, ax = plt.subplots(figsize=(9, 9))
    for p in land.table.values():
        ax.add_patch(plt.Circle(p.xy, p.radius, fill=p.region, alpha=0.15 if p.region else 1.0, color='green' if p.region else 'black'))
        if not p.region:
            ax.annotate(p.id, p.xy, fontsize=6)
    for c in context.everyone():
        if on(c):
            ax.plot(*at(c), 'r*')
            ax.annotate(c.name, at(c), color='red', fontsize=8)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.autoscale_view()
    return fig


# ---------------------------------------------------------------------------- asked of the land
# Each of these is asked as world.<name>(context, ...) and answered by the method of that name of
# the context's Land (sim/land.py), where it is described:
#   the places            places, listed, place, called, find, kind_called, nearest, named_in, has_owner_part, add_place
#   the ground            start, region, regions, region_called, nearest_ground, indoors, sheltered
#   a name into a place   resolve, where
#   nearness              span, at_place, at_its_foot, on_the_spot, together, lying_here, in_earshot, a_step,
#                         can_walk_over, in_sight, in_range, in_the_neighbourhood
#   walking and words     minutes, walk, conveyance, bring, far, away, sight_words, reach_words
#   dangers and pictures  danger_start, danger_reaches, danger_source_near, danger_clear, danger_words,
#                         apart_words, danger_picture_words
def _asked(name):
    def ask(context, *args, **kwargs):
        return getattr(_land(context), name)(*args, **kwargs)
    ask.__name__ = name
    ask.__doc__ = getattr(Land, name).__doc__
    return ask


for _name in ('places', 'listed', 'place', 'called', 'find', 'kind_called', 'nearest', 'named_in', 'has_owner_part', 'add_place',
              'start', 'region', 'regions', 'region_called', 'nearest_ground', 'indoors', 'sheltered',
              'resolve', 'where',
              'span', 'at_place', 'at_its_foot', 'on_the_spot', 'together', 'lying_here', 'in_earshot', 'a_step',
              'can_walk_over', 'in_sight', 'in_range', 'in_the_neighbourhood', 'sees',
              'minutes', 'walk', 'conveyance', 'bring', 'far', 'away', 'sight_words', 'reach_words',
              'danger_start', 'danger_reaches', 'danger_source_near', 'danger_clear', 'danger_words',
              'apart_words', 'danger_picture_words'):
    globals()[_name] = _asked(_name)
