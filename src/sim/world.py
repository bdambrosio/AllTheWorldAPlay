"""The world's space: its places, where each person stands, how far and which way one thing is
from another, and what is in view. Everything above asks here; nothing else reads a position,
the table of places or the ground.

As it stands this is the one way in to the old grid (sim/map.py: a 60x60 field of tiles, named
places scattered on it, a tile for each person), with every distance still counted in tiles
(`tiles`) and every threshold where its caller keeps it. The grid is to give way to places that
hold other places, with positions in metres (docs/physical-world-design.md); what is asked here
stays, and what answers it changes.

A position is whatever `at` gives. Callers hand it back to this module (tiles, bearing, where,
region, put) and do not look inside it."""
import logging
import os
import re

logger = logging.getLogger('simulation_core')

SAME_PLACE = ('', 'same', 'same place', 'same location', 'same as before', 'here', 'unchanged')
AT_PLACE_TILES = 2        # this close to a place is 'at' it


class Place:
    """A named place of the world: its id (the name prompts and labels use), kind ('Spring'),
    description, position, and props: what is so of it that can change (its owner, what is left
    of what it gives, whatever a rule of the world sets)."""
    __slots__ = ('id', 'record')

    def __init__(self, rid, record):
        self.id, self.record = rid, record

    @property
    def kind(self):
        return str(self.record.get('type')).split('.')[-1]

    @property
    def description(self):
        return self.record.get('description', '')

    @property
    def xy(self):
        return tuple(self.record['location'])

    @property
    def props(self):
        return self.record.setdefault('properties', {})

    @property
    def owner(self):
        """Who owns the place: a person's name, a part nobody is yet cast in (stage.Part), or None."""
        owner = self.props.get('owner')
        return owner if owner is None or isinstance(owner, str) or hasattr(owner, 'place') else getattr(owner, 'name', owner)

    def __repr__(self):
        return f'<Place {self.id}>'


def _grid(context):
    return getattr(context, 'map', None)


def build(context, scenario_module):
    """Lay out the world of a performance from its world file."""
    from sim.trace import trace
    if getattr(scenario_module, 'layout', None) and not os.environ.get('ATWAP_GRID'):
        # a world file that says how its places sit is laid out in metres (sim/land.py)
        from sim.land import Land
        context.map, context.land = None, Land(context, scenario_module, seed=int(os.environ['ATWAP_WORLD_SEED']) if os.environ.get('ATWAP_WORLD_SEED') else None)
        trace.event('world_built', seed=context.land.seed, places=[
            {'id': p.id, 'kind': p.kind, 'at': [round(p.xy[0], 1), round(p.xy[1], 1)], 'radius': round(p.radius, 1), 'region': p.region,
             'description': p.description} for p in context.land.table.values()])
        return
    from sim import map as grid_map
    context.map = grid_map.WorldMap(60, 60, scenario_module)
    # the places as laid out, so that what a prompt was shown of them can be made again from a trace
    trace.event('world_built', places=[{'id': p.id, 'kind': p.kind, 'at': list(p.xy), 'description': p.description} for p in places(context)])


def _land(context):
    return getattr(context, 'land', None)


def scenario(context):
    """The world file this performance is played in (plays/scenarios/*.py), or None."""
    land = _land(context)
    return land.scenario if land is not None else getattr(_grid(context), 'scenario_module', None)


# ---------------------------------------------------------------------------- places
def places(context):
    """Every place, in the order the world came to have them."""
    grid = _grid(context)
    return [Place(rid, record) for rid, record in grid.resource_registry.items()] if grid is not None else []


def place(context, rid):
    """The place of this exact id, or None."""
    grid = _grid(context)
    record = grid.resource_registry.get(rid) if grid is not None and rid else None
    return Place(rid, record) if record is not None else None


def called(context, name):
    """The place a name gives, whatever its case; or None."""
    low = str(name or '').lower()
    return next((p for p in places(context) if p.id.lower() == low), None)


def kind_called(context, name):
    """The kind of place a word names in this world ('spring' -> 'Spring'), or None."""
    kind = _grid(context).get_resource_type(name)
    return getattr(kind, 'name', None) if kind is not None else None


def nearest(context, xy, kind=None):
    """(tiles, place) for the place nearest a position, of one kind if given; or None."""
    found = [(tiles(xy, p.xy), p.id, p) for p in places(context) if kind is None or p.kind == kind]
    if not found:
        return None
    d, _, p = min(found, key=lambda f: f[:2])
    return d, p


def named_in(context, text):
    """The ids of the places a text names, longest name first. (A place was looked for as a word
    ending in digits ('Spring3'), which the places an author names ('HuOffice', 'Cafe') are not:
    'Walk to HuOffice to draft the papers breaking Hu's contract' walked Xue to Hu.)"""
    low = (text or '').lower()
    return [p.id for p in sorted(places(context), key=lambda p: len(p.id), reverse=True)
            if re.search(rf'(?<![a-z0-9_]){re.escape(p.id.lower())}(?![a-z0-9_])', low)]


def add_place(context, kind, description, xy):
    """Bring a new place of a kind this world has into it, at a position. Returns the place."""
    grid = _grid(context)
    resource_type = grid.get_resource_type(kind)
    xy = tuple(xy)
    rid = grid._generate_resource_id(resource_type)
    grid.resource_registry[rid] = {'id': rid, 'type': resource_type, 'name': rid,
                                   'description': description or '', 'location': xy, 'properties': {}}
    if 0 <= xy[0] < grid.width and 0 <= xy[1] < grid.height:
        grid.patches[xy[0]][xy[1]].resources[rid] = grid.resource_registry[rid]
    return place(context, rid)


def has_owner_part(context, place):
    """Does this world say a place of this kind has an owner (a mill, a hut)?"""
    for allocation in _grid(context)._resource_rules['allocations']:
        if str(allocation['resource_type']).split('.')[-1] == place.kind:
            return allocation.get('has_npc', False)
    return False


# ---------------------------------------------------------------------------- people
def on(who):
    """Is this person somewhere in the world (a voice with no body on the map is not)?"""
    return getattr(who, 'mapAgent', None) is not None


def at(who):
    """Where a person stands."""
    return (who.mapAgent.x, who.mapAgent.y)


def put(who, xy):
    """Stand a person at a position. (The ledger is told by the caller's event: sim/ledger.py.)"""
    who.mapAgent.x, who.mapAgent.y = xy[0], xy[1]


def enter(context, who, xy, known=False):
    """Give a person a position in this world for the first time, or again. known: one this world
    may already have had a place for, by name."""
    land = _land(context)
    if land is not None:
        from sim.land import Standing
        who.mapAgent = Standing(*xy)
        return
    from sim import map as grid_map
    grid = _grid(context)
    if getattr(who, 'mapAgent', None) is None or known:
        who.mapAgent = (grid.get_agent(who.name) if known else None) or grid_map.Agent(xy[0], xy[1], grid, who.name)
    put(who, xy)


def token(name):
    """A name as one token ('Storm Chaser' -> 'Storm_Chaser')."""
    from sim.map import normalize_name_for_enum_lookup
    return normalize_name_for_enum_lookup(name)


def start(context):
    """Where a play's people stand when nothing says otherwise."""
    return tuple(_grid(context).stage_centre)


def offset(xy, dx, dy):
    """The position this far east and south of another."""
    return (xy[0] + dx, xy[1] + dy)


# ---------------------------------------------------------------------------- distance and direction
def tiles(a, b):
    """How far apart two positions are, in tiles."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def bearing(frm, to):
    """Which way `to` lies from `frm`: 'North', 'Southeast', 'Current'."""
    if isinstance(frm[0], float) or isinstance(to[0], float):
        from sim import land
        return land.bearing(frm, to)
    from sim.map import get_direction_name
    return get_direction_name(to[0] - frm[0], to[1] - frm[1]).name


def compass(text):
    """Is this text only a compass direction ('north', 'to the southwest')?"""
    from sim.map import Direction
    return bool(Direction.from_string(text))


# ---------------------------------------------------------------------------- the ground
def region(context, xy):
    """The kind of ground a position is on ('Forest', 'House')."""
    return _grid(context).patches[xy[0]][xy[1]].terrain_type.name


def regions(context):
    """The kinds of ground this world has."""
    return list(_grid(context).terrain_types.__members__)


def region_called(context, name):
    """The kind of ground a word names in this world, or None."""
    terrain = _grid(context).get_terrain_type(name)
    return terrain.name if terrain is not None else None


def nearest_ground(context, kind, anchor):
    """The position of the nearest ground of a kind (the anchor itself when it stands on it), or None."""
    grid = _grid(context)
    if region(context, anchor) == kind:
        return anchor
    found = [(x, y) for x in range(grid.width) for y in range(grid.height) if grid.patches[x][y].terrain_type.name == kind]
    return min(found, key=lambda t: tiles(anchor, t)) if found else None


def pace(context, xy):
    """What a tile of the ground at a position costs to walk, against open ground at 1.5."""
    costs = getattr(scenario(context), 'infrastructure_rules', {}).get('terrain_costs', {})
    cost = costs.get(region(context, xy), 1.5)
    return 1.5 if cost == float('inf') else cost


# ---------------------------------------------------------------------------- words, and names into places
def where(context, xy):
    """A position in words: 'at Well5', or '6 tiles North of Well5'."""
    found = nearest(context, xy)
    if found is None:
        return f'in {region(context, xy)} terrain'
    d, p = found
    return f'at {p.id}' if d <= AT_PLACE_TILES else f"{d} tiles {bearing(p.xy, xy)} of {p.id}"


def resolve(context, location, anchor):
    """(position, name) of the place a scene location names, seen from `anchor` (where the cast
    is): a place, the nearest place of a kind, or the nearest ground of a kind (the anchor itself
    when it already stands on it). None when the scene stays where the cast is: 'same', or a name
    that is no place in this world."""
    name = str(location or '').strip().strip('#"\'.')
    for suffix in (' terrain', ' resource', ' area'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)].strip()
    if name.lower() in SAME_PLACE:
        return None
    p = called(context, name)
    if p is not None:
        return p.xy, p.id
    kind = kind_called(context, name)
    if kind is not None:
        found = nearest(context, anchor, kind)
        if found:
            return found[1].xy, found[1].id
    ground = region_called(context, name)
    if ground is not None:
        xy = nearest_ground(context, ground, anchor)
        if xy is not None:
            return xy, ground
    logger.warning(f"scene location '{location}' is not a place on this map; the scene stays where the cast is")
    return None


# ---------------------------------------------------------------------------- what is in view
def surroundings(who):
    """What a person sees from where it stands: (the view in words by direction, the ids of the
    places seen, the names of the people seen, the paths seen, one line of summary)."""
    from sim import map as grid_map
    context = who.context
    if _land(context) is not None:
        return _land(context).surroundings(who)
    seen = who.mapAgent.look()
    view = {direction: grid_map.extract_direction_info(_grid(context), seen, direction)
            for direction in ('Current', 'North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest')}
    return grid_map.hash_direction_info(view, world=_grid(context))


def view_report(who):
    """The same, as the full report a narrator is shown."""
    if _land(who.context) is not None:
        return _land(who.context).view_report(who)
    return who.mapAgent.get_detailed_visibility_description()


def find(context, text):
    """The place a reference gives ('#Spring3', 'spring3'), or None."""
    rid = str(text or '').strip().replace('#', '')
    return place(context, rid) or called(context, rid)


def listed(context):
    """(id, place) for every place, in the order the world came to have them."""
    return [(p.id, p) for p in places(context)]


def built(context):
    """Has this performance a world laid out (a play with no map has not)?"""
    return _grid(context) is not None or _land(context) is not None


def put_at(who, rid):
    """Stand a person at a named place (a play file, before the play begins). False when this
    world has no such place: the person stays where it was."""
    p = place(who.context, rid)
    if p is None:
        logger.warning(f"{who.name} was to stand at '{rid}', which is no place in this world")
        return False
    put(who, p.xy)
    return True


def beside(who, other):
    """Stand a person where another stands."""
    put(who, at(other))


def picture(context):
    """The world drawn for whoever is running the play (the server's own window)."""
    land = _land(context)
    if land is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        for p in land.table.values():
            ax.add_patch(plt.Circle(p.xy, p.radius, fill=p.region, alpha=0.15 if p.region else 1.0, color='green' if p.region else 'black'))
            if not p.region:
                ax.annotate(p.id, p.xy, fontsize=6)
        for c in context.everyone():
            if on(c):
                ax.plot(*at(c), 'r*'); ax.annotate(c.name, at(c), color='red', fontsize=8)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.autoscale_view()
        return fig
    from sim.mapview import MapVisualizer
    viz = MapVisualizer(_grid(context))
    viz.draw_terrain_and_infrastructure()
    return viz


# ---------------------------------------------------------------------------- the rules of nearness
# Each is asked by name and answered here, once. On the grid they are counts of tiles, as their
# callers had them; with places that have positions in metres they are the rules agreed for those
# (docs/physical-world-design.md).
EARSHOT_TILES = 4


def span(context, a, b):
    """How far apart two positions are, as a number to sort and compare by (not for prompts)."""
    return tiles(a, b)


def at_place(context, xy, p):
    """Is a position at this place?"""
    return tiles(xy, p.xy) <= AT_PLACE_TILES


def on_the_spot(context, xy, spot):
    """Is a position at a spot (where a scene is set, where someone stands)?"""
    return tiles(xy, spot) <= AT_PLACE_TILES


def at_its_foot(context, xy, p):
    """Is a position right at a place, to use it without a step?"""
    return tiles(xy, p.xy) <= 1


def together(context, a, b):
    """Are two positions near enough to deal with one another: to hand a thing over, to see what
    the other holds, for a fire at one to warm the other?"""
    return tiles(a, b) <= 2


def lying_here(context, at, xy):
    """Is a thing at `at` at hand for someone at `xy`? (A clue laid at a bench two tiles from the
    cafe was listed as lying in the cafe: the same place, not only near.)"""
    d = tiles(at, xy)
    return d <= 1 or (d <= 2 and _nearest_id(context, at) == _nearest_id(context, xy))


def _nearest_id(context, xy):
    found = nearest(context, xy)
    return found[1].id if found else None


def in_earshot(context, a, b):
    """Can someone at one position hear a line said at the other, and witness what is done there?"""
    return tiles(a, b) <= EARSHOT_TILES


def a_step(context, a, b):
    """Is going from one position to the other only a step within the place one is at?"""
    return tiles(a, b) <= EARSHOT_TILES and where(context, a) == where(context, b)


def can_walk_over(context, a, b, cfg=None):
    """Is `b` near enough to `a` to be walked over to in the course of a deed aimed at it?"""
    return tiles(a, b) <= (cfg or {}).get('approach_tiles', 8)


def in_sight(context, xy, p):
    """Is a place in sight from a position?"""
    return tiles(xy, p.xy) <= 10


def in_range(context, xy, p):
    """Is a place near enough to be thought of as somewhere to go and do a thing?"""
    return tiles(xy, p.xy) <= 12


def in_the_neighbourhood(context, xy, p, within=6):
    """Is a place near enough for what has been drawn from it to be known of?"""
    return tiles(xy, p.xy) <= within


def minutes(context, a, b):
    """About how long the walk from one position to another is, as people are told it."""
    return told_minutes(context, tiles(a, b))


def told_minutes(context, n):
    from sim import survival
    return max(1, round(n * (survival.config(context) or {}).get('tile_minutes', 2) * 1.5))


def walk(context, a, b):
    """(minutes, effort) of the walk from `a` to `b` as the clock and the body are charged it."""
    from sim import survival
    n = tiles(a, b)
    try:
        cost = pace(context, b)
    except Exception:
        cost = 1.5
    return n * cost * (survival.config(context) or {}).get('tile_minutes', 2), n


def far(context, a, b):
    """A distance in the words a line of the stage gives it: '6 tiles'."""
    return f'{tiles(a, b)} tiles'


def danger_start(context, xy, rng):
    """Where a danger that comes for someone at a position first stands."""
    return offset(xy, rng.choice([-3, 3]), rng.choice([-2, 2]))


def danger_reaches(context, xy, other):
    """Does a danger come for someone at `xy` also reach someone at `other`?"""
    return tiles(xy, other) <= 4


def danger_source_near(context, xy, p, within=6):
    """Is the place a danger comes from near enough to a position for it to come on?"""
    return tiles(xy, p.xy) <= within


def danger_clear(context, xy, danger_at, clear=8):
    """Has someone at a position got clear of a danger?"""
    return tiles(xy, danger_at) >= clear


def danger_words(context, xy, danger_at):
    """How near a danger is, in words."""
    d = tiles(xy, danger_at)
    return 'close by' if d <= 2 else 'a short way off' if d <= 6 else 'some way off'


def apart_words(context, positions):
    """How far apart the people of a picture stand, in words."""
    spread = max(tiles(a, b) for a in positions for b in positions)
    return next((words for limit, words in ((1, 'side by side'), (3, 'a few steps apart'), (8, 'some way apart')) if spread <= limit), 'far apart')


def danger_picture_words(context, positions, danger_at):
    """How near a danger is to the people of a picture, in words."""
    near = min(tiles(xy, danger_at) for xy in positions)
    return next((words for limit, words in ((2, 'close by'), (6, 'nearby')) if near <= limit), 'in the distance')


def indoors(context, xy):
    """Is a position inside walls?"""
    return region(context, xy) in ('Building', 'House')


def sheltered(context, xy):
    """Is a position under shelter by where it is (inside a hut, a cave, a thicket)?"""
    return False


def away(context, a, b):
    """How far and which way `b` is from `a`, as a person is told it."""
    return f'about {minutes(context, a, b)} minutes walk {bearing(a, b).lower()}'


def reach_words(context, a, b, name):
    """How far a place at `b` is from `name`, who stands at `a`, for a list of places."""
    return f'{far(context, a, b)} from {name}, about {minutes(context, a, b)} min walk'


# With places laid out in metres (sim/land.py), each of these is answered there.
def _answered_by_land(fn):
    def asked(context, *args, **kwargs):
        land = _land(context)
        return getattr(land, fn.__name__)(*args, **kwargs) if land is not None else fn(context, *args, **kwargs)
    asked.__name__, asked.__doc__ = fn.__name__, fn.__doc__
    return asked


for _name in ('places', 'place', 'called', 'kind_called', 'nearest', 'named_in', 'add_place', 'has_owner_part', 'listed', 'find', 'start',
              'region', 'regions', 'region_called', 'nearest_ground', 'pace', 'where', 'resolve', 'span', 'at_place', 'on_the_spot',
              'at_its_foot', 'together', 'lying_here', 'in_earshot', 'a_step', 'can_walk_over', 'in_sight', 'in_range',
              'in_the_neighbourhood', 'minutes', 'told_minutes', 'walk', 'far', 'danger_start', 'danger_reaches', 'danger_source_near',
              'danger_clear', 'danger_words', 'apart_words', 'danger_picture_words', 'indoors', 'sheltered', 'reach_words', 'away'):
    globals()[_name] = _answered_by_land(globals()[_name])
