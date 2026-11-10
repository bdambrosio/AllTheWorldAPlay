"""Where a scene is played: named places, and getting the cast to them.

A scene's 'location' names a place: a resource on the map ('Well5'), a kind of resource
('Well'), a terrain ('Field'), or 'same' for wherever the previous scene ended. Scenes are
continuous: nobody moves when a scene starts unless it names a different place, and then the
cast walks there, which takes time. Only the scene that opens an act may jump to its place.

The playwright prompts get the same picture from gazetteer(): the named places near the cast,
what each offers, and how far each is, so a scene is set where its action can happen.
"""
from __future__ import annotations

import logging

from sim import survival
from sim.trace import trace

logger = logging.getLogger('simulation_core')

SAME_PLACE = ('', 'same', 'same place', 'same location', 'same as before', 'here', 'unchanged')
AT_PLACE_TILES = 2        # this close to a place is 'at' it
GAZETTEER_PLACES = 14     # places listed for the playwright, nearest first

# One wording of the scene 'location' field and of the continuity rule, for every prompt that
# writes scenes (character act plans, the integrated act, a replanned scene).
LOCATION_FIELD = ('"location": string, // a place named under Places (e.g. its exact name), '
                  'or "same" to stay where the previous scene ended')
SCENE_RULES = """Scenes within an act are continuous: each starts where and when the previous one ended.
Set a scene where its action can happen: Places lists what can be done at each named place.
Give a scene a different location only when the action must move there; the cast then walks, which takes the time shown.
A jump in time, or to a distant place, belongs at the start of an act."""


def here(character):
    return (character.mapAgent.x, character.mapAgent.y)


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _direction(frm, to):
    from sim.map import get_direction_name
    return get_direction_name(to[0] - frm[0], to[1] - frm[1]).name


def _nearest_resource(world, xy, of_type=None):
    """(distance, id, resource) of the resource nearest to xy, optionally of one type; or None."""
    found = [(_distance(xy, res['location']), rid, res) for rid, res in world.resource_registry.items()
             if of_type is None or res.get('type') == of_type]
    return min(found, key=lambda f: f[:2]) if found else None


def resolve(context, location, anchor):
    """(xy, name) of the place a scene location names, seen from `anchor` (where the cast is);
    None when the scene stays where the cast is: 'same', a terrain the cast already stands on,
    or a name that is no place on this map."""
    world = context.map
    name = str(location or '').strip().strip('#"\'.')
    for suffix in (' terrain', ' resource', ' area'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)].strip()
    if name.lower() in SAME_PLACE:
        return None
    for rid, res in world.resource_registry.items():
        if rid.lower() == name.lower():
            return tuple(res['location']), rid
    kind = world.get_resource_type(name)
    if kind is not None:
        nearest = _nearest_resource(world, anchor, kind)
        if nearest:
            return tuple(nearest[2]['location']), nearest[1]
    terrain = world.get_terrain_type(name)
    if terrain is not None:
        if world.patches[anchor[0]][anchor[1]].terrain_type == terrain:
            return None
        tiles = [(x, y) for x in range(world.width) for y in range(world.height)
                 if world.patches[x][y].terrain_type == terrain]
        if tiles:
            return min(tiles, key=lambda t: _distance(anchor, t)), terrain.name
    logger.warning(f"scene location '{location}' is not a place on this map; the scene stays where the cast is")
    return None


def where(context, xy):
    """A position in words: 'at Well5', or '6 tiles North of Well5'."""
    nearest = _nearest_resource(context.map, xy)
    terrain = context.map.patches[xy[0]][xy[1]].terrain_type
    if nearest is None:
        return f'in {terrain.name} terrain'
    d, rid, res = nearest
    return f'at {rid}' if d <= AT_PLACE_TILES else f"{d} tiles {_direction(res['location'], xy)} of {rid}"


def set_scene(context, scene, characters, opens_act):
    """Bring the scene's cast together at the scene's place and return the line that says so
    ('' when nobody moves). The place is seen from the first character of the scene: the scene
    stays with that character unless its location names somewhere else. Others join there.
    A scene that opens an act puts its cast in place; any other scene has them walk, and the
    clock moves on by the longest walk."""
    if not characters:
        return ''
    anchor = here(characters[0])
    target = resolve(context, scene.get('location'), anchor)
    xy, place = target if target else (anchor, None)
    scene['place'] = place or where(context, xy).removeprefix('at ')
    movers, longest = [], 0.0
    for character in characters:
        tiles = _distance(here(character), xy)
        if tiles <= AT_PLACE_TILES:   # already there: a step or two needs no telling
            if place:
                character.mapAgent.x, character.mapAgent.y = xy
            continue
        if opens_act:
            character.mapAgent.x, character.mapAgent.y = xy
            if getattr(character, 'body', None) is not None:
                character.body.sheltered = False
        else:
            longest = max(longest, survival.travel(character, *xy, advance_clock=False))
        movers.append((character.name, tiles))
    if not movers:
        return ''
    names = ' and '.join(name for name, _ in movers)
    if opens_act:
        line = f"{names} {'is' if len(movers) == 1 else 'are'} now at {scene['place']}."
    else:
        context.simulation_time += survival.timedelta(minutes=longest)
        line = (f"{names} walk{'s' if len(movers) == 1 else ''} to {scene['place']} "
                f"({max(t for _, t in movers)} tiles, {round(longest)} min).")
    trace.event('scene_place', place=scene['place'], movers=[n for n, _ in movers], walked=not opens_act,
                minutes=round(longest, 1))
    return line


def gazetteer(context):
    """The stage for a playwright prompt: where each cast member is, and the named places
    nearest the cast with what each offers and how far it is. Every kind of place that offers
    an action is listed at least once, however far, so the plan knows it exists."""
    world = context.map
    cast = [c for c in context.actors if getattr(c, 'mapAgent', None) is not None]
    if not cast or not world.resource_registry:
        return ''
    minutes_per_tile = (survival.config(context) or {}).get('tile_minutes', 2) * 1.5
    lines = ['Where the cast is: ' + '; '.join(f'{c.name} {where(context, here(c))}' for c in cast) + '.']
    places = sorted(((min(_distance(here(c), res['location']) for c in cast), rid, res)
                     for rid, res in world.resource_registry.items()), key=lambda p: p[:2])
    listed, kinds = [], set()
    for d, rid, res in places:
        offers = survival.offers(context, res.get('type'))
        if len(listed) < GAZETTEER_PLACES or (offers and res.get('type') not in kinds):
            listed.append((d, rid, res, offers))
            kinds.add(res.get('type'))
    for d, rid, res, offers in listed:
        nearest = min(cast, key=lambda c: _distance(here(c), res['location']))
        reach = 'here' if d <= AT_PLACE_TILES else f'{d} tiles from {nearest.name}, about {round(d * minutes_per_tile)} min walk'
        owner = getattr(res.get('properties', {}).get('owner'), 'name', res.get('properties', {}).get('owner'))
        lines.append(f"- {rid}: {res.get('description') or res.get('type')}"
                     + (f', owned by {owner}' if owner else '') + f' ({reach})'
                     + (f"; can be done there: {', '.join(offers)}" if offers else ''))
    return '\n'.join(lines)
