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
import re

from sim import survival
from sim.trace import trace

logger = logging.getLogger('simulation_core')

SAME_PLACE = ('', 'same', 'same place', 'same location', 'same as before', 'here', 'unchanged')
AT_PLACE_TILES = 2        # this close to a place is 'at' it
GAZETTEER_PLACES = 14     # places listed for the playwright, nearest first

# One wording of the scene 'location' field and of the continuity rule, for every prompt that
# writes scenes (character act plans, the integrated act, a replanned scene).
LOCATION_FIELD = ('"location": string, // a place named under Places (e.g. its exact name), '
                  'or "same" to stay where the previous scene ended\n'
                  ' "tempo": "calm" | "tense" | "urgent", // urgent: danger or a deadline is on them now, and nobody has time for talk\n'
                  ' "business": {"<character name>": ["<action>@<place>"]}, // the physical actions this scene\'s ending depends on, '
                  'as listed under Places (e.g. "drink@Spring2"); {} when it depends on none')
SCENE_RULES = """Scenes within an act are continuous: each starts where and when the previous one ended.
Set a scene where its action can happen: Places lists what can be done at each named place.
Give a scene a different location only when the action must move there; the cast then walks, which takes the time shown.
A jump in time, or to a distant place, belongs at the start of an act.
What a scene's ending says about bodies and things (someone fed, watered, rested, armed, a paper signed or handed over) is true only if a character does the action in that scene: list it under business. Saying it does not make it so."""


# a move that leaves the others behind: said in the act or its reason
ALONE = re.compile(r"\b(flee|flees|fleeing|fled|escap\w+|run(s|ning)? (away|off)|slip(s|ping)? (away|off|out)|sneak\w*|"
                   r"storm(s|ing)? (off|out)|walk(s|ing)? (out|off|away)|away from|leav(e|es|ing) (him|her|them|\w+ behind)|"
                   r"abandon\w*|alone|by (him|her|my)self|hid(e|es|ing) from|scout(s|ing)? ahead|go(es)? ahead)\b", re.I)


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
    """(xy, name) of the place a scene location names, seen from `anchor` (where the cast is):
    a resource, the nearest resource of a kind, or the nearest tile of a terrain (the anchor
    itself when it already stands on that terrain). None when the scene stays where the cast
    is: 'same', or a name that is no place on this map."""
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
            return anchor, terrain.name   # already there: the scene is played where the cast stands
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
    ('' when nobody moves). A scene continues the one before it: unless its location names
    somewhere else, it is played where the characters who were in the previous scene now are,
    and newcomers join them there. (It was played where its first-named character stood, so a
    scene that brought Francoise to Jean walked Jean to Francoise's farm instead.) With nobody
    carried over, it is the previous scene's place; in a first scene, where its first character is.
    A scene that opens an act puts its cast in place; any other scene has them walk, and the
    clock moves on by the longest walk."""
    if not characters:
        return ''
    previous = getattr(context, 'previous_scene', None) or {'cast': (), 'xy': None}
    carried = [c for c in characters if c.name in previous['cast']]
    anchor = here(carried[0]) if carried else previous['xy'] or here(characters[0])
    target = resolve(context, scene.get('location'), anchor)
    xy, place = target if target else (anchor, None)
    scene['place'] = place or where(context, xy).removeprefix('at ')
    context.previous_scene = {'cast': tuple(c.name for c in characters), 'xy': xy}
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
        line = f"{names} walk{'s' if len(movers) == 1 else ''} to {scene['place']} ({round(longest)} min)."
    trace.event('scene_place', place=scene['place'], movers=[n for n, _ in movers], walked=not opens_act,
                minutes=round(longest, 1))
    return line


def gazetteer(context):
    """The stage for a playwright prompt: where each cast member is, and the named places
    nearest the cast with what each offers (and to whom, when a place is someone's own) and
    how far it is. Every kind of place that offers an action is listed at least once, however
    far, so the plan knows it exists."""
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
        offers = survival.offers(context, res, cast)
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


def company(character):
    """The scene partners standing with a character: those who come along when it walks."""
    return [c for c in character.bystanders() if getattr(c, 'mapAgent', None) is not None]


def walk(character, xy, place, alone=False, to=True):
    """A character walks to a tile, and its scene partners come along unless it goes alone (a
    scene's cast stays together: Samantha used to reach the spring an hour ahead of Joe, who was
    then walked there by the next scene). One clock advance for the party. Returns the words for
    the audience, said of the walker ('walks to Spring2 with Joe (14 tiles, 70 min)'), or ''
    when nobody moves."""
    context = character.context
    tiles = _distance(here(character), xy)
    if tiles == 0:
        return ''
    was_with = company(character)
    # a few steps within the place (to the woodpile and back) are taken alone; a real walk takes the party
    party = [] if alone or tiles <= character.EARSHOT_TILES else was_with
    minutes = max(survival.travel(c, *xy, advance_clock=False) for c in [character] + party)
    context.simulation_time += survival.timedelta(minutes=minutes)
    names = ' and '.join(c.name for c in party)
    # steps within a place are not told: a page of 'walks South (15 min)' tracked nothing
    line = '' if tiles <= character.EARSHOT_TILES else (
        f"walks {'to ' if to else ''}{place}" + (f' with {names}' if party else '') + f" ({max(1, round(minutes))} min)")
    for other in was_with:
        if other not in party and not alone:
            continue   # it has only stepped across the place
        other.add_perceptual_input(f'You walk with {character.name} {"to " if to else ""}{place}.' if other in party
                                   else f'You see {character.name} go off {"to " if to else ""}{place}, leaving you.',
                                   percept=False, mode='visual' if other not in party else 'movement')
        other.look()
    trace.event('walk', character=character.name, place=place, party=[c.name for c in party], alone=alone,
                tiles=tiles, minutes=round(minutes, 1))
    return line


def approach(character, target):
    """Before a character acts on or speaks to a scene partner who is out of reach, it walks to
    them (Joe thrust a branch at Samantha from three tiles off, and waved at her from five).
    Returns the words for the audience, or ''."""
    if (target is None or target is character or getattr(target, 'mapAgent', None) is None
            or getattr(character, 'mapAgent', None) is None
            or getattr(target, 'current_scene', None) is not character.context.current_scene
            or character.context.current_scene is None):
        return ''
    if _distance(here(character), here(target)) <= AT_PLACE_TILES:
        return ''
    return walk(character, here(target), target.name, alone=True)


def tempo(character):
    """calm, tense or urgent: the scene's own tempo, and urgent whenever a threat is on the character."""
    scene = character.context.current_scene or {}
    if survival.active_threat_for(character):
        return 'urgent'
    value = str(scene.get('tempo') or 'calm').strip().lower()
    return value if value in ('calm', 'tense', 'urgent') else 'calm'


def speech_tempo(character):
    """A line for the speech prompts when the moment is not calm."""
    return {'urgent': "- This moment is urgent: twelve words at most, and no explaining; but it is still your character's own voice and way with words, not a generic shout.",
            'tense': "- The moment is tense: one sentence, fifteen words at most."}.get(tempo(character), '')


def pending_business(character):
    """The scene's planned physical actions that this character has not yet done
    ('drink@Spring2'): what the scene's ending depends on (the scene 'business' field)."""
    scene = getattr(character, 'current_scene', None) or {}
    business = scene.get('business')
    if not isinstance(business, dict):
        return []
    mine = next((v for k, v in business.items() if str(k).lower() == character.name.lower()), None) or []
    mine = [mine] if isinstance(mine, str) else mine
    done = getattr(character, 'scene_deeds', [])
    return [str(b) for b in mine if str(b).split('@')[0].strip().lower() not in done]


def whereabouts(character):
    """Where the others in the scene are, seen from a character: 'Samantha is 5 tiles East of
    you, out of reach'. The act prompt named who was visible but not how far."""
    scene = character.context.current_scene
    if scene is None or getattr(character, 'mapAgent', None) is None:
        return ''
    out = []
    for other in character.context.actors + character.context.extras + character.context.npcs:
        if other is character or getattr(other, 'current_scene', None) is not scene or getattr(other, 'mapAgent', None) is None:
            continue
        d = _distance(here(character), here(other))
        minutes = round(d * (survival.config(character.context) or {}).get('tile_minutes', 2) * 1.5)
        out.append(f'{other.name} is beside you' if d <= AT_PLACE_TILES
                   else f'{other.name} is about {minutes} minutes\' walk {_direction(here(character), here(other))} of you, out of reach until one of you walks over')
    return '; '.join(out)


def _stems(text, skip=()):
    return {w[:5] for w in re.findall(r"[a-z0-9_]{4,}", text.lower()) if w not in skip}


def stage_line(name, deed, result):
    """What the audience reads for a physical action: the deed, then only what the result adds.
    The result used to follow whole, so a deed was told two or three times over (the actor's
    words, the world's fact, the narrator's restatement) with the body's condition in brackets."""
    deed = re.sub(rf'^\s*{re.escape(name)}\s+', '', (deed or '').strip()).rstrip('. ')
    if '@' in deed:   # an action label given as the deed ('gather berries@Berries9'): the result tells it
        deed = ''
    result = re.sub(r'\s*\([^()]*\bis now\b[^()]*\)', '', result or '').strip()
    skip = {name.lower()}
    seen, kept = [_stems(deed, skip)], []
    for sentence in re.split(r'(?<=[.!?])\s+', result):
        stems = _stems(sentence, skip)
        if not stems or any(len(stems & s) / len(stems) >= 0.6 for s in seen):
            continue
        kept.append(sentence.strip())
        seen.append(stems)
    return ' '.join(([deed + '.'] if deed else []) + kept[:2]).strip()


def plain(context, text):
    """Audience text in the words of the world: the engine's numbered place names become ordinary
    ones ('Fallen_Branch9' -> 'the fallen branch', 'Berries15' -> 'the berries') and tile counts
    go ('walks 4 tiles Southwest to' -> 'walks Southwest to'). Places an author named ('HotelBar')
    are left as they are."""
    world = getattr(context, 'map', None)
    if not text or world is None:
        return text
    def ordinary(match):
        rid = match.group(0)
        if rid not in world.resource_registry:
            return rid
        return 'the ' + re.sub(r'\d+$', '', rid).replace('_', ' ').lower()
    text = re.sub(r'\b[A-Z][A-Za-z_]*\d+\b', ordinary, text)
    text = re.sub(r'\s*\b\d+ tiles?\b(?= )', '', text)
    text = re.sub(r'\b([Tt]he) (\w+ )?the\b', r'\1 \2'.rstrip(), text)   # 'the dense the thicket'
    return re.sub(r'(^\s*[\'"]?|[.!?]\s+)the ', lambda m: m.group(1) + 'The ', text)
