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
import os
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
                  ' "when": "same" | "dawn" | "morning" | "midday" | "afternoon" | "dusk" | "night", // "same": it follows straight on '
                  'from the scene before. Name a part of the day only when what happens in the scene depends on it (a night watch, '
                  'a dawn departure): the play then moves on to the next such time\n'
                  ' "tempo": "calm" | "tense" | "urgent", // urgent: danger or a deadline is on them now, and nobody has time for talk\n'
                  ' "business": {"<character name>": ["<action>@<place>"]}, // the physical actions this scene\'s ending depends on, '
                  'as listed under Places (e.g. "drink@Spring2"); {} when it depends on none')
SCENE_RULES = """Scenes within an act are continuous: each starts where the previous one ended, and when it ended unless its "when" names a later part of the day.
Set a scene where its action can happen: Places lists what can be done at each named place.
Give a scene a different location only when the action must move there; the cast then walks, which takes the time shown.
A scene's title, set-up and ending must fit its "when": a scene called a night watch has "when": "night". When a scene moves the play on, hours pass off stage and the cast is taken to have eaten, drunk and rested as it could: do not spend the scene on that. Move the play on at most once within an act; the clock times you write are not used for this.
A jump to a distant place belongs at the start of an act.
A scene's set-up and ending do not say what becomes of bodies (someone fed, watered, rested, hurt), nor how a danger ends: the world settles that when the scene is played. What a scene needs done physically for its ending to make sense (a drink taken, a thing handed over, a paper signed) is listed under business, and is so only if a character does it in that scene."""


def here(character):
    return (character.mapAgent.x, character.mapAgent.y)


# --------------------------------------------------------------------------- presence
# Who is in a scene. A character could say its goodbyes twice and still be standing there an
# hour later (its exit was a task that had not come up, and words move nobody), or walk out in
# a deed that changed nothing, and then be given a closing drink. And real walks, a misread
# destination or going after a partner, took people across the map while the page kept them at
# the table. A character now leaves a scene when the one who writes its line or deed says so
# (the act and dialog formats ask: leaves, yes or no). Having left, it is off stage until the
# scene ends: nobody's bystander, no cut-ins, no closing business, no further tasks, and an act
# aimed at it is sent back (sim/record.py deeds).
def off_stage(character):
    """Has this character left the scene now being played?"""
    scene = getattr(getattr(character, 'context', None), 'current_scene', None)
    return scene is not None and getattr(character, 'left_scene', None) is scene


def leave(character, party=(), told=False):
    """`character` leaves the scene being played. The others are told; talk with it ends.
    Those of `party`, who are going with it, are not told it has left them (Kidd was told
    'Lanya has left.' as they walked out together, and reasoned from there that he was alone).
    told: the deed that takes it out is already in the scene's record ('Samantha leaves with
    Joe.'); the exit is not recorded again (wrap-ups read 'Samantha leaves with Joe. Samantha
    leaves. Joe leaves.', and the wrap-up is what each character remembers of the scene).
    Returns False when there is no scene to leave or it has left already."""
    context = character.context
    scene = getattr(context, 'current_scene', None)
    if scene is None or off_stage(character) or getattr(character, 'current_scene', None) is not scene:
        return False
    remaining = [c for c in context.actors + context.extras + context.npcs
                 if c is not character and getattr(c, 'current_scene', None) is scene]
    character.left_scene = scene
    character.current_scene = None   # everything that asks 'in this scene?' now answers no
    exit_ = {'time': context.simulation_time.strftime('%H:%M'), 'who': character.name,
             'with': [c.name for c in party if c is not character]}
    for record in ('scene_exits', 'act_exits'):   # what the outcome check reads of who went
        if isinstance(getattr(context, record, None), list):
            getattr(context, record).append(exit_)
    for other in remaining:
        if other not in party:
            other.add_perceptual_input(f'{character.name} has left.', percept=False, mode='visual')
        for a, b in ((other, character), (character, other)):
            model = a.actor_models.get_actor_model(b.name)
            if model is not None and model.dialog.active:
                model.dialog.deactivate_dialog()
            try:
                a._pop_dialog_task(b)   # and the task of talking with it
            except Exception:
                pass
    if not told:
        context.add_event(f'{character.name} leaves')
    trace.event('exit', character=character.name, left_behind=[c.name for c in remaining])
    return True


def clock(context, date=False):
    """The time in the play, with the part of the day in words: '10:12, morning'. Every prompt
    that writes about the world gets this (a set-up written at 10:12 spoke of 'the approaching
    dusk', and the cast then slept through the middle of the day)."""
    t = context.simulation_time
    part = part_of(t)
    return f"{t:%Y-%m-%d %H:%M} ({part})" if date else f"{t:%H:%M}, {part}"


def part_of(t):
    return ('night' if t.hour < 5 else 'dawn' if t.hour < 7 else 'morning' if t.hour < 12 else 'midday' if t.hour < 14
            else 'afternoon' if t.hour < 18 else 'dusk' if t.hour < 20 else 'night')


PART_BEGINS = {'dawn': (5, 30), 'morning': (8, 0), 'midday': (12, 30), 'afternoon': (15, 0), 'dusk': (18, 30), 'night': (21, 0)}
PART_WORDS = (('night', r'night|midnight|dark'), ('dusk', r'dusk|evening|sunset|twilight'), ('dawn', r'dawn|sunrise|daybreak|first light'),
              ('midday', r'midday|noon'), ('afternoon', r'afternoon'), ('morning', r'morning'))


def part_named(when):
    """The part of the day a scene's 'when' names ('night', 'next morning', 'at dusk'), or None
    for 'same', nothing, or words that name none."""
    text = str(when or '').strip().lower()
    return next((part for part, words in PART_WORDS if re.search(rf'\b({words})\b', text)), None)


def next_time(now, when):
    """When the play moves on to for a scene that needs `when`: the next time it is that part of
    the day, or None when it already is (or `when` names none). Never back, and 'next morning'
    said in the morning is tomorrow's."""
    part = part_named(when)
    again = bool(re.search(r'\b(next|tomorrow|following)\b', str(when or '').lower()))
    if part is None or (part == part_of(now) and not again):
        return None
    hour, minute = PART_BEGINS[part]
    then = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if then <= now or again:
        then += survival.timedelta(days=1)
    return then


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


def places_named(context, text):
    """The places on the map that a text names, longest name first. A place was looked for as a
    word ending in digits ('Spring3'), which the places an author names ('HuOffice', 'Cafe') are
    not: 'Walk to HuOffice to draft the papers breaking Hu's contract' walked Xue to Hu."""
    low = (text or '').lower()
    return [rid for rid in sorted(context.map.resource_registry, key=len, reverse=True)
            if re.search(rf'(?<![a-z0-9_]){re.escape(rid.lower())}(?![a-z0-9_])', low)]


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
    left = where(context, here(character))
    was_with = company(character)
    # a few steps within the place (to the woodpile and back) are taken alone; a walk to another
    # place takes the party, however short (Joe walked three tiles to the spring alone, then
    # walked back to say it was 'right there, past the ferns')
    party = [] if alone or (tiles <= character.EARSHOT_TILES and where(context, xy) == left) else was_with
    minutes = max(survival.travel(c, *xy, advance_clock=False) for c in [character] + party)
    context.simulation_time += survival.timedelta(minutes=minutes)
    names = ' and '.join(c.name for c in party)
    # steps within a place are not told: a page of 'walks South (15 min)' tracked nothing. A
    # short walk that ends at another place is (Joe went from the apple tree to the thicket
    # unseen, and the page had him handed a branch at the tree)
    line = '' if tiles <= character.EARSHOT_TILES and where(context, xy) == left else (
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
            'tense': "- The moment is tense: keep it short, a sentence or two, as this person talks when on edge."}.get(tempo(character), '')


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
        minutes = survival.walk_minutes(character.context, d)
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


NUMBER_WORDS = ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a few', 'several')


def _far(count):
    n = int(count) if count.isdigit() else (NUMBER_WORDS.index(count.lower()) + 1 if count.lower() in NUMBER_WORDS[:10] else 4)
    return 'a few paces' if n <= 1 else 'a short way' if n <= 4 else 'some way' if n <= 10 else 'a long way'


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
    # a count of tiles before punctuation or spelled out ('visibility to 32 tiles,', 'three tiles
    # off'): the map's unit has no meaning on the page; how far it is in plain words
    text = re.sub(rf"\b(\d+|{'|'.join(NUMBER_WORDS)}) tiles?\b", lambda m: _far(m.group(1)), text, flags=re.I)
    text = re.sub(r'\b([Tt]he) (\w+ )?the\b', lambda m: f"{m.group(1)} {m.group(2) or ''}".rstrip(), text)   # 'the dense the thicket'
    return re.sub(r'(^\s*[\'"]?|[.!?]\s+)the ', lambda m: m.group(1) + 'The ', text)


# --------------------------------------------------------------------------- pictures
# What an image of the stage is asked for. It used to be the first sentences of the Situation
# text, which since that text is rendered from the world begins with the play's opening
# description: every picture of a play was asked for with the same morning and the same two
# people 'looking around in confusion'. A picture is now of where the play stands: the place,
# the light at this hour, who is there as they look, a danger if one is on them, and for an
# act, what is being done. One style for a play (`image_style` on the context or its scenario).
LIGHT = {'night': 'night, dark, faint moonlight', 'dawn': 'dawn, first grey light', 'morning': 'clear morning light',
         'midday': 'midday, high sun', 'afternoon': 'afternoon light, long shadows beginning', 'dusk': 'dusk, fading light'}
DEFAULT_STYLE = 'Photorealistic, 35mm film'


def style(context):
    return (getattr(context, 'image_style', None) or getattr(getattr(context, 'map', None) and context.map.scenario_module, 'image_style', None)
            or DEFAULT_STYLE)


def _looks(context, character):
    """A character as it looks: the first sentence of its description, without its name. The
    whole name goes ('Hu Manli, a healthy ...' had become 'Manli, a healthy ...')."""
    text = context._image_description(character).rstrip('. ')
    # a family or given name left at the head, before the first comma: 'Manli, a healthy ...'
    head, sep, rest = text.partition(',')
    lead = re.sub(r'^(you are|i am)\s+', '', character.character.strip(), flags=re.I)
    if sep and head.strip() and lead.startswith(f'{character.name} {head.strip()}'):
        text = rest.strip()
    return text


# Terrain a place stands in that makes it a room: a picture of it is an interior.
INDOOR_TERRAIN = {'Building', 'House'}


def _place_name(context, place, res):
    """An author-named place in words: 'XiaoyuOffice' -> "Xiaoyu's office", 'HuDingApartment' ->
    "Hu and Ding's apartment", 'HotelBar' -> 'the hotel bar'. Engine-numbered names go through
    plain()."""
    if not re.fullmatch(r'[A-Z][A-Za-z]*', place or ''):
        return plain(context, place)
    words = re.findall(r'[A-Z][a-z]*', place)
    cast = {c.name for c in list(context.actors) + list(getattr(context, 'extras', []))}
    owners = []
    while words and words[0] in cast:
        owners.append(words.pop(0))
    thing = ' '.join(w.lower() for w in words) or 'place'
    if owners:
        return f"{' and '.join(owners)}'s {thing}"
    return f'the {thing}'


def _place_words(context, cast):
    """(words for the place, whether it is a room)."""
    scene = context.current_scene or {}
    place = scene.get('place') or scene.get('location') or where(context, here(cast[0])).removeprefix('at ')
    res = context.map.resource_registry.get(place)
    words = _place_name(context, place, res) if res else plain(context, place)
    told = (res or {}).get('description')
    indoor = False
    if res and res.get('location'):
        x, y = res['location']
        indoor = context.map.patches[x][y].terrain_type.name in INDOOR_TERRAIN
    return (f"{words} ({told[0].lower() + told[1:]})" if told and told.lower() not in words.lower() else words), indoor


def _on_stage(context):
    """Who is in the scene: the scene plan's cast where there is one (extras too: Xiaoyu was
    never in a picture of his own office), else those marked as in it, else the main cast. The
    scene picture is taken before the cast is marked, and showed the whole main cast."""
    everyone = [c for c in list(context.actors) + list(getattr(context, 'extras', []))
                if getattr(c, 'mapAgent', None) is not None]
    scene = context.current_scene
    named = []
    if isinstance(scene, dict):
        for n in scene.get('action_order') or list((scene.get('characters') or {}).keys()):
            c = next((c for c in everyone if c.name.lower() == str(n).strip().lower()), None)
            if c is not None and c not in named and not off_stage(c):
                named.append(c)
    if named:
        return named
    marked = [c for c in everyone if scene is not None and getattr(c, 'current_scene', None) is scene]
    return marked or [c for c in context.actors if getattr(c, 'mapAgent', None) is not None]


def picture(context, doing=''):
    """The prompt for a picture of the stage as it stands, or None where there is no map to
    stand on. doing: what is being done in the foreground, in plain words. The place comes
    first, and a room is shown from inside: the play's opening sentence ('a modern chinese
    urban setting with a mix of buildings, roads ...') led every picture, after 'Wide view',
    and a meeting in an office was drawn round a desk in the street."""
    world = getattr(context, 'map', None)
    if world is None or not context.actors:
        return None
    on_stage = _on_stage(context)
    if not on_stage:
        return None
    setting = ' '.join(context.initial_state.split('.')[0].split()).strip().rstrip('.')
    place, indoor = _place_words(context, on_stage)
    light = LIGHT[part_of(context.simulation_time)]
    people = '; '.join(_looks(context, c) for c in on_stage[:4])
    threat = getattr(context, 'active_threat', None)
    if indoor:
        # the play's opening sentence describes the town, not the room: given here it brought the street in
        where_ = [f"Interior: inside {place}, {light}"]
    else:
        where_ = ['Wide view', f"{place}, {light}", setting]
    parts = [style(context), *where_,
             f"{'One person' if len(on_stage) == 1 else str(len(on_stage)) + ' people, no others'}: {people}"]
    if threat:
        parts.append(f"Close by, {threat['hazard']['description']}")
    if doing:
        parts.append(f"In the foreground, one of them {plain(context, doing).strip().rstrip('.')[:220]}")
    return '. '.join(p.strip().rstrip('.') for p in parts if p and p.strip()) + '.'


def portrait(context, character, mood=''):
    """The prompt for a character's portrait: how it looks, how it feels, where it is."""
    cast = [character] if getattr(character, 'mapAgent', None) is not None else []
    where_ = f"{_place_words(context, cast)[0]}, {LIGHT[part_of(context.simulation_time)]}" if cast and getattr(context, 'map', None) else ''
    parts = [style(context), f"Head and shoulders portrait of {_looks(context, character)}",
             f"Expression: {mood.strip().rstrip('.')}" if mood and mood.strip() else '',
             f"Behind, out of focus: {where_}" if where_ else '']
    return '. '.join(p.strip().rstrip('.') for p in parts if p) + '.'


# --------------------------------------------------------------------------- pictures from facts
# A picture of the moment the audience has just been told, from what the world keeps: how each
# person looks now (hurt, worn out, what they hold), how they stand to each other, the danger and
# where it is, the deed in front. The prompts above draw everyone as the author first wrote
# them ('a healthy young man' after the boar had gored him) and nothing anyone holds.
#
# picture_facts / portrait_facts gather the facts (they go into the trace with every picture,
# whichever prompt is sent); picture_prompt / portrait_prompt write a prompt from them alone, so
# tools/image_ab.py can draw a recorded moment again. ATWAP_IMAGE_PROMPT=new sends these prompts;
# 'old' (the default until the A/B test is decided) sends picture() and portrait().
IMAGE_PROMPT = os.environ.get('ATWAP_IMAGE_PROMPT', 'old').strip().lower()
PROMPT_BUDGET = 300   # tokens of the image model's encoder; it reads 512 and cuts the rest unseen
APART_WORDS = ((1, 'side by side'), (3, 'a few steps apart'), (8, 'some way apart'))
THREAT_WORDS = ((2, 'close by'), (6, 'nearby'))

LOOKS = """From the description of {{$name}} below, say only how {{$name}} looks to someone seeing them: age, build, hair, face, clothes.
One phrase beginning with 'a' or 'an', 30 words at most. No name, no health or mood, nothing of their past or character.

{{$description}}

End your response with: </end>"""


def ensure_looks(character):
    """How a character looks, written once from its description (one call). Pictures use it in
    place of the description's first sentence, which carries 'healthy' and the like."""
    if getattr(character, 'looks', None):
        return character.looks
    from utils.Messages import UserMessage
    reply = character.llm.ask({'name': character.name, 'description': character.original_character},
                              [UserMessage(content=LOOKS)], tag='looks', temp=0.2, max_tokens=60, stops=['</end>'])
    looks = ' '.join((reply or '').split()).strip().rstrip('.')
    character.looks = looks or None
    return character.looks


def _visible(body):
    """What a picture can show of a body: harm once it is felt, any need at its worst ('exhausted',
    'desperately thirsty'), shelter and dress flags. Not what is carried: that is listed apart."""
    if body is None:
        return ''
    harm = survival._harm(body.cfg)['name']
    parts = []
    for need, value in body.levels.items():
        if (need == harm and value >= survival.NEED_FELT) or value >= 0.85:
            parts.append(survival._word(body.cfg, need, value))
    if body.sheltered:
        parts.append('in shelter')
    return ', '.join(parts + sorted(body.flags))


def _carried(body):
    if body is None:
        return []
    hidden = tuple(body.cfg.get('private_items', ()))
    return [i for i in body.inventory if not (hidden and i.startswith(hidden))]


def _person(context, character):
    body = getattr(character, 'body', None)
    return {'name': character.name, 'looks': getattr(character, 'looks', None) or _looks(context, character),
            'condition': _visible(body), 'holding': _carried(body)}


def _words_for(distance, table, far):
    return next((words for limit, words in table if distance <= limit), far)


def _no_speech(text):
    """A deed without the words spoken in it: a picture cannot show them, and an image model
    letters them onto the picture. The quotation marks say where speech is."""
    text = re.sub(r'["“][^"”]*["”]', '', text or '')
    text = re.sub(r",?\s*\b(and |then )?(says|asks|whispers|shouts|calls|murmurs|mutters)\b:?\s*(?=[,.;]|$)", '', text)
    return ' '.join(text.split()).strip(' ,;:').rstrip('.')


def picture_facts(context, deed='', actor=None, setup=False):
    """What a picture of the stage is to show, as a record built from the world only; None when
    there is no map or nobody on stage. deed: what `actor` is doing in front, as the actor wrote
    it. setup: the scene has just been set, and its set-up sentence is the moment. People are
    named in the prompt, so that a deed ('Joe holds the watch out to Samantha') says who is who."""
    world = getattr(context, 'map', None)
    if world is None or not context.actors:
        return None
    on_stage = _on_stage(context)
    if not on_stage:
        return None
    shown = on_stage[:4]
    place, indoor = _place_words(context, on_stage)
    facts = {'style': style(context), 'place': place, 'indoor': indoor,
             'light': LIGHT[part_of(context.simulation_time)],
             'setting': '' if indoor else ' '.join(context.initial_state.split('.')[0].split()).strip().rstrip('.'),
             'people': [_person(context, c) for c in shown], 'apart': '', 'threat': '', 'deed': '', 'setup': ''}
    if len(shown) > 1:
        spread = max(_distance(here(a), here(b)) for a in shown for b in shown)
        facts['apart'] = _words_for(spread, APART_WORDS, 'far apart')
    threat = getattr(context, 'active_threat', None)
    if threat:
        near = min(_distance(here(c), threat['location']) for c in shown)
        facts['threat'] = f"{threat['hazard']['description']}, {_words_for(near, THREAT_WORDS, 'in the distance')}"
    if deed:
        facts['deed'] = _no_speech(plain(context, f"{actor.name} {deed}" if actor is not None else deed))[:220]
    scene = context.current_scene if isinstance(context.current_scene, dict) else {}
    if setup and scene.get('pre_narrative'):
        facts['setup'] = plain(context, scene['pre_narrative'])
    return facts


def portrait_facts(context, character, mood=''):
    cast = [character] if getattr(character, 'mapAgent', None) is not None and getattr(context, 'map', None) else []
    return {'style': style(context), 'person': _person(context, character), 'mood': (mood or '').strip().rstrip('.'),
            'place': _place_words(context, cast)[0] if cast else '', 'light': LIGHT[part_of(context.simulation_time)]}


def _clause(person, hands=99, named=True):
    held = person['holding'][:hands]
    return ', '.join(p for p in (person['name'] if named else '', person['looks'], person['condition'],
                                 f"carrying {', '.join(held)}" if held else '') if p)


def _join(parts):
    return '. '.join(p.strip().rstrip('.') for p in parts if p and p.strip()) + '.'


_encoder_tokens = None


def prompt_tokens(text):
    """Tokens the image model's text encoder reads for `text` (Qwen3 tokenizer of FLUX.2 klein,
    with its chat wrapper); a word count estimate where that tokenizer cannot be loaded."""
    global _encoder_tokens
    if _encoder_tokens is None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(os.environ.get('KLEIN_MODEL', 'black-forest-labs/FLUX.2-klein-4B'), subfolder='tokenizer')
            _encoder_tokens = lambda t: len(tok(t)['input_ids']) + 12
        except Exception as e:
            logger.warning(f'image prompt tokens estimated from words: {e}')
            _encoder_tokens = lambda t: int(len(t.split()) * 1.4) + 12
    return _encoder_tokens(text)


def picture_prompt(facts):
    """A prompt from picture_facts: style, place and light, one clause per person, how they
    stand, the deed in front, the danger, then the setting and the set-up sentence, which go
    first when the prompt is over budget (people and the deed never do)."""
    people = facts['people']
    who = (f"One person, no others: {_clause(people[0])}" if len(people) == 1 else
           f"{len(people)} people, no others: " + '; '.join(_clause(p) for p in people))
    where_ = ([f"Interior: inside {facts['place']}, {facts['light']}"] if facts['indoor']
              else ['Wide view', f"{facts['place']}, {facts['light']}"])
    head = [facts['style'], *where_, who,
            f"They stand {facts['apart']}" if facts['apart'] else '',
            f"In the foreground, {facts['deed']}" if facts['deed'] else '',
            f"{facts['threat'][0].upper()}{facts['threat'][1:]}" if facts['threat'] else '']
    tail = [facts['setting'], facts['setup']]
    while True:
        prompt = _join(head + tail)
        if not any(tail) or prompt_tokens(prompt) <= PROMPT_BUDGET:
            return prompt
        tail[max(i for i, t in enumerate(tail) if t)] = ''   # the set-up sentence first, then the setting


def portrait_prompt(facts):
    person = facts['person']
    return _join([facts['style'], f"Head and shoulders portrait of {_clause(person, hands=2, named=False)}",
                  f"Expression: {facts['mood']}" if facts['mood'] else '',
                  f"Behind, out of focus: {facts['place']}, {facts['light']}" if facts['place'] else ''])
