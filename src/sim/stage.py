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

from sim import land, ledger, survival, world
from sim.trace import trace

logger = logging.getLogger('simulation_core')

class Part:
    """An owner's part that nobody has been cast in. A world may say a kind of place has an owner
    (a mill, a farm, a hut). Until a play file, the director or a planned scene calls for that
    person (Context.cast_part), the place holds this and no character exists: nobody to be seen
    from afar, described or paid for. `name` is what a world action's refusal shows."""
    def __init__(self, place):
        self.place = place
        self.name = f'the owner of {place}'


def uncast_parts(context):
    """[(place id, Part)] for the owned places nobody has been cast at."""
    return [(rid, res.props['owner']) for rid, res in world.listed(context)
            if isinstance(res.props.get('owner'), Part)]


SAME_PLACE = world.SAME_PLACE
GAZETTEER_PLACES = 14     # places listed for the playwright, nearest first

# One wording of the scene 'location' field and of the continuity rule, for every prompt that
# writes scenes (character act plans, the integrated act, a replanned scene).
LOCATION_FIELD = ('"location": string, // a place named under Places (e.g. its exact name), '
                  'or "same" to stay where the previous scene ended\n'
                  ' "when": "same" | "dawn" | "morning" | "midday" | "afternoon" | "dusk" | "night", // "same": it follows straight on '
                  'from the scene before. Name a part of the day only when what happens in the scene depends on it (a night watch, '
                  'a dawn departure): the play then moves on to the next such time\n'
                  ' "tempo": "calm" | "tense" | "urgent", // urgent: danger or a deadline is on them now, and nobody has time for talk\n'
                  ' "kind": "confrontation" | "shared job" | "lull" | "confession" | "play", // confrontation: one presses, the other resists. '
                  'shared job: something has to be done with the hands, and the talk goes on beside it. lull: the pressure is off; they rest, eat, '
                  'look about, talk of what is not urgent. confession: one tells what they have kept back, or comes near it. play: a game, a joke, a dare')
SCENE_RULES = """Scenes within an act are continuous: each starts where the previous one ended, and when it ended unless its "when" names a later part of the day.
Set a scene where its action can happen: Places lists the places of this world, how far each is, and what the world has rules for there.
Give a scene a different location only when the action must move there; the cast then walks, which takes the time shown.
A scene's title, set-up and ending must fit its "when": a scene called a night watch has "when": "night". When a scene moves the play on, hours pass off stage and the cast is taken to have eaten, drunk and rested as it could: do not spend the scene on that. Move the play on at most once within an act; the clock times you write are not used for this.
A jump to a distant place belongs at the start of an act.
A scene's set-up and ending do not say what becomes of bodies (someone fed, watered, rested, hurt), nor how a danger ends: the world settles that when the scene is played. What a scene needs done physically for its ending to make sense (a drink taken, a thing handed over, a paper signed) is so only if a character does it in that scene.
Someone the cast list says takes part by radio or phone is in a scene by voice only: never at its place, never handed anything or touched.
Not every scene is a confrontation: people are known by how they work, rest, joke and tell things as much as by how they fight.
A character's goal in a scene is something to get, give, find out or settle there. It is never to leave or to get the others moving, unless whether they go is that scene's matter: a scene is played where it is set, and going comes when it is done.
No scene is made only of people keeping things back. Whatever each has to guard, at least one of those in a scene comes to it wanting to tell, give or settle something, and their goal says what: a scene in which every goal is to hold out, put off or find out without giving has nobody to turn it.
The scene in which two people first speak to each other is never a confrontation or a forced choice: nothing is yet between them to give a demand its weight. They do something beside each other, or one shows or offers the other something (a shared job, a lull, play), and what one notices and does not yet ask about is left for a later scene to bring out. An act of three or more scenes has at least one of another kind, and after a danger or a hard scene the next is a lull or a shared job.
What bodies do in a scene is whatever a person could really do there with their body and what is at hand, and the world says what comes of it when the scene is played. A set-up or an ending does not say what a deed came to, nor has one person moved or held by another, nor a thing that is not listed as there; what the world has rules for (buying, selling, pay, what changes hands by deed) happens only where Places lists the rule.
Trust, forgiveness, love or an alliance is never an ending merely declared ("says she trusts him"). It is shown by what it costs: telling something that exposes the teller, giving up a position held until now, or letting go of a way out (staying when one could leave, letting the other lead or keep the watch). A thing handed over counts only when it is a thing the world holds.
A confrontation ends, in that scene, with something given, taken, found out or refused for good. It never ends with the question put off ("deferred", "later", "tomorrow", "on the road", "not dropped"): what a character will not give is refused there, and the next scene goes on from the refusal, not back to the question."""


def here(character):
    return world.at(character)


# --------------------------------------------------------------------------- presence
# Who is in a scene. A character could say its goodbyes twice and still be standing there an
# hour later (its exit was a task that had not come up, and words move nobody), or walk out in
# a deed that changed nothing, and then be given a closing drink. And real walks, a misread
# destination or going after a partner, took people across the map while the page kept them at
# the table. A character now leaves a scene when the one who writes its line or deed says so
# (the act and dialog formats ask: leaves, yes or no). Having left, it is off stage until the
# scene ends: nobody's bystander, no cut-ins, no closing business, no further tasks, and an act
# aimed at it is sent back (sim/record.py deeds).
def remote(character):
    """How a character takes part in scenes from elsewhere ('by radio from mission control on
    Earth'), or ''. A play sets character.remote for someone who is in its scenes by voice
    (radio, phone) and stays at their own place: scenes do not bring them to theirs, they hear
    and are heard by the cast, and nothing is handed to them or done to them in person. Once at
    the scene's own place (the commander who walks out to the gantry), they are there in person.
    character.remote_from: the place they speak from, when it is on the map (the airlock); away
    from it they are wherever they have walked to, in person."""
    words = (getattr(character, 'remote', None) or '').strip()
    if not words or not world.on(character):
        return words
    context = getattr(character, 'context', None)
    xy = (getattr(context, 'previous_scene', None) or {}).get('xy')
    if xy is not None and world.on_the_spot(context, here(character), xy):
        return ''
    home = world.place(context, getattr(character, 'remote_from', None) or '') if context is not None else None
    if home is not None and not world.at_place(context, here(character), home):
        return ''
    return words


def supporting_note(context, plan=True):
    """For the prompts that plan and fit scenes (not the leads' own drafts): the supporting actors,
    what each wants and where each is, and that a scene set there has them in it. '' when the
    play has none in person on the map. plan: for whoever writes an act's scenes, with how such a
    person is listed in one; the stage manager, who fits a scene already written, is not told to
    fill "characters" and "action_order", which its answer does not have."""
    people = [c for c in context.supporting() if world.on(c) and not remote(c)]
    if not people:
        return ''
    def line(c):
        said = ' '.join((getattr(c, 'original_character', None) or getattr(c, 'character', '') or '').split())
        wants = '; '.join(d.text for d in (getattr(c, 'drives', None) or [])[:2])
        return f"- {c.name}: {said}" + (f" Wants: {wants}." if wants else '') + f" Now {where(context, here(c))}."
    if not plan:
        return ("\nPeople of this world who are not at the centre of the play, and where each is:\n" + '\n'.join(line(c) for c in people)
                + '\nA scene set where one of them is has that person in it; they are in no scene set elsewhere unless it has a reason for them to come.')
    return ("\nPeople of this world who are not at the centre of the play, and where each is:\n" + '\n'.join(line(c) for c in people)
            + '\nA scene set where one of them is has that person in it: list them in its "characters", with a goal of their own drawn from what they want, '
              'and in its "action_order". A goal says what that person is after in the scene, never what they tell or what anyone decides. '
              'They are in no scene set elsewhere unless that scene has a reason for them to come.')


def supporting_at(context, xy, but=()):
    """The supporting actors who are at a place in person: they are in any scene set there."""
    return [c for c in context.supporting() if c not in but and world.on(c)
            and not remote(c) and world.on_the_spot(context, here(c), xy)]


def off_stage(character):
    """Has this character left the scene now being played?"""
    scene = getattr(getattr(character, 'context', None), 'current_scene', None)
    return scene is not None and getattr(character, 'left_scene', None) is scene


def not_in_scene(character):
    """While a scene is being played: is this character no part of it? (A line said to someone
    by name was answered by them wherever on the map they were.)"""
    scene = getattr(getattr(character, 'context', None), 'current_scene', None)
    return scene is not None and character.name != 'Viewer' and getattr(character, 'current_scene', None) is not scene


def leave(character, party=(), told=False, to=None):
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
    ledger.of(context).wrote('exit')   # who is in the scene changes only as part of an event (sim/ledger.py)
    exit_ = {'time': context.simulation_time.strftime('%H:%M'), 'who': character.name,
             'with': [c.name for c in party if c is not character], 'to': to,
             'from': where(context, here(character)).removeprefix('at ')}
    for record in ('scene_exits', 'act_exits'):   # what the outcome check reads of who went
        if isinstance(getattr(context, record, None), list):
            getattr(context, record).append(exit_)
    for other in remaining:
        for a, b in ((other, character), (character, other)):
            model = a.actor_models.get_actor_model(b.name)
            if model is not None and model.dialog.active:
                model.dialog.deactivate_dialog()
            try:
                a._pop_dialog_task(b)   # and the task of talking with it
            except Exception:
                pass
    if not told:
        context.add_event(f'{character.name} leaves', doer=character, witnesses=[character] + remaining)
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


_direction = world.bearing
resolve = world.resolve          # (position, name) of the place a scene location names, or None
places_named = world.named_in    # the ids of the places a text names
where = world.where              # a position in words: 'at Well5', '6 tiles North of Well5'


def scene_spot(context, scene, characters):
    """Where the scene is played: (xy, place id or None). Where those who were in the previous scene now are, and
    newcomers join them there (it was played where its first-named character stood, so a scene that brought Francoise
    to Jean walked Jean to Francoise's farm); with nobody carried over, where its own first character (there in
    person) is; where the cast walked out of the last scene for a named place, there; or the place the scene's
    location names (a room or corner of the cast's place made for it, Context.reground_scene)."""
    previous = getattr(context, 'previous_scene', None) or {'cast': (), 'xy': None}
    carried = [c for c in characters if c.name in previous['cast']]
    # the scene is where those there in person are, not where a voice on the radio is
    carried = [c for c in carried if not (getattr(c, 'remote', None) or '').strip()] or carried
    first = next((c for c in characters if not (getattr(c, 'remote', None) or '').strip()), characters[0])
    # with nobody carried over, the scene is where its own cast is ('same' sent a scene set in
    # Fouan's kitchen, with Fouan and his daughter in it, to the farm the last scene was played at)
    anchor = here(carried[0]) if carried else here(first)
    target = resolve(context, scene.get('location'), anchor)
    if target is None:
        # 'same' after the cast walked out for a named place: the scene is where they went
        names = {c.name for c in characters}   # where this scene's own cast went, not someone who walked off alone
        gone_to = [e.get('to') for e in getattr(context, 'last_scene_exits', []) if e.get('to') and e.get('who') in names]
        target = next((t for t in (resolve(context, g, anchor) for g in gone_to) if t), None)
    return target if target else (anchor, None)


def set_scene(context, scene, characters, opens_act):
    """Bring the scene's cast together at the scene's place and return the line that says so
    ('' when nobody moves). A scene continues the one before it: unless its location names
    somewhere else, it is played where the characters who were in the previous scene now are,
    and newcomers join them there. (It was played where its first-named character stood, so a
    scene that brought Francoise to Jean walked Jean to Francoise's farm instead.) With nobody
    carried over, it is where its own first character (there in person) is.
    A scene that opens an act puts its cast in place; any other scene has them walk, and the
    clock moves on by the longest walk."""
    if not characters:
        return ''
    xy, place = scene_spot(context, scene, characters)
    scene['place'] = place or where(context, xy).removeprefix('at ')
    context.previous_scene = {'cast': tuple(c.name for c in characters), 'xy': xy}
    movers, longest, ride = [], 0.0, None
    for character in characters:
        if remote(character):
            # in the scene by voice, from where they are: and indoors there (Novak on Earth was
            # walked onto the gantry, and her suit oxygen began to run down)
            if getattr(character, 'body', None) is not None:
                character.body.sheltered = True
            continue
        if world.on_the_spot(context, here(character), xy):   # already there: a step or two needs no telling
            if place:
                world.put(character, xy)
            continue
        if opens_act:
            world.put(character, xy)
            ledger.of(context).wrote('position')
            if getattr(character, 'body', None) is not None:
                character.body.sheltered = world.sheltered(context, xy)
        else:
            ride = ride or world.conveyance(context, here(character), xy)
            longest = max(longest, survival.travel(character, xy, advance_clock=False))
        movers.append((character.name, 0))
    if not movers:
        return ''
    names = ' and '.join(name for name, _ in movers)
    if opens_act:
        line = f"{names} {'is' if len(movers) == 1 else 'are'} now at {scene['place']}."
    else:
        context.simulation_time += survival.timedelta(minutes=longest)
        if ride is not None:
            world.bring(context, ride, xy)
        line = (f"{names} {ride.by if ride is not None else 'walk'}{'s' if len(movers) == 1 else ''} to {scene['place']}"
                + (f' in {ride.id}' if ride is not None else '') + (f" ({round(longest)} min)." if round(longest) >= 1 else '.'))
    trace.event('scene_place', place=scene['place'], movers=[n for n, _ in movers], walked=not opens_act,
                minutes=round(longest, 1), at=list(xy))
    return line


def gazetteer(context, bearings=False):
    """The stage for a playwright prompt: where each cast member is, and the named places
    nearest the cast with what each offers (and to whom, when a place is someone's own) and
    how far it is. Every kind of place that offers an action is listed at least once, however
    far, so the plan knows it exists. bearings: with which way each lies. (The check of a set-up
    was shown 'Berries9 (southwest)' and the stage manager only how far: it wrote 'northwest', and
    was struck; seven of 117 things struck in 21 plays were a direction or a distance.)"""
    cast = [c for c in context.actors if world.on(c)]
    if not cast or not world.places(context):
        return ''
    lines = ['Where the cast is: ' + '; '.join(f'{c.name} {remote(c) or where(context, here(c))}' for c in cast) + '.']
    cast = [c for c in cast if not remote(c)] or cast   # places are reckoned from those who can walk to them
    places = sorted(((min(world.span(context, here(c), res.xy) for c in cast), rid, res)
                     for rid, res in world.listed(context)), key=lambda p: p[:2])
    listed, kinds = [], set()
    for d, rid, res in places:
        if len(listed) < GAZETTEER_PLACES or res.kind not in kinds:
            listed.append((d, rid, res))
            kinds.add(res.kind)
    for d, rid, res in listed:
        nearest = min(cast, key=lambda c: world.span(context, here(c), res.xy))
        reach = 'here' if world.at_place(context, here(nearest), res) else world.reach_words(context, here(nearest), res.xy, nearest.name)
        if bearings and reach != 'here':
            way = land.bearing(here(nearest), res.xy).lower()
            reach += '' if way == 'current' else f', to the {way}'
        owner = res.owner
        owned = '' if isinstance(owner, Part) else f", owned by {owner}" if owner else ''   # an owner nobody is cast as goes unsaid
        lines.append(f"- {rid}: {res.description or res.kind}"
                     + owned + f' ({reach})')
    shared = []
    if shared:   # what a body can do wherever it is: the planner is held to the same acts as the actors
        lines.append('- Anywhere, with whoever is there: ' + '; '.join(a['name'] for a in shared)
                     + '. With a thing someone holds or that lies there: give, hold out, take, put down.')
    return '\n'.join(lines)


def company(character):
    """The scene partners standing with a character: those who come along when it walks."""
    return [c for c in character.bystanders() if world.on(c)]


GOES = """You are the actor playing {name} in an improvised play. {walker} is setting off for {place}, on foot.

What has just been said and done, in {name}'s hearing and sight:
<lately>
{lately}
</lately>

What {name} is after in this scene: {goal}

Does {name} go along with {walker} now, or stay here? People who are together mostly go when the other sets off. {name} stays when {name} is in the middle of something {name} would not drop, has just said no to going, was not asked along (the other has said it is going by itself, or will be back), or has a reason of {name}'s own to stay. Where {walker} goes is {walker}'s to say; whether {name} goes is {name}'s.

Answer with these two lines only:
#goes yes or no
#why four to eight words
</end>"""


def goes_along(other, walker, place):
    """Does `other` go along when `walker` sets off for another place? Its own actor says. (The
    one setting off named who came, and nobody asked them: Samantha walked Joe off the rock he
    had just begun to climb, and a stranger slipping away took the whole scene with him.)"""
    from utils.Messages import UserMessage
    from utils import hash_utils
    context = other.context
    direction = getattr(other, 'direction', None) or {}
    lately = '\n'.join(context.witnessed(other, n=8)) or 'Nothing yet.'
    try:
        reply = other.llm.ask({}, [UserMessage(content=GOES.format(name=other.name, walker=walker.name, place=plain(context, f'x {place}')[2:] or place,
                                                                  lately=lately, goal=direction.get('goal') or 'not said'))],
                              tag='stage.goes_along', temp=0.2, stops=['</end>'], max_tokens=40) or ''
    except Exception:
        reply = ''
    goes = not (hash_utils.find('goes', reply) or 'yes').strip().lower().startswith('n')   # unread: together, as before
    trace.event('goes_along', character=other.name, walker=walker.name, place=place, goes=goes, why=(hash_utils.find('why', reply) or '').strip()[:80])
    return goes


def walk(character, xy, place, alone=False, to=True):
    """A character walks to a tile, and its scene partners come along unless it goes alone (a
    scene's cast stays together: Samantha used to reach the spring an hour ahead of Joe, who was
    then walked there by the next scene). One clock advance for the party. Returns the words for
    the audience, said of the walker ('walks to Spring2 with Joe (14 tiles, 70 min)'), or ''
    when nobody moves."""
    context = character.context
    start = here(character)
    if start == tuple(xy):
        return ''
    step = world.a_step(context, start, xy)   # within the place it is at
    left = where(context, here(character))
    was_with = company(character)
    # a few steps within the place (to the woodpile and back) are taken alone; a walk to another
    # place takes the party, however short (Joe walked three tiles to the spring alone, then
    # walked back to say it was 'right there, past the ferns')
    # each of those with it says for itself whether it comes (goes_along)
    party = [] if alone or step else [c for c in was_with if goes_along(c, character, place)]
    stayed = [] if alone or step else [c for c in was_with if c not in party]
    if not (alone or step):
        character.walk_asked = list(getattr(character, 'walk_asked', None) or []) + was_with
        character.walk_stayed = list(getattr(character, 'walk_stayed', None) or []) + stayed
    ride = None if step else world.conveyance(context, start, xy)   # a car at hand, for a way too long to walk
    minutes = max(survival.travel(c, xy, advance_clock=False) for c in [character] + party)
    if ride is not None:
        world.bring(context, ride, xy)
    context.simulation_time += survival.timedelta(minutes=minutes)
    names = ' and '.join(c.name for c in party)
    # steps within a place are not told: a page of 'walks South (15 min)' tracked nothing. A
    # short walk that ends at another place is (Joe went from the apple tree to the thicket
    # unseen, and the page had him handed a branch at the tree)
    # a place of the same kind as the one left is another of them, and which way it lies ('walks to
    # the fallen branch' was said of a second branch, 160 metres north of the one they stood at)
    dest, was_at = world.place(context, place), world.nearest(context, start)
    if to and dest is not None and was_at is not None and world.at_place(context, start, was_at[1]) and was_at[1].kind == dest.kind and was_at[1].id != dest.id:
        where_to = f"{world.bearing(start, xy).lower()} to another {dest.kind.replace('_', ' ').lower()} ({dest.id})"
    else:
        where_to = f"{'to ' if to else ''}{place}"
    line = '' if step else (
        f"{ride.by + 's' if ride is not None else 'walks'} {where_to}" + (f' with {names}' if party else '')
        + (f' in {ride.id}' if ride is not None else '') + f" ({max(1, round(minutes))} min)")
    for other in was_with:
        if other not in party and not alone:
            continue   # it has only stepped across the place
        other.look()
    # a scene is where its cast stands: when all of it has walked on, the scene has, and whoever
    # reads its place reads the new one (a scene walked from the thicket to the apple tree and
    # went on being 'at Thicket7')
    scene = getattr(context, 'current_scene', None)
    if isinstance(scene, dict) and getattr(character, 'current_scene', None) is scene and where(context, xy) != left:
        on_stage = [c for c in context.actors + context.extras + context.npcs if getattr(c, 'current_scene', None) is scene and not off_stage(c) and not remote(c)]
        if on_stage and all(here(c) == tuple(xy) for c in on_stage):
            scene['place'] = where(context, xy).removeprefix('at ')
            if isinstance(getattr(context, 'previous_scene', None), dict):
                context.previous_scene['xy'] = tuple(xy)
            trace.event('scene_moved', place=scene['place'])
    # one who walks off to another place while someone stays behind has gone out of the scene, with
    # those who went along: the scene is where the ones who stayed are
    if stayed and isinstance(scene, dict) and getattr(character, 'current_scene', None) is scene and where(context, xy) != left:
        for c in [character] + party:
            leave(c, [x for x in [character] + party if x is not c], told=True, to=place)
    # who saw it go: those it walked with and those it left (the walk's line is recorded by the walker)
    character.walk_witnesses = [character] + was_with
    trace.event('walk', character=character.name, place=place, party=[c.name for c in party], alone=alone,
                tiles=world.span(context, start, xy), minutes=round(minutes, 1))
    return line


def approach(character, target):
    """Before a character acts on or speaks to a scene partner who is out of reach, it walks to
    them (Joe thrust a branch at Samantha from three tiles off, and waved at her from five).
    Returns the words for the audience, or ''."""
    if (target is None or target is character or not world.on(target)
            or not world.on(character)
            or getattr(target, 'current_scene', None) is not character.context.current_scene
            or character.context.current_scene is None):
        return ''
    if remote(character) or remote(target):
        return ''   # spoken to over the radio, not walked to
    if world.on_the_spot(character.context, here(character), here(target)):
        return ''
    return walk(character, here(target), target.name, alone=True)


def tempo(character):
    """calm, tense or urgent: the scene's own tempo, and urgent whenever a threat is on the character."""
    scene = character.context.current_scene or {}
    if survival.active_threat_for(character):
        return 'urgent'
    value = str(scene.get('tempo') or 'calm').strip().lower()
    return value if value in ('calm', 'tense', 'urgent') else 'calm'


# What sort of scene this is, for the actor: every scene was a confrontation ('press her on ...'),
# played 'on edge' at twenty words a line, so a card's 'chatty, rambling, philosophical when relaxed'
# was never heard, and nothing was learned of anyone except under questioning.
KINDS = {'shared job': ("This scene is a shared job: something has to be done with the hands, and the talk goes on beside the work, "
                        "about it and about other things. Nobody is cornering anybody.", "25 words at most"),
         'lull': ("This scene is a lull: the pressure is off for now. People rest, eat, look about, talk of what is not urgent and let "
                  "each other be. Nobody presses or tests anybody; what is learned of the other is learned by the way.",
                  "40 words at most, as this person talks at ease"),
         'confession': ("In this scene someone tells what they have kept back, or comes near it and turns away. It is slow: the one "
                        "telling chooses how much, and the one hearing listens more than asks.", "40 words at most, unhurried"),
         'play': ("This scene is play: a game, a joke, a dare, a contest with nothing riding on it. It is light; what it shows of each "
                  "is shown by how they play.", "25 words at most")}


def kind(character):
    """The scene's kind as the plan gave it ('lull', 'shared job', ...), '' for a confrontation, an
    unknown kind, or whenever a threat is on the character."""
    scene = character.context.current_scene or {}
    value = ' '.join(str(scene.get('kind') or '').strip().lower().replace('_', ' ').split())
    return value if value in KINDS and tempo(character) != 'urgent' else ''


def kind_note(character):
    return KINDS[kind(character)][0] if kind(character) else ''


def say_limit(character):
    """How long a spoken turn may be, for the act prompt's one length rule: by the scene's kind,
    and in a confrontation by the moment's tempo."""
    if kind(character):
        return KINDS[kind(character)][1]
    return {'urgent': "12 words at most, no explaining, still in the character's own voice",
            'tense': "20 words at most: a sentence or two, as this person talks on edge"}.get(tempo(character), "30 words at most")


def pending_business(character):
    """The scene's planned physical actions that this character has not yet done
    ('drink@Spring2'): what the scene's ending depends on (the scene 'business' field)."""
    scene = getattr(character, 'current_scene', None) or {}
    business = scene.get('business')
    # a scene's business was labels of a catalogue of actions; what is done on a stage is now the resolver's to say
    return []


def whereabouts(character):
    """Where the others in the scene are, seen from a character: 'Samantha is 5 tiles East of
    you, out of reach'. The act prompt named who was visible but not how far."""
    scene = character.context.current_scene
    if scene is None or not world.on(character):
        return ''
    out = []
    for other in character.context.actors + character.context.extras + character.context.npcs:
        if other is character or getattr(other, 'current_scene', None) is not scene or not world.on(other):
            continue
        minutes = world.minutes(character.context, here(character), here(other))
        out.append(f'{other.name} is beside you' if world.on_the_spot(character.context, here(character), here(other))
                   else f'{other.name} is about {minutes} minutes\' walk {_direction(here(character), here(other))} of you, out of reach until one of you walks over')
    return '; '.join(out)


def _stems(text, skip=()):
    return {w[:5] for w in re.findall(r"[a-z0-9_]{4,}", text.lower()) if w not in skip}


def unnamed(name, text):
    """`text` without its doer's name at its head, for a line the name is put before ('Samantha,
    sitting...' was put down 'Samantha Samantha, sitting'; 'Receptionist The Receptionist sorts')."""
    return re.sub(rf'^\s*(?:[Tt]he\s+)?{re.escape(name)}[\s,]+', '', text or '')


def stage_line(name, deed, result):
    """What the audience reads for a physical action: the deed, then only what the result adds.
    The result used to follow whole, so a deed was told two or three times over (the actor's
    words, the world's fact, the narrator's restatement) with the body's condition in brackets."""
    deed = unnamed(name, (deed or '').strip()).rstrip('. ')
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
    go ('walks 4 tiles Southwest to' -> 'walks Southwest to'). A place an author named ('HotelBar')
    is said as the world file's `said` gives it, or left as it is."""
    if not text or not world.built(context):
        return text
    def ordinary(match):
        rid = match.group(0)
        if world.place(context, rid) is None:
            return rid
        # 'BusStop1' -> 'the bus stop': a kind named in run-together words is said as words
        return 'the ' + re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', re.sub(r'\d+$', '', rid)).replace('_', ' ').lower()
    # a place's own name given in brackets after words for it is for the record, not the page
    text = re.sub(r'\s*\(([A-Z][A-Za-z_]*\d*)\)', lambda m: '' if world.place(context, m.group(1)) is not None else m.group(0), text)
    text = re.sub(r'\b[A-Z][A-Za-z_]*\d+\b', ordinary, text)
    # a place the author named is said as the world file says it is said ('HuDingApartment' -> "Hu and Ding's apartment")
    said = getattr(world.scenario(context), 'said', None) or {}
    if said:
        text = re.sub(r'\b(' + '|'.join(re.escape(n) for n in sorted(said, key=len, reverse=True)) + r')\b', lambda m: said[m.group(1)], text)
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


def light(context):
    """The light of the part of day now, in the world's own words where it has them (a
    scenario's image_light: the Moon has a black sky at every hour, and 'clear morning light'
    drew an Earth desert)."""
    part = part_of(context.simulation_time)
    own = getattr(world.scenario(context), 'image_light', None) or {}
    return own.get(part) or LIGHT[part]


def style(context):
    return (getattr(context, 'image_style', None) or getattr(world.scenario(context), 'image_style', None)
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


def _place_name(context, place, res):
    """An author-named place in words: 'XiaoyuOffice' -> "Xiaoyu's office", 'HuDingApartment' ->
    "Hu and Ding's apartment", 'HotelBar' -> 'the hotel bar'. Engine-numbered names go through
    plain()."""
    if not re.fullmatch(r'[A-Z][A-Za-z]*', place or ''):
        return plain(context, place)
    words = re.findall(r'[A-Z][a-z]*', place)
    cast = {c.name for c in list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', []))}
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
    res = world.place(context, place)
    words = _place_name(context, place, res) if res else plain(context, place)
    told = res.description if res else None
    indoor = bool(res) and world.indoors(context, res.xy)
    return (f"{words} ({told[0].lower() + told[1:]})" if told and told.lower() not in words.lower() else words), indoor


def _on_stage(context):
    """Who is in the scene: the scene plan's cast where there is one (extras too: Xiaoyu was
    never in a picture of his own office), else those marked as in it, else the main cast. The
    scene picture is taken before the cast is marked, and showed the whole main cast. Those in
    the scene by radio are not in its picture."""
    everyone = [c for c in context.everyone() if world.on(c) and not remote(c)]
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
    return marked or [c for c in everyone if c in context.actors]


def picture(context, doing=''):
    """The prompt for a picture of the stage as it stands, or None where there is no map to
    stand on. doing: what is being done in the foreground, in plain words. The place comes
    first, and a room is shown from inside: the play's opening sentence ('a modern chinese
    urban setting with a mix of buildings, roads ...') led every picture, after 'Wide view',
    and a meeting in an office was drawn round a desk in the street."""
    if not world.built(context) or not context.actors:
        return None
    on_stage = _on_stage(context)
    if not on_stage:
        return None
    setting = ' '.join(context.initial_state.split('.')[0].split()).strip().rstrip('.')
    place, indoor = _place_words(context, on_stage)
    lit = light(context)
    people = '; '.join(_looks(context, c) for c in on_stage[:4])
    threat = getattr(context, 'active_threat', None)
    if indoor:
        # the play's opening sentence describes the town, not the room: given here it brought the street in
        where_ = [f"Interior: inside {place}, {lit}"]
    else:
        where_ = ['Wide view', f"{place}, {lit}", setting]
    parts = [style(context), *where_,
             f"{'One person' if len(on_stage) == 1 else str(len(on_stage)) + ' people, no others'}: {people}"]
    if threat:
        parts.append(f"Close by, {threat['hazard']['description']}")
    if doing:
        parts.append(f"In the foreground, one of them {plain(context, doing).strip().rstrip('.')[:220]}")
    return '. '.join(p.strip().rstrip('.') for p in parts if p and p.strip()) + '.'


def portrait(context, character, mood=''):
    """The prompt for a character's portrait: how it looks, how it feels, where it is."""
    cast = [character] if world.on(character) else []
    where_ = f"{_place_words(context, cast)[0]}, {light(context)}" if cast and getattr(context, 'map', None) else ''
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
# tools/image_ab.py can draw a recorded moment again. These prompts are sent; ATWAP_IMAGE_PROMPT=old
# sends picture() and portrait() instead (they described a lunar engineer by her job, from the
# description's first sentence, and the picture was a desert building site with hardhats).
IMAGE_PROMPT = os.environ.get('ATWAP_IMAGE_PROMPT', 'new').strip().lower()
PROMPT_BUDGET = 300   # tokens of the image model's encoder; it reads 512 and cuts the rest unseen

LOOKS = """From the description of {{$name}} below, say only how {{$name}} looks to someone seeing them: age, build, hair, face, clothes.
One phrase beginning with 'a' or 'an', 30 words at most. No name, no health or mood, nothing of their past or character.

{{$description}}

End your response with: </end>"""


def ensure_looks(character):
    """How a character looks, written once from its description (one call). Pictures use it in
    place of the description's first sentence, which carries 'healthy' and the like."""
    if getattr(character, 'looks', None) or getattr(character, '_looks_asked', False):
        return getattr(character, 'looks', None)
    character._looks_asked = True   # once: an empty reply is not asked again at every picture
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
    # looks are written here when nothing has written them yet: the opening picture is taken
    # when the play loads, before any scene, and described Kidd by his description's first
    # sentence ('a 27 year old bisexual male of mixed racial...') while his portrait had his looks
    if not getattr(character, 'looks', None) and getattr(character, 'llm', None) is not None:
        try:
            ensure_looks(character)
        except Exception as e:
            logger.warning(f'looks for {character.name}: {e}')
    return {'name': character.name, 'looks': getattr(character, 'looks', None) or _looks(context, character),
            'condition': _visible(body), 'holding': _carried(body)}


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
    if not world.built(context) or not context.actors:
        return None
    on_stage = _on_stage(context)
    if not on_stage:
        return None
    shown = on_stage[:4]
    place, indoor = _place_words(context, on_stage)
    facts = {'style': style(context), 'place': place, 'indoor': indoor,
             'light': light(context),
             'setting': '' if indoor else ' '.join(context.initial_state.split('.')[0].split()).strip().rstrip('.'),
             'people': [_person(context, c) for c in shown], 'apart': '', 'threat': '', 'deed': '', 'setup': ''}
    if len(shown) > 1:
        facts['apart'] = world.apart_words(context, [here(c) for c in shown])
    threat = getattr(context, 'active_threat', None)
    if threat:
        facts['threat'] = f"{threat['hazard']['description']}, {world.danger_picture_words(context, [here(c) for c in shown], threat['location'])}"
    if deed:
        facts['deed'] = _no_speech(plain(context, f"{actor.name} {deed}" if actor is not None else deed))[:220]
    scene = context.current_scene if isinstance(context.current_scene, dict) else {}
    if setup and scene.get('pre_narrative'):
        facts['setup'] = plain(context, scene['pre_narrative'])
    return facts


def portrait_facts(context, character, mood=''):
    cast = [character] if world.on(character) and world.built(context) else []
    place = _place_words(context, cast)[0] if cast else ''
    by = remote(character)
    if by:
        # behind someone on the radio is where they speak from, not the scene's place (Novak at
        # her mission-control consoles was painted on the gantry)
        place = re.sub(r'^by\b.*?\bfrom\s+', '', by) or by
    return {'style': style(context), 'person': _person(context, character), 'mood': (mood or '').strip().rstrip('.'),
            'place': place, 'light': '' if by else light(context)}


def _clause(person, hands=99, named=True):
    held = person['holding'][:hands]
    looks = person['looks']
    if named and re.match(r'(A|An|The) ', looks or ''):   # 'Joe, a young man', not 'Joe, A young man'
        looks = looks[0].lower() + looks[1:]
    return ', '.join(p for p in (person['name'] if named else '', looks, person['condition'],
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
    """A prompt from picture_facts: style, place and light, the setting, one clause per person,
    how they stand, the deed in front, the danger, then the set-up sentence, which goes when
    the prompt is over budget (people and the deed never do)."""
    people = facts['people']
    deed = facts['deed']
    # The one doing the deed is told once, with the deed: 'They stand side by side. In the
    # foreground, Joe kneels at the spring' was drawn as the pair standing and a second Joe
    # kneeling in front of them (six pictures in eight of one run).
    doer = next((p for p in people if deed and re.match(rf"{re.escape(p['name'])}\b", deed)), None)
    if doer is not None:
        rest = [p for p in people if p is not doer]
        act = deed[len(doer['name']):].strip()
        who = (f"{len(people)} {'person' if len(people) == 1 else 'people'}, no others: in the foreground, {_clause(doer)}, {act}"
               + (('; behind, ' + '; '.join(_clause(p) for p in rest)) if rest else ''))
    else:
        who = (f"One person, no others: {_clause(people[0])}" if len(people) == 1 else
               f"{len(people)} people, no others: " + '; '.join(_clause(p) for p in people))
    where_ = ([f"Interior: inside {facts['place']}, {facts['light']}"] if facts['indoor']
              else ['Wide view', f"{facts['place']}, {facts['light']}"])
    head = [facts['style'], *where_, facts['setting'], who,
            f"They stand {facts['apart']}" if facts['apart'] and doer is None else '',
            f"In the foreground, {deed}" if deed and doer is None else '',
            f"{facts['threat'][0].upper()}{facts['threat'][1:]}" if facts['threat'] else '']
    # the world the place is in stays (a lunar base: dropped, the gantry was drawn on Earth);
    # the set-up sentence goes first when the prompt is over budget
    tail = [facts['setup']]
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
