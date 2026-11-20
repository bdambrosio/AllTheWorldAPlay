"""What a deed does, said once by one call that is shown the whole stage (2026-11-20).

A deed used to exist twice: as the actor's words, which the audience was shown, and as a list of
labels a judge matched to it, which the world applied. The two were written by different calls and
never set side by side: a judge's 'show leaf@Kidd' matched no listed label and a plain pick-up was
refused; a deed that picked up a strip lying at another place was passed as 'kneel + examine' and
shown as written; 'leaves with Joe' was read by one of three code paths that dropped Joe. Every
one of those was a model that had the situation right and code that lost it at a label, a field or
a pattern.

Here one call is given the stage as it stands (record.of), the scene so far and the deed, and says
what the deed does as effects in the ledger's own terms, with the line the audience sees. Code
does not read the deed's words. It checks that the effects can be applied (the thing is there, the
person is there, the world's action is one listed and lacks nothing) and applies them; what
changes a body's needs or finds what a place holds is still the world's own action, applied by
survival._apply. The page is written from what was done, so the two cannot differ.

Tested offline on 225 recorded deeds (logs/review-20261120/resolver): Claude Opus 5.5 gave 409 of
409 well-formed answers, no thing from nothing, no thing taken from another's hands, about two
seconds a call."""
from __future__ import annotations

import re

from sim import ablation, stage, survival
from sim.trace import trace
from utils import json_reply
from utils.Messages import UserMessage

RESOLVE = """You keep the physical world of an improvised play. A character is about to do a deed. Say what the deed does in this world as it now stands.

<world>
{world}
</world>

<stage>
{stage}
</stage>

<the_scene_so_far>
{scene}
</the_scene_so_far>

{name} is about to:
<deed>
{deed}
</deed>
{leaves}
How this world works:
- It is the world the setting describes. What an ordinary person could do there with their body and what is at hand can be done, though it take time and effort (a fire worked up from dry wood, a shelter of boughs); what they could not, cannot, however the deed is worded.
- The stage above is the whole truth of who holds what and what lies here. A deed can use only what its doer holds, what lies here, what another person here gives or holds out, and what this place gives. A thing that lies at another place, however near, is not here: nobody picks it up from where they stand.
- Ordinary handling of a thing that is here needs no leave: pick it up, put it down, hand it over, hold it out or up to be seen, pocket it, open, shut, fold, tear, wipe or turn it, set it on something, read it. Each is an effect: say it.
- What changes a body's needs (eating, drinking, resting, tending a wound) and what finds things a place holds (searching it, picking from it) happens only by one of the world's actions listed on the stage, named exactly as listed. An action marked as lacking something cannot be done. With a danger on the stage, what is done about it is one of the answers listed for it.
- A touch is done as it is written: taking a hand or an arm, a hand on a shoulder, pressing a thing into someone's hand (the thing is then theirs to keep or give back). The one touched may draw away afterwards; that is theirs to play.
- Nobody is moved, held back, stopped or searched against their will. To lead, draw, pull or beckon someone along is to ask: they answer for themselves, and where they stand or go is no effect of this deed. A thing another person holds comes away only if they give it or have held it out.
- Nothing comes from nothing: a new thing is one this place gives (by a listed action), or is made from things that are here.
- A look, a gesture, a step or two, a change of posture changes nothing that lasts, except the posture.
- A deed is about the things it names: where the thing it means is not here, that part is not done, and no other thing is used in its place.
- Where one part of the deed cannot be done and the rest still makes sense without it, the rest is done and that part is left out. Where nothing of it can be done, it cannot be done.

Answer in JSON only:
{{"can": true or false,
  "why": "<if it cannot: the world's reason, ten words at most; else empty>",
  "effects": [<one entry for each lasting thing the deed does, in order; empty if it changes nothing that lasts>
     {{"thing": "<a thing, by its exact name on the stage>", "to": "<the person who now holds it, by name, or the word here for lying here>"}},
     {{"thing": "<exact name>", "state": "<how the thing itself is left, two to five words>"}},
     {{"thing": "<exact name>", "held_out_to": "<the person it is held out or shown to>"}},
     {{"thing": "<exact name of a written thing>", "read_by": "{name}"}},
     {{"world": "<one of the world's actions, the whole of the part before the bracket, with its @place or @person, exactly as listed>"}},
     {{"who": "{name}", "posture": "<standing, sitting, kneeling, lying down>"}},
     {{"who": "{name}", "touches": "<the person touched>", "how": "<two to four words>"}},
     {{"mark": "<a mark the deed leaves on the place or on something that is part of it, two to five words>"}},
     {{"who": "{name}", "goes": "<a place named on the stage, or the words out of the scene>", "with": ["<those {name} asks along>"]}},
     {{"new": "<a thing now made or picked>", "from": "<the listed action or the things here it comes from>"}}],
  "left_out": "<the part of the deed that is not done, in a few words; empty if all of it is done>",
  "page": "<what the audience sees happen: {name}'s deed in the third person and present tense, only what is done, no words spoken, 30 words at most>"}}
</end>"""

AGAIN = """

Your answer could not be applied to the stage:
{faults}
Answer again, in the same form, from the stage as it is given."""


def _low(value):
    """A name as the stage has it: no state in brackets, no article, lower case."""
    return re.sub(r'^(a|an|the|some)\s+', '', re.sub(r'\s*\(.*', '', str(value or '')).strip().lower())


def people(context, character):
    """Those in the scene with `character`, itself first."""
    scene = getattr(context, 'current_scene', None)
    if scene is not None:
        cast = [c for c in context.actors + context.extras + context.npcs
                if getattr(c, 'current_scene', None) is scene and not stage.off_stage(c)]
    else:
        cast = [character] + list(character.bystanders())
    return [character] + [c for c in cast if c is not character]


def find(context, character, name):
    """Where a thing named on the stage is: ('mine' | 'here' | 'theirs', the thing's own name, its holder) or None."""
    low = _low(name)
    for item in character.body.inventory:
        if _low(item) == low:
            return 'mine', str(item), character
    for prop in survival.lying(context, stage.here(character)):
        if _low(prop['name']) == low:
            return 'here', prop['name'], None
    # in another's hands: the one who has held it out to this character before any other who holds
    # a thing of the same name (two people carried berries, one held his out, and the taking was
    # refused because the other had not)
    theirs = [(str(item), other) for other in people(context, character)[1:]
              for item in (other.body.inventory if getattr(other, 'body', None) is not None else []) if _low(item) == low]
    offered = [(item, other) for item, other in theirs if (item, other.name, character.name) in context.held_out]
    if offered or theirs:
        item, other = (offered or theirs)[0]
        return 'theirs', item, other
    return None


def _person(context, character, name):
    low = _low(name)
    return next((c for c in people(context, character) if c.name.lower() == low), None)


def faults(context, character, found):
    """Why these effects cannot be applied to the stage as it stands: [] when they can. Nothing
    here reads the deed: it is whether the things, people and actions named are there."""
    out, made, gained = [], set(), set()   # made: new in this deed; gained: come into the doer's hands earlier in this deed
    for e in found.get('effects') or []:
        if not isinstance(e, dict):
            out.append('an effect is not one of the forms')
            continue
        if e.get('new'):
            made.add(_low(e['new']))
            gained.add(_low(e['new']))
        elif e.get('world'):
            chosen = survival._chosen(context, character, str(e['world']))
            if chosen is None:
                out.append(f"'{e['world']}' is not one of the world's actions listed")
            elif chosen[0] == 'action':
                _, action, near, target = chosen
                lack = survival._missing(action, character.body)
                # what the action needs may be what an earlier part of this deed puts in the doer's hands
                # ('takes a few berries and eats them')
                needs = [str(action.get(k) or '').lstrip('*').lower() for k in ('requires', 'transfer')]
                if lack and lack.startswith('has no') and any(n and any(g == n or g.endswith(n) for g in gained) for n in needs):
                    lack = ''
                if lack:
                    out.append(f"'{e['world']}' cannot be done: {character.name} {lack}")
                elif action.get('resource_types') and (near is None or not survival._can_approach(context, character, near[2])):
                    out.append(f"'{e['world']}' is at a place too far to do from here")
                elif action.get('target') == 'character' and target is None:
                    out.append(f"'{e['world']}' names nobody who is here")
        elif e.get('thing'):
            where = find(context, character, e['thing'])
            if where is None and _low(e['thing']) not in made:
                out.append(f"there is no '{e['thing']}' held by anyone here or lying here")
                continue
            for key in ('to', 'held_out_to'):
                if e.get(key) and _low(e[key]) != 'here' and _person(context, character, e[key]) is None:
                    out.append(f"nobody called '{e[key]}' is here")
            if where and where[0] == 'theirs' and e.get('to') and _low(e['to']) != where[2].name.lower() \
                    and (where[1], where[2].name, character.name) not in context.held_out:
                out.append(f"{where[2].name} holds the {where[1]} and has not held it out to {character.name}")
            if e.get('to') and _low(e['to']) == character.name.lower():
                gained.add(_low(e['thing']))
            if e.get('read_by') and where and not survival.writing_of(context, where[1]):
                out.append(f"the {where[1]} has nothing written on it")
        elif e.get('touches'):
            if _person(context, character, e['touches']) is None:
                out.append(f"nobody called '{e['touches']}' is here")
        elif e.get('goes'):
            if _low(e.get('who')) not in ('', character.name.lower()):
                out.append(f"only {character.name} goes anywhere by {character.name}'s deed")
            for name in e.get('with') or []:
                if _person(context, character, name) is None:
                    out.append(f"nobody called '{name}' is here")
        elif e.get('posture'):
            if _low(e.get('who')) not in ('', character.name.lower()):
                out.append(f"only {character.name}'s posture changes by {character.name}'s deed")
        elif not e.get('mark'):
            out.append('an effect is not one of the forms')
    return out


def _ask(context, character, act, note=''):
    from sim import keeper, record
    direction = getattr(character, 'direction', None) or {}
    scene = '\n'.join(filter(None, [f"It opened: {direction['setup']}" if direction.get('setup') else '',
                                    *context.witnessed(character, n=14)])) or 'Nothing yet.'
    party = act.leaves if isinstance(getattr(act, 'leaves', None), list) else []
    to = getattr(act, 'leaves_to', None)
    leaves = '' if not getattr(act, 'leaves', None) else \
        f"\nIts actor marks it as going out of the scene{' with ' + ', '.join(c.name for c in party) if party else ' alone'}{', for ' + to if to else ''}.\n"
    prompt = RESOLVE.format(world=f"{' '.join((context.initial_state or '').split())}\nThis world keeps count of: {keeper._tracks(context)}.",
                            stage=record.of(context, character, people(context, character), full=True), scene=scene,
                            name=character.name, deed=act.action, leaves=leaves) + note
    reply = context.llm.ask({}, [UserMessage(content=prompt)], tag='stage.resolve_deed', temp=0.2, stops=['</end>'], max_tokens=700)
    found = json_reply.parse(reply) if reply else None
    return found if isinstance(found, dict) and 'can' in found else None


def resolve(context, character, act):
    """What this deed does, decided before it is played: {'can', 'why', 'effects', 'left_out',
    'page'}. An answer that cannot be applied is sent back once with what is wrong with it; one
    that still cannot be, or cannot be read, is a deed that cannot be done. A deed of which
    nothing is done and something is left out cannot be done either: the actor is told, and the
    page is not given a line about nothing."""
    found = _ask(context, character, act)
    wrong = faults(context, character, found) if found else ['the answer could not be read']
    if found and wrong:
        trace.event('resolve_again', character=character.name, deed=act.action[:200], faults=wrong)
        again = _ask(context, character, act, AGAIN.format(faults='\n'.join(f'- {w}.' for w in wrong)))
        if again is not None:
            found, wrong = again, faults(context, character, again)
    if not found or wrong:
        found = {'can': False, 'why': wrong[0], 'effects': [], 'left_out': '', 'page': ''}
    found['effects'] = [e for e in found.get('effects') or [] if isinstance(e, dict)]
    found['left_out'], found['page'] = str(found.get('left_out') or '').strip(), str(found.get('page') or '').strip()
    if found.get('can') and not found['effects'] and found['left_out']:
        found.update(can=False, why=str(found.get('why') or '').strip() or f"not here: {found['left_out']}")
    if found.get('can') and not found['page']:
        found.update(can=False, why='the deed came to nothing that could be shown')
    found['can'], found['why'] = bool(found.get('can')), str(found.get('why') or '').strip()
    trace.event('resolved', character=character.name, deed=act.action[:300], can=found['can'], why=found['why'],
                effects=found['effects'], left_out=found['left_out'], page=found['page'])
    return found


def leaving(context, character, found):
    """Who goes out of the scene by this deed, and where to: (None | 'alone' | [characters], place or None).
    Going to another place is going out of the scene: a scene has one place."""
    goes = next((e for e in found.get('effects') or [] if e.get('goes')), None)
    if goes is None:
        return None, None
    party = [c for c in (_person(context, character, n) for n in goes.get('with') or []) if c is not None and c is not character]
    to = str(goes['goes']).strip()
    return (party or 'alone'), (None if 'scene' in to.lower() else to)


POSTURES = {'sitting': 'sitting', 'kneeling': 'kneeling', 'crouching': 'kneeling', 'lying down': 'lying down', 'lying': 'lying down'}


def apply(context, character, act):
    """Do what the deed was found to do. Returns the world's own sentences for what it settled
    (what a world action found or eased, what a written thing says): the rest is on the page
    already, in the resolver's line."""
    from sim import keeper
    found, facts, me = act.found, [], character.name
    worlded = False
    for e in found['effects']:
        if e.get('world'):
            chosen = survival._chosen(context, character, str(e['world']))
            if chosen and chosen[0] == 'threat':
                fact = survival._respond_to_threat(context, character, survival.active_threat_for(character), act.action.lower(), chosen=chosen[1])
            elif chosen:
                fact = survival._apply(context, character, chosen[1], chosen[2], chosen[3], 'model', act.action)
            else:
                fact = None
            worlded = True
            if fact:
                facts.append(fact)
        elif e.get('thing'):
            where = find(context, character, e['thing'])
            if where is None:
                continue   # made earlier in this deed and not come by: nothing to move
            kind, name, holder = where
            to = _low(e.get('to'))
            other = _person(context, character, e.get('to') or e.get('held_out_to') or '')
            if e.get('to'):
                if kind == 'here' and to != 'here':
                    survival._prop_do(context, character, 'take', name, None)
                    kind = 'mine'
                if kind == 'theirs' and to == me.lower():
                    survival._prop_do(context, character, 'take from', name, holder)
                elif kind == 'mine' and to == 'here':
                    survival._prop_do(context, character, 'put down', name, None)
                elif kind == 'mine' and other is not None and other is not character:
                    survival._prop_do(context, character, 'give', name, other)
            elif e.get('held_out_to') and other is not None and other is not character:
                if kind == 'here':
                    survival._prop_do(context, character, 'take', name, None)
                    kind = 'mine'
                if kind == 'mine':
                    survival._prop_do(context, character, 'offer', name, other)
            elif e.get('read_by'):
                survival._prop_do(context, character, 'read', name, None)
                facts.append(f"The {name} says: {survival.writing_of(context, name)}")
            elif e.get('state') and kind in ('mine', 'here'):
                survival._handle(context, character, name, e['state'])
        elif e.get('posture'):
            poses, _ = survival._arrangement(context)
            word = POSTURES.get(_low(e['posture']))
            poses.pop(me, None) if word is None else poses.__setitem__(me, word)
            survival._bear(context)
        elif e.get('touches'):
            other = _person(context, character, e['touches'])
            if other is not None and other is not character:
                _, between = survival._arrangement(context)
                # the stage keeps that they are touching ('touching Lanya'); how is on the page
                between[:] = [b for b in between if not (b[0] == me and b[2] == other.name and b[3] == 'contact')] + [(me, 'touching', other.name, 'contact')]
                survival._bear(context)
        elif e.get('mark'):
            survival.add_thing(context, ' '.join(str(e['mark']).split()), at=stage.here(character), fixed=True)
        elif e.get('new') and not worlded:
            # made from what is here, not given by a world action (whose own gain put it in hand): the
            # keeper says what kind of thing it is and how it behaves in time, where that matters
            # (a fire); a thing it does not model is simply there
            kept = keeper.consider(context, character, act.action) if ablation.enabled('keeper') else {'can': False}
            fact, done = keeper.perform(context, character, act.action) if kept.get('can') else ('', False)
            if not done:   # it is here; a later effect of the deed puts it in a hand
                survival.add_thing(context, str(e['new']), at=stage.here(character))
            elif fact:
                facts.append(fact)
    character.last_affordance = None   # what was done is told by the page line, not by the last part's own sentence
    if found['left_out']:
        character.add_perceptual_input(f"Of what you meant to do, this was not done, because it cannot be here: {found['left_out']}.",
                                       percept=False, mode='internal')
    return ' '.join(facts)
