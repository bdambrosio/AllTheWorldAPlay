"""The world's record, and the check that keeps narration to it.

The mechanics (sim/survival.py) hold what is true of bodies, things, places, stock and dangers.
A narrator is told all of it and then writes freely, and what it wrote used to be shown,
remembered and built on unchecked: a pile of berries nobody held, a fever in an unhurt body, a
boar struck five times after it had fled. settle() puts one short check between the narrator
and everything downstream. What the record does not support is written again without it, and
failing that, cut. Nothing is checked in a world without mechanics: there is no record there.
"""
from __future__ import annotations

import re

from sim import ablation, stage, survival
from sim.trace import trace
from utils.Messages import UserMessage

NEARBY_TILES = 10   # places this near are in sight: they may be seen, pointed at, spoken of
BODILY = (r"hung|starv|fed\\b|thirst|quench|parch|tired|exhaust|rested|fatigue|sleep|hurt|wound|injur|pain|ache|sore|swol|swell|"
          r"sprain|twist|broke|bleed|blood|bruis|fever|sick|ill\\b|nause|cramp|limp|lame|burn|bandag|sling|faint|dizz|weak")
KINDS = ('item', 'condition', 'presence', 'place', 'time', 'stock', 'outcome')

CHECK = """You keep the record of a simulated world. A narrator has written what follows from a character's action. Find what the narration states that the record does not support.

What {{$name}} tried (an attempt, not a fact):
<action>
{{$action}}
</action>

What the world's rules made of it ('none': the rules changed nothing):
<settled>
{{$facts}}
</settled>

<record>
{{$record}}
</record>

<narration>
{{$narration}}
</narration>

A statement is unsupported when it is one of these:
- item: someone has, gains, uses, eats, drinks from, drops or hands over a thing that Holdings and Settled do not give them (a tool, weapon, food, water, medicine, container, device), or finds such a thing lying there. Picking, touching or handling what grows or lies at a place the character is at (a berry at the berry bush, a branch at a fallen branch), or something of a kind it carries, is supported.
- condition: of the body only (fear, shock, a scare, relief and calm are of the mind and never a condition): a wound, illness or pain from injury, or a need met or unmet (hungry, fed, thirsty, quenched, tired, rested), that Bodies and Settled do not show. A stronger or weaker word for what Bodies shows (exhausted for tired) is supported.
- presence: an animal or person that Present, Danger and Settled do not mention; or the danger listed under Danger gone, struck or beaten when Settled does not say so. (A danger leaving as Settled says is supported. Anyone listed under Present is there, wherever the narration puts them within the place. Places, plants, wind, noises and things in the landscape are never 'presence'.)
- place: a character standing at, or having moved to, a place other than where Present puts them, when Settled has no walk. Seeing, facing, pointing at, turning or taking a few steps toward a place in sight is not moving there.
- time: light, dark or a time of day that does not fit Time.
- stock: how much is left somewhere, other than as Stock says.
- outcome: only when Settled refuses the action (it says 'cannot', 'must move there first', 'still has it', 'nothing to') and the narration has it done all the same; or Settled has it done and the narration has it fail. When Settled is 'none' there is no outcome claim: one character touching, pushing, pinning, gripping or striking another, or handling a thing it holds, is allowed.
Everything the record does not keep is allowed: the ground, plants, weather that fits the time, the clothes a person wears, sounds, gestures, how a thing looks or feels, whatever the action itself says the character does with its own body, and everything of the mind: what anyone thinks, feels, fears, wants, knows, remembers or has forgotten is never a 'condition'.

Most narration has nothing unsupported. When you are not sure, it is supported.

Respond with one line for each unsupported statement, giving the narration's own words exactly:
#claim <kind>: <the exact words from the narration>
or, when there is none, the single line:
#ok
</end>
"""


def of(context, actor, people=None):
    """The record as it bears on what may be said of `actor` and those with it (or of `people`,
    a scene's cast)."""
    people = [c for c in (people or [actor] + actor.bystanders()) if getattr(c, 'mapAgent', None) is not None]
    at = stage.here(actor)
    near = sorted((stage._distance(at, res['location']), rid, res) for rid, res in context.map.resource_registry.items())
    places = [f"{rid} ({'here' if d <= stage.AT_PLACE_TILES else stage._direction(at, res['location']).lower()})"
              for d, rid, res in near if d <= NEARBY_TILES]
    return '\n'.join([
        f'Time: {stage.clock(context)}',
        'Present: ' + '; '.join(f'{c.name} {stage.where(context, stage.here(c))}' for c in people) + '. Nobody else.',
        'Places in sight: ' + (', '.join(places[:16]) or 'none named'),
        'Bodies: ' + ('; '.join(f'{c.name}: {c.body.summary(public=True)}' for c in people if getattr(c, 'body', None) is not None)
                      or 'not tracked'),
        'Holdings: ' + ('; '.join(f"{c.name}: {', '.join(c.body.inventory) or 'nothing'}" for c in people
                                   if getattr(c, 'body', None) is not None) or 'not tracked'),
        'Stock: ' + (survival.stock(actor) or 'nothing has been used up'),
        'Danger: ' + (danger(actor) or 'none'),
    ])


def situation(context, lately=''):
    """The Situation every prompt reads, in a world with mechanics: where and when things stand,
    from the world itself. It was free text that each narration added to and an LLM rewrote
    after every scene, so whatever a narrator once wrote was the situation from then on."""
    cast = [c for c in list(context.actors) + list(getattr(context, 'extras', [])) if getattr(c, 'mapAgent', None) is not None]
    lines = [f'It is now {stage.clock(context, date=True)}.',
             'Where everyone is: ' + '; '.join(f'{c.name} {stage.where(context, stage.here(c))}' for c in cast) + '.']
    bodies = [f'{c.name}: {c.body.summary(public=True)}' for c in cast if getattr(c, 'body', None) is not None]
    if bodies:
        lines.append('Bodies and what each carries: ' + '; '.join(bodies) + '.')
    threat = getattr(context, 'active_threat', None)
    lines.append('Danger: ' + (f"{threat['hazard']['description']}, threatening {', '.join(threat['targets'])}" if threat else 'none at present') + '.')
    used = '; '.join(f'{rid}: {survival._left(res)}' for rid, res in context.map.resource_registry.items() if survival._left(res))
    if used:
        lines.append(f'Drawn on so far: {used}.')
    if lately:
        lines.append(f'Lately: {lately}')
    return '\n'.join(lines)


def danger(actor):
    """The threat on a character now, in words, or ''."""
    threat = survival.active_threat_for(actor)
    return f"{threat['hazard']['description']}, {survival.threat_position(actor, threat)}, not yet driven off" if threat else ''


def check(context, actor, action, facts, narration, people=None):
    """The claims in `narration` that the record does not support: [(kind, words)]. None when
    the check itself failed (the narration then stands: a broken check must not silence the play)."""
    reply = context.llm.ask({'name': actor.name, 'action': action, 'facts': facts or 'none', 'narration': narration,
                             # what was just settled is part of the record (a boar that Settled drives off is no longer under Danger)
                             'record': of(context, actor, people) + f"\nJust settled, and true: {facts or 'nothing'}"},
                            [UserMessage(content=CHECK)], tag='record.check', temp=0.1, stops=['</end>'], max_tokens=160)
    if reply is None:
        return None
    claims = []
    names = [c.name.lower() for c in (people or [actor] + actor.bystanders())]
    hazards = [h['name'].lower() for h in (survival.config(context) or {}).get('hazards', [])]
    for kind, words in re.findall(r'^\s*#claim\s+(\w+)\s*:\s*(.+?)\s*$', reply, re.M):
        kind, words, low = kind.lower(), words.strip(' "\''), words.lower()
        if kind not in KINDS or not words.strip('.'):
            continue
        # two things the check gets wrong often enough to settle here: someone who is present
        # called a 'presence', and a state of mind ('recovered from a scare') called a condition
        if kind == 'presence' and any(n in low for n in names) and not any(h in low for h in hazards):
            continue
        if kind == 'condition' and not re.search(BODILY, low):
            continue
        claims.append((kind, words))
    return claims


def rejection(claims):
    """The note that sends a narration back to be written again."""
    return ("\nYour earlier answer was turned down: the world does not hold these things, so they may not be stated or implied:\n"
            + '\n'.join(f'- {words}' for _, words in claims)
            + "\nWrite the result again without them. If little is left to say, say little.\n")


def _words(text):
    return {w for w in re.findall(r"[a-z']{4,}", text.lower())}


def cut(text, claims):
    """`text` without the sentences that carry a claim; '' when a claim cannot be placed (then
    nothing of the narration can be trusted to be clear of it)."""
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
    drop = set()
    for _, words in claims:
        flat = ' '.join(words.lower().split()).rstrip('.')
        hit = [i for i, s in enumerate(sentences) if flat and flat in ' '.join(s.lower().split())]
        if not hit:
            want = _words(words)
            hit = [i for i, s in enumerate(sentences) if want and len(want & _words(s)) / len(want) >= 0.6]
        if not hit:
            return ''
        drop.update(hit)
    return ' '.join(s for i, s in enumerate(sentences) if i not in drop)


def settle(context, actor, action, facts, narration, rewrite, kind='do'):
    """`narration` as the world admits it. rewrite(note) asks the narrator again with a note on
    what was turned down. One check, at most one rewrite and re-check, then the unsupported
    sentences are cut. Every finding is traced (kind 'narration_check')."""
    if (not narration or not survival.config(context) or getattr(actor, 'body', None) is None
            or not ablation.enabled('narration_gate')):
        return narration
    claims = check(context, actor, action, facts, narration)
    if not claims:
        trace.event('narration_check', character=actor.name, of=kind, outcome='ok' if claims is not None else 'unchecked', claims=[])
        return narration
    again = rewrite(rejection(claims))
    still = check(context, actor, action, facts, again) if again else None
    if again and still == []:
        result, outcome = again, 'rewritten'
    elif again and still:
        result, outcome = cut(again, still), 'cut'
    else:
        result, outcome = cut(narration, claims), 'cut'
    trace.event('narration_check', character=actor.name, of=kind, outcome=outcome,
                claims=[f'{k}: {w}' for k, w in claims], still=[f'{k}: {w}' for k, w in (still or [])],
                before=narration, after=result)
    return result
