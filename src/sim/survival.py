"""Survival stage business: bodies, affordances, travel time and hazards.

Opt-in per scenario: a scenario module (plays/scenarios/*.py) that defines `survival = {...}`
gets it; other plays are unchanged. The mechanics decide what happens (numbers, depletion,
success rolls); the LLM only narrates, given the facts (Context.do).

    needs           hunger / thirst / fatigue rise per hour of simulation time; injury does not
    actions         Do verbs matched against resources in reach (drink at a Spring, eat Berries,
                    take a Fallen_Branch, shelter / sleep in a Cave, rest, build a fire ...)
    hazards         rolled per hour (time of day, terrain, unsheltered, no fire); resolved by a
                    defend / flee / hide Do, or they strike at their deadline

Config shape (see plays/scenarios/forest.py):
    survival = {'initial': {need: level}, 'rates_per_hour': {need: rate},
                'tile_minutes': minutes per map tile at terrain cost 1,
                'actions': [{'name', 'verbs', 'resource_types' (None = anywhere), 'named_verbs', 'effects',
                             'minutes', 'deplete', 'gain', 'requires', 'consumes', 'sets',
                             'risk': {'chance', 'effects', 'text'}, 'text'}],
                'hazards': [{'name', 'description', 'when' ('day'|'night'|'any'), 'terrain',
                             'chance_per_hour', 'blocked_by', 'minutes_to_strike', 'harm',
                             'retreat_text', 'strike_text',
                             'responses': {name: {'verbs', 'success', 'bonus': {item: +p}}}}]}
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sim.trace import trace

NEEDS = ('hunger', 'thirst', 'fatigue')
LEVELS = {  # need -> words for (<0.3, <0.6, <0.85, >=0.85)
    'thirst': ('not thirsty', 'thirsty', 'very thirsty', 'desperately thirsty'),
    'hunger': ('not hungry', 'hungry', 'very hungry', 'starving'),
    'fatigue': ('rested', 'tired', 'very tired', 'exhausted'),
    'injury': ('unhurt', 'bruised', 'injured', 'badly hurt'),
}
THRESHOLD_PERCEPTS = {
    ('thirst', 0.5): "Your mouth is dry; you have not had water for a while.",
    ('thirst', 0.8): "Thirst is becoming hard to ignore; your head aches and your lips are cracked.",
    ('hunger', 0.5): "Your stomach is empty and growling.",
    ('hunger', 0.8): "Hunger makes you light-headed and weak.",
    ('fatigue', 0.5): "Your legs are heavy; you are getting tired.",
    ('fatigue', 0.8): "You are exhausted; every step is an effort and you need rest.",
}
_rng = random.Random()


def config(context) -> Optional[dict]:
    module = getattr(getattr(context, 'map', None), 'scenario_module', None)
    return getattr(module, 'survival', None)


def _word(need, value):
    words = LEVELS[need]
    return words[0] if value < 0.3 else words[1] if value < 0.6 else words[2] if value < 0.85 else words[3]


@dataclass
class Body:
    hunger: float = 0.2
    thirst: float = 0.2
    fatigue: float = 0.2
    injury: float = 0.0
    inventory: list = field(default_factory=list)
    sheltered: bool = False
    last_tick: Optional[datetime] = None
    warned: set = field(default_factory=set)

    def summary(self):
        parts = [_word(n, getattr(self, n)) for n in ('thirst', 'hunger', 'fatigue', 'injury')]
        if self.inventory:
            parts.append('carrying ' + ', '.join(self.inventory))
        if self.sheltered:
            parts.append('in shelter')
        return '; '.join(parts)

    def adjust(self, need, delta):
        old = getattr(self, need)
        setattr(self, need, min(1.0, max(0.0, old + delta)))
        if delta < 0:   # allow a fresh warning next time the need rises past a threshold
            self.warned = {w for w in self.warned if not (w[0] == need and getattr(self, need) < w[1])}


def attach(context):
    """Give every cast member a Body when the scenario opts in (Context.__init__)."""
    cfg = config(context)
    if not cfg:
        return
    # what the area offers and threatens, in the shared situation every planning prompt reads
    types = sorted({str(r.get('type')) for r in context.map.resource_registry.values()})
    dangers = [h['description'] + (' at night' if h.get('when') == 'night' else '') for h in cfg.get('hazards', [])]
    brief = ('\nThe area offers: ' + ', '.join(t.replace('_', ' ').lower() for t in types) + '.' +
             (' Dangers: ' + '; '.join(dangers) + '.' if dangers else '') +
             ' Thirst, hunger and fatigue grow as the hours pass.')
    if brief not in context.current_state:
        context.current_state += brief
    for character in context.actors + context.extras:
        if getattr(character, 'body', None) is None and getattr(character, 'mapAgent', None) is not None:
            initial = cfg.get('initial', {})
            character.body = Body(**{n: initial.get(n, 0.2) for n in NEEDS}, last_tick=context.simulation_time)


def scene_clock_reset(context):
    """A scene sets its own clock (often hours after the last one): time between scenes passes
    off-stage, so needs and hazard rolls restart from the scene's start time (Context scene start)."""
    now = context.simulation_time
    for character in context.actors + context.extras:
        if getattr(character, 'body', None) is not None:
            character.body.last_tick = now
    if hasattr(context, '_hazard_last'):
        context._hazard_last = now


def tick(character):
    """Advance needs by elapsed simulation time; threshold crossings become internal percepts."""
    body = getattr(character, 'body', None)
    cfg = config(character.context) if character.context else None
    if body is None or not cfg:
        return
    now = character.context.simulation_time
    if body.last_tick is None or now <= body.last_tick:
        body.last_tick = now
        return
    hours = (now - body.last_tick).total_seconds() / 3600
    body.last_tick = now
    rates = cfg.get('rates_per_hour', {})
    for need in NEEDS:
        rate = rates.get(need, 0.0) * (0.3 if body.sheltered and need == 'fatigue' else 1.0)
        body.adjust(need, rate * hours)
        crossed = [(level, text) for (n, level), text in THRESHOLD_PERCEPTS.items()
                   if n == need and getattr(body, need) >= level and (need, level) not in body.warned]
        if crossed:   # several thresholds in one tick: one percept, the most severe
            body.warned.update((need, level) for level, _ in crossed)
            character.add_perceptual_input(max(crossed)[1], mode='internal')
            trace.event('body', character=character.name, need=need, level=round(getattr(body, need), 2))
    for need in ('thirst', 'hunger'):   # neglect hurts
        if getattr(body, need) >= 0.95:
            body.adjust('injury', 0.05 * hours)


# --------------------------------------------------------------------------- places
def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _direction(frm, to):
    from sim.map import get_direction_name
    return get_direction_name(to[0] - frm[0], to[1] - frm[1]).name


def _resources_of(context, types):
    for rid, res in context.map.resource_registry.items():
        if str(res.get('type')) in types:
            yield rid, res


def _here(character):
    return (character.mapAgent.x, character.mapAgent.y)


def _nearest(character, types, mentioned=None, view=12):
    """(resource id, resource, distance) of the best matching resource within view, or None."""
    here = _here(character)
    found = []
    for rid, res in _resources_of(character.context, types):
        d = _distance(here, res['location'])
        if d <= view:
            # an id the act names first, then a type it names ('berries' over an apple tree), then distance
            named_type = bool(mentioned) and _names_type(mentioned, [str(res.get('type'))])
            found.append((rid.lower() not in (mentioned or ''), not named_type, d, rid, res))
    if not found:
        return None
    found.sort(key=lambda f: f[:3])
    _, _, d, rid, res = found[0]
    return rid, res, d


def _inflected(word):
    """Regex for a verb with ordinary inflections: drink(s/ing), snap(ped/ping), tak(e/es/ing)."""
    stem = word[:-1] if word.endswith('e') and len(word) > 3 else word
    tail = r'(?:e|es|ed|ing)?' if stem != word else r'(?:s|es|ed|d|ing|' + re.escape(word[-1]) + r'(?:ed|ing))?'
    return re.escape(stem) + tail


def _matches(text, verbs):
    """A verb or phrase whose first word is inflected ('scoop water' matches 'scooping water',
    'drink' matches drinks/drinking, but not 'east')."""
    for v in verbs:
        first, *rest = v.split()
        pattern = r'\b' + _inflected(first) + ''.join(r'\s+' + re.escape(w) for w in rest) + r'\b'
        if re.search(pattern, text):
            return True
    return False


# extra words naming a resource type in an act ("water" for a Spring), besides the type itself
TYPE_NOUNS = {'Spring': ['water', 'stream', 'pool', 'spring'], 'Berries': ['berry', 'berries'],
              'Apple_Tree': ['apple', 'apples'], 'Mushrooms': ['mushroom', 'mushrooms', 'fungus', 'fungi'],
              'Fallen_Branch': ['branch', 'branches', 'stick', 'wood', 'log'], 'Thicket': ['thicket', 'bush', 'undergrowth'],
              'Cave': ['cave', 'cavern'], 'Hut': ['hut', 'cabin', 'shack']}


def _names_type(text, types):
    words = [w for t in types for w in TYPE_NOUNS.get(t, []) + [t.lower(), t.lower().replace('_', ' ')]]
    return any(re.search(r'\b' + re.escape(w) + r'\d*\b', text) for w in words)   # 'berries3' names Berries


# --------------------------------------------------------------------------- actions
def available_actions(character):
    """One line listing what the character could physically do here (act and dialog prompts)."""
    cfg = config(character.context) if character.context else None
    body = getattr(character, 'body', None)
    if not cfg or body is None:
        return ''
    out, seen = [], set()
    for action in cfg.get('actions', []):
        types = action.get('resource_types')
        if not types:
            needs = action.get('requires')
            if not needs or needs in body.inventory:
                out.append(action['name'])
            elif needs:
                out.append(f"{action['name']} (needs a {needs})")
            continue
        near = _nearest(character, types)
        if near is None:
            continue
        rid, res, d = near
        if (action['name'], rid) in seen:
            continue
        seen.add((action['name'], rid))
        depleted = res.get('properties', {}).get('remaining') == 0
        if depleted:
            continue
        if d <= 1:
            out.append(f"{action['name']} at {rid} (here)")
        else:
            out.append(f"{action['name']} at {rid} ({d} tiles {_direction(_here(character), res['location'])}; first Move: {rid})")
    threat = active_threat_for(character)
    if threat:
        h = threat['hazard']
        out.insert(0, f"THREAT: {h['description']} is {threat_position(character, threat)}; "
                      f"you could {', '.join(h.get('responses', {}).keys())}")
    return '; '.join(out[:10])


def resolve_do(context, character, act_text):
    """Apply the mechanics of a Do, if it is a survival action. Returns a fact line for the
    narration, or None when the act is not a survival action (the LLM decides as before)."""
    cfg = config(context)
    body = getattr(character, 'body', None)
    if not cfg or body is None:
        return None
    text = (act_text or '').lower()

    threat = active_threat_for(character)
    if threat:
        fact = _respond_to_threat(context, character, threat, text)
        if fact:
            return fact

    # candidate actions: verbs match; rank those whose resource the act names, then those with a
    # resource in reach, then config order (so 'eat berries' is not taken as eating mushrooms)
    candidates = []
    for order, action in enumerate(cfg.get('actions', [])):
        types = action.get('resource_types')
        named = bool(types) and _names_type(text, types)
        # 'named_verbs' are too generic alone ('cups his hands'): they count when the act names the thing
        if not (_matches(text, action.get('verbs', [])) or (named and _matches(text, action.get('named_verbs', [])))):
            continue
        near = _nearest(character, types, mentioned=text) if types else None
        if types and not named and any(_names_type(text, a.get('resource_types') or [])
                                       for a in cfg.get('actions', []) if a is not action and a.get('resource_types')):
            continue   # the act names a different kind of thing
        in_reach = (not types) or (near is not None and near[2] <= 1)
        candidates.append((not named, not in_reach, order, action, near))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[:3])
    _, _, _, action, near = candidates[0]
    types = action.get('resource_types')
    resource_id = res = None
    if types:
        if near is None:
            fact = f"There is no {' or '.join(types).replace('_', ' ').lower()} in sight here."
            _record(character, action, fact, applied=False)
            return fact
        resource_id, res, d = near
        if d > 1:
            fact = (f"{resource_id} is {d} tiles {_direction(_here(character), res['location'])}; "
                    f"{character.name} must move there first.")
            _record(character, action, fact, applied=False)
            return fact
        props = res.setdefault('properties', {})
        if action.get('deplete') and props.setdefault('remaining', action['deplete']) <= 0:
            fact = f"There is nothing left at {resource_id}."
            _record(character, action, fact, applied=False)
            return fact
    if action.get('requires') and action['requires'] not in body.inventory:
        fact = f"{character.name} cannot {action['name']} without a {action['requires']}."
        _record(character, action, fact, applied=False)
        return fact

    for need, delta in (action.get('effects') or {}).items():
        body.adjust(need, delta)
    if action.get('gain'):
        body.inventory.append(action['gain'])
    if action.get('consumes') and action['consumes'] in body.inventory:
        body.inventory.remove(action['consumes'])
    if action.get('sets') == 'sheltered':
        body.sheltered = True
    if action.get('sets') == 'fire':
        context.survival_fire = {'at': _here(character), 'until': context.simulation_time + timedelta(hours=3)}
    if res is not None and action.get('deplete'):
        res['properties']['remaining'] -= 1
    context.simulation_time += timedelta(minutes=action.get('minutes', 10))
    body.last_tick = context.simulation_time   # needs during the action are folded into its effect

    where = f" at {resource_id}" if resource_id else ''
    fact = f"{character.name} {action.get('text', action['name']).format(resource=resource_id or 'here')}{'' if action.get('text') else where}."
    risk = action.get('risk')
    if risk and _rng.random() < risk.get('chance', 0):
        for need, delta in risk.get('effects', {}).items():
            body.adjust(need, delta)
        fact += ' ' + risk.get('text', '')
    fact += f" ({character.name} is now {body.summary()}.)"
    _record(character, action, fact, applied=True, resource=resource_id)
    return fact


def _record(character, action, fact, applied, resource=None):
    trace.event('affordance', character=character.name, action=action['name'], applied=applied,
                resource=resource, fact=fact)


def travel(character, x, y):
    """Move a character to (x, y), advancing time by distance x terrain cost (survival plays)."""
    context = character.context
    cfg = config(context)
    start = _here(character)
    tiles = _distance(start, (x, y))
    costs = getattr(context.map.scenario_module, 'infrastructure_rules', {}).get('terrain_costs', {})
    try:
        terrain = context.map.patches[x][y].terrain_type.name
        cost = costs.get(terrain, 1.5)
        cost = 1.5 if cost == float('inf') else cost
    except Exception:
        cost = 1.5
    minutes = tiles * cost * (cfg or {}).get('tile_minutes', 2)
    character.mapAgent.x, character.mapAgent.y = x, y
    if getattr(character, 'body', None) is not None:
        character.body.sheltered = False
        character.body.adjust('fatigue', 0.01 * tiles)
    context.simulation_time += timedelta(minutes=minutes)
    trace.event('travel', character=character.name, tiles=tiles, minutes=round(minutes, 1))
    return minutes


# --------------------------------------------------------------------------- hazards
def _is_night(t):
    return t.hour < 6 or t.hour >= 20


def _blocked(context, character, hazard):
    blocks = hazard.get('blocked_by', [])
    body = character.body
    fire = getattr(context, 'survival_fire', None)
    near_fire = bool(fire and fire['until'] > context.simulation_time and _distance(fire['at'], _here(character)) <= 2)
    return ('sheltered' in blocks and body.sheltered) or ('fire' in blocks and near_fire)


def active_threat_for(character):
    threat = getattr(character.context, 'active_threat', None)
    return threat if threat and character.name in threat['targets'] else None


def threat_position(character, threat):
    d = _distance(_here(character), threat['location'])
    return 'right here' if d == 0 else f"{d} tiles {_direction(_here(character), threat['location'])}"


def hazard_tick(context):
    """Roll hazard onsets and strikes for the elapsed simulation time (run_scene calls this)."""
    cfg = config(context)
    if not cfg:
        return
    now = context.simulation_time
    last = getattr(context, '_hazard_last', None)
    context._hazard_last = now
    if last is None or now <= last:
        return
    hours = (now - last).total_seconds() / 3600
    cast = [c for c in context.actors if getattr(c, 'body', None) is not None]

    threat = getattr(context, 'active_threat', None)
    if threat:
        targets = [c for c in cast if c.name in threat['targets']]
        if all(_blocked(context, c, threat['hazard']) for c in targets):
            _end_threat(context, threat, threat['hazard'].get('retreat_text', f"The {threat['hazard']['name']} gives up and leaves."))
        elif now >= threat['deadline']:
            harm = threat['hazard'].get('harm', 0.3)
            for c in targets:
                c.body.adjust('injury', harm)
            _end_threat(context, threat, threat['hazard'].get('strike_text', f"The {threat['hazard']['name']} attacks.")
                        + f" ({', '.join(c.name + ' is now ' + c.body.summary() for c in targets)})", struck=True)
        return

    for hazard in cfg.get('hazards', []):
        when = hazard.get('when', 'any')
        if (when == 'night' and not _is_night(now)) or (when == 'day' and _is_night(now)):
            continue
        exposed = []
        for c in cast:
            try:
                terrain = context.map.patches[c.mapAgent.x][c.mapAgent.y].terrain_type.name
            except Exception:
                terrain = None
            if hazard.get('terrain') and terrain not in hazard['terrain']:
                continue
            if not _blocked(context, c, hazard):
                exposed.append(c)
        if not exposed:
            continue
        p = 1 - (1 - hazard.get('chance_per_hour', 0.1)) ** hours
        if _rng.random() >= p:
            continue
        anchor = exposed[0]
        loc = (anchor.mapAgent.x + _rng.choice([-3, 3]), anchor.mapAgent.y + _rng.choice([-2, 2]))
        targets = [c.name for c in exposed if _distance(_here(c), _here(anchor)) <= 4]
        context.active_threat = {'hazard': hazard, 'targets': targets, 'location': loc,
                                 'deadline': now + timedelta(minutes=hazard.get('minutes_to_strike', 45))}
        text = f"{hazard['description'].capitalize()} appears {threat_position(anchor, context.active_threat)} of {anchor.name}, watching {', '.join(targets)}."
        context.add_event(text)
        context.current_state += '\n' + text
        for c in exposed:
            c.add_perceptual_input(text, mode='visual')
        trace.event('hazard', phase='onset', hazard=hazard['name'], targets=targets)
        return


def _end_threat(context, threat, text, struck=False):
    context.active_threat = None
    context.add_event(text)
    context.current_state += '\n' + text
    for c in context.actors:
        if c.name in threat['targets']:
            c.add_perceptual_input(text, mode='visual')
    trace.event('hazard', phase='struck' if struck else 'ended', hazard=threat['hazard']['name'], text=text)


def _respond_to_threat(context, character, threat, text):
    hazard = threat['hazard']
    for name, response in hazard.get('responses', {}).items():
        if not _matches(text, response.get('verbs', [])):
            continue
        p = response.get('success', 0.5)
        for item, bonus in response.get('bonus', {}).items():
            if item == 'fire' and getattr(context, 'survival_fire', None):
                p += bonus
            elif item in character.body.inventory:
                p += bonus
        context.simulation_time += timedelta(minutes=5)
        if _rng.random() < min(0.95, p):
            fact = f"{character.name} tries to {name}: it works. " + hazard.get('retreat_text', f"The {hazard['name']} backs off.")
            _end_threat(context, threat, fact)
        else:
            character.body.adjust('injury', hazard.get('harm', 0.3) / 2)
            fact = (f"{character.name} tries to {name}, but the {hazard['name']} is not deterred and lunges; "
                    f"{character.name} is now {character.body.summary()}.")
            context.add_event(fact)
        trace.event('hazard', phase='response', hazard=hazard['name'], character=character.name, response=name, fact=fact)
        return fact
    return None
