"""Stage business: bodies, affordances, travel time and hazards, defined per world.

Opt-in per scenario: a scenario module (plays/scenarios/*.py) that defines `affordances = {...}`
(older name: `survival`) gets it; other plays are unchanged. The mechanics decide what happens
(numbers, stock, success rolls, who holds what); the LLM only narrates, given the facts
(Context.do). A scenario may also define `stakes = "..."`, the sentence the planning and act
prompts use to say what is at stake in that world (see stakes_text).

    needs           per-world needs (thirst, oxygen, ...) rise per hour of simulation time; the
                    harm need (injury) rises only from hazards, risks and neglect
    actions         Do acts matched to the actions on offer: by the act's #affordance label
                    (drink@Spring3, give bread@Fouan), else by verbs and the thing named.
                    Categories: survival, explore, social, work
    hazards         rolled per hour (time of day, terrain, unsheltered, no fire, flags);
                    resolved by a response Do, or they strike at their deadline

Config shape (see plays/scenarios/forest.py, rural.py, apocalypse.py ...):
    affordances = {
        'needs': {need: {'initial', 'per_hour', 'words': (4 words for <.3, <.6, <.85, >=.85),
                         'percepts': {level: text}, 'neglect_harm': harm per hour at >= .95,
                         'shelter_factor': rate multiplier while sheltered}},
        'harm': {'name': 'injury', 'words': (...)},
        'travel_need': need raised 0.01 per tile walked (optional),
        'currency': 'francs', 'initial_money': 0, 'initial_inventory': [...],
        'tile_minutes': minutes per tile at terrain cost 1, 'approach_tiles': 8,
        'type_nouns': {ResourceType: [words that name it in an act]},
        'actions': [{'name', 'category', 'verbs', 'named_verbs' (count only when the thing is
                     named), 'resource_types' (None = anywhere), 'target': 'character',
                     'effects': {need: delta}, 'target_effects': {need: delta}, 'minutes',
                     'deplete', 'gain', 'requires', 'consumes', 'transfer': item, 'transfer_money',
                     'finds': [{'item', 'chance'}], 'earn', 'cost',
                     'sets' ('sheltered' | 'fire' | any flag), 'requires_flag',
                     'sets_property': {key: value}, 'requires_property': {key: value or None;
                     '{actor}'; for owner, '{household}' = the actor or a parent or spouse},
                     'risk': {'chance', 'effects', 'text'}, 'text'}],
        'hazards': [{'name', 'description', 'when' ('day'|'night'|'any'), 'terrain',
                     'chance_per_hour', 'blocked_by' (sheltered, fire, flags), 'minutes_to_strike',
                     'harm', 'retreat_text', 'strike_text', 'fail_text',
                     'responses': {name: {'verbs', 'success', 'bonus': {item: +p}}}}]}
    Text templates may use {resource}, {target}, {actor}, {item}.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sim.trace import trace

DEFAULT_HARM = {'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')}
CO_PRESENT_TILES = 2       # a target character must be this close for a social action
_rng = random.Random()


def config(context) -> Optional[dict]:
    module = getattr(getattr(context, 'map', None), 'scenario_module', None)
    return getattr(module, 'affordances', None) or getattr(module, 'survival', None)


def stakes_text(context, fallback=''):
    """The scenario's sentence on what is at stake physically and socially in its world, for the
    planning and act prompts; `fallback` (the engine's old wording) when the scenario has none."""
    module = getattr(getattr(context, 'map', None), 'scenario_module', None)
    return (getattr(module, 'stakes', None) or fallback).strip()


def act_note(character):
    """One line for the act prompt's here-and-now block, only when the world has affordances."""
    if getattr(character, 'body', None) is None or not config(character.context):
        return ''
    return ("#Needs, threats and what this place offers are real: a concrete Do or Move that "
            "deals with them is often the right act.")


def _harm(cfg):
    return {**DEFAULT_HARM, **(cfg.get('harm') or {})}


def _need_defs(cfg):
    return cfg.get('needs') or {}


def _word(cfg, need, value):
    words = (_need_defs(cfg).get(need) or {}).get('words')
    if need == _harm(cfg)['name']:
        words = _harm(cfg)['words']
    if not words:
        return f'{need} {value:.1f}'
    return words[0] if value < 0.3 else words[1] if value < 0.6 else words[2] if value < 0.85 else words[3]


@dataclass
class Body:
    levels: dict = field(default_factory=dict)      # need -> 0..1, including the harm need
    inventory: list = field(default_factory=list)
    money: float = 0
    flags: set = field(default_factory=set)         # 'dressed', 'suited', ... set by actions
    sheltered: bool = False
    last_tick: Optional[datetime] = None
    warned: set = field(default_factory=set)
    cfg: dict = field(default_factory=dict, repr=False)

    def level(self, need):
        return self.levels.get(need, 0.0)

    def summary(self):
        parts = [_word(self.cfg, n, v) for n, v in self.levels.items()]
        if self.inventory:
            parts.append('carrying ' + ', '.join(self.inventory))
        if self.cfg.get('currency'):
            parts.append(f"{self.money:g} {self.cfg['currency']}")
        if self.sheltered:
            parts.append('in shelter')
        parts += sorted(self.flags)
        return '; '.join(parts)

    def adjust(self, need, delta):
        old = self.levels.get(need, 0.0)
        self.levels[need] = min(1.0, max(0.0, old + delta))
        if delta < 0:   # allow a fresh warning next time the need rises past a threshold
            self.warned = {w for w in self.warned if not (w[0] == need and self.levels[need] < w[1])}


def attach(context):
    """Give every cast member a Body when the scenario opts in (Context.__init__)."""
    cfg = config(context)
    if not cfg:
        return
    # what the area offers and threatens, in the shared situation every planning prompt reads
    types = sorted({str(r.get('type')) for r in context.map.resource_registry.values()})
    dangers = [h['description'] + (' at night' if h.get('when') == 'night' else '') for h in cfg.get('hazards', [])]
    rising = [n for n, d in _need_defs(cfg).items() if d.get('per_hour')]
    brief = ('\nThe area offers: ' + ', '.join(t.replace('_', ' ').lower() for t in types) + '.' +
             (' Dangers: ' + '; '.join(dangers) + '.' if dangers else '') +
             (' ' + ', '.join(rising).capitalize() + ' grow as the hours pass.' if rising else ''))
    if brief not in context.current_state:
        context.current_state += brief
    harm = _harm(cfg)['name']
    for character in context.actors + context.extras:
        if getattr(character, 'body', None) is None and getattr(character, 'mapAgent', None) is not None:
            levels = {n: d.get('initial', 0.2) for n, d in _need_defs(cfg).items()}
            levels[harm] = 0.0
            character.body = Body(levels=levels, inventory=list(cfg.get('initial_inventory', [])),
                                  money=cfg.get('initial_money', 0), last_tick=context.simulation_time, cfg=cfg)


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
    for need, spec in _need_defs(cfg).items():
        rate = spec.get('per_hour', 0.0) * (spec.get('shelter_factor', 1.0) if body.sheltered else 1.0)
        body.adjust(need, rate * hours)
        crossed = [(level, text) for level, text in (spec.get('percepts') or {}).items()
                   if body.level(need) >= level and (need, level) not in body.warned]
        if crossed:   # several thresholds in one tick: one percept, the most severe
            body.warned.update((need, level) for level, _ in crossed)
            character.add_perceptual_input(max(crossed)[1], mode='internal')
            trace.event('body', character=character.name, need=need, level=round(body.level(need), 2))
        if spec.get('neglect_harm') and body.level(need) >= 0.95:   # neglect hurts
            body.adjust(_harm(cfg)['name'], spec['neglect_harm'] * hours)


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
    cfg = config(character.context) or {}
    found = []
    for rid, res in _resources_of(character.context, types):
        d = _distance(here, res['location'])
        if d <= view:
            # an id the act names first, then a type it names ('berries' over an apple tree), then distance
            named_type = bool(mentioned) and _names_type(cfg, mentioned, [str(res.get('type'))])
            found.append((rid.lower() not in (mentioned or ''), not named_type, d, rid, res))
    if not found:
        return None
    found.sort(key=lambda f: f[:3])
    _, _, d, rid, res = found[0]
    return rid, res, d


def _present(character, name=None):
    """Other cast members within CO_PRESENT_TILES (with bodies), or the one named."""
    context = character.context
    out = []
    for other in context.actors + context.extras:
        if other is character or getattr(other, 'body', None) is None or getattr(other, 'mapAgent', None) is None:
            continue
        if name and other.name.lower() != name.lower():
            continue
        out.append((other, _distance(_here(character), _here(other))))
    return out


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


def _names_type(cfg, text, types):
    """Does the act name one of these resource types? (the type, its id form, or a configured noun)"""
    nouns = cfg.get('type_nouns') or {}
    words = [w for t in types for w in nouns.get(t, []) + [t.lower(), t.lower().replace('_', ' ')]]
    return any(re.search(r'\b' + re.escape(w) + r'\d*\b', text) for w in words)   # 'berries3' names Berries


def _fmt(template, **kw):
    try:
        return template.format(**kw)
    except (KeyError, IndexError):
        return template


FAMILY_OWNER_RELATIONS = ('father of', 'mother of', 'husband of', 'wife of')


def _prop_value(value):
    """A resource property as config and text use it: an owner is stored as a map agent
    (Context gives each owned resource an owner NPC), and compared and shown by name."""
    return getattr(value, 'name', value)


def _household_owns(character, owner):
    """True when `owner` (a name) is the character or a parent or spouse of the character
    (reference manager relationships, e.g. declared by the play)."""
    if not owner:
        return False
    if owner == character.name:
        return True
    manager = getattr(character.context, 'reference_manager', None)
    relations = manager.get_relationships(owner) if manager else []
    return any(rel in FAMILY_OWNER_RELATIONS and other == character.name for rel, other in relations)


def _prop_ok(res, wanted, character):
    """requires_property check. Value None: the property must be unset. '{actor}': it must be the
    actor's name. '{household}' (for 'owner'): the actor, or the actor's parent or spouse."""
    props = (res or {}).get('properties', {})
    for key, value in (wanted or {}).items():
        have = _prop_value(props.get(key))
        if value == '{household}':
            if not _household_owns(character, have):
                return False
            continue
        value = _fmt(value, actor=character.name) if isinstance(value, str) else value
        if value is None and have:
            return False
        if value is not None and have != value:
            return False
    return True


# --------------------------------------------------------------------------- actions
def _missing(action, body):
    """Why the actor cannot do this action now (requirements on its own body), or ''."""
    if action.get('requires') and action['requires'] not in body.inventory:
        return f"needs a {action['requires']}"
    if action.get('transfer') and action['transfer'] not in body.inventory:
        return f"needs a {action['transfer']} to give"
    if action.get('transfer_money') and body.money < action['transfer_money']:
        return f"needs {action['transfer_money']:g} {body.cfg.get('currency', '')} to give".rstrip()
    if action.get('cost') and body.money < action['cost']:
        return f"costs {action['cost']:g} {body.cfg.get('currency', '')}".strip()
    if action.get('requires_flag') and action['requires_flag'] not in body.flags:
        return f"needs to be {action['requires_flag']} first"
    return ''


def available_actions(character):
    """One line listing what the character could physically do here (act and dialog prompts)."""
    cfg = config(character.context) if character.context else None
    body = getattr(character, 'body', None)
    if not cfg or body is None:
        return ''
    out, seen = [], set()
    for action in cfg.get('actions', []):
        missing = _missing(action, body)
        note = f" ({missing})" if missing else ''
        if action.get('target') == 'character':
            for other, d in _present(character):
                if d <= CO_PRESENT_TILES:
                    out.append(f"{action['name']}@{other.name}{note}")
            continue
        types = action.get('resource_types')
        if not types:
            out.append(action['name'] + note)
            continue
        near = _nearest(character, types)
        if near is None:
            continue
        rid, res, d = near
        if (action['name'], rid) in seen:
            continue
        seen.add((action['name'], rid))
        if res.get('properties', {}).get('remaining') == 0:
            continue
        if not _prop_ok(res, action.get('requires_property'), character):
            continue
        if d <= 1:
            out.append(f"{action['name']}@{rid} (here{'; ' + missing if missing else ''})")
        else:
            out.append(f"{action['name']}@{rid} ({d} tiles {_direction(_here(character), res['location'])}; "
                       f"first Move: {rid}{'; ' + missing if missing else ''})")
    threat = active_threat_for(character)
    if threat:
        h = threat['hazard']
        out.insert(0, f"THREAT: {h['description']} is {threat_position(character, threat)}; "
                      f"you could {', '.join(h.get('responses', {}).keys())}")
    return '; '.join(out[:12])


def affordance_prompt_line(character):
    """The optional #affordance field of the act format, for worlds with affordances only."""
    if getattr(character, 'body', None) is None or not config(character.context):
        return ''
    offered = available_actions(character)
    example = next((p.split(' (')[0] for p in offered.split('; ') if p and not p.startswith('THREAT')), 'rest')
    return ("\n#affordance for a Do that performs one of the actions listed under 'You could do here', "
            f"that action's exact label (e.g. {example}); omit for any other act.")


def _chosen(context, character, label):
    """('threat', response) or ('action', action, near, target) for an #affordance label, or None."""
    cfg = config(context)
    label = label.strip().strip('\'"`.').split(' (')[0]
    name, _, ref = label.partition('@')
    name, ref = name.strip().lower(), ref.strip()
    threat = active_threat_for(character)
    if threat and name in threat['hazard'].get('responses', {}):
        return ('threat', name)
    for action in cfg.get('actions', []):
        if action['name'].lower() != name:
            continue
        if action.get('target') == 'character':
            present = _present(character, ref) if ref else _present(character)
            return ('action', action, None, min(present, key=lambda p: p[1]) if present else None)
        types = action.get('resource_types')
        if not types:
            return ('action', action, None, None)
        res = context.map.resource_registry.get(ref) if ref else None
        if res is not None and str(res.get('type')) in types:
            return ('action', action, (ref, res, _distance(_here(character), res['location'])), None)
        return ('action', action, _nearest(character, types, mentioned=ref.lower() if ref else None), None)
    return None


def resolve_do(context, character, act_text, affordance=None, targets=None):
    """Apply the mechanics of a Do, if it is one of the world's actions. Returns a fact line for the
    narration, or None when it is not (the LLM decides as before). `targets`: the act's characters."""
    cfg = config(context)
    body = getattr(character, 'body', None)
    if not cfg or body is None:
        return None
    text = (act_text or '').lower()
    # the act's #affordance label, when it gave a valid one, decides; text matching is the fallback
    chosen = _chosen(context, character, affordance) if affordance else None
    via = 'label' if chosen else 'text'

    threat = active_threat_for(character)
    if threat and not (chosen and chosen[0] == 'action'):
        fact = _respond_to_threat(context, character, threat, text,
                                  chosen=chosen[1] if chosen else None)
        if fact:
            return fact

    if chosen and chosen[0] == 'action':
        _, action, near, target = chosen
        return _apply(context, character, action, near, target, via)

    # candidate actions: verbs match; rank those whose resource (or person) the act names, then
    # those in reach, then config order (so 'eat berries' is not taken as eating mushrooms)
    names = [t.name.lower() for t in (targets or []) if hasattr(t, 'name')]
    candidates = []
    for order, action in enumerate(cfg.get('actions', [])):
        types = action.get('resource_types')
        if action.get('target') == 'character':
            present = [p for p in _present(character) if p[0].name.lower() in names or p[0].name.lower() in text]
            named = bool(present)
            if not (named and (_matches(text, action.get('verbs', [])) or _matches(text, action.get('named_verbs', [])))):
                continue
            if action.get('transfer') and action['transfer'].lower() not in text:
                continue   # 'gives Joe a look' is not giving him the bread
            target = min(present, key=lambda p: p[1])
            # first: it needs its own verb, a present person the act names, and the item given
            candidates.append((False, False, target[1] > CO_PRESENT_TILES, order, action, None, target))
            continue
        named = bool(types) and _names_type(cfg, text, types)
        # 'named_verbs' are too generic alone ('cups his hands'): they count when the act names the thing
        if not (_matches(text, action.get('verbs', [])) or (named and _matches(text, action.get('named_verbs', [])))):
            continue
        near = _nearest(character, types, mentioned=text) if types else None
        if types and not named and any(_names_type(cfg, text, a.get('resource_types') or [])
                                       for a in cfg.get('actions', []) if a is not action and a.get('resource_types')):
            continue   # the act names a different kind of thing
        in_reach = (not types) or (near is not None and near[2] <= 1)
        candidates.append((True, not named, not in_reach, order, action, near, None))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[:4])
    _, _, _, _, action, near, target = candidates[0]
    return _apply(context, character, action, near, target, via)


def _can_approach(context, character, d):
    """A Do aimed at a resource a short walk away (config 'approach_tiles', default 8), with no
    threat active, walks there first instead of being refused. No line-of-sight test: the
    'You could do here' offer lists resources by distance alone, and in forest the map's
    line of sight blocked targets 5 tiles off that the character had just been offered."""
    return d <= (config(context) or {}).get('approach_tiles', 8) and not active_threat_for(character)


def _refuse(character, action, fact, via, target=None):
    _record(character, action, fact, via, applied=False, target=target)
    return fact


def _apply(context, character, action, near, target, via):
    """Check reach, stock and requirements, then apply the action's effects; returns the fact line."""
    cfg = config(context)
    body = character.body
    types = action.get('resource_types')
    resource_id = res = None
    other = None
    approach = ''
    if action.get('target') == 'character':
        if target is None:
            return _refuse(character, action, f"There is no one here to {action['name']}.", via)
        other, d = target
        if d > CO_PRESENT_TILES:
            if not _can_approach(context, character, d):
                return _refuse(character, action, f"{other.name} is {d} tiles away; {character.name} must go to them first.",
                               via, other.name)
            minutes = travel(character, *_here(other))
            approach = f"{character.name} walks {d} tiles to {other.name} ({round(minutes)} min). "
    elif types:
        if near is None:
            return _refuse(character, action, f"There is no {' or '.join(types).replace('_', ' ').lower()} in sight here.", via)
        resource_id, res, d = near
        if d > 1 and _can_approach(context, character, d):
            # the narration has usually put the character there already: walk over, then act
            heading = _direction(_here(character), res['location'])
            minutes = travel(character, *res['location'])
            approach = f"{character.name} walks {d} tiles {heading} to {resource_id} ({round(minutes)} min). "
            d = 0
        if d > 1:
            return _refuse(character, action, f"{resource_id} is {d} tiles {_direction(_here(character), res['location'])}; "
                                              f"{character.name} must move there first.", via)
        props = res.setdefault('properties', {})
        if action.get('deplete') and props.setdefault('remaining', action['deplete']) <= 0:
            return _refuse(character, action, approach + f"There is nothing left at {resource_id}.", via)
        if not _prop_ok(res, action.get('requires_property'), character):
            return _refuse(character, action, approach + f"{character.name} cannot {action['name']} at {resource_id} "
                                                         f"({', '.join(f'{k}: {_prop_value(v)}' for k, v in props.items() if k != 'remaining') or 'not possible now'}).", via)
    missing = _missing(action, body)
    if missing:
        return _refuse(character, action, approach + f"{character.name} cannot {action['name']}: {missing}.", via,
                       other.name if other else None)

    # ---- effects
    for need, delta in (action.get('effects') or {}).items():
        body.adjust(need, delta)
    item = action.get('transfer')
    if item and other is not None:
        body.inventory.remove(item)
        other.body.inventory.append(item)
    if action.get('transfer_money') and other is not None:
        body.money -= action['transfer_money']
        other.body.money += action['transfer_money']
    if other is not None:
        for need, delta in (action.get('target_effects') or {}).items():
            other.body.adjust(need, delta)
    if action.get('consumes') and action['consumes'] in body.inventory:
        body.inventory.remove(action['consumes'])
    if action.get('gain'):
        body.inventory.append(action['gain'])
    found = None
    if action.get('finds'):
        roll = _rng.random()
        for f in action['finds']:
            roll -= f.get('chance', 0)
            if roll < 0:
                found = f['item']
                body.inventory.append(found)
                break
    if action.get('cost'):
        body.money -= action['cost']
    if action.get('earn'):
        body.money += action['earn']
    flag = action.get('sets')
    if flag == 'sheltered':
        body.sheltered = True
    elif flag == 'fire':
        context.survival_fire = {'at': _here(character), 'until': context.simulation_time + timedelta(hours=3)}
    elif flag:
        body.flags.add(flag)
    for key, value in (action.get('sets_property') or {}).items():
        if res is not None:
            res['properties'][key] = _fmt(value, actor=character.name) if isinstance(value, str) else value
    if res is not None and action.get('deplete'):
        res['properties']['remaining'] -= 1
    context.simulation_time += timedelta(minutes=action.get('minutes', 10))
    body.last_tick = context.simulation_time   # needs during the action are folded into its effect

    kw = dict(resource=resource_id or 'here', target=other.name if other else '', actor=character.name, item=item or found or '')
    where = f" at {resource_id}" if resource_id else ''
    fact = approach + f"{character.name} {_fmt(action.get('text', action['name']), **kw)}{'' if action.get('text') else where}."
    if action.get('finds'):
        fact += f" {character.name} finds {found}." if found else f" {character.name} finds nothing useful."
    risk = action.get('risk')
    if risk and _rng.random() < risk.get('chance', 0):
        for need, delta in risk.get('effects', {}).items():
            body.adjust(need, delta)
        fact += ' ' + _fmt(risk.get('text', ''), **kw)
    fact += f" ({character.name} is now {body.summary()}"
    fact += f"; {other.name} is now {other.body.summary()}.)" if other is not None else ".)"
    _record(character, action, fact, via, applied=True, resource=resource_id, approached=bool(approach),
            target=other.name if other else None, found=found)
    return fact


def _record(character, action, fact, via, applied, resource=None, approached=False, target=None, found=None):
    trace.event('affordance', character=character.name, action=action['name'],
                category=action.get('category', 'survival'), applied=applied, resource=resource,
                target=target, found=found, fact=fact, via=via, approached=approached)


def travel(character, x, y):
    """Move a character to (x, y), advancing time by distance x terrain cost (affordance worlds)."""
    context = character.context
    cfg = config(context) or {}
    start = _here(character)
    tiles = _distance(start, (x, y))
    costs = getattr(context.map.scenario_module, 'infrastructure_rules', {}).get('terrain_costs', {})
    try:
        terrain = context.map.patches[x][y].terrain_type.name
        cost = costs.get(terrain, 1.5)
        cost = 1.5 if cost == float('inf') else cost
    except Exception:
        cost = 1.5
    minutes = tiles * cost * cfg.get('tile_minutes', 2)
    character.mapAgent.x, character.mapAgent.y = x, y
    if getattr(character, 'body', None) is not None:
        character.body.sheltered = False
        if cfg.get('travel_need'):
            character.body.adjust(cfg['travel_need'], 0.01 * tiles)
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
    return (('sheltered' in blocks and body.sheltered) or ('fire' in blocks and near_fire)
            or any(b in body.flags for b in blocks))


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
    harm_need = _harm(cfg)['name']

    threat = getattr(context, 'active_threat', None)
    if threat:
        targets = [c for c in cast if c.name in threat['targets']]
        if all(_blocked(context, c, threat['hazard']) for c in targets):
            _end_threat(context, threat, threat['hazard'].get('retreat_text', f"The {threat['hazard']['name']} passes."))
        elif now >= threat['deadline']:
            harm = threat['hazard'].get('harm', 0.3)
            for c in targets:
                c.body.adjust(harm_need, harm)
            _end_threat(context, threat, threat['hazard'].get('strike_text', f"The {threat['hazard']['name']} strikes.")
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
        text = f"{hazard['description'][0].upper() + hazard['description'][1:]} appears {threat_position(anchor, context.active_threat)} of {anchor.name}, threatening {', '.join(targets)}."
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


def _respond_to_threat(context, character, threat, text, chosen=None):
    hazard = threat['hazard']
    cfg = config(context)
    for name, response in hazard.get('responses', {}).items():
        if chosen is not None and name != chosen:
            continue
        if chosen is None and not _matches(text, response.get('verbs', [])):
            continue
        p = response.get('success', 0.5)
        for item, bonus in response.get('bonus', {}).items():
            if item == 'fire' and getattr(context, 'survival_fire', None):
                p += bonus
            elif item in character.body.inventory or item in character.body.flags:
                p += bonus
        context.simulation_time += timedelta(minutes=5)
        if _rng.random() < min(0.95, p):
            fact = f"{character.name} tries to {name}: it works. " + hazard.get('retreat_text', f"The {hazard['name']} passes.")
            _end_threat(context, threat, fact)
        else:
            character.body.adjust(_harm(cfg)['name'], hazard.get('harm', 0.3) / 2)
            fail = _fmt(hazard.get('fail_text', f"the {hazard['name']} is not deterred"), actor=character.name)
            fact = f"{character.name} tries to {name}, but {fail}; {character.name} is now {character.body.summary()}."
            context.add_event(fact)
        trace.event('hazard', phase='response', hazard=hazard['name'], character=character.name, response=name, fact=fact)
        return fact
    return None
