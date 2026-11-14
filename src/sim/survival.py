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
                     'risk': {'chance', 'effects', 'text'}, 'text',
                     'success': {'chance', 'text'} (the deed can fail: on a failed roll the
                     time and effects are spent, but sets_property is not applied, and the
                     text tells the failure),
                     items ('requires', 'consumes', 'transfer') may be '*suffix': any held item
                     ending so ("*contract" is "Hu's contract"); 'gain' may use {actor}, {item},
                     'remote': True (a 'target' action that needs no nearness: a phone message),
                     'conveys': suffix (with 'transfer': an item named '<ResourceId><suffix>',
                     a deed, makes the target that resource's owner),
                     'target_gain': item the target receives (may use {said}: the words the
                     act quotes), 'reveals': prefix (lists the target's items that start so:
                     the messages on someone's phone)}],
        'private_items': [prefixes of items others do not see a character carrying],
        'hazards': [{'name', 'description', 'when' ('day'|'night'|'any'), 'terrain',
                     'chance_per_hour', 'blocked_by' (sheltered, fire, flags), 'minutes_to_strike',
                     'harm', 'retreat_text', 'strike_text', 'fail_text',
                     'escape_tiles' (8): a threat is in one place; a target this far from it
                     when it strikes is spared, and it ends once every target is clear,
                     'source': {'resource_types', 'requires_property', 'within' (tiles, 6)}:
                     the hazard comes from a resource in that state (a line not yet repaired);
                     it threatens only characters near one, appears there, and ends
                     ('resolved_text') when the resource leaves that state,
                     'responses': {name: {'verbs', 'success', 'bonus': {item: +p},
                                          'sets_property' (on the source, when it works)}}}]}
    Text templates may use {resource}, {target}, {actor}, {item}.
"""
from __future__ import annotations

import random
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sim.trace import trace

DEFAULT_HARM = {'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')}
CO_PRESENT_TILES = 2       # a target character must be this close for a social action
NEED_FELT = 0.3            # below this a need is not felt ('not hungry'): Body words change here
_rng = random.Random()


def config(context) -> Optional[dict]:
    module = getattr(getattr(context, 'map', None), 'scenario_module', None)
    return getattr(module, 'affordances', None) or getattr(module, 'survival', None)


def stakes_text(context, fallback=''):
    """The scenario's sentence on what is at stake physically and socially in its world, for the
    planning and act prompts; `fallback` (the engine's old wording) when the scenario has none."""
    module = getattr(getattr(context, 'map', None), 'scenario_module', None)
    return (getattr(module, 'stakes', None) or fallback).strip() + (f'\n{PLAN_RULE}' if config(context) else '')


# For every prompt that writes or rewrites the plan. A plan wrote 'the boar swerves, goring Joe's
# leg' as a scene's ending: the actor played the wound from his first line, the world's own boar
# then struck the other character, and the play carried two accounts of who was hurt.
PLAN_RULE = ("What happens to bodies, and how a danger ends, is decided by the world when the scene is played, never by the plan. "
             "A plan may put the characters in a danger's way. It does not say that someone is wounded, bitten, healed, fed or killed, "
             "nor that an animal is driven off or strikes. "
             "States and endings say where the characters stand with each other and with the central question: "
             "what has been chosen, said, refused, found out or lost between them. "
             "Ordinary eating, drinking and rest happen between scenes where the means are at hand; a scene is about them "
             "only when a need is desperate or what meets it is out of reach.")


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

    def summary(self, public=False):
        """The body in words. public: as someone else sees it, without the items the world
        keeps private ('private_items' prefixes: the messages on a phone)."""
        parts = [_word(self.cfg, n, v) for n, v in self.levels.items()]
        hidden = tuple(self.cfg.get('private_items', ())) if public else ()
        carried = [i for i in self.inventory if not (hidden and i.startswith(hidden))]
        if carried:
            parts.append('carrying ' + ', '.join(carried))
        if self.cfg.get('currency'):
            parts.append(f"{self.money:g} {self.cfg['currency']}")
        if self.sheltered:
            parts.append('in shelter')
        parts += sorted(self.flags)
        return '; '.join(parts)

    def condition(self):
        """How the body is, in words, without what it carries: 'very thirsty; tired; in shelter'."""
        return '; '.join([_word(self.cfg, n, v) for n, v in self.levels.items()]
                         + (['in shelter'] if self.sheltered else []) + sorted(self.flags))

    def visible(self):
        """The body as another person sees it: hurt or not, what is carried, sheltered or not.
        Thirst, hunger and tiredness are the body's own to know."""
        harm = _harm(self.cfg)['name']
        hidden = tuple(self.cfg.get('private_items', ()))
        carried = [i for i in self.inventory if not (hidden and i.startswith(hidden))]
        parts = [_word(self.cfg, harm, self.levels.get(harm, 0.0))] if harm in self.levels else []
        if carried:
            parts.append('carrying ' + ', '.join(carried))
        if self.sheltered:
            parts.append('in shelter')
        return '; '.join(parts) or 'nothing remarkable'

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
    # what threatens and wears on the cast, in the shared situation every planning prompt reads.
    # What a place offers is told where it is true: per place to the playwright (stage.gazetteer)
    # and as 'You could do here' to each character.
    dangers = [h['description'] + (' at night' if h.get('when') == 'night' else '') for h in cfg.get('hazards', [])]
    rising = [n for n, d in _need_defs(cfg).items() if d.get('per_hour')]
    brief = '\n' + ' '.join(filter(None, ['Dangers: ' + '; '.join(dangers) + '.' if dangers else '',
                                          ', '.join(rising).capitalize() + ' grow as the hours pass.' if rising else '']))
    if brief.strip() and brief not in context.current_state:
        context.current_state += brief
    for character in context.actors + context.extras:
        give_body(context, character)


def give_body(context, character):
    """A Body for one character, if the world has affordances and it has none yet. NPCs get one
    when a play gives them a part (Context.set_resource_owner): a father who can hold a deed."""
    cfg = config(context)
    if cfg and getattr(character, 'body', None) is None and getattr(character, 'mapAgent', None) is not None:
        levels = {n: d.get('initial', 0.2) for n, d in _need_defs(cfg).items()}
        levels[_harm(cfg)['name']] = 0.0
        character.body = Body(levels=levels, inventory=list(cfg.get('initial_inventory', [])),
                              money=cfg.get('initial_money', 0), last_tick=context.simulation_time, cfg=cfg)


def _embodied(context):
    """Everyone with a body: the cast, and NPCs given one."""
    return [c for c in context.actors + context.extras + context.npcs if getattr(c, 'body', None) is not None]




REAL_JUMP_HOURS = 2   # a gap this long is a jump in the play, not the pause between two scenes


def _spans_night(start, end):
    """Does the time from start to end take in the small hours?"""
    t = start
    while t <= end:
        if t.hour >= 23 or t.hour < 4:
            return True
        t += timedelta(hours=1)
    return False


PASSING_TILES = 8    # what meets a need this near the cast at a scene's end is used off the page


def meet_in_passing(context, cast):
    """A scene is over: each of `cast` meets an ordinary need off the page where something that
    meets it is near (a spring for thirst, berries for hunger, a hut for rest), as people do
    between the things a play shows. A need at its worst word stays for the play to deal with,
    and so does one with nothing near. (Day 2 of a play went on spring and apple runs while the
    lead the characters had found sat idle, and a man was 'parched' through three scenes at a
    spring because no scene was about drinking.) Returns {name: [needs met]}."""
    cfg = config(context) or {}
    met = {}
    for character in cast:
        body = getattr(character, 'body', None)
        if body is None or getattr(character, 'mapAgent', None) is None:
            continue
        for need, spec in _need_defs(cfg).items():
            level = body.levels.get(need, 0.0)
            if spec.get('per_hour', 0.0) <= 0 or not NEED_FELT <= level < 0.85:
                continue
            for action in cfg.get('actions', []):
                eases = (action.get('effects') or {}).get(need, 0.0) < 0
                if not eases or action.get('requires') or action.get('target') or not action.get('resource_types'):
                    continue
                near = _stocked(character, action['resource_types']) if action.get('deplete') else _nearest(character, action['resource_types'])
                if near is not None and near[2] <= PASSING_TILES:
                    body.adjust(need, (NEED_FELT - 0.05) - level)
                    met.setdefault(character.name, []).append(need)
                    break
    for name, needs in met.items():
        character = next(c for c in cast if c.name == name)
        character.add_perceptual_input(f"Between scenes you saw to it: {', '.join(needs)} no longer pressing.", percept=False, mode='internal')
    if met:
        trace.event('met_in_passing', met=met)
    return met


def pass_offstage(context, gap):
    """Time passes off stage, to context.simulation_time + gap (the caller moves the clock).
    Needs do not rise off stage. Over a real jump the cast is taken to have looked after itself:
    hunger and thirst that were felt are met, and tiredness too when the gap takes in a night.
    Injuries stay. (Needs used to rise across a gap, so a play that moved on to dusk for a night
    watch opened on two people who had to find water first; and frozen, a character hungry in the
    morning was still hungry at dusk having done nothing about it. A plan that wants them hungry
    says nothing of bodies: the world keeps them.) A danger left standing is over, and a fire burns out.
    Returns the names of the needs that were met."""
    now = context.simulation_time + gap
    hours = gap.total_seconds() / 3600
    cfg = config(context) or {}
    jump = hours >= REAL_JUMP_HOURS
    night = _spans_night(context.simulation_time, now)
    met = set()
    for character in _embodied(context):
        body = character.body
        if jump:
            for need, spec in _need_defs(cfg).items():
                level = body.levels.get(need, 0.0)
                if spec.get('per_hour', 0.0) <= 0 or level < NEED_FELT:
                    continue
                if re.search(r'fatig|tired|sleep|rest', need) and not night:
                    continue
                body.adjust(need, (NEED_FELT - 0.05) - level)
                met.add(need)
        body.last_tick = now
    if hasattr(context, '_hazard_last'):
        context._hazard_last = now
    threat = getattr(context, 'active_threat', None)
    if threat and (jump or now >= threat['deadline']):   # it does not strike, or wait hours for, a cast that is not on stage
        _end_threat(context, threat, threat['hazard'].get('retreat_text', f"The {threat['hazard']['name']} passes."), tell=False)
    fire = getattr(context, 'survival_fire', None)
    if fire and fire['until'] <= now:
        context.survival_fire = None
    return met


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
    for other in _embodied(context):
        if other is character or getattr(other, 'mapAgent', None) is None:
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
    actor's name. '{household}' (for 'owner'): the actor, or the actor's parent or spouse;
    '!{household}': anyone else's."""
    props = (res or {}).get('properties', {})
    for key, value in (wanted or {}).items():
        have = _prop_value(props.get(key))
        if value in ('{household}', '!{household}'):
            if _household_owns(character, have) != (value == '{household}'):
                return False
            continue
        value = _fmt(value, actor=character.name) if isinstance(value, str) else value
        if value is None and have:
            return False
        if value is not None and have != value:
            return False
    return True


# --------------------------------------------------------------------------- actions
def _held(body, spec):
    """The item in the inventory that an item spec names: the item itself, or for '*suffix' the
    first held item ending with the suffix. None when nothing matches."""
    if not spec:
        return None
    if spec.startswith('*'):
        return next((i for i in body.inventory if i.endswith(spec[1:])), None)
    return spec if spec in body.inventory else None


def _missing(action, body):
    """Why the actor cannot do this action now (requirements on its own body), or ''."""
    if action.get('requires') and not _held(body, action['requires']):
        return f"has no {action['requires'].lstrip('*')}"
    gain = action.get('gain')
    if gain and '{' not in gain and gain in body.inventory and not action.get('effects'):
        return f"already has {'' if gain.endswith('s') else 'a '}{gain}"   # one branch in hand is enough: no taking it again
    if action.get('transfer') and not _held(body, action['transfer']):
        return f"has no {action['transfer'].lstrip('*')} to give"
    if action.get('transfer_money') and body.money < action['transfer_money']:
        return f"needs {action['transfer_money']:g} {body.cfg.get('currency', '')} to give".rstrip()
    if action.get('cost') and body.money < action['cost']:
        return f"costs {action['cost']:g} {body.cfg.get('currency', '')}".strip()
    if action.get('requires_flag') and action['requires_flag'] not in body.flags:
        return f"needs to be {action['requires_flag']} first"
    if _sated(action, body):
        return 'is ' + ' and '.join(_word(body.cfg, need, body.levels.get(need, 0.0)) for need in action['effects'])
    return ''


def _sated(action, body):
    """Would this action only use up food the body does not want? (Joe ate at three bushes in a
    row while 'not hungry'.) Such an act is no action of the world's: nothing is used and nothing
    is refused, so chewing on is just chewing. (Refused, 'chews the apple loudly' after a first
    bite was told as 'a loud crunching noise against the air'.)"""
    effects = action.get('effects') or {}
    return bool(effects) and bool(action.get('deplete') or action.get('consumes')) and all(
        delta < 0 and body.levels.get(need, 0.0) < NEED_FELT for need, delta in effects.items())


def _stocked(character, types):
    """The nearest resource of these types that still has something left, else the nearest."""
    here = _here(character)
    found = sorted((res.get('properties', {}).get('remaining') == 0, _distance(here, res['location']), rid, res)
                   for rid, res in _resources_of(character.context, types) if _distance(here, res['location']) <= 12)
    return (found[0][2], found[0][3], found[0][1]) if found else None


def _left(res):
    """What is left of a resource that actions use up, in words; '' when it is untouched."""
    remaining = res.get('properties', {}).get('remaining')
    return '' if remaining is None else 'none left' if remaining <= 0 else 'nearly gone' if remaining == 1 else 'some left'


def stock(character, within=6):
    """What is left at the nearby places that have been drawn on, for the narrator: nobody was
    shown this, so depletion was invented ('the bare stem of the depleted bush' after one meal)
    while a bush that really was empty looked full."""
    if not config(character.context) or getattr(character, 'mapAgent', None) is None:
        return ''
    here = _here(character)
    return '; '.join(f'{rid}: {_left(res)}' for rid, res in character.context.map.resource_registry.items()
                     if _left(res) and _distance(here, res['location']) <= within)


def _unseen(character, rid):
    """True when the character is new to this country (Context.strangers) and has not yet seen
    this place: it is then the actor's to know of (a stage direction), not the character's."""
    if not getattr(character, 'unknown_land', False):
        return False
    known = getattr(getattr(character, 'resource_models', None), 'known_resources', None)
    return known is not None and rid not in known


def unseen_places(character):
    """The places behind the stage directions of available_actions(part='unseen')."""
    return sorted(set(re.findall(r'@(\w+) \(', available_actions(character, part='unseen'))))


def available_actions(character, part='all'):
    """One line listing what the character could physically do here (act and dialog prompts):
    offered_actions, joined."""
    return '; '.join(offered_actions(character, part))


def offered_actions(character, part='all'):
    """What the character could physically do here, one entry for each thing.
    part='known': what the character itself can know of (at hand, or at a place it has seen);
    part='unseen': what lies at places it has not seen, for the director's notes to its actor.
    The two are the same list divided, and differ from 'all' only for a character new to the
    country. (Joe, woken in a forest he had never seen, was shown 'drink@Spring2 (4 tiles
    Northeast)' among his surroundings and told Samantha where the spring was.)"""
    cfg = config(character.context) if character.context else None
    body = getattr(character, 'body', None)
    if not cfg or body is None:
        return []
    out, seen = [], set()
    for action in cfg.get('actions', []):
        missing = _missing(action, body)
        note = f" ({missing})" if missing else ''
        if action.get('target') == 'character':
            if action.get('remote'):
                out.append((bool(missing), f"{action['name']}@<anyone, wherever they are>{note}"))
                continue
            for other, d in _present(character):
                if d <= CO_PRESENT_TILES:
                    out.append((bool(missing), f"{action['name']}@{other.name}{note}"))
            continue
        types = action.get('resource_types')
        if not types:
            out.append((bool(missing), action['name'] + note))
            continue
        if action.get('requires_property'):
            # whose place it is decides: offer it where this character may do it (a character
            # knows the way to its own farm), not at whichever is nearest
            own = _eligible(character, action)
            near = (own[1], own[2], own[0]) if own else None
        else:
            # the nearest with something left: an empty bush used to hide a full one further on
            near = _stocked(character, types) if action.get('deplete') else _nearest(character, types)
        if near is None:
            continue
        rid, res, d = near
        if (action['name'], rid) in seen:
            continue
        seen.add((action['name'], rid))
        if res.get('properties', {}).get('remaining') == 0:
            continue
        if action.get('deplete') and _left(res) == 'nearly gone':
            missing = (missing + '; ' if missing else '') + 'nearly gone'
        if d <= 1:
            if getattr(character, 'unknown_land', False) and hasattr(character, 'resource_models'):
                character.resource_models.add_resource_model(rid)   # standing at a place is having seen it
            out.append((bool(missing), f"{action['name']}@{rid} (here{'; ' + missing if missing else ''})", False))
        else:
            out.append((1 + bool(missing), f"{action['name']}@{rid} (about {walk_minutes(character.context, d)} minutes' walk {_direction(_here(character), res['location'])}; "
                                           f"first Move: {rid}{'; ' + missing if missing else ''})", _unseen(character, rid)))
    # what can be done now comes first; what needs a walk or something not held is cut first
    # (the list was cut at 12 in config order, which dropped every action with a person: a
    # father standing with his son was never offered 'sign over the farm')
    out = [o if len(o) == 3 else (*o, False) for o in out]
    out = sorted(out, key=lambda o: o[0])[:14]
    if part == 'unseen':
        return [text for _, text, unseen in out if unseen]
    out = [text for _, text, unseen in out if part == 'all' or not unseen]
    threat = active_threat_for(character)
    if threat:
        h = threat['hazard']
        out.insert(0, f"THREAT: {h['description']} is {threat_position(character, threat)}; "
                      f"you could {', '.join(h.get('responses', {}).keys())}")
    return out


def affordance_prompt_line(character):
    """The optional #affordance field of the act format, for worlds with affordances only."""
    if getattr(character, 'body', None) is None or not config(character.context):
        return ''
    offered = available_actions(character)
    example = next((p.split(' (')[0] for p in offered.split('; ') if p and not p.startswith('THREAT')), 'rest')
    return ("\n#affordance only for a Do whose action text itself performs one of the actions listed under 'You could do here' "
            f"(the text says you {example.split('@')[0]}, not that you look at or think about it): that action's exact label "
            f"(e.g. {example}); omit for any other act.")


_CLAIM = re.compile(r"\bI(?:'m|’m| am)\s+(?:\w+ly\s+)?(\w+ing)\b", re.I)


def spoken_deed(context, character, line):
    """A spoken line that says the speaker is doing one of the world's actions, here and now
    ("I'm drinking"), does it, so that what was said and the body agree. (A whole scene of
    drinking at the spring was only talk, and left both of them thirsty.) Only a first-person
    'I am ...ing' of an action on a resource within reach. Returns the fact line, or None."""
    cfg = config(context)
    if not cfg or getattr(character, 'body', None) is None:
        return None
    for claim in _CLAIM.finditer(line or ''):
        word = claim.group(1).lower()
        for action in cfg.get('actions', []):
            types = action.get('resource_types')
            if not types or action.get('target') or action.get('remote') or not _matches(word, action.get('verbs', [])):
                continue
            near = _nearest(character, types)
            if near is not None and near[2] <= CO_PRESENT_TILES and not _missing(action, character.body):
                character.last_affordance = None
                fact = _apply(context, character, action, near, None, 'speech', line)
                return fact if fact and (character.last_affordance or {}).get('applied') else None   # refused: it was only talk
    return None


def do_business(context, character, label):
    """Carry out a planned scene action ('drink@Spring1') for a character who has not done it.
    Returns the fact line, or None when the label names no action of this world."""
    if not config(context) or getattr(character, 'body', None) is None:
        return None
    chosen = _chosen(context, character, label)
    if not chosen or chosen[0] != 'action':
        return None
    _, action, near, target = chosen
    if action.get('sets') == 'sheltered' and character.body.sheltered:
        return None   # asleep in the thicket is already sheltered there
    character.last_affordance = None
    fact = _apply(context, character, action, near, target, 'business', '')
    return fact if fact and (character.last_affordance or {}).get('applied') else None


def acts_at_a_distance(character, act):
    """Is this Do one of the world's remote actions (sending a message)? Then the actor does
    not walk to its target first."""
    cfg = config(character.context)
    if not cfg:
        return False
    label = (getattr(act, 'affordance', None) or '').split('@')[0].strip().lower()
    text = (getattr(act, 'action', '') or '').lower()
    return any(a.get('remote') and (a['name'].lower() == label or _matches(text, a.get('verbs', [])))
               for a in cfg.get('actions', []))


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
            d = _distance(_here(character), res['location'])
            at_hand = _nearest(character, types)
            if d > 1 and at_hand is not None and at_hand[2] <= 1:
                # a deed is done where the doer stands: searching under the cot in the hut was
                # resolved to a cave out of sight, and 'the cave searched carefully' was told
                return ('action', action, at_hand, None)
            return ('action', action, (ref, res, d), None)
        return ('action', action, _nearest(character, types, mentioned=ref.lower() if ref else None), None)
    return None


def holdings(character):
    """Who is carrying what, among the character and those with it, as anyone can see: for
    the narrator, so that things the mechanics track stay with whoever holds them."""
    body = getattr(character, 'body', None)
    if body is None or not config(character.context):
        return ''
    people = [character] + [o for o, d in _present(character) if d <= CO_PRESENT_TILES]
    here = ', '.join(p['name'] for p in lying(character.context, _here(character)))
    return '\n'.join([f"{p.name}: {', '.join(i for i in p.body.inventory if not i.startswith(tuple(p.body.cfg.get('private_items', ())) or ('\0',))) or 'nothing'}"
                      for p in people] + ([f'Lying here, for anyone to take: {here}'] if here else []))


# --------------------------------------------------------------------------- things that come into the world
# A plan, or a character's deed, sometimes brings in a thing the world does not hold. It is then
# either made real here, in the world's own terms, or refused (sim/record.py decides which).
# Made real, a thing is one of: a prop (something nobody acts on through the world's rules: a
# sickle, a phone), an item the world's actions use, a place of a kind the world has, or a
# change to a body. Props are held, or lie where they were left, and change hands by the three
# deeds in prop_offers / _prop_do.
def item_kinds(context):
    """The things this world's actions use (gain, need, hand over or find), by name."""
    kinds = []
    for action in (config(context) or {}).get('actions', []):
        for spec in [action.get(k) for k in ('gain', 'requires', 'consumes', 'transfer')] + [f.get('item') for f in action.get('finds') or []]:
            if spec and '{' not in spec and spec.lstrip('*') not in kinds:
                kinds.append(spec.lstrip('*'))
    return kinds


def place_kinds(context):
    """{kind of place: what can be done there}, for the kinds this map has and its actions use."""
    kinds = {}
    for action in (config(context) or {}).get('actions', []):
        for kind in action.get('resource_types') or []:
            if context.map.get_resource_type(kind) is not None:
                kinds.setdefault(kind, []).append(action['name'])
    return kinds


def whereabouts_of(context, name):
    """Where a thing is now, in a few words: 'held by Samantha', 'lying at Spring1', or '' when
    the world does not hold it. Matched as record.admit matches a thing ('watch', 'wristwatch')."""
    name = re.sub(r'^(a|an|the|some)\s+', '', (name or '').strip().lower())[:40]   # as add_thing names it
    same = lambda held: held.lower() == name or held.lower().endswith(' ' + name) or name.endswith(' ' + held.lower())
    for person in list(context.actors) + list(getattr(context, 'extras', [])):
        body = getattr(person, 'body', None)
        if body is not None and any(same(i) for i in body.inventory):
            return f'held by {person.name}'
    for prop in getattr(context, 'props', []):
        if same(prop['name']):
            near = _nearest_resource_id(context, prop['at'])
            return f"lying at {near}" + (', not yet found' if prop.get('hidden') else '') if near else 'lying out of sight'
    return ''


def _nearest_resource_id(context, xy):
    best = min(((_distance(xy, res['location']), rid) for rid, res in context.map.resource_registry.items()
                if res.get('location')), default=None)
    return best[1] if best else ''


def lying(context, xy, hidden=False):
    """The things lying at or near a spot, in plain sight; hidden=True for those that are found
    only by searching (a clue the director placed, see searchable)."""
    return [p for p in getattr(context, 'props', [])
            if _distance(p['at'], xy) <= CO_PRESENT_TILES and bool(p.get('hidden')) == hidden]


def searchable(context, res):
    """Whether this world has an action that finds things at a place of this kind."""
    return any(a.get('finds') and (not a.get('resource_types') or str(res.get('type')) in a['resource_types'])
               for a in (config(context) or {}).get('actions', []))


def add_thing(context, name, holder=None, at=None, hidden=False):
    """A thing comes into the world: in `holder`'s hands, or lying at `at` (hidden: not seen
    until someone searches there). Returns the fact."""
    name = re.sub(r'^(a|an|the|some)\s+', '', name.strip().lower())[:40]
    if not name:
        return ''
    if holder is not None and getattr(holder, 'body', None) is not None:
        if name not in holder.body.inventory:
            holder.body.inventory.append(name)
        fact = f"{holder.name} has {_a(name)}."
    else:
        context.props = getattr(context, 'props', [])
        if not any(p['name'] == name for p in lying(context, at)):
            context.props.append({'name': name, 'at': tuple(at), **({'hidden': True} if hidden else {})})
        fact = f"There is {_a(name)} here."
    trace.event('thing_added', name=name, holder=holder.name if holder is not None else None)
    return fact


def _a(name):
    return name if name.endswith('s') else f"{'an' if name[0] in 'aeiou' else 'a'} {name}"


def add_place(context, kind, description, xy):
    """A place of one of the world's own kinds comes into the map at xy (a pool is a Spring:
    it can be drunk from, by the same rules). Returns (id, fact), or (None, '') for a kind this
    world does not have."""
    world = context.map
    resource_type = world.get_resource_type(kind)
    if resource_type is None or kind not in place_kinds(context):
        return None, ''
    rid = world._generate_resource_id(resource_type)
    world.resource_registry[rid] = {'id': rid, 'type': resource_type, 'name': rid,
                                    'description': description or f'a {kind.lower()}', 'location': tuple(xy), 'properties': {}}
    try:
        world.patches[xy[0]][xy[1]].resources[rid] = world.resource_registry[rid]
        world._resource_counters[resource_type] = world._resource_counters.get(resource_type, 0) + 1
    except Exception:
        pass
    trace.event('place_added', id=rid, type=kind, description=description)
    return rid, f"There is {description or _a(kind.lower())} here ({rid}): one can {', '.join(place_kinds(context)[kind])} there."


def change_body(character, need, worse, size):
    """A body changes as a plan says (a twisted ankle is an injury the body then has). Returns the fact."""
    body = getattr(character, 'body', None)
    cfg = config(character.context) or {}
    need = _harm(cfg)['name'] if need in ('injury', 'harm', 'hurt', 'wound') else need
    if body is None or need not in body.levels:
        return ''
    body.adjust(need, (1 if worse else -1) * {'slightly': 0.2, 'clearly': 0.4, 'badly': 0.6}.get(size, 0.3))
    trace.event('body_changed', character=character.name, need=need, level=round(body.levels[need], 2))
    return f"{character.name} is now {body.summary(public=True)}."


# Things that no action of this world covers (a shoe found in a hut, a tally stick, a page) are
# handed over, held out, taken and put down all the same. What can be done with them is read from
# the world each time (who holds what, what lies here, who is in reach) and offered to the same
# judgement that picks any other action of a deed (choose_action).
# (This was done by matching the deed's verbs: 'give', 'hand', 'take', 'grab', a list of words for
# the ground. 'Samantha takes the shoe from Joe's hands' matched none of the world's actions it
# was offered and was settled as searching the hut; a stick was told as handed over, twice,
# while the world kept it where it was.)
PROP_GLOSS = {'give': "puts it in {other}'s hands", 'offer': 'offers or shows it; {other} may then take it',
              'share': '{other} has it too and you keep it; when either of you uses it up, it is used up for both, and does both good',
              'put down': 'leaves it lying here', 'take': 'it is lying here',
              'take from': '{other} is holding it: it comes away only if {other} has held it out'}


def prop_offers(character):
    """{label: (kind, item, other)}: what the character could do now with the things it holds,
    the things lying here, and the things those in reach are holding."""
    context, body = character.context, getattr(character, 'body', None)
    cfg = config(context) if context else None
    if not cfg or body is None:
        return {}
    private = tuple(cfg.get('private_items', ())) or ('\0',)
    near = [o for o, d in _present(character) if d <= CO_PRESENT_TILES and getattr(o, 'body', None) is not None]
    out = {}
    for item in dict.fromkeys(body.inventory):
        for other in near:
            if not _give_label(cfg, item, other.name):   # the world's own 'give berries' stands where it has one
                out[f'give {item}@{other.name}'] = ('give', item, other)
            out[f'hold out {item}@{other.name}'] = ('offer', item, other)
            if item not in other.body.inventory and not item.startswith(private):
                out[f'share {item}@{other.name}'] = ('share', item, other)
        out[f'put down {item}'] = ('put down', item, None)
    for prop in lying(context, _here(character)):
        out.setdefault(f"take {prop['name']}", ('take', prop['name'], None))
    for other in near:
        for item in dict.fromkeys(other.body.inventory):
            if not item.startswith(private):
                out[f'take {item}@{other.name}'] = ('take from', item, other)
    return out


def _prop_do(context, character, kind, item, other):
    """Do one of prop_offers. A thing in someone's hands is theirs to give: it is taken from
    them only when they have held it out to the taker (context.held_out, kept for the scene);
    otherwise the reach is an attempt, and the holder is told what is wanted."""
    body = character.body
    held_out = context.__dict__.setdefault('held_out', [])
    forget = lambda holder: held_out.__setitem__(slice(None), [h for h in held_out if not (h[0] == item and h[1] == holder.name)])
    if kind == 'give' and item in body.inventory:
        body.inventory.remove(item)
        other.body.inventory.append(item)
        forget(character)
        fact = f"{character.name} hands {other.name} the {item}."
        _record(character, {'name': 'hand over', 'category': 'social'}, fact, 'model', applied=True, target=other.name)
    elif kind == 'share' and item in body.inventory and item not in other.body.inventory:
        # Shared, a thing is held by both: 'apple', not 'an apple' (an inventory is of mass nouns,
        # and nothing here says what can be divided: Lanya could not share a can). The two
        # holdings are tied (context.shared): used up by one, it is used up for both, and does
        # both the good it does (shared_out). Given away or put down, only that one's goes.
        other.body.inventory.append(item)
        ties = context.__dict__.setdefault('shared', [])
        tie = next((t for t in ties if t[0] == item and character.name in t[1]), None)
        if tie is None:
            ties.append((item, {character.name, other.name}))
        else:
            tie[1].add(other.name)
        fact = f"{character.name} shares the {item} with {other.name}; both now have it."
        _record(character, {'name': 'share', 'category': 'social'}, fact, 'model', applied=True, target=other.name)
    elif kind == 'offer' and item in body.inventory:
        if (item, character.name, other.name) not in held_out:
            held_out.append((item, character.name, other.name))
        fact = f"{character.name} holds out the {item} to {other.name}; {character.name} still has it."
        _record(character, {'name': 'hold out', 'category': 'social'}, fact, 'model', applied=True, target=other.name)
    elif kind == 'put down' and item in body.inventory:
        body.inventory.remove(item)
        forget(character)
        context.props = getattr(context, 'props', []) + [{'name': item, 'at': _here(character)}]
        fact = f"{character.name} puts down the {item}."
        _record(character, {'name': 'put down', 'category': 'survival'}, fact, 'model', applied=True)
    elif kind == 'take':
        prop = next((p for p in lying(context, _here(character)) if p['name'] == item), None)
        if prop is None:
            return None
        context.props.remove(prop)
        body.inventory.append(item)
        fact = f"{character.name} takes the {item}."
        _record(character, {'name': 'pick up', 'category': 'survival'}, fact, 'model', applied=True)
    elif kind == 'take from' and item in other.body.inventory:
        if (item, other.name, character.name) in held_out:
            other.body.inventory.remove(item)
            body.inventory.append(item)
            forget(other)
            fact = f"{character.name} takes the {item} that {other.name} holds out."
            _record(character, {'name': 'take', 'category': 'social'}, fact, 'model', applied=True, target=other.name)
        else:
            label = _give_label(config(context), item, character.name) or f'give {item}@{character.name}'
            other.add_perceptual_input(f"{character.name} wants the {item} you are holding. It is yours to hand over or keep "
                                       f"(to hand it over: {label}).", mode='internal')
            fact = f"{character.name} reaches for the {item} that {other.name} is holding; {other.name} still has it."
            _record(character, {'name': 'ask for', 'category': 'social'}, fact, 'model', applied=False, target=other.name)
    else:
        return None
    trace.event('prop_deed', character=character.name, deed=kind, item=item, other=other.name if other is not None else None)
    return fact


def shared_out(context, character, item, effects):
    """`character` has used up `item`. Those it was shared with (see _prop_do 'share') and who
    still hold it lose it too and get the same good of it. Returns their names."""
    ties = getattr(context, 'shared', None) or []
    tie = next((t for t in ties if t[0] == item and character.name in t[1]), None)
    if tie is None:
        return []
    ties.remove(tie)
    names = []
    for other in context.actors + context.extras + context.npcs:
        if other is character or other.name not in tie[1] or item not in getattr(getattr(other, 'body', None), 'inventory', []):
            continue
        other.body.inventory.remove(item)
        for need, delta in effects.items():
            other.body.adjust(need, delta)
        names.append(other.name)
    trace.event('shared_used', character=character.name, item=item, others=names)
    return names


def _give_label(cfg, item, to_name):
    """The label of this world's action by which a holder hands `item` to someone, or ''."""
    for action in cfg.get('actions', []):
        spec = action.get('transfer')
        if spec and action.get('target') == 'character' and (item.endswith(spec[1:]) if spec.startswith('*') else item == spec):
            return f"{action['name']}@{to_name}"
    return ''


DEED_ACTION = """A character in a play does the deed below. The world it lives in has rules for some actions, listed under Actions. Say which one of them, if any, the deed itself performs.

{{$name}} does:
<deed>
{{$deed}}
</deed>

Actions ({{$name}} could do these now; each is 'name@where', with how far off it is or what it does):
{{$offered}}

A deed performs an action only when doing what it says is doing that action: drinking from the spring is drinking; looking at the spring, talking about water, sitting back, leaning, gesturing or waiting is none of them. A thing is given only when the deed puts it in the other's hands; holding it out, offering it or showing it is 'hold out'. Looking at, turning over or using a thing already in hand is none of them. An action some way off is performed only if the deed goes there to do it ("walks to the spring and drinks"). If the deed does one of them and more besides, name the one it does. When you are not sure, it is none.

Respond ONLY with JSON, no prose, in this shape:
{"action": "<the action exactly as listed, the part before the bracket, e.g. drink@Spring1>"}
or, when the deed performs none of them:
{"action": null}
</end>
"""


def choose_action(context, character, act_text, label=None):
    """The world action a deed performs, as a label for _chosen ('drink@Spring1', a threat
    response), or None. Asked of the model, with the actor's own #affordance label as a hint:
    the deed was matched to actions by their verbs ('sits back against the counter' was resting,
    at a bus stop six tiles off, and walked her out of the scene), and an actor's label was
    checked by the same verbs. None when the reply cannot be read: the deed is then the
    narrator's, and no rule applies (the safe side: nothing is used up, nobody walks)."""
    from utils import json_reply
    from utils.Messages import UserMessage
    # each offer is one line (the joined line was split on '; ', which an entry may contain:
    # 'take a branch@Fallen_Branch4 (9 tiles Southeast; first Move: ...)' reached the model as
    # two broken ones), with what can be done with the things held and lying here (prop_offers)
    offered = [o.strip() for o in offered_actions(character) if o.strip()]
    offered += [f"{name} ({PROP_GLOSS[kind].format(other=other.name if other is not None else '')})"
                for name, (kind, item, other) in prop_offers(character).items()]
    if not offered:
        return None
    # (the actor's own label for the deed was given here as a hint, and drew the answer to it:
    # two deeds with a shoe, labelled 'search', were settled as searching)
    reply = context.llm.ask({'name': character.name, 'deed': act_text,
                             'offered': '\n'.join(f'- {o}' for o in offered)},
                            [UserMessage(content=DEED_ACTION)], tag='survival.choose_action', temp=0.1,
                            stops=['</end>'], max_tokens=60)
    found = json_reply.parse(reply)
    if not isinstance(found, dict):
        trace.event('check_unreadable', of='deed_action', reply=(reply or '')[:300])
        return None
    chosen = found.get('action')
    if not isinstance(chosen, str) or not chosen.strip() or chosen.strip().lower() in ('null', 'none'):
        if label:
            trace.event('affordance_label_rejected', character=character.name, label=label, act_text=(act_text or '')[:200])
        return None
    chosen = chosen.strip().split(' (')[0]
    names = {o.split(' (')[0].split(': ')[0].lower() for o in offered}
    threat = active_threat_for(character)
    responses = set(threat['hazard'].get('responses', {}).keys()) if threat else set()
    if chosen.lower() not in names and chosen.split('@')[0].lower() not in responses:
        trace.event('check_unreadable', of='deed_action', reply=f'not offered: {chosen}')
        return None
    return chosen


def resolve_do(context, character, act_text, affordance=None, targets=None):
    """Apply the mechanics of a Do, if it is one of the world's actions. Returns a fact line for the
    narration, or None when it is not (the LLM decides as before). `targets`: the act's characters."""
    cfg = config(context)
    body = getattr(character, 'body', None)
    if not cfg or body is None:
        return None
    text = (act_text or '').lower().replace('’', "'").replace('‘', "'")
    # which of the world's actions the deed performs is the model's to say (choose_action)
    label = choose_action(context, character, act_text, affordance)
    prop = next((v for k, v in prop_offers(character).items() if k.lower() == label.lower()), None) if label else None
    if prop:   # a thing handed over, held out, taken or put down
        return _prop_do(context, character, *prop)
    chosen = _chosen(context, character, label) if label else None
    via = 'label' if affordance and label else 'model'

    threat = active_threat_for(character)
    if threat and not (chosen and chosen[0] == 'action'):
        fact = _respond_to_threat(context, character, threat, text,
                                  chosen=chosen[1] if chosen else None)
        if fact:
            return fact

    if chosen and chosen[0] == 'action':
        _, action, near, target = chosen
        return _apply(context, character, action, near, target, via, act_text or '')

    return None


def _can_approach(context, character, d):
    """A Do aimed at a resource a short walk away (config 'approach_tiles', default 8), with no
    threat active, walks there first instead of being refused. No line-of-sight test: the
    'You could do here' offer lists resources by distance alone, and in forest the map's
    line of sight blocked targets 5 tiles off that the character had just been offered."""
    return d <= (config(context) or {}).get('approach_tiles', 8) and not active_threat_for(character)


def _refuse(character, action, fact, via, target=None):
    _record(character, action, fact, via, applied=False, target=target)
    return fact


def _said(act_text):
    """The words an act carries (a message's content): what follows a colon, else what is in
    quotes (outermost, so an apostrophe inside does not end it)."""
    text = act_text or ''
    if ':' in text:
        said = text.split(':', 1)[1]
    else:
        quoted = re.search(r'["“](.+)["”]', text) or re.search(r"['‘](.+)['’]", text)
        said = quoted.group(1) if quoted else ''
    return said.strip().strip('"“”\'‘’').strip()[:160]


def _apply(context, character, action, near, target, via, act_text=''):
    """Check reach, stock and requirements, then apply the action's effects; returns the fact line."""
    cfg = config(context)
    body = character.body
    types = action.get('resource_types')
    resource_id = res = None
    other = None
    approach = ''
    if _sated(action, body):
        return None   # not a deed of the world's: the narration tells it as it is
    lacking = _missing(action, body)
    if lacking:   # before any walk: Joe walked to the mushrooms to be told he was not hungry
        return _refuse(character, action, f"{character.name} cannot {action['name']}: {lacking}.", via,
                       target[0].name if target else None)
    if action.get('target') == 'character':
        if target is None:
            return _refuse(character, action, f"There is no one here to {action['name']}.", via)
        other, d = target
        if d > CO_PRESENT_TILES and not action.get('remote'):
            if not _can_approach(context, character, d):
                return _refuse(character, action, f"{other.name} is {d} tiles away; {character.name} must go to them first.",
                               via, other.name)
            minutes = travel(character, *_here(other))
            approach = f"{character.name} walks {d} tiles to {other.name} ({round(minutes)} min). "
        relief = action.get('target_effects') or {}
        if relief and not (action.get('transfer') or action.get('transfer_money') or action.get('target_gain') or action.get('consumes')) and all(
                delta < 0 and other.body.levels.get(need, 0.0) < NEED_FELT for need, delta in relief.items()):
            # Joe's shoulder was bandaged twice while his body was 'unhurt'
            state = ' and '.join(_word(cfg, need, other.body.levels.get(need, 0.0)) for need in relief)
            return _refuse(character, action, approach + f"{other.name} is {state}: there is nothing to {action['name'].split()[0]}.",
                           via, other.name)
    elif types:
        if near is None:
            return _refuse(character, action, f"There is no {' or '.join(types).replace('_', ' ').lower()} in sight here.", via)
        resource_id, res, d = near
        if d > 1 and _can_approach(context, character, d):
            # the narration has usually put the character there already: walk over, then act
            minutes = travel(character, *res['location'])
            approach = f"{character.name} walks {d} tiles to {resource_id} ({round(minutes)} min). "
            d = 0
        if d > 1:
            return _refuse(character, action, f"{resource_id} is {d} tiles {_direction(_here(character), res['location'])}; "
                                              f"{character.name} must move there first.", via)
        props = res.setdefault('properties', {})
        if action.get('deplete') and props.setdefault('remaining', action['deplete']) <= 0:
            return _refuse(character, action, approach + f"There is nothing left at {resource_id}.", via)
        if not _prop_ok(res, action.get('requires_property'), character):
            own = _eligible(character, action)
            elsewhere = (f" {character.name} can do that at {own[1]}, {own[0]} tiles "
                         f"{_direction(_here(character), own[2]['location'])}." if own else '')
            return _refuse(character, action, approach + f"{character.name} cannot {action['name']} at {resource_id} "
                                                         f"({', '.join(f'{k}: {_prop_value(v)}' for k, v in props.items() if k != 'remaining') or 'not possible now'})."
                                                         + elsewhere, via)
    missing = _missing(action, body)
    if missing:
        return _refuse(character, action, approach + f"{character.name} cannot {action['name']}: {missing}.", via,
                       other.name if other else None)

    # ---- effects
    for need, delta in (action.get('effects') or {}).items():
        body.adjust(need, delta)
    item = _held(body, action.get('transfer'))
    if item and other is not None:
        body.inventory.remove(item)
        other.body.inventory.append(item)
        conveyed = context.map.resource_registry.get(item.removesuffix(action['conveys'])) if action.get('conveys') else None
        if conveyed is not None:
            conveyed.setdefault('properties', {})['owner'] = other.mapAgent
            context.reference_manager.declare_relationship(item.removesuffix(action['conveys']), 'owned by', other.name, 'owner of')
    if action.get('transfer_money') and other is not None:
        body.money -= action['transfer_money']
        other.body.money += action['transfer_money']
    if other is not None:
        for need, delta in (action.get('target_effects') or {}).items():
            was = other.body.level(need)
            other.body.adjust(need, delta)
            if need == _harm(cfg)['name'] and delta < 0 and was >= NEED_FELT:
                # a wound that is tended is still a wound: one bandaging took a gored leg to
                # 'unhurt' within the scene, and the next deed on that leg was refused
                other.body.levels[need] = max(other.body.level(need), NEED_FELT)
    used = _held(body, action.get('consumes'))
    shared_with = []
    if used:
        body.inventory.remove(used)
        shared_with = shared_out(context, character, used, action.get('effects') or {})
    item = item or used or _held(body, action.get('requires'))
    if action.get('gain'):
        body.inventory.append(_fmt(action['gain'], actor=character.name, item=item or ''))
    if action.get('target_gain') and other is not None:
        received = _fmt(action['target_gain'], actor=character.name, item=item or '', said=_said(act_text)).rstrip(': ')
        other.body.inventory.append(received)
        other.add_perceptual_input(f'You receive a {received}', mode='internal')
    revealed = None
    if action.get('reveals') and other is not None:
        revealed = [i for i in other.body.inventory if i.startswith(action['reveals'])]
    found = None
    concealed = lying(context, _here(character), hidden=True) if action.get('finds') else []
    if concealed:
        # what was put there to be found is found before the luck of the search is tried; it
        # stays where it lies, in plain sight from now on, for anyone to take or point at
        concealed[0].pop('hidden', None)
        found = concealed[0]['name']
        trace.event('thing_found', name=found, character=character.name)
    elif action.get('finds'):
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
    # a deed that may not come off (a weld with a torch, where a kit's seal is sure): the try
    # is made and paid for, and the place is left as it was
    success = action.get('success')
    failed = bool(success) and _rng.random() >= success.get('chance', 1.0)
    for key, value in ({} if failed else action.get('sets_property') or {}).items():
        if res is not None:
            res['properties'][key] = _fmt(value, actor=character.name) if isinstance(value, str) else value
    if res is not None and action.get('deplete'):
        res['properties']['remaining'] -= 1
    context.simulation_time += timedelta(minutes=action.get('minutes', 10))
    body.last_tick = context.simulation_time   # needs during the action are folded into its effect

    kw = dict(resource=resource_id or 'here', target=other.name if other else '', actor=character.name, item=item or found or '')
    where = f" at {resource_id}" if resource_id else ''
    fact = approach + f"{character.name} {_fmt(action.get('text', action['name']), **kw)}{'' if action.get('text') else where}."
    if failed:
        fact = approach + f"{character.name} tries to {action['name']}{where}. " + _fmt(success.get('text', 'It does not take.'), **kw)
    if shared_with:
        fact += f" It was shared with {' and '.join(shared_with)}: it is used up for both, and did them the same good."
    if action.get('finds'):
        fact += f" {character.name} finds {found}." if found else f" {character.name} finds nothing useful."
    if revealed is not None:
        fact += f" {character.name} finds: {', '.join(revealed)}." if revealed else f" {character.name} finds nothing there."
    risk = action.get('risk')
    if risk and _rng.random() < risk.get('chance', 0):
        for need, delta in risk.get('effects', {}).items():
            body.adjust(need, delta)
        fact += ' ' + _fmt(risk.get('text', ''), **kw)
    fact += f" ({character.name} is now {body.summary()}"
    fact += f"; {other.name} is now {other.body.summary(public=True)}.)" if other is not None else ".)"
    _record(character, action, fact, via, applied=True, resource=resource_id, approached=bool(approach),
            target=other.name if other else None, found=found)
    return fact


def _record(character, action, fact, via, applied, resource=None, approached=False, target=None, found=None):
    character.last_affordance = {'applied': applied, 'fact': fact}   # a refusal is not told to the audience
    if applied:   # for the scene's planned business (stage.pending_business)
        character.scene_deeds = getattr(character, 'scene_deeds', []) + [action['name'].lower()]
    trace.event('affordance', character=character.name, action=action['name'],
                category=action.get('category', 'survival'), applied=applied, resource=resource,
                target=target, found=found, fact=fact, via=via, approached=approached)


def offers(context, res, cast=()):
    """What can be done at a resource, for the playwright's list of places. An action open to
    only some of the cast there (a family's own farm) names them; one open to none is left out."""
    cfg = config(context) or {}
    cast = [c for c in cast if getattr(c, 'body', None) is not None]
    out = []
    for action in cfg.get('actions', []):
        if str(res.get('type')) not in (action.get('resource_types') or []):
            continue
        able = [c for c in cast if _prop_ok(res, action.get('requires_property'), c)]
        if not cast or len(able) == len(cast):
            out.append(action['name'])
        elif able:
            out.append(f"{action['name']} ({', '.join(c.name for c in able)} only)")
    return out


def _eligible(character, action):
    """(distance, id, resource) of the nearest resource of the action's types where the
    character meets the action's requires_property, however far; None when there is none."""
    here = _here(character)
    found = [(_distance(here, res['location']), rid, res)
             for rid, res in _resources_of(character.context, action.get('resource_types') or [])
             if _prop_ok(res, action.get('requires_property'), character)]
    return min(found, key=lambda f: f[:2]) if found else None


def walk_minutes(context, tiles):
    """About how long a walk of `tiles` takes, for what characters are told (stage.whereabouts says
    it the same way). Actors were told '3 tiles Southeast' and guessed: the spring was 'three
    minutes' away, then 'nine', then a fifteen-minute walk."""
    return max(1, round(tiles * (config(context) or {}).get('tile_minutes', 2) * 1.5))


def travel(character, x, y, advance_clock=True):
    """Walk a character to (x, y); returns the minutes it takes (distance x terrain cost) and,
    unless told not to (several characters walking together), moves the clock on by them."""
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
    if advance_clock:
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


def _hazard_source(context, character, hazard):
    """The resource this hazard comes from, for this character: (id, resource) of the nearest
    one of the source's types, in the required state, within its reach; None when there is
    none. A hazard without a 'source' is not tied to a resource (callers check that first)."""
    source = hazard['source']
    best = None
    for rid, res in _resources_of(context, source.get('resource_types') or []):
        d = _distance(_here(character), res['location'])
        if d <= source.get('within', 6) and _prop_ok(res, source.get('requires_property'), character) \
                and (best is None or d < best[0]):
            best = (d, rid, res)
    return best[1:] if best else None


def _source_gone(context, threat):
    """True when the threat came from a resource that is no longer in the hazard's state."""
    rid = threat.get('source')
    if not rid:
        return False
    res = context.map.resource_registry.get(rid)
    wanted = threat['hazard']['source'].get('requires_property')
    return res is None or not all(_prop_ok(res, wanted, c) for c in context.actors if c.name in threat['targets'])


def active_threat_for(character):
    threat = getattr(character.context, 'active_threat', None)
    return threat if threat and character.name in threat['targets'] else None


def threat_position(character, threat):
    d = _distance(_here(character), threat['location'])
    # in words a person would use: '3 tiles Northeast' went into set-ups ('the boar stands three
    # tiles off') and from there onto the page
    far = 'close by' if d <= 2 else 'a short way off' if d <= 6 else 'some way off'
    return 'right here' if d == 0 else f"{far} to the {_direction(_here(character), threat['location']).lower()}"


def _where(anchor, threat):
    if threat.get('source'):
        return f"at {threat['source']}"
    return f"{threat_position(anchor, threat)} of {anchor.name}"


BODY_NOTE = r'\s*\([^()]*\bis now\b[^()]*\)'   # '(Joe is now thirsty; ...)': for characters and the narrator


def _tell(context, text):
    """A line for the audience about a hazard. Onsets, strikes and retreats went to the characters
    only: a wolf bit both of them in the last scene and the page never said so."""
    text = re.sub(BODY_NOTE, '', text).strip()
    if text:
        context.message_queue.put({'name': '', 'text': f'    {text}'})


def _start_threat(context, hazard, anchor, exposed):
    """A hazard comes on: it threatens those of `exposed` near `anchor` until answered or its deadline."""
    now = context.simulation_time
    context._hazards_started = getattr(context, '_hazards_started', 0) + 1
    source_id = None
    if hazard.get('source') and _hazard_source(context, anchor, hazard) is not None:
        source_id, source = _hazard_source(context, anchor, hazard)
        loc = tuple(source['location'])
    else:
        loc = (anchor.mapAgent.x + _rng.choice([-3, 3]), anchor.mapAgent.y + _rng.choice([-2, 2]))
    targets = [c.name for c in exposed if _distance(_here(c), _here(anchor)) <= 4]
    context.active_threat = {'hazard': hazard, 'targets': targets, 'location': loc, 'source': source_id,
                             'deadline': now + timedelta(minutes=hazard.get('minutes_to_strike', 45))}
    text = f"{hazard['description'][0].upper() + hazard['description'][1:]} appears {_where(anchor, context.active_threat)}, threatening {', '.join(targets)}."
    context.add_event(text)   # seen by those in the scene (the record)
    context.current_state += '\n' + text
    described = hazard['description'][0].upper() + hazard['description'][1:]
    _tell(context, f"{described} appears at {source_id}." if source_id else
          f"{described} appears to the {_direction(_here(anchor), loc).lower()}.")
    trace.event('hazard', phase='onset', hazard=hazard['name'], targets=targets)


MAX_HAZARDS_PER_PLAY = 1   # from every source together: chance, the plan, the director


def hazard_spent(context):
    """Has this play had its danger? Bruce: dangers are to be rare. A play had a boar, the boar
    again and a wolf by its second act, each from a set-up that merely named one."""
    return getattr(context, '_hazards_started', 0) >= MAX_HAZARDS_PER_PLAY


def _present_forms(verb):
    stem = verb[:-1] if verb.endswith('e') else verb
    doubled = verb + verb[-1] if re.fullmatch(r'.*[^aeiou][aeiou][bdgmnprt]', verb) and len(verb) <= 5 else stem
    return rf"{verb}e?s|{stem}ing|{doubled}ing"


ARRIVING = '|'.join(_present_forms(v) for v in (
    'appear', 'emerge', 'burst', 'charge', 'attack', 'lunge', 'approach', 'circle', 'stalk', 'prowl', 'howl', 'snarl', 'growl',
    'leap', 'rush', 'close', 'bear', 'break', 'strike', 'arrive', 'come', 'step', 'crouch', 'creep', 'roll', 'race', 'sweep',
    'hit', 'begin', 'start', 'pelt', 'ping', 'rain', 'hammer', 'vent', 'bound', 'jump', 'crash', 'descend', 'spread', 'loom')) + r"|is upon|are upon|is here|are here"


def arrives(hazard, text):
    """Does this text have the hazard arriving now: named as the one doing it, in the present?
    ('A wolf emerges from the dark'; not 'safety against the wolves and boars', not 'the bruise
    from the boar's charge', not 'before the boar returns'.) The earlier test staged a hazard
    unless the sentence said it was gone, and failed three times on three phrasings."""
    name = re.escape(hazard['name'].lower())
    return bool(re.search(rf"\b{name}s?\b(?!['’])(?:\s+(?!not\b|never\b)[\w-]+){{0,3}}?\s+(?:{ARRIVING})\b", (text or '').lower()))


def brought_on(context, scene):
    """The hazard this scene's set-up brings on, if it may: a real one is then started
    (stage_hazard). None when the set-up brings none on, a threat is already active, or the
    play has had its danger."""
    cfg = config(context)
    if not cfg or getattr(context, 'active_threat', None) or hazard_spent(context):
        return None
    told = f"{scene.get('scene_title', '')}. {scene.get('pre_narrative', '')}"
    return next((h for h in cfg.get('hazards', []) if arrives(h, told)), None)


def stage_hazard(context, scene, cast):
    """When a scene's set-up brings on one of the world's hazards ('The boar bursts from the
    undergrowth'), it is a real one: it threatens the cast, they must answer it, and it strikes
    or is driven off by the world's rules. (The planned boar was only words: it never acted,
    and the scene was about a branch.)"""
    hazard = brought_on(context, scene)
    exposed = [c for c in cast if getattr(c, 'body', None) is not None and getattr(c, 'mapAgent', None) is not None]
    if hazard is None or not exposed:
        return
    for c in exposed:
        c.body.sheltered = False   # the scene has it upon them
    context._hazard_last = context.simulation_time
    _start_threat(context, hazard, exposed[0], exposed)


# per scene from the second act on: about one play in two has one, none has two
# (ATWAP_DIRECTOR_HAZARD overrides it, for testing)
DIRECTOR_HAZARD_CHANCE = float(os.environ.get('ATWAP_DIRECTOR_HAZARD', 0.08))


def director_hazard(context, cast, act_number):
    """The director brings on one of the world's hazards for the scene about to be played: a
    real one, by the world's rules, at most once in a play and in about half of plays. (Chance
    alone gave a daytime play no danger at all, and a narrator may no longer invent one.)
    Returns the hazard's name, or None."""
    cfg = config(context)
    try:
        act_number = int(act_number)
    except (TypeError, ValueError):
        return None
    if (not cfg or act_number < 2 or hazard_spent(context)
            or getattr(context, 'active_threat', None) or _rng.random() >= DIRECTOR_HAZARD_CHANCE):
        return None
    now = context.simulation_time
    exposed = [c for c in cast if getattr(c, 'body', None) is not None and getattr(c, 'mapAgent', None) is not None]
    fitting = []
    for hazard in cfg.get('hazards', []):
        when = hazard.get('when', 'any')
        if (when == 'night' and not _is_night(now)) or (when == 'day' and _is_night(now)):
            continue
        if hazard.get('source') and not any(_hazard_source(context, c, hazard) for c in exposed):
            continue
        fitting.append(hazard)
    if not exposed or not fitting:
        return None
    hazard = _rng.choice(fitting)
    for c in exposed:
        c.body.sheltered = False   # it comes on them where they are
    context._director_hazard = hazard['name']
    context._hazard_last = now
    trace.event('director_hazard', hazard=hazard['name'], act=act_number)
    _start_threat(context, hazard, exposed[0], exposed)
    return hazard['name']


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
        if _source_gone(context, threat):
            _end_threat(context, threat, threat['hazard'].get('resolved_text') or threat['hazard'].get(
                'retreat_text', f"The {threat['hazard']['name']} is over."))
        elif all(_blocked(context, c, threat['hazard']) for c in targets):
            _end_threat(context, threat, threat['hazard'].get('retreat_text', f"The {threat['hazard']['name']} passes."))
        else:
            clear = threat['hazard'].get('escape_tiles', 8)
            caught = [c for c in targets if _distance(_here(c), threat['location']) < clear]
            if not caught:
                _end_threat(context, threat, f"{' and '.join(c.name for c in targets)} "
                                             f"{'is' if len(targets) == 1 else 'are'} clear of the {threat['hazard']['name']}.")
            elif now >= threat['deadline']:
                # it falls on one of them, the nearest, and the page says who ('its tusks catching
                # a leg' was nobody's leg, and both bodies were hurt)
                harm = threat['hazard'].get('harm', 0.3)
                victim = min(caught, key=lambda c: _distance(_here(c), threat['location']))
                victim.body.adjust(harm_need, harm)
                _end_threat(context, threat, threat['hazard'].get('strike_text', f"The {threat['hazard']['name']} strikes.")
                            + f" It is {victim.name} who is hurt. ({victim.name} is now {victim.body.summary()})", struck=True)
        return

    if hazard_spent(context):
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
            if hazard.get('source') and _hazard_source(context, c, hazard) is None:
                continue
            if not _blocked(context, c, hazard):
                exposed.append(c)
        if not exposed:
            continue
        p = 1 - (1 - hazard.get('chance_per_hour', 0.1)) ** hours
        if _rng.random() >= p:
            continue
        _start_threat(context, hazard, exposed[0], exposed)
        return


def _end_threat(context, threat, text, struck=False, tell=True):
    context.active_threat = None
    context.add_event(text)
    context.current_state += '\n' + text
    if tell:   # not when a character's own act ended it: that act's line says so
        _tell(context, text)
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
            source = context.map.resource_registry.get(threat.get('source')) if threat.get('source') else None
            for key, value in (response.get('sets_property') or {}).items():
                if source is not None:
                    source.setdefault('properties', {})[key] = _fmt(value, actor=character.name) if isinstance(value, str) else value
            fact = f"{character.name} tries to {name}, and it works. " + hazard.get('retreat_text', f"The {hazard['name']} passes.")
            _end_threat(context, threat, fact, tell=False)
        else:
            # A stand that fails costs something the body shows and the page says. (Half a strike
            # left Joe 'unhurt' after the boar ran through him; two scenes later the plan had him
            # clutching his side, and the wound the last act turned on had happened off stage.)
            harm_need = _harm(cfg)['name']
            level = character.body.levels.get(harm_need, 0.0)
            character.body.adjust(harm_need, max(hazard.get('harm', 0.3) / 2, NEED_FELT + 0.02 - level))
            fail = _fmt(hazard.get('fail_text', f"the {hazard['name']} is not deterred"), actor=character.name)
            fact = (f"{character.name} tries to {name}, but {fail}. {character.name} takes the force of it and is "
                    f"{_word(cfg, harm_need, character.body.levels[harm_need])}. ({character.name} is now {character.body.summary()}.)")
            context.add_event(fact)
        trace.event('hazard', phase='response', hazard=hazard['name'], character=character.name, response=name, fact=fact)
        return fact
    return None
