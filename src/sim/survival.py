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
from sim import ablation, keeper, ledger, world

DEFAULT_HARM = {'name': 'injury', 'words': ('unhurt', 'bruised', 'injured', 'badly hurt')}
NEED_FELT = 0.3            # below this a need is not felt ('not hungry'): Body words change here
_rng = random.Random()


def config(context) -> Optional[dict]:
    module = world.scenario(context)
    return getattr(module, 'affordances', None) or getattr(module, 'survival', None)


def stakes_text(context, fallback=''):
    """The scenario's sentence on what is at stake physically and socially in its world, for the
    planning and act prompts; `fallback` (the engine's old wording) when the scenario has none."""
    module = world.scenario(context)
    # a world stated in facts: the planner plans from the same facts the world rules by
    return (getattr(module, 'stakes', None) or fallback).strip() + (f'\n{facts_note(context)}' if facts_note(context) else '') \
        + (f'\n{STYLES[style(context)][1]}' if dangers(context) else '') + (f'\n{PLAN_RULE}' if config(context) else '')


# For every prompt that writes or rewrites the plan. A plan wrote 'the boar swerves, goring Joe's
# leg' as a scene's ending: the actor played the wound from his first line, the world's own boar
# then struck the other character, and the play carried two accounts of who was hurt.
PLAN_RULE = ("What happens to bodies, and how a danger ends, is decided by the world when the scene is played, never by the plan. "
             "A plan may put the characters in a danger's way. It does not say that someone is wounded, bitten, healed, fed or killed, "
             "nor that an animal is driven off or strikes. "
             "States and endings say where the characters stand with each other and with the central question: "
             "what has been chosen, said, refused, found out or lost between them. "
             "Bodies' needs rise with time and are met only by what the characters do on stage; a scene need not be about them "
             "unless a need is desperate or what meets it is out of reach.")


# ---------------------------------------------------------------------------- states the world keeps
# A state is kept only where something in the world file depends on it (a rule that needs someone
# dressed). The world file says what it means and what brings it about ('states': name ->
# {'means', 'by', 'at': kinds of place, 'minutes', 'needs': other states}); the stage shows it;
# the resolver says when a deed makes it so or ends it; what requires it refuses without it.
# --------------------------------------------------------------------------- a world stated in facts
# A world file with `facts` says in sentences how its world works, and in `held` the few things no
# deed or word gets past; it lists no actions, no ways and no states, and what a deed does there is
# the resolver's to say each time (docs/world-facts-design.md). A world file without `facts` is
# still ruled by its tables, until it is converted.
HOW_FAR = {'a little': 0.15, 'well': 0.45, 'fully': 0.9}
LIKELY = {'rare': 0.05, 'unlikely': 0.2, 'as likely as not': 0.5, 'likely': 0.8, 'almost certain': 0.95}


def facts_note(context):
    """How this world works and what is held in it, as its world file states them; '' for a world of tables."""
    module = world.scenario(context)
    facts, held = list(getattr(module, 'facts', None) or []), list(getattr(module, 'held', None) or [])
    if not facts:
        return ''
    threats = dangers(context)
    return ('The facts of this world:\n' + '\n'.join(f'- {f}' for f in facts)
            + ('\nWhat threatens people here:\n' + '\n'.join(f'- {d}' for d in threats) if threats else '')
            + ('\nHeld (no deed or word gets past these):\n' + '\n'.join(f'- {h}' for h in held) if held else ''))


def dangers(context):
    """What threatens people in this world, as its world file states it in sentences (docs/dangers-design.md)."""
    return list(getattr(world.scenario(context), 'dangers', None) or [])


# How a performance is to be played, as its director chose (Context.director_style): how often the
# director brings a danger upon a scene, and what the planner is told. Never more than one danger in an act.
STYLES = {'quiet': (0.0, "The director's style for this performance is quiet: a play of people and what passes between them. "
                         "No danger comes upon them unless the story itself calls for one."),
          'mixed': (0.15, "The director's style for this performance is mixed: mostly between people, with the world pressing on them now and then."),
          'thrilling': (0.6, "The director's style for this performance is thrilling: the world presses hard, and a danger comes upon the cast in most acts.")}


def style(context):
    return getattr(context, 'style', None) if getattr(context, 'style', None) in STYLES else 'mixed'


def come_on(context, what, cast, minutes=30):
    """A danger, said in words, comes upon the cast: brought by a scene's set-up or by the director.
    It is then a state of the stage until a deed ends it, those it is upon get clear of it, or it
    comes to a head (hazard_tick). Returns it, or None when none may come on now."""
    what = ' '.join(str(what or '').split()).strip(' .')
    exposed = [c for c in cast if getattr(c, 'body', None) is not None and world.on(c)]
    if not what or not exposed or not may_come_on(context):
        return None
    context._hazard_last = context.simulation_time
    _start_threat(context, {'name': what, 'description': what, 'minutes_to_strike': minutes, 'stated': True}, exposed[0], exposed)
    return context.active_threat


def danger_over(context, how=''):
    """The danger on the stage is over, by a deed as the resolver found it: told in that deed's own line."""
    threat = getattr(context, 'active_threat', None)
    if not threat:
        return
    text = ' '.join(str(how or '').split()) or f"{_described(threat['hazard'])} is gone."
    _end_threat(context, threat, text, tell=False)
    context.add_event(text)
    if getattr(context, 'current_scene', None) is not None:
        context.scene_since = list(getattr(context, 'scene_since', None) or []) + [text]


def stands(context, fact, about=(), known=None):
    """Something now stands between people: kept in these words. known: who knows it (everyone it
    is about, when not given)."""
    fact = ' '.join(str(fact or '').split())
    book = ledger.of(context)
    if not fact or any(s['fact'].lower() == fact.lower() and not s['ended'] for s in book.standing):
        return None
    about = [getattr(c, 'name', c) for c in about or []]
    entry = {'fact': fact, 'about': about, 'known': [getattr(c, 'name', c) for c in (known if known is not None else about)],
             'since': context.simulation_time, 'ended': None, 'event': book.open[-1] if book.open else None}
    book.standing.append(entry)
    trace.event('stands', fact=fact, about=about, known=entry['known'])
    return entry


def no_longer_stands(context, words):
    """The standing fact these words give has been ended by what was just done."""
    low = ' '.join(str(words or '').lower().split()).strip(' .')
    for s in ledger.of(context).standing:
        if low and not s['ended'] and s['fact'].lower().strip(' .') == low:
            s['ended'] = context.simulation_time
            ledger.of(context).wrote('standing')
            trace.event('stands_ended', fact=s['fact'])
            return s
    return None


def stands_note(context, reader=None):
    """What stands, oldest first: all of it (reader None: the world's own calls and the planners),
    or what `reader` knows of it. '' when nothing does."""
    name = getattr(reader, 'name', reader)
    rows = [s for s in ledger.of(context).standing if not s['ended'] and (name is None or name in s['known'])][-24:]
    return '\n'.join(f"- {s['fact']}" + ('' if name is not None else f" (known to {', '.join(s['known']) or 'nobody yet'})") for s in rows)


def places_near(character, cut=14):
    """The places this character knows of, nearest first, with how far each is: what its actor is
    told in a world played without a catalogue ('Spring2 (here); Thicket9 (about 9 minutes' walk west)')."""
    context = character.context
    here, unseen = _here(character), set(unseen_places(character))
    near = sorted((world.span(context, here, res.xy), rid, res) for rid, res in world.listed(context) if rid not in unseen)
    return '; '.join(f"{rid} ({'here' if world.at_place(context, here, res) else world.away(context, here, res.xy)})"
                     for d, rid, res in near[:cut])


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


def wear(context, thing, wearer):
    """Put a thing on a person, worn or bound on (a coat round them, a bandage tied on): it is then
    theirs, on the body rather than in a hand, until it is taken off or handed on."""
    found = _thing(context, thing, wearer)
    if found is None:
        return False
    ledger.of(context).put(found, holder=wearer.name)
    found['worn'] = True
    return True


def need_called(body, said):
    """The need of `body` that `said` names: by the need's own name, or by one of the words the
    world tells it by (_word). Bodies are shown as 'hungry; thirsty', so a resolver naming a
    need as it was shown ('hungry') named 'hunger', and was refused it as no need at all."""
    said = ' '.join(str(said or '').lower().split())
    harm = _harm(body.cfg)
    for need in body.levels:
        words = harm['words'] if need == harm['name'] else (_need_defs(body.cfg).get(need) or {}).get('words') or ()
        if said == need.lower() or said in {w.lower() for w in words}:
            return need
    return None


@dataclass
class Body:
    levels: dict = field(default_factory=dict)      # need -> 0..1, including the harm need
    held: list = field(default_factory=list, repr=False)   # what it holds before it is bound to the stage's ledger (bind)
    money: float = 0
    flags: set = field(default_factory=set)         # 'dressed', 'suited', ... set by actions
    sheltered: bool = False
    last_tick: Optional[datetime] = None
    warned: set = field(default_factory=set)
    cfg: dict = field(default_factory=dict, repr=False)
    bearing: list = field(default_factory=list)     # 'sitting', 'holding the arm of Joe': this scene's (shared actions)

    def bind(self, context, holder):
        """From now on what this body holds is kept in the stage's ledger (sim/ledger.py)."""
        held, self.held = list(self.held), []
        self._inventory = ledger.Inventory(ledger.of(context), holder)
        for name in held:
            self._inventory.append(name)

    @property
    def inventory(self):
        mine = getattr(self, '_inventory', None)
        return mine if mine is not None else self.held

    @inventory.setter
    def inventory(self, names):
        mine = getattr(self, '_inventory', None)
        if names is mine or names is self.held:   # 'body.inventory += [...]' hands the same view back
            return
        if mine is None:
            self.held = list(names)
            return
        for name in list(mine):
            mine.remove(name)
        for name in names:
            mine.append(name)

    def level(self, need):
        return self.levels.get(need, 0.0)

    def summary(self, public=False):
        """The body in words. public: as someone else sees it, without the items the world
        keeps private ('private_items' prefixes: the messages on a phone)."""
        parts = [_word(self.cfg, n, v) for n, v in self.levels.items()]
        hidden = tuple(self.cfg.get('private_items', ())) if public else ()
        parts += self._held_words(hidden)
        if self.cfg.get('currency'):
            parts.append(f"{self.money:g} {self.cfg['currency']}")
        if self.sheltered:
            parts.append('in shelter')
        parts += sorted(self.flags) + list(self.bearing)
        return '; '.join(parts)

    def _held_words(self, hidden=()):
        """['carrying a and b', 'wearing c']: what is in hand or carried, and what is on the body."""
        held = [i for i in self.inventory if not (hidden and i.startswith(hidden))]
        worn = [i for i in held if (getattr(i, 'thing', None) or {}).get('worn')]
        carried = [i for i in held if i not in worn]
        # ' and ', not a comma: a body's words are joined by ';' and a thing's name may hold a comma
        return ([('carrying ' + ' and '.join(carried))] if carried else []) + ([('wearing ' + ' and '.join(worn))] if worn else [])

    def condition(self):
        """How the body is, in words, without what it carries: 'very thirsty; tired; in shelter'."""
        return '; '.join([_word(self.cfg, n, v) for n, v in self.levels.items()]
                         + (['in shelter'] if self.sheltered else []) + sorted(self.flags) + list(self.bearing))

    def visible(self):
        """The body as another person sees it: hurt or not, what is carried, sheltered or not.
        Thirst, hunger and tiredness are the body's own to know."""
        harm = _harm(self.cfg)['name']
        hidden = tuple(self.cfg.get('private_items', ()))
        parts = [_word(self.cfg, harm, self.levels.get(harm, 0.0))] if harm in self.levels else []
        parts += self._held_words(hidden)
        if self.sheltered:
            parts.append('in shelter')
        parts += list(self.bearing)
        return '; '.join(parts) or 'nothing remarkable'

    def adjust(self, need, delta):
        old = self.levels.get(need, 0.0)
        self.levels[need] = min(1.0, max(0.0, old + delta))
        book = getattr(getattr(self, '_inventory', None), '_ledger', None)
        if book is not None and self.levels[need] != old:
            book.wrote('needs')   # a body's needs change only as part of an event (sim/ledger.py)
        if delta < 0:   # allow a fresh warning next time the need rises past a threshold
            self.warned = {w for w in self.warned if not (w[0] == need and self.levels[need] < w[1])}


def attach(context):
    """Give every cast member a Body when the scenario opts in (Context.__init__)."""
    cfg = config(context)
    if not cfg:
        return
    # what wears on the cast, in the shared situation every planning prompt reads (what threatens
    # them is among the world's facts: facts_note)
    rising = [n for n, d in _need_defs(cfg).items() if d.get('per_hour')]
    brief = '\n' + (', '.join(rising).capitalize() + ' grow as the hours pass.' if rising else '')
    if brief.strip() and brief not in context.current_state:
        context.current_state += brief
    for character in context.everyone():
        give_body(context, character)


def give_body(context, character):
    """A Body for one character, if the world has affordances and it has none yet. A supporting
    actor gets one when cast (Context.cast_part): a father who can hold a deed."""
    cfg = config(context)
    if cfg and getattr(character, 'body', None) is None and world.on(character):
        levels = {n: d.get('initial', 0.2) for n, d in _need_defs(cfg).items()}
        levels[_harm(cfg)['name']] = 0.0
        character.body = Body(levels=levels, held=list(cfg.get('initial_inventory', [])),
                              money=cfg.get('initial_money', 0), last_tick=context.simulation_time, cfg=cfg)
        character.body.bind(context, character.name)


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
        # ... and as the things near it ease or worsen it, by what the world has come to know of them
        body.adjust(need, (rate + keeper.drift(character.context, character, need)) * hours)
        crossed = [(level, text) for level, text in (spec.get('percepts') or {}).items()
                   if body.level(need) >= level and (need, level) not in body.warned]
        if crossed:   # several thresholds in one tick: one percept, the most severe
            body.warned.update((need, level) for level, _ in crossed)
            character.add_perceptual_input(max(crossed)[1], mode='internal')
            trace.event('body', character=character.name, need=need, level=round(body.level(need), 2))
        if spec.get('neglect_harm') and body.level(need) >= 0.95:   # neglect hurts
            body.adjust(_harm(cfg)['name'], spec['neglect_harm'] * hours)


# --------------------------------------------------------------------------- places
_direction = world.bearing


def _here(character):
    return world.at(character)


def _present(character, name=None):
    """Other cast members within CO_PRESENT_TILES (with bodies), or the one named."""
    context = character.context
    out = []
    for other in _embodied(context):
        if other is character or not world.on(other):
            continue
        if name and other.name.lower() != name.lower():
            continue
        out.append((other, world.span(context, _here(character), _here(other))))
    return out


def _with(character, other):
    """Is `other` near enough to `character` to be dealt with (handed a thing, tended, paid)?"""
    return world.together(character.context, _here(character), _here(other))


def _fmt(template, **kw):
    try:
        return template.format(**kw)
    except (KeyError, IndexError):
        return template


def _unseen(character, rid):
    """True when the character is new to this country (Context.strangers) and has not yet seen
    this place: it is then the actor's to know of (a stage direction), not the character's."""
    if not getattr(character, 'unknown_land', False):
        return False
    known = getattr(getattr(character, 'resource_models', None), 'known_resources', None)
    return known is not None and rid not in known


def unseen_places(character, cut=8):
    """The places a stranger to this country has not yet seen, nearest first: its actor's to know of
    (stage directions), not the character's."""
    if not getattr(character, 'unknown_land', False) or not world.on(character):
        return []
    context, here = character.context, _here(character)
    near = sorted((world.span(context, here, res.xy), rid) for rid, res in world.listed(context) if _unseen(character, rid))
    return [rid for d, rid in near[:cut]]


def unseen_note(character):
    """Where the places a stranger has not seen lie, for its actor: 'Spring2 (about 9 minutes' walk west)'."""
    context, here = character.context, _here(character) if world.on(character) else None
    return '; '.join(f"{rid} ({world.away(context, here, world.place(context, rid).xy)})" for rid in unseen_places(character))


def holdings(character):
    """Who is carrying what, among the character and those with it, as anyone can see: for
    the narrator, so that things the mechanics track stay with whoever holds them."""
    body = getattr(character, 'body', None)
    if body is None or not config(character.context):
        return ''
    people = [character] + [o for o, d in _present(character) if _with(character, o)]
    here = '; '.join(named(character.context, p) for p in lying(character.context, _here(character)))
    return '\n'.join([f"{p.name}: {'; '.join(named(character.context, i) for i in p.body.inventory if not i.startswith(tuple(p.body.cfg.get('private_items', ())) or ('\0',))) or 'nothing'}"
                      for p in people] + ([f'Lying here, for anyone to take: {here}'] if here else []))


def whereabouts_of(context, name):
    """Where a thing is now, in a few words: 'held by Samantha', 'lying at Spring1', or '' when
    the world does not hold it. Matched as record.admit matches a thing ('watch', 'wristwatch')."""
    name = re.sub(r'^(a|an|the|some)\s+', '', (name or '').strip().lower())[:40]   # as add_thing names it
    same = lambda held: held.lower() == name or held.lower().endswith(' ' + name) or name.endswith(' ' + held.lower())
    for person in list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', [])):
        body = getattr(person, 'body', None)
        if body is not None and any(same(i) for i in body.inventory):
            return f'held by {person.name}'
    for prop in getattr(context, 'props', []):
        if same(prop['name']):
            near = _nearest_resource_id(context, prop['at'])
            return (f"lying at {near}" + (f", {prop['spot']}" if prop.get('spot') else '')
                    + (', not yet found' if prop.get('hidden') else '')) if near else 'lying out of sight'
    return ''


# --------------------------------------------------------------------------- written things
# A letter, a deed, a tally: what it says is a fact of the world, settled once, not made up by
# whoever reads it out. (The tally in laTerre was 'the signed field tally' and nothing more: the
# father said Fouan wanted his north field, the daughter read that her hand and Fouan's north
# field went to the Marquarts, and the deal she read gave the son what he wanted.) The world's
# narrators and checks are told it; a character knows it once it has read it or been told.
def write(context, name, text):
    """Settle what a written thing says."""
    name = re.sub(r'^(a|an|the|some)\s+', '', (name or '').strip().lower())[:40]
    if name and (text or '').strip():
        context.writings[name] = text.strip()
        trace.event('writing', name=name, says=text.strip()[:300])


def writing_of(context, name):
    """What a written thing says, or ''. Matched as whereabouts_of matches a thing."""
    name = re.sub(r'^(a|an|the|some)\s+', '', (name or '').strip().lower())[:40]
    for key, text in (getattr(context, 'writings', None) or {}).items():
        if key == name or key.endswith(' ' + name) or name.endswith(' ' + key):
            return text
    return ''


def read_by(context, character):
    """What the written things `character` has read say, a line each ('' if none)."""
    return '\n'.join(f"{name}: it says: {writing_of(context, name)}"
                     for name, who in (getattr(context, 'readers', None) or {}).items() if character.name in who)


def writings_note(context, xy=None, people=()):
    """What the written things lying near `xy` (all, if None) or held by `people` say, a line each."""
    # at a place: only what is in sight there (a narrator told what a still-hidden paper says would read it out);
    # with no place (the planners): all of them
    names = [p['name'] for p in getattr(context, 'props', [])
             if xy is None or (at_hand(context, p['at'], xy) and not p.get('hidden'))]
    names += [i for c in people if getattr(c, 'body', None) is not None for i in c.body.inventory]
    seen, lines = set(), []
    for name in names:
        text = writing_of(context, name)
        if text and name not in seen:
            seen.add(name)
            lines.append(f"{name}: it says: {text}")
    return '\n'.join(lines)


def things_note(context, xy=None, people=()):
    """Where the things of the play are now, for any prompt that writes about them: each thing
    lying loose, with its place and where in it (prop spot), and what each of `people` carries.
    xy: only those lying at or near that spot. (A set-up put the scarf on a peg, a search found it
    swept under the floor, and the next set-up laid it on the floor again while the actors took it
    to be in Samantha's hands: nobody writing of it was told where the world had it.)"""
    props = [p for p in getattr(context, 'props', []) if xy is None or at_hand(context, p['at'], xy)]
    lines = [f"{p['name']}: {whereabouts_of(context, p['name'])}" for p in props]
    marks = marks_note(context, xy)
    if marks:   # not things anyone left to be found: what the cast's own deeds did to a place
        lines.append("traces of the cast's own deeds (nobody else made them): " + '; '.join(marks))
    for person in people:
        body = getattr(person, 'body', None)
        private = tuple((body.cfg.get('private_items', ()) if body is not None else ())) or ('\0',)
        held = [i for i in (body.inventory if body is not None else []) if not i.startswith(private)]
        if held:
            lines.append(f"carried by {person.name}: {', '.join(held)}")
    return '\n'.join(lines)


def _nearest_resource_id(context, xy):
    found = world.nearest(context, xy)
    return found[1].id if found else ''


def at_hand(context, at, xy):
    """Whether a thing lying at `at` is at the spot `xy`: beside it, or within reach and at the
    same place. (A clue laid at a bench two tiles from the cafe was listed as lying in the cafe
    and picked up there, an act before the scene written for it.)"""
    return world.lying_here(context, at, xy)


def lying(context, xy, hidden=False):
    """The things lying at or near a spot, in plain sight; hidden=True for those that are found
    only by searching (a clue the director placed, see searchable)."""
    return [p for p in getattr(context, 'props', [])
            if at_hand(context, p['at'], xy) and bool(p.get('hidden')) == hidden]


def searchable(context, res):
    """Whether what is left at a place of this kind is found only by searching it (its kind is
    marked `search` in the world file): a clue put there is then hidden until someone looks."""
    return bool((getattr(world.scenario(context), 'kinds', None) or {}).get(res.kind, {}).get('search'))


# A thing has a state as well as a holder or a place: how it has been left by whoever last
# handled it ('snapped into short lengths', 'edge lifted', 'folded small'). Ordinary handling of
# a thing that is held or lying here is always open to a body (classify_deed 'handle'), and the
# world keeps what it leaves: it is shown wherever the thing is listed. (Anything that changed a
# thing was refused unless the world had an action for it: in one play twenty deeds, among them
# lifting the edge of a canvas, snapping a branch and breaking an apple in two.) A thing may
# also be fixed where it is, a mark or a part of the place: looked at and handled, never taken.
def _thing(context, thing, character=None):
    """The thing meant: itself, the one a held name names (ledger.Name), or by a bare name the one
    this character holds, else one lying where it is, else any of that name."""
    if isinstance(thing, ledger.Thing):
        return thing
    if getattr(thing, 'thing', None) is not None:
        return thing.thing
    name = str(thing).lower()
    things = ledger.of(context).things
    if character is not None:
        here = _here(character)
        found = next((t for t in things if t['name'].lower() == name and t.get('holder') == character.name), None) \
            or next((t for t in things if t['name'].lower() == name and t.get('at') is not None and at_hand(context, t['at'], here)), None)
        if found is not None:
            return found
    return next((t for t in things if t['name'].lower() == name), None)


def state_of(context, thing, character=None):
    found = _thing(context, thing, character)
    return (found or {}).get('state', '')


def set_state(context, thing, state, character=None):
    """How a thing now is, by whatever left it so (handling, or a change its model has: sim/keeper.py)."""
    found = _thing(context, thing, character)
    if found is not None:
        found['state'], found['since'] = state, context.simulation_time
        opened = ledger.of(context).open
        found.state_by = opened[-1] if opened else None
    return found


def whose(event, reader):
    """Whose doing an event of the play was, as `reader` may know it: 'by you at 08:01', 'by
    Samantha at 08:01', 'you did not see by whom'; '' when it was nobody's deed (time, the
    director, how things stood before the play). reader: a character, or None for those who are
    told everything (the stage's own calls, the planners). A fact is told with whose doing it
    is in the same breath: listed bare, a woman's own fingermarks in the mud were read by her
    partner, by the check on his line and by the planner as a stranger's."""
    if not event or not event.get('doer') or event.get('cause') not in ('deed', 'line', 'walk', 'leave'):
        return ''
    at = event['time'].strftime('%H:%M') if hasattr(event.get('time'), 'strftime') else ''
    if reader is None or any(reader.name in e['witnesses'] for e in event['entries']) or reader.name == event['doer']:
        who = 'you' if reader is not None and reader.name == event['doer'] else event['doer']
        return f'by {who}' + (f' at {at}' if at else '')
    return 'you did not see by whom'


def named(context, thing, character=None, reader=False):
    """A thing as it is listed for whoever acts, judges or tells: its name, where in the place it
    is, and how it has been left. thing: a thing, a held name, or a bare name.
    reader: whoever it is listed for (a character; None for those told everything): what a deed
    of the play made, or left in the state it is in, is then listed as that person's doing
    (whose). False: listed without."""
    found = _thing(context, thing, character) or {}
    name = found.get('name') or str(thing)
    state, made = found.get('state'), ''
    if reader is not False and isinstance(found, ledger.Thing):
        by = whose(found.state_by, reader) if state else ''
        state = f'{state}, {by}' if by else state
        # a thing made in the play (a fire, a marker) is told as its maker's; one only picked or
        # taken from what the place gives is nobody's making, and where it is says enough
        got = any(isinstance(e, dict) and e.get('world') for e in (found.made or {}).get('effects') or [])
        # likewise one taken up from what a place plainly has (a suit from the closet, a cup in a
        # kitchen): listed as 'made by Alex', the suit he put on was ruled not to be clothes from the closet
        land = getattr(context, 'land', None)
        got = got or any(isinstance(e, dict) and e.get('new') and land is not None and land.named_in(str(e.get('from') or ''))
                         for e in (found.made or {}).get('effects') or [])
        by = '' if got or (found.made or {}).get('effects') is None else whose(found.made, reader)
        made = f'made {by}' if by.startswith('by') else by
    notes = [n for n in (found.get('spot') if found.get('at') is not None else '', 'worn' if found.get('worn') else '',
                         'part of the place' if found.get('fixed') else '', made, state) if n]
    return name + (f" ({'; '.join(notes)})" if notes else '')


def _said(words):
    """A trace or a state as it is kept, from the words a ruling gives; one quoted back from the stage comes
    with whose doing it is ('door bolted (left by Marisol at 18:18)'), which is not part of it."""
    return ' '.join(re.sub(r'\s*\((?:left |made )?by [^)]*\)\s*$', '', str(words or '')).split()).strip(' .')[:60]


def leave_mark(context, character, words, on=None, ends=None):
    """A deed leaves a trace on the place where its doer stands (or on `on`, a thing that is part
    of it): kept as that deed's doing (ledger.marks), not as a thing lying there. ends: an earlier
    trace here that this one undoes or takes the place of, word for word; it is then no longer there."""
    words = ' '.join(str(words).split()).strip(' .')[:60]
    book, at = ledger.of(context), _here(character)
    if _said(ends):
        old = [m for m in book.marks if m['words'].lower() == _said(ends).lower() and at_hand(context, m['at'], at)]
        for m in old:
            book.marks.remove(m)
        trace.event('mark_ended' if old else 'ends_unmatched', character=character.name, mark=_said(ends), by=words)
    if not words or any(m['words'].lower() == words.lower() and at_hand(context, m['at'], at) for m in book.marks):
        return
    book.marks.append({'words': words, 'at': tuple(at), 'on': on, 'event': book.open[-1] if book.open else None})
    trace.event('mark_left', character=character.name, mark=words, on=on)


def marks_note(context, xy=None, reader=None):
    """The traces deeds have left at or near `xy` (everywhere, if None, each with its place),
    each as whose doing it is for `reader` (whose): ["fingermarks in the mud (left by Samantha at 08:01)"]."""
    out = []
    for m in ledger.of(context).marks:
        if xy is not None and not at_hand(context, m['at'], xy):
            continue
        by = whose(m['event'], reader)
        place = f"at {_nearest_resource_id(context, m['at'])}: " if xy is None else ''
        out.append(f"{place}{m['words']}" + (f" (left {by})" if by.startswith('by') else f' ({by})' if by else ''))
    return out


def _handle(context, character, thing, state):
    """Ordinary handling of a thing held or lying here: the thing is left as `state`."""
    state = ' '.join(str(state).split()).strip(' .')[:60]
    set_state(context, thing, state, character)
    fact = f"The {thing} is now {state}."
    _record(character, {'name': 'handle', 'category': 'survival'}, fact, 'model', applied=True)
    trace.event('thing_handled', character=character.name, thing=thing, state=state)
    return fact


def add_thing(context, name, holder=None, at=None, hidden=False, says=None, fixed=False):
    """A thing comes into the world: in `holder`'s hands, or lying at `at` (hidden: not seen
    until someone searches there; fixed: a mark or a part of the place, not to be carried off).
    says: what it says, if it is a written thing (write). Returns the fact."""
    name = re.sub(r'^(a|an|the|some)\s+', '', name.strip().lower())[:40]
    if not name:
        return ''
    if says:
        write(context, name, says)
    if holder is not None and getattr(holder, 'body', None) is not None:
        if name not in holder.body.inventory:
            holder.body.inventory.append(name)
        fact = f"{holder.name} has {_a(name)}."
    else:
        if not any(p['name'] == name for p in lying(context, at)):
            context.props.append({'name': name, 'at': tuple(at), **({'hidden': True} if hidden else {}), **({'fixed': True} if fixed else {})})
        fact = f"There is {_a(name)} here."
    trace.event('thing_added', name=name, holder=holder.name if holder is not None else None)
    return fact


def _a(name):
    return name if name.endswith('s') else f"{'an' if name[0] in 'aeiou' else 'a'} {name}"


def add_place(context, kind, description, xy):
    """A place of one of the world's own kinds comes into the map at xy. Returns (id, fact), or
    (None, '') for a kind this world does not have."""
    if world.kind_called(context, kind) is None:
        return None, ''
    rid = world.add_place(context, kind, description or f'a {kind.lower()}', xy).id
    trace.event('place_added', id=rid, type=kind, description=description)
    return rid, f"There is {description or _a(kind.lower())} here ({rid})."


# Things that no action of this world covers (a shoe found in a hut, a tally stick, a page) are
# handed over, held out, taken and put down all the same. What can be done with them is read from
# the world each time (who holds what, what lies here, who is in reach) and offered to the same
# judgement that says what a deed is (classify_deed).
# (This was done by matching the deed's verbs: 'give', 'hand', 'take', 'grab', a list of words for
# the ground. 'Samantha takes the shoe from Joe's hands' matched none of the world's actions it
# was offered and was settled as searching the hut; a stick was told as handed over, twice,
# while the world kept it where it was.)
PROP_GLOSS = {'read': 'a written thing, here or in hand: you learn what it says',
              'give': "puts it in {other}'s hands", 'offer': 'offers or shows it; {other} may then take it',
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
    near = [o for o, d in _present(character) if _with(character, o) and getattr(o, 'body', None) is not None]
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
        if not prop.get('fixed'):   # a mark or a part of the place is not carried off
            out.setdefault(f"take {prop['name']}", ('take', prop['name'], None))
    for name in list(dict.fromkeys(body.inventory)) + [p['name'] for p in lying(context, _here(character))]:
        if writing_of(context, name):
            out[f'read {name}'] = ('read', name, None)
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
    held_out = context.held_out
    forget = lambda holder: held_out.__setitem__(slice(None), [h for h in held_out if not (h[0] == item and h[1] == holder.name)])
    if kind == 'give' and item in body.inventory:
        other.body.inventory.append(body.inventory.remove(item))   # the same thing, in other hands
        forget(character)
        fact = f"{character.name} hands {other.name} the {item}."
        _record(character, {'name': 'hand over', 'category': 'social'}, fact, 'model', applied=True, target=other.name)
    elif kind == 'share' and item in body.inventory and item not in other.body.inventory:
        # Shared, a thing is held by both: 'apple', not 'an apple' (an inventory is of mass nouns,
        # and nothing here says what can be divided: Lanya could not share a can). The two
        # holdings are tied (context.shared): used up by one, it is used up for both, and does
        # both the good it does (shared_out). Given away or put down, only that one's goes.
        other.body.inventory.append(item)
        ties = context.shared
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
    elif kind == 'read' and writing_of(context, item):
        # the reader knows it from now on (read_by); the words are the reader's own, not the room's:
        # they go to its memory, and the record says only that it read
        context.readers.setdefault(item, set()).add(character.name)
        character.add_perceptual_input(f"You read the {item}. It says: {writing_of(context, item)}", percept=False, mode='internal')
        fact = f"{character.name} reads the {item}."
        _record(character, {'name': 'read', 'category': 'explore'}, fact, 'model', applied=True)
    elif kind == 'put down' and item in body.inventory:
        ledger.of(context).put(body.inventory.remove(item).thing, at=_here(character))   # the same thing, lying here now
        forget(character)
        fact = f"{character.name} puts down the {item}."
        _record(character, {'name': 'put down', 'category': 'survival'}, fact, 'model', applied=True)
    elif kind == 'take':
        prop = next((p for p in lying(context, _here(character)) if p['name'] == item), None)
        if prop is None:
            return None
        body.inventory.append(prop)   # the same thing, in hand now
        fact = f"{character.name} takes the {item}."
        _record(character, {'name': 'pick up', 'category': 'survival'}, fact, 'model', applied=True)
    elif kind == 'take from' and item in other.body.inventory:
        if (item, other.name, character.name) in held_out:
            body.inventory.append(other.body.inventory.remove(item))
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


# ---------------------------------------------------------------------- shared stage actions
# What a body can do anywhere (plays/scenarios/shared.py): gestures, which change nothing, and
# posture, position and contact, which last until the doer moves or the scene ends. They are
# kept on the context (poses, between) and worded onto each body (Body.bearing), so that every
# prompt that shows a body shows how it is arranged. (A deed that was none of the world's
# actions was the narrator's to tell, and whatever it told became so: a shirt came off, could
# not then be handed over because the world held no shirt, and was told as vanished.)
def shared_actions(context):
    """The shared actions this world takes: all of them unless it says otherwise."""
    cfg = config(context)
    if not cfg or cfg.get('shared') is False:
        return []
    from plays.scenarios import shared
    skip = {a['name'].lower() for a in cfg.get('actions', [])} | {w.lower() for w in cfg.get('without', ())}
    return [a for a in shared.actions if a['name'].lower() not in skip]


def _arrangement(context):
    return context.poses, context.between


def _bear(context):
    """Word each body's arrangement, and how deeds have left it to the eye, from the context's (Body.bearing)."""
    poses, between = _arrangement(context)
    states = ledger.of(context).states
    for c in _embodied(context):
        c.body.bearing = ([poses[c.name]] if c.name in poses else []) + [f'{words} {b}' for a, words, b, _ in between if a == c.name] \
            + [s['words'] for s in states if s['who'] == c.name]


def leave_state(context, who, words, placed=False, ends=None, by=None):
    """A deed leaves a person some way that others can see and that lasts (fingers inked, up on the rock): kept
    in the ruling's words and shown with the body's bearing. placed: it is how or where the person is placed
    here, and is over when they move off (unpose); else it goes with them until a deed ends it. ends: a state
    of theirs that this undoes or takes the place of, word for word."""
    words, states = _said(words), ledger.of(context).states
    if _said(ends):
        old = [s for s in states if s['who'] == who.name and s['words'].lower() == _said(ends).lower()]
        for s in old:
            states.remove(s)
        trace.event('state_ended' if old else 'ends_unmatched', character=who.name, state=_said(ends), by=words)
    if words and not any(s['who'] == who.name and s['words'].lower() == words.lower() for s in states):
        states.append({'who': who.name, 'words': words, 'placed': bool(placed), 'since': context.simulation_time})
        trace.event('state_left', character=who.name, state=words, placed=bool(placed), by=getattr(by, 'name', None))
    _bear(context)


def read_out(context, reader, item, hearers):
    """A writing read aloud is known from then on to each who hears it (context.readers), as to its reader."""
    for other in hearers:
        if other.name not in context.readers.setdefault(item, set()):
            context.readers[item].add(other.name)
            other.add_perceptual_input(f"{reader.name} reads the {item} aloud. It says: {writing_of(context, item)}", percept=False, mode='internal')
    if hearers:
        trace.event('read_out', character=reader.name, thing=item, to=[o.name for o in hearers])


def unpose(context, character=None):
    """Whoever moves off is no longer sitting, nor touching or placed by anyone; with no
    character, nobody is (a new scene)."""
    poses, between = _arrangement(context)
    states = ledger.of(context).states
    if character is None:
        poses.clear()
        between.clear()
        states[:] = [s for s in states if not s['placed']]
    else:
        poses.pop(character.name, None)
        between[:] = [e for e in between if character.name not in (e[0], e[2])]
        states[:] = [s for s in states if not (s['placed'] and s['who'] == character.name)]   # how it was placed there is over
    _bear(context)


def shared_offers(character):
    """{label: (action, other, item)}: the shared actions the character could do now."""
    context, body = character.context, getattr(character, 'body', None)
    acts = shared_actions(context) if context else []
    if not acts or body is None:
        return {}
    poses, between = _arrangement(context)
    me = character.name
    near = [o for o, d in _present(character) if _with(character, o) and getattr(o, 'body', None) is not None]
    out = {}
    for action in acts:
        if action.get('target') == 'character':
            for other in near:
                if action.get('clears') and not any(k == 'contact' and {a, b} == {me, other.name} for a, _, b, k in between):
                    continue   # nothing to let go of
                if action.get('item'):
                    for item in dict.fromkeys(body.inventory):
                        out[f"{action['name']} {item}@{other.name}"] = (action, other, item)
                else:
                    out[f"{action['name']}@{other.name}"] = (action, other, None)
        elif action.get('item'):
            for item in dict.fromkeys(body.inventory):
                out[f"{action['name']} {item}"] = (action, None, item)
        elif action.get('clears'):
            if me in poses:
                out[action['name']] = (action, None, None)
        elif poses.get(me) != action.get('sets') or action['kind'] == 'gesture':
            out[action['name']] = (action, None, None)
    return out


def shared_line(character):
    """The shared actions as one standing line for an act prompt: the names once, not one entry
    for each person (the list of what can be done here is cut at fourteen)."""
    offers = shared_offers(character)
    return ''   # no standing list of what a body can do anywhere: a deed is told in words
    alone = list(dict.fromkeys(a['name'] + (' <a thing you hold>' if a.get('item') else '') for a, other, _ in offers.values() if other is None))
    toward = list(dict.fromkeys(a['name'] + (' <a thing you hold>' if a.get('item') else '') for a, other, _ in offers.values() if other is not None))
    who = list(dict.fromkeys(other.name for _, other, _ in offers.values() if other is not None))
    line = 'Anywhere, you can also: ' + '; '.join(alone)
    if toward:
        line += f". With someone in reach ({', '.join(who)}), as 'name@who': " + '; '.join(toward)
    return line + '.'


def _shared_do(context, character, action, other, item):
    """Do a shared action: nothing for a gesture; the arrangement for the rest. Returns the fact
    line for the record ('' for a gesture: the actor's own words are all there is to it)."""
    poses, between = _arrangement(context)
    me, kind = character.name, action['kind']
    fact = _fmt(action['fact'], actor=me, item=item or '') if action.get('fact') else ''
    if kind == 'posture':
        if action.get('clears'):
            poses.pop(me, None)
            fact = f'{me} stands up.'
        else:
            poses[me] = action['sets']
            fact = f"{me} is {action['sets']}."
    elif kind == 'position':
        poses.pop(me, None)   # it takes being on one's feet
        between[:] = [e for e in between if not (e[0] == me and e[3] == 'position')] + [(me, action['sets'], other.name, 'position')]
        fact = f"{me} is {action['sets']} {other.name}."
    elif kind == 'contact':
        if action.get('clears'):
            between[:] = [e for e in between if not (e[3] == 'contact' and {e[0], e[2]} == {me, other.name})]
            fact = f'{me} and {other.name} are no longer touching.'
        else:
            between[:] = [e for e in between if not (e[0] == me and e[2] == other.name and e[3] == 'contact')] + [(me, action['sets'], other.name, 'contact')]
            fact = f"{me} is {action['sets']} {other.name}."
    _bear(context)
    _record(character, {'name': action['name'], 'category': 'stage'}, fact or f"{me}: {action['name']}.", 'model', applied=True,
            target=other.name if other is not None else None)
    trace.event('shared_deed', character=me, action=action['name'], sort=kind, other=other.name if other is not None else None, item=item)
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


DEED_ACTION = """A character in a play is about to do the deed below. In this world a physical act is made of the actions listed under Actions, or is a small passing act that changes nothing. Say what the deed is.

{{$name}} is about to:
<deed>
{{$deed}}
</deed>
{{$claimed}}
{{$name}} holds: {{$holding}}. Lying here: {{$lying}}.

Actions ({{$name}} could do these now; each is 'name@where' or 'name@who', with how far off it is or what it does):
{{$offered}}

- A deed is an action only when doing what it says is doing that action: drinking from the spring is drinking; looking at the spring or talking about water is not. A thing is given only when the deed puts it in the other's hands; holding it out or offering it is 'hold out'; holding it up to be seen is 'show'. An action some way off is done only if the deed goes there to do it.
- A deed may be two or three of the actions done one after another ("kneels at the water and drinks", "sits down and takes her hand"): name each, in order. How it is done (slowly, without a word, eyes on her face) and what the doer then waits for are manner, not a further act.
- passing: a small act that is none of the listed actions and leaves nothing that anyone would later need to know: it puts nothing away, opens, shuts, breaks, lights or makes nothing, moves no thing from where it is kept, changes no garment, passes no thing between hands, and does nothing to another person by force. Tracing a print with a fingertip, patting the ground beside oneself, setting a foot beside a mark to compare, tapping a thing someone holds, squaring one's shoulders, wiping one's hands on one's clothes, walking a slow circle round someone where they stand: each is passing. (Putting a held thing in a pocket is the action 'stow', where it is listed.) It is seen, and then it is gone. It may stand alone or be one part of a deed.
- The contact actions only touch, except 'lead': taking someone by the hand or arm and starting off with them, or drawing them to their feet or aside, is 'lead' where it is listed, and they come only if they choose. A deed that shoves, pins, drags, carries or holds back another person against their will is not an action and not passing.
- handle: ordinary handling of ONE thing listed as held or lying here, done with the hands and with what is held, that leaves that thing lastingly different, so that someone coming later would find it so, and leaves nothing else different: opens or shuts it, folds or unfolds it, turns back an edge and leaves it so, snaps or breaks it, splits it in two, ties, wraps, wipes or empties it. Turning a thing over, weighing it in the hand, holding it up, or holding it against another thing to compare leaves nothing different: those are passing. Write handling as 'handle <the thing exactly as listed>: <how the thing is left, two to five words>' ("handle branch: snapped into short lengths"). It may stand alone or be one part of a deed.
- It is not handling, and not passing, when the deed makes a thing that was not there, lights, feeds, banks or puts out a fire, opens a way or a hiding place, repairs a thing so that it works again, eats, drinks or treats a body, or carries off a thing marked as part of the place: each of those is one of the listed actions where this world has it, and cannot be done where it has not.
- A deed, or a part of one, that changes a garment or the place itself (takes off a shirt, shuts a tap, wedges a door, digs, moves earth), or that uses or changes a thing not listed as held or lying here, is not an action, not handling and not passing.
- An action marked as one {{$name}} cannot do now ('has no canned food', 'nothing to give') is not done, whatever the deed says.
- If any part of the deed is neither a listed action nor passing, the whole deed cannot be done: the answer is null. When you are not sure, it is null.

Respond ONLY with JSON, no prose, in one of these shapes:
{"actions": ["<a listed action exactly as listed, the part before the bracket, e.g. drink@Spring1; or handle <thing>: <how it is left>; or the word passing>", ...]}   (one to three, in order)
{"actions": null, "why": "<what in the deed is neither, six words at most>"}
</end>
"""


def deed_offers(character):
    """Everything the character could do now as lines for a judge or for the actor who is sent
    back: what can be done with things, and the shared actions."""
    offered = [f"{name} ({PROP_GLOSS[kind].format(other=other.name if other is not None else '')})"
                for name, (kind, item, other) in prop_offers(character).items()]
    offered += [f"{name} ({action['gloss'].format(other=other.name if other is not None else '')})"
                for name, (action, other, item) in shared_offers(character).items()]
    offered += keeper.offer_lines(character)   # what the things here allow, as the world has come to model them
    return offered


PASSING = 'passing'
MANY = 3   # the actions one deed may be made of


def classify_deed(context, character, act_text, label=None):
    """What a deed is, judged before it is played: (parts, why).
      (['kneel', 'drink@Spring1'], '')   the actions it is made of, in order: world actions,
                                         things handed or taken, shared actions, threat responses,
                                         and 'passing' for a small act that changes nothing
      ([], why)                          it cannot be done in this world
    label: the actor's own #affordance, shown to the judge as a claim to check. An unreadable
    reply is ([], ...): what cannot be judged is not done. (The judgement used to be made when
    the deed was played, and a deed that was none of the world's actions was then the
    narrator's to tell as it liked.)"""
    from utils import json_reply
    from utils.Messages import UserMessage
    offered = deed_offers(character)
    body = character.body
    reply = context.llm.ask({'name': character.name, 'deed': act_text,
                             'claimed': f"\nIts actor calls it: {label}\n" if label else '',
                             'holding': '; '.join(named(context, i) for i in body.inventory) or 'nothing',
                             'lying': '; '.join(named(context, p) for p in lying(context, _here(character))) or 'nothing',
                             'offered': '\n'.join(f'- {o}' for o in offered) or '- none here'},
                            [UserMessage(content=DEED_ACTION)], tag='survival.classify_deed', temp=0.1,
                            stops=['</end>'], max_tokens=130)
    found = json_reply.parse(reply)
    if not isinstance(found, dict):
        trace.event('check_unreadable', of='deed_action', reply=(reply or '')[:300])
        return [], 'the deed could not be judged'
    parts = found.get('actions', found.get('action'))
    if isinstance(parts, str):
        parts = [parts]
    parts = [str(x).strip().split(' (')[0] for x in parts or [] if str(x).strip() and str(x).strip().lower() not in ('null', 'none')]
    if not parts:
        return [], str(found.get('why') or 'no action here covers it').strip()
    if len(parts) > MANY:
        return [], 'it is too many acts for one deed'
    names = {o.split(' (')[0].split(': ')[0].lower() for o in offered}
    threat = active_threat_for(character)
    responses = set(threat['hazard'].get('responses', {}).keys()) if threat else set()
    things = {i.lower(): i for i in list(body.inventory) + [p['name'] for p in lying(context, _here(character))]}
    for n, part in enumerate(parts):
        handled = re.match(r'handle\s+(.+?)\s*:\s*(.+)$', part, re.I)
        if handled:   # ordinary handling of a thing that is held or lying here
            thing = things.get(handled.group(1).strip().lower()) or things.get(re.sub(r'^(the|a|an)\s+', '', handled.group(1).strip().lower()))
            if thing is None:
                return [], 'it handles a thing that is not here'
            parts[n] = f'handle {thing}: {handled.group(2).strip()}'
            continue
        if part.lower() not in names and part.split('@')[0].lower() in names:
            parts[n] = part = part.split('@')[0]   # 'examine@Spring1': an act that takes no target, given one
        if part.lower() != PASSING and part.lower() not in names and part.split('@')[0].lower() not in responses:
            trace.event('check_unreadable', of='deed_action', reply=f'not offered: {part}')
            return [], 'no action here covers it'
    return [PASSING if part.lower() == PASSING else part for part in parts], ''


def resolve_do(context, character, act_text, affordance=None, targets=None, resolved=None):
    """Apply the mechanics of a Do: each action it is made of, in order. Returns the fact lines
    for the narration ('' when its parts leave none), or None when the deed is none of the
    actions on offer. `targets`: the act's characters. `resolved`: the parts the deed was judged
    to be before it was chosen (record.judge: 'kneel + drink@Spring1'); a deed that comes here
    unjudged is judged now. Afterwards context.deed_kind is the weightiest thing done ('world',
    'thing', 'shared', 'passing', or None) and context.deed_passing says whether a part of it
    was a passing act, which is not kept beyond the scene."""
    cfg = config(context)
    body = getattr(character, 'body', None)
    context.deed_kind, context.deed_passing = None, False
    if not cfg or body is None:
        return None
    text = (act_text or '').lower().replace('’', "'").replace('‘', "'")
    parts = [p.strip() for p in resolved.split(' + ') if p.strip()] if resolved else classify_deed(context, character, act_text, affordance)[0]
    kinds, facts = set(), []
    for label in parts:
        if label.lower() == PASSING:
            kinds.add(PASSING)
            continue
        kind, fact = _do_one(context, character, label, text, act_text or '', 'label' if affordance else 'model')
        if kind:
            kinds.add(kind)
            if fact:
                facts.append(fact)
    context.deed_passing = PASSING in kinds
    context.deed_kind = next((k for k in ('world', 'thing', 'shared', PASSING) if k in kinds), None)
    return ' '.join(facts) if kinds else None


def _do_one(context, character, label, text, act_text, via):
    """One action of a deed, by its label: (kind, fact line), or (None, None) when the label is
    no longer something the character can do (an earlier part of the deed changed that)."""
    handled = re.match(r'handle\s+(.+?)\s*:\s*(.+)$', label, re.I)
    if handled:
        return 'thing', _handle(context, character, handled.group(1), handled.group(2))
    if label == keeper.KEPT or label.lower() in {k.lower() for k in keeper.offers(character)}:
        # a deed the keeper placed, or a change a modelled thing offers: the models say what it makes and
        # changes. What does not come off (a lack, a try that fails) is the doer's to know, like any refusal.
        fact, done = keeper.perform(context, character, act_text) if label == keeper.KEPT else keeper.do_offer(context, character, label)
        _record(character, {'name': 'kept deed' if label == keeper.KEPT else label, 'category': 'survival'}, fact, via, applied=done)
        return 'world', fact
    prop = next((v for k, v in prop_offers(character).items() if k.lower() == label.lower()), None)
    if prop:   # a thing handed over, held out, taken or put down
        return 'thing', _prop_do(context, character, *prop)
    shared = next((v for k, v in shared_offers(character).items() if k.lower() == label.lower()), None)
    if shared:
        return 'shared', _shared_do(context, character, *shared)
    return None, None


def _record(character, action, fact, via, applied, resource=None, approached=False, target=None, found=None):
    character.last_affordance = {'applied': applied, 'fact': fact}   # a refusal is not told to the audience
    if applied:   # for the scene's planned business (stage.pending_business)
        character.scene_deeds = getattr(character, 'scene_deeds', []) + [action['name'].lower()]
    trace.event('affordance', character=character.name, action=action['name'],
                category=action.get('category', 'survival'), applied=applied, resource=resource,
                target=target, found=found, fact=fact, via=via, approached=approached)


def travel(character, to, advance_clock=True):
    """Walk a character to a position; returns the minutes it takes (distance x what the ground
    there costs) and, unless told not to (several characters walking together), moves the clock
    on by them."""
    context = character.context
    cfg = config(context) or {}
    start = _here(character)
    moved = tuple(start) != tuple(to)
    if moved:
        unpose(context, character)   # whoever walks off is no longer sitting, nor holding anyone's arm
    minutes, effort = world.walk(context, start, to)
    world.put(character, to)
    if moved:
        ledger.of(context).wrote('position')
    if getattr(character, 'body', None) is not None:
        character.body.sheltered = world.sheltered(context, to)   # by where it now is: inside a hut, a cave, a thicket
        if cfg.get('travel_need'):
            character.body.adjust(cfg['travel_need'], 0.01 * effort)
    if advance_clock:
        context.simulation_time += timedelta(minutes=minutes)
    trace.event('travel', character=character.name, tiles=effort, minutes=round(minutes, 1), to=list(to))
    return minutes


def active_threat_for(character):
    threat = getattr(character.context, 'active_threat', None)
    return threat if threat and character.name in threat['targets'] else None


def threat_position(character, threat):
    # in words a person would use: '3 tiles Northeast' went into set-ups ('the boar stands three
    # tiles off') and from there onto the page
    far = world.danger_words(character.context, _here(character), threat['location'])
    return 'right here' if _here(character) == tuple(threat['location']) else f"{far} to the {_direction(_here(character), threat['location']).lower()}"


def _where(anchor, threat):
    if threat.get('source'):
        return f"at {threat['source']}"
    return f"{threat_position(anchor, threat)} of {anchor.name}"


BODY_NOTE = r'\s*\([^()]*\bis now\b[^()]*\)'   # '(Joe is now thirsty; ...)': for characters and the narrator


# A danger, in one place (docs/dangers-design.md). The world file says in sentences what threatens
# people there. One comes on because the play brings it: a scene's set-up has it upon the cast
# (resolver.resolve_setup says what; stage_hazard starts it) or the director brings it on
# (director_danger), never more than one in an act. It is then a state of the stage, in words. A
# deed against it is ruled like any deed, put to chance on the world's odds (sim/resolver.py); it
# ends when a deed ends it (danger_over), when those it is upon are out of its reach, or when its
# time runs out and the world says what it does (resolver.befalls). It is kept in the ledger
# (Context.active_threat), and each of these is a change to the stage inside an event.


def _told(context, text):
    """What a danger does, told: to those present now (the record, Context.add_event) and to the
    audience once the act it belongs to is on the page (the outermost open event: Context.close_event
    puts it after that event's own lines). Outside any event it goes to the page at once."""
    text = ' '.join(str(text or '').split())
    if not text:
        return
    context.add_event(text)
    book = ledger.of(context)
    if book.open:
        book.open[0].setdefault('after', []).append(text)
    else:
        context.to_page('', f'    {text}', kind=None, transcript=False)


def _described(hazard):
    return hazard['description'][0].upper() + hazard['description'][1:]


def _start_threat(context, hazard, anchor, exposed):
    """A hazard comes on: it threatens those of `exposed` near `anchor` until answered or its deadline."""
    now = context.simulation_time
    context._hazards_started = getattr(context, '_hazards_started', 0) + 1
    context._act_dangers = getattr(context, '_act_dangers', 0) + 1
    source_id = None
    loc = world.danger_start(context, _here(anchor), _rng)
    targets = [c.name for c in exposed if world.danger_reaches(context, _here(anchor), _here(c))]
    for c in exposed:
        if c.name in targets:
            c.body.sheltered = False   # it comes on them where they are
    context.active_threat = {'hazard': hazard, 'targets': targets, 'location': loc, 'source': source_id,
                             'deadline': now + timedelta(minutes=hazard.get('minutes_to_strike', 45))}
    _told(context, f"{_described(hazard)} appears {_where(anchor, context.active_threat)}, threatening {', '.join(targets)}.")
    trace.event('hazard', phase='onset', hazard=hazard['name'], targets=targets)


def hazard_spent(context):
    """Has this act had its danger? Never more than one in an act (Context.run_narrative_act
    counts from nothing). A play had a boar, the boar again and a wolf by its second act, each
    from a set-up that merely named one."""
    return getattr(context, '_act_dangers', 0) >= 1


def may_come_on(context):
    """Whether a danger may come on now: none is on the cast, and the play has not had its one."""
    return bool(config(context)) and not getattr(context, 'active_threat', None) and not hazard_spent(context)


def stage_hazard(context, scene, cast):
    """The danger a scene's set-up has upon the cast (scene['danger'], from resolver.resolve_setup),
    started with the scene: it threatens the cast, they must answer it, and it strikes or is driven
    off by the world's rules. (A planned boar was only words: it never acted, and the scene was
    about a branch.)"""
    come_on(context, scene.get('danger'), cast)   # in words, as the set-up check gave it


def director_danger(context, cast, scene=None):
    """The director brings a danger upon the scene about to be played, as often as the style it
    chose for this performance has it and never twice in an act: one of those this world says
    threaten people, as it would come at this hour and place. Returns what it is, or None."""
    chance = STYLES[style(context)][0]
    if not may_come_on(context) or _rng.random() >= chance:
        return None
    exposed = [c for c in cast if getattr(c, 'body', None) is not None and world.on(c)]
    if not exposed:
        return None
    from utils.Messages import UserMessage
    from sim import stage
    where = '; '.join(f"{c.name} {stage.where(context, _here(c))} ({'under shelter' if c.body.sheltered else 'in the open'})" for c in exposed)
    prompt = (f"You are the director of an improvised play. You have chosen to bring a danger upon the cast in the scene about to be played.\n\n"
              f"The scene: '{(scene or {}).get('scene_title', '')}'. {(scene or {}).get('pre_narrative', '')}\n"
              f"It is {stage.clock(context)}. {where}.\n\nWhat threatens people in this world:\n" + '\n'.join(f'- {d}' for d in dangers(context))
              + "\n\nWhich of these could come upon them here and now? Answer with what the cast would see of it, in three to eight words "
                "('a gaunt grey wolf at the clearing's edge'), or with the one word none if none of them comes at this hour or in such a place.\n</end>")
    reply = (context.llm.ask({}, [UserMessage(content=prompt)], tag='director.danger', temp=0.5, stops=['</end>'], max_tokens=60) or '').strip().strip('"\'.')
    if not reply or reply.lower().startswith('none') or len(reply.split()) > 14:
        trace.event('director_danger', what='', reply=reply[:120])
        return None
    context._director_hazard = reply
    trace.event('director_danger', what=reply, style=style(context))
    come_on(context, reply, exposed)
    return reply


def hazard_tick(context):
    """Between one act and the next: has the danger on the cast been got clear of, or has its time run out?"""
    cfg = config(context)
    if not cfg:
        return
    now = context.simulation_time
    last = getattr(context, '_hazard_last', None)
    context._hazard_last = now
    if last is None or now <= last:
        return
    cast = [c for c in context.actors if getattr(c, 'body', None) is not None]
    threat = getattr(context, 'active_threat', None)
    # those it is upon get clear of a danger by going out of its reach; left to itself until its
    # time runs out, it comes to a head, and the world says what it does (resolver.befalls). No
    # dice bring one on: the play does (docs/dangers-design.md)
    if threat:
        targets = [c for c in cast if c.name in threat['targets']]
        caught = [c for c in targets if not world.danger_clear(context, _here(c), threat['location'])]
        if not caught:
            _end_threat(context, threat, f"{' and '.join(c.name for c in targets) or 'They'} {'is' if len(targets) == 1 else 'are'} out of reach of it.")
        elif now >= threat['deadline']:
            from sim import resolver
            victim = min(caught, key=lambda c: world.span(context, _here(c), threat['location']))
            resolver.befalls(context, victim, threat)


def _end_threat(context, threat, text, struck=False, tell=True):
    """The danger is over. tell=False: it ended off stage, and nobody saw it go."""
    context.active_threat = None
    if tell:
        _told(context, text)
        if getattr(context, 'current_scene', None) is not None:
            # a scene's set-up is how it opened: what has ended since is said beside it (Character.director_notes)
            context.scene_since = list(getattr(context, 'scene_since', None) or []) + [text]
    trace.event('hazard', phase='struck' if struck else 'ended', hazard=threat['hazard']['name'], text=text)
    if getattr(context, 'current_scene', None) is not None:
        from sim import scene   # a scene that opened under the danger and goes on is given its owner now
        scene.reopen(context)


def danger_line(character):
    """The danger on `character`, in one line, or ''."""
    threat = active_threat_for(character)
    return f"{threat['hazard']['description']} is {threat_position(character, threat)}" if threat else ''


