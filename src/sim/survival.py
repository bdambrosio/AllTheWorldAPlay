"""Stage business: bodies, things, places, travel time and what is settled between people, per world.

Opt-in per scenario: a scenario module (plays/scenarios/*.py) that defines `affordances = {...}` (older name:
`survival`) gets bodies and the world layer; other plays are unchanged. The world keeps who holds what, what lies
where, how a thing has been left, where each person is and how they look; what a deed does to it is the
resolver's to say (sim/resolver.py). It keeps no needs, money or dangers (2026-11-28): what happens to bodies is
played, by deeds.

    affordances = {'tile_minutes': minutes per tile at terrain cost 1, 'initial_inventory': [...]}
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

_rng = random.Random()


def color(context):
    """The world file's color, labelled for a prompt, or '': informative text that helps the calls that write the
    play (the director, the planners, the stage manager, the actors) and is never a rule. It goes where the setting
    goes in those calls, and to no ruling, keeper or check: nothing is refused or put right by it (2026-11-27)."""
    text = ' '.join(str(getattr(world.scenario(context), 'color', None) or '').split())
    return (f"Color of this world (atmosphere to use or leave; the sketch, the stage and the director's facts come first, "
            f"and nothing here is a rule): {text}") if text else ''


def setting(context):
    """The play's setting as the calls that write the play read it: the play file's own, then the world's color."""
    return '\n'.join(filter(None, [(context.initial_state or '').strip(), color(context)]))


def config(context) -> Optional[dict]:
    module = world.scenario(context)
    return getattr(module, 'affordances', None) or getattr(module, 'survival', None)


# For every prompt that writes or rewrites the plan. A plan wrote 'the boar swerves, goring Joe's
# leg' as a scene's ending: the actor played the wound from his first line, the world's own boar
# then struck the other character, and the play carried two accounts of who was hurt.
PLAN_RULE = ("What happens to bodies is played, when the scene is played, never decided by the plan: a plan does not say that "
             "someone is wounded, bitten, healed, fed or killed, nor that an animal is driven off or strikes. "
             "States and endings say where the characters stand with each other and with the central question: "
             "what has been chosen, said, refused, found out or lost between them.")


# How likely a deed is to go as its doer means, as the resolver says it; the engine draws (resolver._draw).
LIKELY = {'rare': 0.05, 'unlikely': 0.2, 'as likely as not': 0.5, 'likely': 0.8, 'almost certain': 0.95}


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


# Who knows a thing that stands was written once, when it came to stand, and nothing added to it:
# a director's fact told on the stage in part one was still 'known to Agnes' alone at the end of
# part two, for the ruling on a line, the check of a set-up, the stage manager and the planner,
# while the thread had closed its question because they had been told. (The stage manager then had
# Agnes keep back in act 4 that she carried them, and had Samantha not know who wrote the names she
# had been told of in act 3.) What is told or shown on the stage is kept with the thing itself, by
# one writer (director.told_in, after each scene), and every reader has it from here.
def told_to(entry, name=None):
    """Who has been told a standing thing on the stage: {name: what of it ('all', or the part in a
    few words)}; or for one name, what that one has been told of it ('' for nothing)."""
    parts = {}
    for t in entry.get('told') or []:
        parts.setdefault(t['who'], []).append(t.get('what', 'all'))
    out = {who: 'all' if 'all' in said else '; '.join(said[-3:]) for who, said in parts.items()}   # the latest parts: they are read in prompts
    return out if name is None else out.get(name, '')


def has_it(entry, name):
    """Does this person have the whole of a standing thing: knew it, or has been told all of it?"""
    return name in (entry.get('known') or []) or told_to(entry, name) == 'all'


def tell(context, entry, who, by='', what='all', words='', scene=None):
    """`who` has been told a standing thing, or seen it, on the stage. Returns the names newly told."""
    what = ' '.join(str(what or 'all').split()).strip(' .') or 'all'
    what = 'all' if what.lower() in ('all', 'all of it', 'the whole', 'whole') else what
    new = []
    for name in who:
        had = told_to(entry, name)
        if name in (entry.get('known') or []) or had == 'all' or (what != 'all' and what.lower() in had.lower()):
            continue
        entry.setdefault('told', []).append({'who': name, 'by': by, 'what': what, 'words': words, 'scene': scene, 'at': context.simulation_time})
        new.append(name)
    if new:
        ledger.of(context).wrote('standing')
        trace.event('told', fact=entry['fact'], who=new, by=by, what=what, words=words[:200])
    return new


def knowing(entry):
    """Who knows a standing thing and who has been told it, in words: 'known to Agnes; told on the
    stage to Samantha and Joe (Samantha: that she was carried)'."""
    told = told_to(entry)
    whole = [n for n, w in told.items() if w == 'all']
    part = [f'{n}: {w}' for n, w in told.items() if w != 'all']
    return (f"known to {', '.join(entry.get('known') or []) or 'nobody yet'}"
            + (f"; told on the stage, all of it, to {', '.join(whole)}" if whole else '')
            + (f"; told on the stage in part to {'; '.join(part)}" if part else ''))


def standing(context, fact):
    """The standing entry for these words, or None."""
    low = ' '.join(str(fact or '').lower().split()).strip(' .')
    return next((s for s in ledger.of(context).standing if not s['ended'] and s['fact'].lower().strip(' .') == low), None)


def stands_note(context, reader=None, without=()):
    """What stands, oldest first: all of it (reader None: the world's own calls and the planners),
    with who knows each and who has been told it on the stage; or what `reader` knows or has been
    told of it. '' when nothing does. without: facts the same prompt already gives, word for word,
    which are not printed again."""
    name = getattr(reader, 'name', reader)
    if name is None:
        return '\n'.join(f"- {s['fact']} ({knowing(s)})" for s in [s for s in ledger.of(context).standing if not s['ended'] and s['fact'] not in without][-24:])
    rows = [s for s in ledger.of(context).standing if not s['ended'] and (name in s['known'] or told_to(s, name))][-24:]
    def mine(s):
        if name in s['known']:
            return f"- {s['fact']}"
        by = ', '.join(dict.fromkeys(t['by'] for t in s.get('told') or [] if t['who'] == name and t.get('by') and t['by'].lower() != 'seen'))
        what = told_to(s, name)
        if what == 'all':
            return f"- {s['fact']} ({by} told you)" if by else f"- {s['fact']} (you have seen for yourself)"
        # told in part, the part and no more: the whole fact was printed with '(of this you have seen for yourself only:
        # his hands had mud on them)', and its actor sang of the lost hours nobody had told her (bellona-20261126-115753)
        return f"- {by} told you, and no more than this: {what}" if by else f"- You have seen for yourself, and no more than this: {what}"
    return '\n'.join(mine(s) for s in rows)


def places_near(character, cut=14):
    """The places this character knows of, nearest first, with how far each is: what its actor is
    told in a world played without a catalogue ('Spring2 (here); Thicket9 (about 9 minutes' walk west)')."""
    context = character.context
    here, unseen = _here(character), set(unseen_places(character))
    near = sorted((world.span(context, here, res.xy), rid, res) for rid, res in world.listed(context) if rid not in unseen)
    return '; '.join(f"{rid} ({'here' if world.at_place(context, here, res) else world.away(context, here, res.xy)})"
                     for d, rid, res in near[:cut])


def wear(context, thing, wearer):
    """Put a thing on a person, worn or bound on (a coat round them, a bandage tied on): it is then
    theirs, on the body rather than in a hand, until it is taken off or handed on."""
    found = _thing(context, thing, wearer)
    if found is None:
        return False
    ledger.of(context).put(found, holder=wearer.name)
    found['worn'] = True
    return True


@dataclass
class Body:
    held: list = field(default_factory=list, repr=False)   # what it holds before it is bound to the stage's ledger (bind)
    flags: set = field(default_factory=set)         # 'dressed', 'suited', ... set by actions
    sheltered: bool = False
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

    def summary(self, public=False, held=True):
        """The body in words. public: as someone else sees it, without the items the world
        keeps private ('private_items' prefixes: the messages on a phone). held: with what it carries
        and wears (False where the same prompt lists that beside it)."""
        hidden = tuple(self.cfg.get('private_items', ())) if public else ()
        parts = self._held_words(hidden) if held else []
        if self.sheltered:
            parts.append('in shelter')
        parts += sorted(self.flags) + list(self.bearing)
        return '; '.join(parts) or ('empty-handed' if held else 'nothing to remark')

    def _held_words(self, hidden=()):
        """['carrying a and b', 'wearing c']: what is in hand or carried, and what is on the body."""
        held = [i for i in self.inventory if not (hidden and i.startswith(hidden))]
        worn = [i for i in held if (getattr(i, 'thing', None) or {}).get('worn')]
        carried = [i for i in held if i not in worn]
        # ' and ', not a comma: a body's words are joined by ';' and a thing's name may hold a comma
        return ([('carrying ' + ' and '.join(carried))] if carried else []) + ([('wearing ' + ' and '.join(worn))] if worn else [])

    def condition(self):
        """How the body is, in words, without what it carries: 'in shelter; sitting'."""
        return '; '.join((['in shelter'] if self.sheltered else []) + sorted(self.flags) + list(self.bearing))

    def visible(self):
        """The body as another person sees it: what is carried, sheltered or not, how it is placed."""
        hidden = tuple(self.cfg.get('private_items', ()))
        parts = self._held_words(hidden)
        if self.sheltered:
            parts.append('in shelter')
        parts += list(self.bearing)
        return '; '.join(parts) or 'nothing remarkable'


def attach(context):
    """Give every cast member a Body when the scenario opts in (Context.__init__)."""
    cfg = config(context)
    if not cfg:
        return
    for character in context.everyone():
        give_body(context, character)


def give_body(context, character):
    """A Body for one character, if the world has affordances and it has none yet. A supporting
    actor gets one when cast (Context.cast_part): a father who can hold a deed."""
    cfg = config(context)
    if cfg and getattr(character, 'body', None) is None and world.on(character):
        character.body = Body(held=list(cfg.get('initial_inventory', [])), cfg=cfg)
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
    """Time passes off stage, to context.simulation_time + gap (the caller moves the clock). The world keeps no needs
    and no dangers (2026-11-28): nothing of a body changes. Returns the needs met: none."""
    return set()


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


def writings_note(context, xy=None, people=(), readers=False):
    """What the written things lying near `xy` (all, if None) or held by `people` say, a line each. readers: with who
    has read each (the stage's own account: a set of notes had a man keep back what a page said that nobody had read)."""
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
            read = sorted((getattr(context, 'readers', None) or {}).get(name) or ())
            lines.append(f"{name}: it says: {text}" + ((f" (read by {', '.join(read)})" if read else ' (read by nobody yet)') if readers else ''))
    return '\n'.join(lines)


def things_note(context, xy=None, people=(), states=False):
    """Where the things of the play are now, for any prompt that writes about them: each thing
    lying loose, with its place and where in it (prop spot), and what each of `people` carries.
    xy: only those lying at or near that spot. (A set-up put the scarf on a peg, a search found it
    swept under the floor, and the next set-up laid it on the floor again while the actors took it
    to be in Samantha's hands: nobody writing of it was told where the world had it.)
    states: each thing with how it has been left, as the stage lists it (named). The stage manager was
    given bare names: 'water skin' for one the stage had as 'unstoppered, still heavy', and 'spring
    water' for a handful long since drunk; it wrote 'the empty water skin' and 'a palmful of spring
    water cupped in her hands', and both were struck."""
    props = [p for p in getattr(context, 'props', []) if xy is None or at_hand(context, p['at'], xy)]
    how = (lambda p: f" ({p['state']})" if isinstance(p, dict) and p.get('state') else '') if states else (lambda p: '')
    lines = [f"{p['name']}: {whereabouts_of(context, p['name'])}{how(p)}" for p in props]
    marks = marks_note(context, xy)
    if marks:   # not things anyone left to be found: what the cast's own deeds did to a place
        lines.append("traces of the cast's own deeds (nobody else made them): " + '; '.join(marks))
    for person in people:
        body = getattr(person, 'body', None)
        private = tuple((body.cfg.get('private_items', ()) if body is not None else ())) or ('\0',)
        held = [i for i in (body.inventory if body is not None else []) if not i.startswith(private)]
        if held:
            lines.append(f"carried by {person.name}: " + ('; '.join(named(context, i, reader=None) for i in held) if states else ', '.join(held)))
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
    if not words:
        return
    # The same trace left again is this deed's doing, and the latest thing done to the place: it used to be
    # dropped as already there. A door bolted, unbolted and bolted again (bellona-20261122-153935) stayed on the
    # stage as 'unbolted, open', the later of the two traces kept, and eight rulings after it held 'bolted' to be
    # not so. The earlier one is taken out and this one stands last, with its own time and doer.
    again = [m for m in book.marks if m['words'].lower() == words.lower() and at_hand(context, m['at'], at)]
    for m in again:
        book.marks.remove(m)
    book.marks.append({'words': words, 'at': tuple(at), 'on': on, 'event': book.open[-1] if book.open else None})
    trace.event('mark_left', character=character.name, mark=words, on=on, again=bool(again))


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


def _give_label(cfg, item, to_name):
    """The label of this world's action by which a holder hands `item` to someone, or ''."""
    for action in cfg.get('actions', []):
        spec = action.get('transfer')
        if spec and action.get('target') == 'character' and (item.endswith(spec[1:]) if spec.startswith('*') else item == spec):
            return f"{action['name']}@{to_name}"
    return ''


PASSING = 'passing'   # a deed that changes nothing in the world (Context.do)


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
    if advance_clock:
        context.simulation_time += timedelta(minutes=minutes)
    trace.event('travel', character=character.name, tiles=effort, minutes=round(minutes, 1), to=list(to))
    return minutes


BODY_NOTE = r'\s*\([^()]*\bis now\b[^()]*\)'   # '(Joe is now thirsty; ...)': for characters and the narrator
