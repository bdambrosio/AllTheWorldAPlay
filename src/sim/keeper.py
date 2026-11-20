"""The world-keeper: how a thing works is decided when a deed first needs it, not written ahead.

A world's file says what its places are and what its bodies need. It does not say how a
campfire, a basket or a contract works, and no list of such things would ever be complete: of
140 deeds the stage refused in two weeks of plays, almost none was something a person could not
do (docs/physical-world-design.md). So when a deed cannot be placed among the actions the world
and the stage already have, one call puts it to the keeper with the setting, the stage as it is
and the models made so far. The keeper answers with

  a model   of a kind of thing: its states, what it is made from, what must be at hand, how it
            changes by a deed and in time, how long, at what chance, and what it does to
            anything this world tracks; or an addition to a model that exists
  what the deed does, in those terms
  or that it cannot happen in this world, and why, or what is lacking

The model is kept for the performance (context.models) and no longer: nothing is remembered
from one performance to another. From then on code applies it: the changes a thing
allows in the state it is in are offered beside the world's actions (offers), done without a
model call (perform), and the thing moves on in time by itself (age). A later deed that goes
beyond a model extends it.

What is fixed is a few laws given to the keeper every time (LAWS), and what code holds of them:
what a deed uses up must be on the stage, time is spent, and a model already kept is added to,
never rewritten.

Not yet played: tried on recorded and invented deeds only (logs/review-20261119/physical).
"""
import os, re, json, random
from datetime import timedelta

from sim.trace import trace

_rng = random.Random()
KEEPS_OFF = 'keeps off'
KEPT = 'kept'   # the part a deed is made of when the keeper placed it (record.judge, survival._do_one)

LAWS = """Laws of every world you keep:
- It is the world the setting describes. What an ordinary person could do there with their body and what is at hand can be done; what they could not, cannot, however the deed is worded.
- Where the stage already offers an action that is what the deed does (listed under 'This place offers'), the deed is that action: it needs no model. Say so in "does", with the action exactly as listed.
- Nothing comes from nothing. A thing that is made is made from things that are on the stage, or is something the record below already speaks of as there, or is something a place of that kind plainly has lying about (a stone on a hillside, a chair in a cafe). A thing nobody would expect there, and that nothing has mentioned, is not there.
- Everything takes time, and only what an ordinary person often fails at may fail: give a chance below 1 for that alone.
- What is settled stays settled. You may add a state, a way of changing or a need to a model; you do not take back what a model already says.
- A model says what a thing is like, never what anyone will do with it or how the play goes.
- An effect is said only on something this world tracks, by its name as listed."""

ASK = """You keep the physical world of an improvised play. A character is about to do something the world cannot yet place. Decide, now, how the world works for this deed, and no further than the deed needs.

<world>
{world}
This world tracks: {tracks}.
</world>

<stage>
{stage}
</stage>

<lately>
{lately}
</lately>

<models_so_far>
{models}
</models_so_far>

{name} is about to:
<deed>
{deed}
</deed>

""" + LAWS + """

Answer in JSON only:
{{"can": true or false,
  "why": "<if it cannot: the world's reason, ten words at most>",
  "lacking": ["<a thing the deed needs that is not on the stage; empty if none>"],
  "models": [{{"kind": "<one or two words: a kind of thing, new or one of the models so far>",
              "portable": true or false,
              "states": ["<every state of it, old and new, one or two words each>"], "starts": "<the state a new one begins in>",
              "made_from": ["<things on the stage used up in making one>"], "needs": ["<things that must be at hand to make one, not used up>"],
              "minutes": 0, "chance": 1.0,
              "changes": [{{"by": "<two to four words, a verb first>", "from": "<state>", "to": "<state, or gone>", "uses_up": [], "needs": [], "minutes": 0, "chance": 1.0}}],
              "in_time": [{{"from": "<state>", "to": "<state, or gone>", "after_minutes": 0}}],
              "effects": [{{"while": "<state>", "on": "<a thing this world tracks, as listed>", "does": "keeps off | eases | worsens"}}]}}],
  "does": [{{"thing": "<the thing: its name on the stage, or a short name for one now made or now found to be here>", "kind": "<its model>", "by": "<make, here, or one of that model's changes>", "ends": "<in hand, or here>"}},
           {{"action": "<or, in place of those three: an action the stage already offers, exactly as listed>"}}]}}
Give only the models this deed makes or adds to; in a model that exists give only what is added, and its full list of states. "does" has one entry for each thing the deed makes, finds or changes, in order, three at most. "by" is the one word make when the deed makes the thing; the one word here when the thing was there already and the deed only comes upon it or takes it up; otherwise a change, in that model's own words for it. "ends" is where the thing is when the deed is done: in hand when {name} is left holding or carrying it, here when it is left where {name} stands.
</end>"""


# ------------------------------------------------------------------ the models, kept
def models(context):
    """{kind: model}: what this world has been found to hold in this performance. Nothing is
    carried from one performance to another: each night the world is found out afresh."""
    if getattr(context, 'models', None) is None:
        context.models = {}
    return context.models


def _words(value):
    return [' '.join(str(v).split())[:60] for v in (value if isinstance(value, list) else []) if str(v).strip()]


def _clean(change):
    try:
        chance = min(1.0, max(0.05, float(change.get('chance', 1.0))))
        minutes = min(480, max(0, int(float(change.get('minutes', 0) or 0))))
    except (TypeError, ValueError):
        chance, minutes = 1.0, 0
    return {'by': ' '.join(str(change.get('by') or '').lower().split())[:40], 'from': str(change.get('from') or '').strip().lower(),
            'to': str(change.get('to') or '').strip().lower(), 'uses_up': _words(change.get('uses_up')), 'needs': _words(change.get('needs')),
            'minutes': minutes, 'chance': chance}


def keep(context, found):
    """Take the keeper's models into the world's own: a new kind as given, an existing one added
    to. Nothing a model already says is replaced. Returns the kinds touched."""
    kept, touched = models(context), []
    for m in found if isinstance(found, list) else []:
        kind = ' '.join(str((m or {}).get('kind') or '').lower().split())[:40]
        if not kind:
            continue
        new = kind not in kept
        model = kept.setdefault(kind, {'kind': kind, 'portable': bool(m.get('portable')), 'states': [], 'starts': '', 'made_from': _words(m.get('made_from')),
                                       'needs': _words(m.get('needs')), 'changes': [], 'in_time': [], 'effects': [],
                                       **{k: v for k, v in _clean({**m, 'by': 'make'}).items() if k in ('minutes', 'chance')}})
        for s in _words(m.get('states')):
            if s.lower() not in model['states']:
                model['states'].append(s.lower())
        if new:
            model['starts'] = str(m.get('starts') or (model['states'][0] if model['states'] else 'there')).strip().lower()
            if model['starts'] not in model['states']:
                model['states'].insert(0, model['starts'])
        for c in m.get('changes') or []:
            c = _clean(c) if isinstance(c, dict) else None
            if c and c['by'] and c['to'] and not any(x['by'] == c['by'] and x['from'] == c['from'] for x in model['changes']):
                model['changes'].append(c)
        for t in m.get('in_time') or []:
            try:
                t = {'from': str(t['from']).strip().lower(), 'to': str(t['to']).strip().lower(), 'after_minutes': max(1, int(float(t['after_minutes'])))}
            except (KeyError, TypeError, ValueError):
                continue
            if not any(x['from'] == t['from'] for x in model['in_time']):   # how long a state lasts is settled once
                model['in_time'].append(t)
        for e in m.get('effects') or []:
            if isinstance(e, dict) and e.get('while') and e.get('on') and e not in model['effects']:
                model['effects'].append({'while': str(e['while']).strip().lower(), 'on': str(e['on']).strip().lower(), 'does': str(e.get('does') or '').strip().lower()})
        touched.append(kind)
        trace.event('model_kept', name=kind, new=new, model=model)
    return touched


# ------------------------------------------------------------------ things that have a model
# A thing in the ledger (sim/ledger.py) that the world has a model for carries 'kind' (the
# model's), 'state' and 'since' (when it came to that state).
def _here(character):
    return (character.mapAgent.x, character.mapAgent.y)


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _on_stage(context, character):
    """{name: thing}: what this character holds, and what lies where it stands."""
    from sim import survival, ledger
    out = {t['name'].lower(): t for t in survival.lying(context, _here(character))}
    out.update({t['name'].lower(): t for t in ledger.of(context).held(character.name)})
    return out


def _find(context, character, wanted):
    """The thing on the stage a model's words mean ('branch' for 'a dry branch'), or None."""
    things, low = _on_stage(context, character), wanted.lower()
    return things.get(low) or next((t for n, t in things.items() if low in n or n in low), None)


def age(context):
    """Move each modelled thing on in time: a fire burns down whether or not anyone looks."""
    from sim import ledger
    stage_ledger = ledger.of(context)
    for thing in [t for t in stage_ledger.things if t.get('kind')]:
        model = models(context).get(thing['kind'])
        while model:
            step = next((t for t in model['in_time'] if t['from'] == thing.get('state')), None)
            if not step or context.simulation_time < thing['since'] + timedelta(minutes=step['after_minutes']):
                break
            if step['to'] == 'gone':
                stage_ledger.remove(thing)
                break
            thing['state'], thing['since'] = step['to'], thing['since'] + timedelta(minutes=step['after_minutes'])


def offers(character):
    """{label: (thing, change)}: what the modelled things here allow in the state each is in, for
    the deed judge and the actor ('lay a branch on fire')."""
    context = character.context
    age(context)
    out = {}
    for name, thing in _on_stage(context, character).items():
        model = models(context).get(thing.get('kind'))
        for change in (model or {}).get('changes', []):
            if change['from'] in ('', thing.get('state')):
                out[f"{change['by']} {name}"] = (thing, change)
    return out


def offer_lines(character):
    return [f"{label} (the {thing['name']} is {thing.get('state')}; then {c['to']}"
            + (f"; uses up {', '.join(c['uses_up'])}" if c['uses_up'] else '') + (f"; {c['minutes']} minutes" if c['minutes'] else '') + ')'
            for label, (thing, c) in offers(character).items()]


def keeps_off(context, character, hazard_name):
    """Is there a thing near this character whose model keeps this danger off in its present state?"""
    from sim import ledger
    age(context)
    for thing in ledger.of(context).loose():
        model = models(context).get(thing.get('kind'))
        if model and _distance(thing['at'], _here(character)) <= 2 and any(
                e['while'] == thing.get('state') and KEEPS_OFF in e['does'] and (hazard_name.lower() in e['on'] or e['on'] in hazard_name.lower())
                for e in model['effects']):
            return True
    return False


DRIFT = 0.05   # what a thing near a body adds to or takes from a need in an hour, where its model says it eases or worsens it


def drift(context, character, need):
    from sim import ledger
    total = 0.0
    for thing in ledger.of(context).loose() + ledger.of(context).held(character.name):
        model = models(context).get(thing.get('kind'))
        if model and (thing.get('holder') or _distance(thing['at'], _here(character)) <= 2):
            for e in model['effects']:
                if e['while'] == thing.get('state') and need.lower() in e['on']:
                    total += -DRIFT if 'ease' in e['does'] else DRIFT if 'worsen' in e['does'] else 0.0
    return total


# ------------------------------------------------------------------ asking the keeper
def _tracks(context):
    from sim import survival
    cfg = survival.config(context) or {}
    needs = list(survival._need_defs(cfg)) + [survival._harm(cfg)['name']]
    dangers = [h['name'] + (f" (kept off by: {', '.join(h['blocked_by'])})" if h.get('blocked_by') else '') for h in cfg.get('hazards', [])]
    return '; '.join(filter(None, [', '.join(needs), ('dangers: ' + ', '.join(dangers)) if dangers else '',
                                   cfg.get('currency', ''), 'what each person carries', 'what lies at each place']))


def _stage(context, character):
    from sim import survival, stage
    age(context)
    return '\n'.join(filter(None, [f"{stage.where(context, _here(character)).capitalize()}, {stage.clock(context)}.",
                                   survival.holdings(character), 'This place offers: ' + (survival.available_actions(character, part='known') or 'nothing listed')]))


def _by(value):
    """How a deed comes by a thing, as the keeper gave it: 'make', 'here', or a change's name. (The
    form once read 'here (it was there already)', the keeper copied it whole, and nothing matched.)"""
    by = ' '.join(re.sub(r'\(.*?\)', ' ', str(value or '').lower()).split())
    return 'here' if by.startswith('here') else by


def consider(context, character, deed):
    """Put a deed the stage could not place to the keeper. Returns {'can', 'why', 'does'}: 'does'
    is what code is to do when the deed is played (perform). Models it writes are kept at once:
    how the world works does not depend on whether this deed is the one chosen."""
    from sim import survival
    from utils import json_reply
    from utils.Messages import UserMessage
    asked = context.__dict__.setdefault('_kept_deeds', {})
    key = (character.name, deed)
    if key in asked:
        return asked[key]
    known = models(context)
    reply = context.llm.ask({}, [UserMessage(content=ASK.format(
        world=' '.join((context.initial_state or '').split()), tracks=_tracks(context), stage=_stage(context, character),
        lately='\n'.join(context.witnessed(character, n=8)) or 'Nothing yet.',
        models='\n'.join(json.dumps(m) for m in known.values()) or 'none', name=character.name, deed=deed))],
        tag='keeper.model', temp=0.2, stops=['</end>'], max_tokens=900)
    found = json_reply.parse(reply)
    if not isinstance(found, dict):
        trace.event('check_unreadable', of='keeper', reply=(reply or '')[:300])
        return asked.setdefault(key, {'can': False, 'why': 'no action here covers it', 'does': []})
    lacking = _words(found.get('lacking'))
    if not found.get('can'):
        why = str(found.get('why') or '').strip() or 'it cannot be done here'
        out = {'can': False, 'why': why + (f" (lacking: {', '.join(lacking)})" if lacking and not any(l.lower() in why.lower() for l in lacking) else ''), 'does': []}
    else:
        keep(context, found.get('models'))
        does = [({'action': ' '.join(str(d['action']).split())} if d.get('action') else
                 {'thing': ' '.join(str(d.get('thing') or '').lower().split())[:40], 'kind': ' '.join(str(d.get('kind') or '').lower().split()),
                  'by': _by(d.get('by')), 'ends': 'in hand' if 'hand' in str(d.get('ends') or '').lower() else 'here'})
                for d in found.get('does') or [] if isinstance(d, dict) and (d.get('thing') or d.get('action'))][:3]
        # Taking up a thing that lies here is the stage's own 'take' and needs no model. (The keeper
        # is not shown the offers for things, said 'creased notebook leaf: here, in hand' of a kind
        # it gave no model for, and the deed was refused: 'no action here covers it'.)
        takes = {k.lower(): k for k in survival.prop_offers(character)}
        does = [{'action': takes[f"take {d['thing']}"]} if not d.get('action') and d['kind'] not in models(context) and d['by'] == 'here'
                and d['ends'] == 'in hand' and f"take {d['thing']}" in takes else d for d in does]
        does = [d for d in does if d.get('action') or d['kind'] in models(context)]
        out = {'can': bool(does), 'why': '' if does else 'no action here covers it', 'does': does}
    trace.event('keeper', character=character.name, deed=deed[:200], can=out['can'], why=out['why'], does=out['does'])
    return asked.setdefault(key, out)


# ------------------------------------------------------------------ doing what a model says
def _real(context, wanted):
    """Is this a thing the world keeps count of (so that its absence is a lack), or only a manner or
    a part of the place ('both hands', 'dry earth') that the keeper judged to be there?"""
    from sim import survival, ledger
    known = {k.lower() for k in survival.item_kinds(context)} | set(models(context)) | {t['name'].lower() for t in ledger.of(context).things}
    return any(k == wanted.lower() or k in wanted.lower() for k in known)


def _place_gives(context, character, wanted):
    """Is this something the place the character stands at is, or is a kind of ('fallen branch' at
    Fallen_Branch5)? Then it is at hand, and making from it does not use the place up."""
    low = wanted.lower()
    for res in context.map.resource_registry.values():
        if _distance(res['location'], _here(character)) <= 2:
            for word in (str(res.get('type', '')), str(res.get('name', ''))):
                word = re.sub(r'\d+$', '', word.split('.')[-1]).replace('_', ' ').strip().lower()
                if word and (word in low or low in word):
                    return True
    return False


def _spend(context, character, name, change, verb):
    """What a change or a making costs, as its model has it: what must be at hand, what is used up,
    the time, the chance. Returns '' when it comes off, else the fact of why it did not (time is
    spent on a try that fails; nothing is used up)."""
    from sim import ledger
    for need in change['needs']:
        if _real(context, need) and _find(context, character, need) is None and not _place_gives(context, character, need):
            return f"{character.name} cannot {verb} the {name}: there is no {need} at hand."
    for item in change['uses_up']:
        if _real(context, item) and _find(context, character, item) is None and not _place_gives(context, character, item):
            return f"{character.name} cannot {verb} the {name}: there is no {item} to use."
    context.simulation_time += timedelta(minutes=change['minutes'])
    if _rng.random() >= change['chance']:
        return f"{character.name} tries to {verb} the {name}, and it does not come off."
    for item in change['uses_up']:
        used = _find(context, character, item)
        if used is not None:
            ledger.of(context).remove(used)
    return ''


def _apply(context, character, thing, change):
    """One change to one thing that has a model: (fact, done)."""
    from sim import ledger
    failed = _spend(context, character, thing['name'], change, change['by'])
    if failed:
        return failed, False
    if change['to'] == 'gone':
        ledger.of(context).remove(thing)
        return f"The {thing['name']} is gone.", True
    thing['state'], thing['since'] = change['to'], context.simulation_time
    return f"The {thing['name']} is now {change['to']}.", True


def perform(context, character, deed):
    """Play a deed the keeper placed: make, find or change each thing as the models say, and leave
    it in the doer's hand or where they stand, as the keeper said. Returns (the fact, whether it
    came off): what did not come off is the doer's to know, not an event. A deed that leaves the
    world as it was did not come off: the narrator is not left to say that it did."""
    from sim import survival, ledger
    facts, came_off, did = [], True, False
    for d in (context.__dict__.get('_kept_deeds', {}).get((character.name, deed)) or {}).get('does', []):
        if d.get('action'):   # an action the stage already had: done as that action
            fact = survival._do_one(context, character, d['action'], deed.lower(), deed, 'model')[1]
            if fact:
                facts.append(fact)
                did = True
            continue
        model, name = models(context).get(d['kind']), d['thing']
        if model is None:
            continue
        thing = _on_stage(context, character).get(name)
        change = next((c for c in model['changes'] if c['by'] == d['by']), None)
        if thing is None:
            # not on the stage yet: made now, or found to have been here (a deed that changes a
            # thing nobody has yet taken up, 'picks mushrooms', comes upon it first)
            if d['by'] == 'make':
                failed = _spend(context, character, name, {'needs': model['needs'], 'uses_up': model['made_from'], 'minutes': model.get('minutes', 0),
                                                           'chance': model.get('chance', 1.0)}, 'make')
                if failed:
                    facts, came_off = [failed], False
                    trace.event('thing_modelled', character=character.name, thing=name, model=d['kind'], by='make', done=False, why=failed)
                    break
            elif d['by'] != 'here' and change is None:
                continue
            survival.add_thing(context, name, at=_here(character), fixed=not model['portable'])
            thing = _on_stage(context, character).get(name)
            if thing is None:
                continue
            starts = change['from'] if change and change['from'] in model['states'] else model['starts']
            thing.update(kind=d['kind'], state=starts, since=context.simulation_time)
            did = True
            facts.append(f"{character.name} makes a {name}: it is {starts}." if d['by'] == 'make' else f"There is a {name} here: it is {starts}.")
            trace.event('thing_modelled', character=character.name, thing=name, model=d['kind'], by=d['by'] if d['by'] in ('make', 'here') else 'here', done=True, state=starts)
        elif not thing.get('kind'):   # a thing already on the stage, now known for what it is
            thing.update(kind=d['kind'], state=thing.get('state') if thing.get('state') in model['states'] else model['starts'], since=context.simulation_time)
        if change is not None and d['by'] not in ('make', 'here'):
            change = next((c for c in model['changes'] if c['by'] == d['by'] and c['from'] in ('', thing.get('state'))), None)
            if change is not None:
                fact, done = _apply(context, character, thing, change)
                trace.event('thing_changed', character=character.name, thing=name, by=d['by'], done=done, state=thing.get('state'))
                if not done:
                    facts, came_off = [fact], False
                    break
                facts.append(fact)
                did = True
        # where the deed leaves it
        still = any(t is thing for t in ledger.of(context).things)
        if still and d.get('ends') == 'in hand' and model['portable'] and thing.get('holder') != character.name:
            ledger.of(context).put(thing, holder=character.name)
            facts.append(f"{character.name} has the {name}.")
            did = True
    if came_off and not did:
        return f"{character.name} tries it, and nothing comes of it: things are as they were.", False
    return ' '.join(facts), came_off


def do_offer(context, character, label):
    """A change the judge placed from offers(): done by code, no keeper call."""
    found = next((v for k, v in offers(character).items() if k.lower() == label.lower()), None)
    if not found:
        return None, False
    thing, change = found
    fact, done = _apply(context, character, thing, change)
    trace.event('thing_changed', character=character.name, thing=thing['name'], by=change['by'], done=done, state=thing.get('state'))
    return fact, done
