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

The model is kept (context.models, and from night to night in the world's own file under
~/.local/share/alltheworldaplay/worlds/). From then on code applies it: the changes a thing
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
  "does": [{{"thing": "<the thing: its name on the stage, or a short name for one now made or now found to be here>", "kind": "<its model>", "by": "<make, here (it was there already), or one of that model's changes>"}}]}}
Give only the models this deed makes or adds to; in a model that exists give only what is added, and its full list of states. "does" has one entry for each thing the deed makes, finds or changes, in order, three at most.
</end>"""


# ------------------------------------------------------------------ the models, kept
def _path(context):
    module = getattr(getattr(context, 'map', None), 'scenario_module', None)
    name = (getattr(module, '__name__', '') or 'world').split('.')[-1]
    return os.path.join(os.path.expanduser(os.environ.get('ATWAP_WORLDS_DIR', '~/.local/share/alltheworldaplay/worlds')), f'{name}.models.json')


def models(context):
    """{kind: model}: what this world has been found to hold, tonight and on earlier nights."""
    if getattr(context, 'models', None) is None:
        context.models = {}
        if not os.environ.get('ATWAP_WORLD_FRESH'):
            try:
                context.models = json.load(open(_path(context)))
            except (OSError, ValueError):
                pass
    return context.models


def _save(context):
    try:
        os.makedirs(os.path.dirname(_path(context)), exist_ok=True)
        json.dump(context.models, open(_path(context), 'w'), indent=1)
    except OSError:
        pass


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
    if touched:
        _save(context)
    return touched


# ------------------------------------------------------------------ things that have a model
def _things(context):
    return context.__dict__.setdefault('things', {})   # {name: {'kind', 'state', 'since'}}


def _here(character):
    return (character.mapAgent.x, character.mapAgent.y)


def _on_stage(context, character):
    """{name: 'held' | 'here'}: the things this character holds and those lying where it is."""
    from sim import survival
    out = {p['name'].lower(): 'here' for p in survival.lying(context, _here(character))}
    out.update({i.lower(): 'held' for i in getattr(getattr(character, 'body', None), 'inventory', [])})
    return out


def age(context):
    """Move each modelled thing on in time: a fire burns down whether or not anyone looks."""
    from sim import survival
    for name, thing in list(_things(context).items()):
        model = models(context).get(thing['kind'])
        while model:
            step = next((t for t in model['in_time'] if t['from'] == thing['state']), None)
            if not step or context.simulation_time < thing['since'] + timedelta(minutes=step['after_minutes']):
                break
            thing['since'] += timedelta(minutes=step['after_minutes'])
            if step['to'] == 'gone':
                _remove(context, name)
                break
            thing['state'] = step['to']
            survival.set_state(context, name, step['to'])


def _remove(context, name, holder=None):
    """A thing leaves the world: used up, burned, gone."""
    _things(context).pop(name.lower(), None)
    getattr(context, 'thing_states', {}).pop(name.lower(), None)
    body = getattr(holder, 'body', None)
    held = next((i for i in (body.inventory if body is not None else []) if i.lower() == name.lower()), None)
    if held:
        body.inventory.remove(held)
        return True
    prop = next((p for p in getattr(context, 'props', []) if p['name'].lower() == name.lower()
                 and (holder is None or survival_distance(p['at'], _here(holder)) <= 2)), None)
    if prop:
        context.props.remove(prop)
    return bool(prop)


def survival_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def offers(character):
    """{label: (thing, change)}: what the modelled things here allow in the state each is in, for
    the deed judge and the actor ('lay a branch on fire')."""
    context = character.context
    age(context)
    out = {}
    for name in _on_stage(context, character):
        thing = _things(context).get(name)
        model = models(context).get(thing['kind']) if thing else None
        for change in (model or {}).get('changes', []):
            if change['from'] in ('', thing['state']):
                out[f"{change['by']} {name}"] = (name, change)
    return out


def offer_lines(character):
    return [f"{label} (the {name} is {_things(character.context)[name]['state']}; then {c['to']}"
            + (f"; uses up {', '.join(c['uses_up'])}" if c['uses_up'] else '') + (f"; {c['minutes']} minutes" if c['minutes'] else '') + ')'
            for label, (name, c) in offers(character).items()]


def keeps_off(context, character, hazard_name):
    """Is there a thing near this character whose model keeps this danger off in its present state?"""
    age(context)
    for prop in getattr(context, 'props', []):
        thing = _things(context).get(prop['name'].lower())
        model = models(context).get(thing['kind']) if thing else None
        if model and survival_distance(prop['at'], _here(character)) <= 2 and any(
                e['while'] == thing['state'] and KEEPS_OFF in e['does'] and (hazard_name.lower() in e['on'] or e['on'] in hazard_name.lower())
                for e in model['effects']):
            return True
    return False


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


def consider(context, character, deed):
    """Put a deed the stage could not place to the keeper. Returns {'can', 'why', 'does'}: 'does'
    is what code is to do when the deed is played (perform). Models it writes are kept at once:
    how the world works does not depend on whether this deed is the one chosen."""
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
        does = [{'thing': ' '.join(str(d.get('thing') or '').lower().split())[:40], 'kind': ' '.join(str(d.get('kind') or '').lower().split()), 'by': ' '.join(str(d.get('by') or '').lower().split())}
                for d in found.get('does') or [] if isinstance(d, dict) and d.get('thing')][:3]
        does = [d for d in does if d['kind'] in models(context)]
        out = {'can': bool(does), 'why': '' if does else 'no action here covers it', 'does': does}
    trace.event('keeper', character=character.name, deed=deed[:200], can=out['can'], why=out['why'], does=out['does'])
    return asked.setdefault(key, out)


# ------------------------------------------------------------------ doing what a model says
def _take(context, character, wanted):
    """Use up one thing of this name from the doer's hands or from where it stands; False when there is none."""
    stage_things = _on_stage(context, character)
    name = next((n for n in stage_things if n == wanted.lower()), None) or next((n for n in stage_things if wanted.lower() in n or n in wanted.lower()), None)
    return bool(name) and _remove(context, name, holder=character)


def _has(context, character, wanted):
    stage_things = _on_stage(context, character)
    return any(n == wanted.lower() or wanted.lower() in n or n in wanted.lower() for n in stage_things)


def _real(context, wanted):
    """Is this a thing the world keeps count of (so that its absence is a lack), or only a manner or
    a part of the place ('both hands', 'dry earth') that the keeper judged to be there?"""
    from sim import survival
    known = {k.lower() for k in survival.item_kinds(context)} | set(models(context)) | set(_things(context)) \
        | {p['name'].lower() for p in getattr(context, 'props', [])}
    return any(k == wanted.lower() or k in wanted.lower() for k in known)


def _spend(context, character, thing, change, verb):
    """What a change or a making costs, as its model has it: what must be at hand, what is used up,
    the time, the chance. Returns '' when it comes off, else the fact of why it did not (time is
    spent on a try that fails; nothing is used up)."""
    for need in change['needs']:
        if _real(context, need) and not _has(context, character, need):
            return f"{character.name} cannot {verb} the {thing}: there is no {need} at hand."
    for item in change['uses_up']:
        if _real(context, item) and not _has(context, character, item):
            return f"{character.name} cannot {verb} the {thing}: there is no {item} to use."
    context.simulation_time += timedelta(minutes=change['minutes'])
    if _rng.random() >= change['chance']:
        return f"{character.name} tries to {verb} the {thing}, and it does not come off."
    for item in change['uses_up']:
        _take(context, character, item)
    return ''


def _apply(context, character, thing, change):
    """One change to one thing that has a model: (fact, done)."""
    from sim import survival
    failed = _spend(context, character, thing, change, change['by'])
    if failed:
        return failed, False
    if change['to'] == 'gone':
        _remove(context, thing, holder=character)
        return f"The {thing} is gone.", True
    _things(context)[thing].update(state=change['to'], since=context.simulation_time)
    survival.set_state(context, thing, change['to'])
    return f"The {thing} is now {change['to']}.", True


def perform(context, character, deed):
    """Play a deed the keeper placed: make, find or change each thing as the models say. Returns
    the fact for the narration."""
    from sim import survival
    facts = []
    for d in (context.__dict__.get('_kept_deeds', {}).get((character.name, deed)) or {}).get('does', []):
        model, name = models(context).get(d['kind']), d['thing']
        if model is None:
            continue
        if d['by'] in ('make', 'here') and name not in _on_stage(context, character):
            if d['by'] == 'make':
                failed = _spend(context, character, name, {'needs': model['needs'], 'uses_up': model['made_from'], 'minutes': model.get('minutes', 0),
                                                           'chance': model.get('chance', 1.0)}, 'make')
                if failed:
                    facts.append(failed)
                    trace.event('thing_modelled', character=character.name, thing=name, model=d['kind'], by='make', done=False, why=failed)
                    break
            if model['portable'] and d['by'] == 'make':
                survival.add_thing(context, name, holder=character)
            else:
                survival.add_thing(context, name, at=_here(character), fixed=not model['portable'])
            _things(context)[name] = {'kind': d['kind'], 'state': model['starts'], 'since': context.simulation_time}
            survival.set_state(context, name, model['starts'])
            facts.append(f"{character.name} makes a {name}: it is {model['starts']}." if d['by'] == 'make' else f"There is a {name} here: it is {model['starts']}.")
            trace.event('thing_modelled', character=character.name, thing=name, model=d['kind'], by=d['by'], done=True, state=model['starts'])
            continue
        if name not in _things(context):   # a thing already on the stage, now known for what it is
            if name not in _on_stage(context, character):
                continue
            _things(context)[name] = {'kind': d['kind'], 'state': model['starts'], 'since': context.simulation_time}
            survival.set_state(context, name, model['starts'])
        change = next((c for c in model['changes'] if c['by'] == d['by'] and c['from'] in ('', _things(context)[name]['state'])), None)
        if change is None:
            continue
        fact, done = _apply(context, character, name, change)
        facts.append(fact)
        trace.event('thing_changed', character=character.name, thing=name, by=d['by'], done=done, state=_things(context).get(name, {}).get('state'))
        if not done:
            break
    return ' '.join(facts)


def do_offer(context, character, label):
    """A change the judge placed from offers(): done by code, no keeper call."""
    found = next((v for k, v in offers(character).items() if k.lower() == label.lower()), None)
    if not found:
        return None
    name, change = found
    fact, done = _apply(context, character, name, change)
    trace.event('thing_changed', character=character.name, thing=name, by=change['by'], done=done, state=_things(context).get(name, {}).get('state'))
    return fact
