"""A performance that can be continued: the state of the play, kept at the end of each part.

A performance used to end with nothing it could be picked up from: the trace is a log of what
changed, and everything else (the stage's ledger, what is so behind the play and who has come to
it, what each person remembers) lived only in the running engine. At the end of a part the state
of the play is now written to one readable file, a thread:

    ~/.local/share/alltheworldaplay/threads/<thread>.json
    {'thread', 'play', 'parts': [the state at the end of each part, the last being where it stands]}

A thread is one performance played in several sittings: it is kept for that thread alone, never
for a play or a world, so a new night of the same play still starts from the sketch and knows
nothing of it.

The file is a readable account, not a copy of the engine's objects: the engine changes between
sittings, and a part is continued by loading the sketch and applying this to it.

  questions  what the thread has asked, each with the part that raised it, whether what is so
             behind it has been settled, what the leads have come to, and the part that closed it
  so         the director's facts and who knows each, the things that show them and where each is
  story      each act as it came out, the choices it turned on, what the play has planted
  transcript the part as it was played, line by line: what the director of the next part reads
  people     the cast, the director's person among them: where each is, their body's needs, what
             they hold, whose names they do not know, and what they carry forward (digest)
  stage      the ledger: things, traces, how people have been left to the eye, what has been told,
             promised and what stands
  world      the places (those made in play among them), what things have been found to be

Writing a part: state() and, for each person, digest() (the one model call here). Taking it up
again: the sketch is loaded as on any night, apply() puts the play back as the part left it, and
the director opens the next part from what the play has left open (open_part, settle_part; the
prompts are in sim/director.py). Whatever a viewer says to a character is part of the performance:
said after the last act, it is kept too (write(again=True)).
"""
import os, re, json
from datetime import datetime, date

DIR = os.path.expanduser('~/.local/share/alltheworldaplay/threads')

DIGEST = """A play is performed in parts. A part has just ended, and the next may be played on another day, by actors who remember nothing of this one. For one of its people you write what that person carries out of this part into the next: what they would remember of it.

<person>
{who}: {description}
</person>

<what_this_person_witnessed>
{witnessed}
</what_this_person_witnessed>

<their_own_thoughts>
{thoughts}
</their_own_thoughts>

<what_they_carried_into_this_part>
{carried}
</what_they_carried_into_this_part>

Write what {who} carries forward, in three kinds of line.

- lived: this part as {who} lived it, in order, in five to eight plain sentences: what happened, where, and what came of it. What changed something, not every exchange.
- knows: for each other person {who} dealt with, what {who} has learned of them (what they told of their life, what they did, what they refused) and how things now stand between the two.
- holds: each thing still in force for {who} as the part ends: a promise {who} made or was made, a warning or condition {who} gave or was put under (where not to go, what not to ask), a question {who} asked that was not answered, something {who} said they mean to do. No line where there is none.

All of it:
- is only what the record above shows {who} seeing, hearing, doing or thinking, and what {who} carried into this part. Nothing {who} was not there for, and nothing guessed about what another person thinks, feels or hides.
- takes the place of what {who} carried into this part: the parts before are kept in two or three sentences at the head of lived, what {who} knew of a person is kept where this part did not change it, and what was in force is kept unless this part ended it.
- keeps what a person said as what they said, close to their words where the words matter (a place named, a term set, what was done to them): not a stronger or a looser telling of it. What was only offered, meant or promised is told as that, not as done.
- adds nothing to make the record fit together. Where two things in it do not agree, or one is left unexplained, each is kept as it was seen or said, and no thing, event or reason is supplied to join them. There are as many of a thing as the record shows.
- is in the first person, as {who} would tell it, by the names {who} knows people by. Someone whose name {who} has not heard is called as {who} would call them.
- is in plain words, in the past tense for what happened, with no judgement of the play and nothing about what will happen next.

Respond with these lines only:
#lived <five to eight sentences>
#knows <name, or how {who} calls them> | <one or two sentences>   (one line for each person)
#holds <one sentence>   (one line for each thing still in force, or no line)
</end>"""


def _plain(value):
    """A value of the engine as plain data: times as text, sets and tuples as lists, and nothing
    that is one of the engine's own objects (an event a change belongs to, an owner's part)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat(timespec='seconds') if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items() if k != 'event' and _keeps(v)}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value if _keeps(v))
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value if _keeps(v)]
    return None


def _keeps(value):
    return isinstance(value, (str, int, float, bool, datetime, date, dict, set, frozenset, list, tuple)) or value is None


def _where(context, xy):
    from sim import survival
    return survival._nearest_resource_id(context, xy) if xy is not None else ''


# ------------------------------------------------------------------ the state of the play
def questions(context, part, earlier=()):
    """The thread's questions, with this part's own added. A play with a director stands on one;
    a play without one has its central question and what its episode left open."""
    sp = context.screenplay
    out = [dict(q) for q in earlier]
    def add(text, settled, come_to=None):
        text = ' '.join(str(text or '').split())
        if text and not any(q['text'] == text for q in out):
            out.append({'text': text, 'raised': part, 'settled': bool(settled), 'come_to': dict(come_to or {}), 'closed': None,
                        **({'answer': list(sp.answers)} if getattr(sp, 'answers', None) and text == ' '.join(str(getattr(context, 'standing_question', '') or '').split()) else {})})
    takes = [n for n in (getattr(context, 'thread_takes', None) or []) if 0 < n <= len(out)]
    if takes:   # a continued part stands on questions already in the list: what the leads have now come to is theirs
        for n in takes:
            out[n - 1].update(come_to=dict(sp.known), stood_on=part, asked_as=context.standing_question)
    elif getattr(context, 'standing_question', ''):
        add(context.standing_question, any(not f.get('of') for f in sp.facts), sp.known)
    else:
        add(sp.question.get('question') or sp.question_text, False)
        add((context.episode or {}).get('open'), any(f.get('open') for f in sp.facts))
    return out


def so(context):
    """What is so behind the play: the facts and who knows each, and what shows them."""
    from sim import ledger
    sp, book = context.screenplay, ledger.of(context)
    def clue(c):
        thing = next((t for t in book.things if t['name'].lower() == c['thing'].lower()), None)
        now = {} if thing is None else {'holder': thing.get('holder'), 'place': _where(context, thing.get('at')) if thing.get('at') is not None else None,
                                         'hidden': bool(thing.get('hidden'))}
        return {**_plain(c), 'now': now or 'no longer in the world'}
    from sim import survival
    def fact(f):   # with who has been told it on the stage, as the stage keeps it (survival.told_to)
        entry = None if f.get('of') else survival.standing(context, f['text'])
        return {**_plain(f), **({'told': survival.told_to(entry)} if entry is not None and survival.told_to(entry) else {})}
    return {'facts': [fact(f) for f in sp.facts], 'clues': [clue(c) for c in sp.clues], 'marks': _plain(sp.marks)}


def story(context):
    sp = context.screenplay
    return {'acts': [{'number': a.number, 'kind': a.kind, 'title': a.title, 'summary': a.summary, 'choices': _plain(a.choices)} for a in sp.performed()],
            'planted': _plain(sp.threads)}


def person(context, who):
    from sim import world, ledger
    book, body = ledger.of(context), getattr(who, 'body', None)
    xy = world.at(who) if world.on(who) else None
    return {'name': who.name, 'lead': who in context.actors,
            'description': ' '.join((getattr(who, 'original_character', None) or who.character or '').split()),
            'lately': ' '.join(str(getattr(who, 'character', '') or '').split()) if getattr(who, 'character', None) != getattr(who, 'original_character', None) else '',
            'wants': [{'text': d.text, 'activation': round(float(getattr(d, 'activation', 0) or 0), 2)} for d in (who.drives or [])],
            'at': _plain(xy), 'place': _where(context, xy),
            'needs': {n: round(float(v), 2) for n, v in (body.levels if body is not None else {}).items()},
            'holds': [str(n) for n in (body.inventory if body is not None else [])],
            'sheltered': bool(getattr(body, 'sheltered', False)), 'flags': sorted(getattr(body, 'flags', ()) or ()),
            'to_the_eye': [s['words'] for s in book.states if s['who'] == who.name],
            'does_not_know_the_name_of': sorted(getattr(who, 'unknown_names', None) or ()),
            # the places this person has seen: not kept, a person who had sheltered in a cave through part one was told in
            # part three, as a stranger to the country, that she 'has not seen the cave' (lost-20261125-154614)
            'has_seen': sorted(str(r) for r in (getattr(getattr(who, 'resource_models', None), 'known_resources', None) or {})),
            'carries_forward': {}}


def stage_state(context):
    """The stage's ledger, each list as it is kept."""
    from sim import ledger
    book = ledger.of(context)
    def thing(t):
        d = _plain(dict(t))
        if t.get('at') is not None:
            d['place'] = _where(context, t['at'])
        return d
    def mark(m):
        return {**_plain(m), 'place': _where(context, m.get('at'))}
    return {'things': [thing(t) for t in book.things], 'traces': [mark(m) for m in book.marks],
            'states': _plain(list(book.states)), 'told': _plain(list(book.told)), 'promised': _plain(list(book.promised)),
            'standing': _plain(list(book.standing)), 'writings': _plain(dict(book.writings)), 'readers': _plain(dict(book.readers)),
            'shared': _plain(list(book.shared)), 'danger': _plain(book.danger),
            'absent': [{**_plain({k: v for k, v in a.items() if k != 'at'}), 'place': _where(context, a['at']), 'xy': _plain(a['at'])} for a in book.absent]}


def world_state(context):
    from sim import world, keeper
    land = world._land(context)
    def place(p):
        owner = p.props.get('owner')
        return {'id': p.id, 'kind': p.kind, 'description': p.description, 'at': _plain(p.xy),
                'owner': owner if isinstance(owner, str) else None, 'made': _plain(p.made) if _keeps(p.made) else bool(p.made),
                'names': sorted(p.names), 'props': _plain({k: v for k, v in p.props.items() if k != 'owner'})}
    module = getattr(context, 'scenario_module', None)
    return {'scenario': getattr(module, '__name__', None), 'seed': getattr(land, 'seed', None),
            'places': [place(p) for p in land.places()] if land is not None else [],
            'models': _plain(keeper.models(context))}


def state(context, part=1, earlier=()):
    """The state of the play as plain data (no model is called: carries_forward is left empty)."""
    from sim.trace import trace
    return {'part': part, 'run': trace.run_id, 'written': datetime.now().isoformat(timespec='seconds'), 'source': 'engine',
            'clock': context.simulation_time.isoformat(timespec='seconds'),
            'setting': ' '.join(str(context.initial_state or '').split()),
            'questions': questions(context, part, earlier), 'so': so(context), 'story': story(context),
            'transcript': [context.entry_line(e) for e in getattr(context, 'record_entries', []) if e.get('text')],   # the part as played, for the director of the next
            'people': [person(context, c) for c in context.everyone()],
            'stage': stage_state(context), 'world': world_state(context)}


# ------------------------------------------------------------------ what each person carries forward
def read_digest(text):
    """A digest reply as {'lived', 'knows': {name: ...}, 'holds': [...]}."""
    lines = lambda tag: [m.strip() for m in re.findall(rf'^\s*#{tag}\s+(.+?)\s*$', text or '', re.M)]
    knows = {}
    for line in lines('knows'):
        name, _, what = line.partition('|')
        if name.strip() and what.strip():
            knows[name.strip()] = what.strip()
    holds = [h for h in lines('holds') if h.lower().strip(' .') not in ('none', 'nothing')]
    return {'lived': ' '.join(lines('lived')), 'knows': knows, 'holds': holds}


# A digest is read once against the record before it is kept, and written again where it is at
# fault. (The first one kept gave Samantha 'a second button': she had said, at another spot, that
# the one in her hand came off there, and the digest supplied a button to make the two agree.
# She would have remembered it so in every later part.)
DIGEST_CHECK = """A play is performed in parts. A part has just ended, and what one of its people carries into the next has been written from the record of what that person witnessed. That person will remember the part only as this says it, so before it is kept you check it against the record.

<person>
{who}
</person>

<what_this_person_witnessed>
{witnessed}
</what_this_person_witnessed>

<their_own_thoughts>
{thoughts}
</their_own_thoughts>

<what_they_carried_into_this_part>
{carried}
</what_they_carried_into_this_part>

<written>
{written}
</written>

A statement in what is written is at fault when:
- it tells of a thing, a person, an event, a number or a reason that the record and what was carried in do not show, or says the contrary of what they show;
- it tells as done what was only offered, meant, asked or promised;
- it gives as said or done by one person what was said or done by another.
It is not at fault for leaving something out, for telling in fewer words what the record tells in many, or for giving as {who}'s own belief what {who} said or thought.

Respond with one line for each statement at fault, and nothing for the others:
#fault <the words at fault, as written> | <what the record shows>
If none is at fault, respond with the single line:
#none
</end>"""

AGAIN = """

A first writing of this was checked against the record and found at fault in these places:
{faults}
Write it again whole, the same where it was sound, with each of these as the record shows it."""


def digest_text(found):
    """A digest in the line forms it was written in."""
    return '\n'.join([f"#lived {found['lived']}"] + [f"#knows {n} | {w}" for n, w in found['knows'].items()] + [f"#holds {h}" for h in found['holds']])


def checked_digest(ask, who, description, witnessed, thoughts, carried=None):
    """A digest written, read once against the record, and written again if it was at fault.
    ask(prompt, max_tokens) -> reply. Returns (digest, faults [(words, what the record shows)])."""
    prompt = digest_prompt(who, description, witnessed, thoughts, carried)
    found = read_digest(ask(prompt, 900) or '')
    if not found['lived']:
        return found, []
    reply = ask(DIGEST_CHECK.format(who=who, witnessed=witnessed or 'Nothing.', thoughts=thoughts or 'None kept.', written=digest_text(found),
                                    carried=card(carried).partition('\n')[2] or 'Nothing: this was the first part.'), 500) or ''
    faults = [tuple(x.strip() for x in f.partition('|')[::2]) for f in re.findall(r'^\s*#fault\s+(.+?)\s*$', reply, re.M)]
    if faults:
        again = read_digest(ask(prompt + AGAIN.format(faults='\n'.join(f'- "{w}": {shows}' for w, shows in faults)), 900) or '')
        if again['lived']:
            found = again
        else:   # the writing found at fault is the one kept: said so (digest), not passed off as put right
            faults = [(w, shows + ' [not put right: the second writing could not be read]') for w, shows in faults]
    return found, faults


def digest_prompt(who, description, witnessed, thoughts, carried=None):
    return DIGEST.format(who=who, description=description, witnessed=witnessed or 'Nothing.', thoughts=thoughts or 'None kept.',
                         carried=card(carried).partition('\n')[2] or 'Nothing: this was the first part.')


def digest(context, who):
    """What one person carries out of this part: written by the planner from what that person
    witnessed (the play's record) and their own thoughts. {} when it could not be made."""
    from utils.Messages import UserMessage
    witnessed = '\n'.join(context.witnessed(who))
    try:
        thoughts = '\n'.join(m.to_string() for m in who.structured_memory.get_recent(40))
    except Exception:
        thoughts = ''
    carried = getattr(who, 'carried', None) or {}
    if who.name == 'Viewer' or (not witnessed and not thoughts):   # the voice of whoever watches carries nothing: it is not played
        return carried
    description = ' '.join((getattr(who, 'original_character', None) or who.character or '').split())
    ask = lambda prompt, room: context.llm.ask({}, [UserMessage(content=prompt)], tag='thread.digest', temp=0.3, stops=['</end>'], max_tokens=room)
    found, faults = checked_digest(ask, who.name, description, witnessed, thoughts, carried)
    if faults:
        from sim.trace import trace
        kept = any('[not put right' in shows for _, shows in faults)
        trace.event('digest_fault_kept' if kept else 'digest_put_right', character=who.name, faults=[{'words': w, 'shows': shows} for w, shows in faults])
    return found if found['lived'] else carried   # what could not be written again is not lost


# ------------------------------------------------------------------ the file
def path(thread):
    return os.path.join(DIR, f'{thread}.json')


def load(thread):
    try:
        with open(path(thread)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def listing():
    """The threads that can be continued, the latest written first: [{'thread', 'play', 'parts', 'clock', 'written', 'question'}]."""
    out = []
    try:
        names = [n[:-5] for n in os.listdir(DIR) if n.endswith('.json')]
    except OSError:
        return out
    for name in names:
        record = load(name)
        if not record or not record.get('parts'):
            continue
        last = record['parts'][-1]
        still = [q['text'] for q in last.get('questions') or [] if not q.get('closed')]
        out.append({'thread': record['thread'], 'play': record.get('play', ''), 'parts': len(record['parts']), 'clock': last.get('clock'),
                    'written': last.get('written') or '', 'question': (still or [''])[-1]})
    return sorted(out, key=lambda t: t['written'], reverse=True)


def save(record):
    os.makedirs(DIR, exist_ok=True)
    tmp = path(record['thread']) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path(record['thread']))
    return path(record['thread'])


def write(context, again=False, only=None):
    """A part has ended: add its state to the thread it belongs to (a new thread, named for this
    run, when the performance is not a continuation). Returns the file's path, None on failure:
    a thread that could not be written never stops a play."""
    from sim.trace import trace
    try:
        thread = getattr(context, 'thread', None) or trace.run_id
        if not thread:
            return None
        record = load(thread) or {'thread': thread, 'play': os.path.basename(str(getattr(context, 'play_file', '') or '')), 'parts': []}
        last = record['parts'][-1] if record['parts'] else {}
        if again and record['parts'] and record['parts'][-1].get('run') == trace.run_id:   # the same part, kept again with what has been said since
            kept = {p['name']: p.get('carries_forward') or {} for p in record['parts'].pop()['people']}
            last = record['parts'][-1] if record['parts'] else {}
        else:
            kept = {}
        part = state(context, part=len(record['parts']) + 1, earlier=getattr(context, 'thread_questions', None) or last.get('questions') or ())
        for entry, who in zip(part['people'], context.everyone()):
            if only and who.name != only and kept.get(who.name):   # only the one spoken to has more to carry
                entry['carries_forward'] = kept[who.name]
                continue
            try:
                entry['carries_forward'] = digest(context, who)
            except Exception as e:
                trace.event('thread_digest_failed', character=who.name, error=f'{type(e).__name__}: {e}')
        record['parts'].append(part)
        where = save(record)
        context.thread_kept = True
        trace.event('thread_written', thread=thread, part=part['part'], path=where,
                    people=[p['name'] for p in part['people']], digests=sum(1 for p in part['people'] if p['carries_forward']),
                    things=len(part['stage']['things']), places=len(part['world']['places']))
        return where
    except Exception as e:
        trace.event('thread_failed', error=f'{type(e).__name__}: {e}')
        return None


# ------------------------------------------------------------------ taking a part up again
# A part is continued by loading the play's sketch as on any night and then applying the thread to
# it: the sketch gives the people as written and the world's kinds and rules, the thread gives how
# things stood when the last part ended. apply() reports what it could not put back, so that a
# thread written by an older engine fails in the open.
CARRIED = "What has happened to you here so far, as you remember it:"


def card(carried):
    """What a person carries forward, as it stands on their own card (Character.get_character_description):
    their own prompts only. '' when they carry nothing."""
    if not carried or not carried.get('lived'):
        return ''
    lines = [CARRIED, carried['lived']]
    lines += [f"Of {name}: {what}" for name, what in (carried.get('knows') or {}).items()]
    if carried.get('holds'):
        lines += ['Still in force for you:'] + [f"- {h}" for h in carried['holds']]
    return '\n'.join(lines)


def _when(value):
    try:
        return datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError:
        return value


def seed(record):
    """The seed the thread's world was laid out from: a part is continued in the same country
    (world.build reads ATWAP_WORLD_SEED when the sketch is loaded)."""
    return ((record.get('parts') or [{}])[-1].get('world') or {}).get('seed')


def apply(context, record):
    """Put the play back as the thread's last part left it, on a context just built from the sketch
    and not yet begun. Returns {'thread', 'part', 'missing': [what could not be put back]}."""
    from sim import world, ledger, survival
    from sim.trace import trace
    part, missing = record['parts'][-1], []
    book, land, sp = ledger.of(context), world._land(context), context.screenplay
    # the world: the same country, with the places the play made and what it came to call them
    for p in part['world']['places']:
        have = land.table.get(p['id'])
        if have is None and p.get('made') and p.get('at'):
            have = land.add_place(p['kind'], p.get('description', ''), tuple(p['at']))
            if have.id != p['id']:
                missing.append(f"place {p['id']} came back as {have.id}")
        if have is None:
            missing.append(f"place {p['id']}")
            continue
        if p.get('at') and max(abs(a - b) for a, b in zip(have.xy, p['at'])) > 1:
            missing.append(f"place {p['id']} lies elsewhere in this world (laid out from another seed)")
        have.names |= set(p.get('names') or ())
        have.props.update(p.get('props') or {})
    context.models = dict(part['world'].get('models') or {})
    context.simulation_time = _when(part['clock'])
    # the people: the director's own among them, each where they were and as they were
    spot = lambda entry: tuple(entry['at']) if entry.get('at') else getattr(land.table.get(entry.get('place')), 'xy', None)
    for entry in part['people']:
        who = next((c for c in context.everyone() if c.name == entry['name']), None)
        if who is None and entry['name'] == 'Viewer':   # the voice of whoever watches: it comes into the play when it next speaks
            continue
        if who is None and not entry.get('lead'):
            owned = next((p['id'] for p in part['world']['places'] if p.get('owner') == entry['name']), None)
            try:
                who = context.cast_part(entry['name'], entry.get('description') or entry['name'], place=owned,
                                        wants=[w['text'] for w in entry.get('wants') or []][:2], owns=bool(owned), xy=spot(entry))
            except Exception as e:
                missing.append(f"{entry['name']} could not be cast: {type(e).__name__}: {e}")
        if who is None:
            missing.append(f"{entry['name']} is not in the sketch")
            continue
        if spot(entry) and world.on(who):
            world.put(who, spot(entry))
        wants = [w['text'] for w in entry.get('wants') or []]
        if wants and wants != [d.text for d in (who.drives or [])]:
            who.set_drives(wants)
        for d, w in zip(who.drives or [], entry.get('wants') or []):
            if w.get('activation') is not None:
                d.activation = float(w['activation'])
        if entry.get('lately'):
            who.character = entry['lately']
        body = getattr(who, 'body', None)
        if body is not None:
            body.levels.update({n: float(v) for n, v in (entry.get('needs') or {}).items() if n in body.levels})
            body.sheltered, body.flags, body.last_tick = bool(entry.get('sheltered')), set(entry.get('flags') or ()), context.simulation_time
        who.unknown_names = set(entry.get('does_not_know_the_name_of') or ())
        for rid in entry.get('has_seen') or ():   # a thread written before this was kept has none: the person starts as it did
            if getattr(who, 'resource_models', None) is not None:
                who.resource_models.add_resource_model(rid)
        who.carried = dict(entry.get('carries_forward') or {})
        who.own_past = next((f['text'] for f in part['so']['facts'] if f.get('of') == who.name), getattr(who, 'own_past', ''))
        who.last_sense_time = context.simulation_time
    names = {c.name for c in context.everyone()}
    # the stage: every thing where it was, and what has been left, told, promised and settled
    for t in part['stage']['things']:
        at = tuple(t['at']) if t.get('at') else getattr(land.table.get(t.get('place')), 'xy', None)
        holder = t.get('holder') if t.get('holder') in names else None
        if holder is None and at is None:   # a thread rebuilt from a trace knows only who was last seen holding it
            holder = next((e['name'] for e in part['people'] if t['name'] in (e.get('holds') or []) and e['name'] in names), None)
        if holder is None and at is None:
            missing.append(f"thing '{t['name']}' is nowhere (neither held nor lying at a place)")
            continue
        book.add(t['name'], holder=holder, at=None if holder else at,
                 **{k: _when(v) for k, v in t.items() if k not in ('name', 'holder', 'at', 'place')})
    for m in part['stage']['traces']:
        at = tuple(m['at']) if m.get('at') else getattr(land.table.get(m.get('place')), 'xy', None)
        if at is None:
            missing.append(f"trace '{m['words']}' has no place")
            continue
        book.marks.append({'words': m['words'], 'at': at, 'on': m.get('on'), 'event': None})
    for s in part['stage']['states']:
        book.states.append({'who': s['who'], 'words': s['words'], 'placed': bool(s.get('placed')), 'since': _when(s.get('since'))})
    for t in part['stage']['told']:
        book.told.append({'who': t['who'], 'to': list(t.get('to') or []), 'fact': t['fact'], 'event': None})
    for p in part['stage']['promised']:
        book.promised.append({'who': p['who'], 'to': list(p.get('to') or []), 'what': p['what'], 'kind': p.get('kind', 'later'), 'event': None})
    for s in part['stage']['standing']:
        book.standing.append({'fact': s['fact'], 'about': list(s.get('about') or []), 'known': list(s.get('known') or []),
                              'told': [dict(t) for t in s.get('told') or []],   # who has been told it on the stage, in the parts before
                              'since': _when(s.get('since')), 'ended': _when(s.get('ended')), 'event': None})
    for a in part['stage'].get('absent') or []:   # what was found not to be there, in the parts before (sim/ruling.py)
        if a.get('xy') is not None:
            book.absent.append({'what': a['what'], 'at': tuple(a['xy']), 'since': _when(a.get('since')), 'seen_by': list(a.get('seen_by') or []),
                                'by': a.get('by', ''), 'ended': _when(a.get('ended'))})
    for name, says in (part['stage'].get('writings') or {}).items():
        book.writings[name] = says
    for name, who in (part['stage'].get('readers') or {}).items():
        book.readers[name] = list(who) if isinstance(who, list) else who
    for item in part['stage'].get('shared') or []:
        if isinstance(item, list) and len(item) == 2:
            book.shared.append((item[0], set(item[1])))
    # what is so behind the play, and how far the leads have come to it
    sp.facts = [dict(f) for f in part['so']['facts']]
    sp.clues = [{k: v for k, v in c.items() if k != 'now'} for c in part['so']['clues']]
    sp.marks = [dict(m) for m in part['so'].get('marks') or []]
    sp.threads = [dict(t) for t in part['story'].get('planted') or []]
    sp.standing = bool(sp.facts)
    known = {}
    for q in part.get('questions') or []:
        known.update(q.get('come_to') or {})
    sp.known = known
    context.thread, context.thread_parts = record['thread'], record['parts']
    from sim import director
    sp.earlier = director.story_text(record['parts'])   # for whoever plans an act: what has been played, not only in this part
    for who in context.everyone():
        try:
            who.look()
        except Exception:
            pass
    trace.event('thread_applied', thread=record['thread'], part=part['part'], source=part.get('source'), missing=missing,
                people=sorted(names), things=len(book.things), clock=part['clock'])
    return {'thread': record['thread'], 'part': part['part'], 'missing': missing}


# ------------------------------------------------------------------ opening the next part
def sketch(context):
    """The sketch as the director is given it, with the kinds of place a new place may be."""
    from sim import world
    out = context.sketch_for_director()
    land = world._land(context)
    kinds = sorted(k for k, said in (getattr(land, 'kinds', None) or {}).items() if not said.get('region'))
    if kinds:
        out['places'] += f"\nKinds of place this world has: {', '.join(kinds)}."
    return out


def _promise_line(p):
    return f"{p['who']}" + (f" to {', '.join(p['to'])}" if p.get('to') else '') + f" ({p.get('kind', 'later')}): {p['what']}"


def ending(context):
    """How the last part left things, for the director who opens the next: its last act, the hour,
    where everyone is and what their bodies want, and what is still promised."""
    from sim import ledger, survival, world
    last = context.thread_parts[-1]
    acts = (last.get('story') or {}).get('acts') or []
    lines = [f"The last act, '{acts[-1].get('title', '')}': {acts[-1].get('summary', '')}" if acts else 'No act was recorded.',
             f"It ended at {context.simulation_time.strftime('%Y-%m-%d %H:%M')} ({context.simulation_time.strftime('%A')})."]
    for c in context.everyone():
        if not world.on(c):
            continue
        body = getattr(c, 'body', None)
        felt = [n for n, v in (body.levels if body is not None else {}).items() if v >= survival.NEED_FELT]
        lines.append(f"{c.name} is at {survival._nearest_resource_id(context, world.at(c))}"
                     + (f", holding {', '.join(str(i) for i in body.inventory)}" if body is not None and list(body.inventory) else '')
                     + (f"; feels {', '.join(felt)}" if felt else '') + '.')
    promised = [_promise_line(p) for p in ledger.of(context).promised]
    if promised:
        lines.append('Still promised or agreed:\n' + '\n'.join(f'- {p}' for p in promised))
    return '\n'.join(lines)


def open_part(context, final=None):
    """Before a continued part's first act: read the part before (questions closed and raised,
    promises spent), choose the question this one stands on and when it opens, and let that time
    pass. Returns the question; the first open question of the thread, as asked, when the director
    could not write one."""
    from datetime import timedelta
    from sim import director, ledger, survival
    from sim.trace import trace
    parts, book = context.thread_parts, ledger.of(context)
    final = bool(getattr(context, 'thread_final', False) or os.environ.get('ATWAP_THREAD_FINAL')) if final is None else final
    number = len(parts) + 1
    questions = [dict(q) for q in parts[-1].get('questions') or []]
    for q in questions:
        q.setdefault('whose', 'part')
    said = sketch(context)
    read = director.close_part(context, said, parts, [_promise_line(p) for p in book.promised]) or {'closed': [], 'open': [], 'spent': []}
    for n, how in read['closed']:
        if 0 < n <= len(questions) and not questions[n - 1].get('closed'):
            questions[n - 1].update(closed=parts[-1].get('part', number - 1), closed_by=how)
    numbered = director.standing_facts(parts[-1]['so']['facts'])   # as CLOSE was shown them: a question is settled by the fact it names
    named = lambda k: [numbered[k - 1]['text']] if isinstance(k, int) and 0 < k <= len(numbered) else []
    for n, status, why, fact in read.get('still') or []:   # each open question, read again against what is now so
        if 0 < n <= len(questions) and not questions[n - 1].get('closed'):
            questions[n - 1]['settled'] = status == 'settled' and bool(named(fact))
            questions[n - 1]['answer'] = named(fact) if questions[n - 1]['settled'] else []
            if status == 'void':   # it supposes what stands rules out: no part can answer it
                questions[n - 1].update(closed=parts[-1].get('part', number - 1), closed_by=f'void: {why}'.strip(' :'), void=True)
    for q in read['open']:
        if not any(o['text'] == q['text'] for o in questions):
            questions.append({'text': q['text'], 'raised': parts[-1].get('part', number - 1), 'whose': q['whose'], 'settled': bool(named(q.get('fact'))),
                              'answer': named(q.get('fact')), 'raised_by': q['raised_by'], 'come_to': {}, 'closed': None})
    spent = [p for n, p in enumerate(list(book.promised), 1) if n in read['spent']]
    from sim import ruling
    for p in spent:
        book.promised.remove(p)
        ruling.released(context, p, 'kept, broken or overtaken in the part before')   # those who held it are told (sim/ruling.py)
    looks = [f"{s['who']}: {s['words']}" for s in book.states]
    opened = director.next_part(context, said, parts, questions, ending(context), looks, final=final)
    context.thread_questions, context.thread_takes = questions, []
    context.thread_recap = context.thread_between = ''
    if not opened:
        still = [q for q in questions if not q.get('closed')]
        trace.event('thread_part_opened', part=number, failed=True)
        if still:
            context.thread_takes = [questions.index(still[0]) + 1]
        return still[0]['text'] if still else ''
    context.thread_takes = [n for n in opened['takes'] if 0 < n <= len(questions) and not questions[n - 1].get('closed')]
    when = _when(opened['opens']) if opened['opens'] else context.simulation_time
    gap = when - context.simulation_time if isinstance(when, datetime) else timedelta(0)
    if gap > timedelta(0):   # the time between: bodies are seen to, how people were placed is over, and what would not have lasted is gone
        survival.pass_offstage(context, gap)
        context.simulation_time = when
        survival.unpose(context)
        gone = [s for n, s in enumerate(list(book.states), 1) if n in opened['gone'] and s in book.states]
        for s in gone:
            book.states.remove(s)
        for c in context.everyone():
            if getattr(c, 'body', None) is not None:
                c.body.last_tick = when
            c.last_sense_time = when
            if opened['between'] and getattr(c, 'carried', None) and c.carried.get('lived'):
                c.carried['lived'] += f" Since then: {opened['between']}"
    context.thread_recap, context.thread_between = opened['recap'], opened['between'] if gap > timedelta(0) else ''
    trace.event('thread_part_opened', part=number, final=final, question=opened['question'], takes=context.thread_takes, opens=opened['opens'],
                gap_hours=round(gap.total_seconds() / 3600, 1), between=context.thread_between, closed=read['closed'],
                raised=[q['text'] for q in read['open']], promises_spent=len(spent), looks_gone=opened['gone'])
    return opened['question']


def another_question(context):
    """Nothing could be settled behind the question this part took. The director who chose it chooses again, once,
    from the questions still open without it; the hour the part opens at stands, and so does what has been told of
    the time between. Returns the new question, or '' when there is no other or none could be written. (The question
    was read to the audience and then played with nothing behind it: sim/ruling.py, clauses 1 and 4.)"""
    from sim import director, ruling
    from sim.trace import trace
    questions, takes = getattr(context, 'thread_questions', None) or [], list(getattr(context, 'thread_takes', None) or [])
    if not takes:
        return ''
    ruling.unmet(context, 'part', '', f'the question taken for this part: {context.standing_question}', 'nothing that fits what stands could be settled behind it')
    for n in takes:
        questions[n - 1]['unanswered'] = len(context.thread_parts) + 1   # read again by the director who opens the next part (CLOSE, 'still')
    offered = [dict(q, closed=q.get('closed') or ('set aside' if n in takes else None)) for n, q in enumerate(questions, 1)]
    if all(q.get('closed') for q in offered):
        return ''
    opened = director.next_part(context, sketch(context), context.thread_parts, offered, ending(context), [], final=bool(getattr(context, 'thread_final', False)))
    if not opened:
        return ''
    context.thread_takes = [n for n in opened['takes'] if 0 < n <= len(questions) and not offered[n - 1].get('closed')]
    trace.event('thread_question_again', set_aside=takes, question=opened['question'], takes=context.thread_takes)
    return opened['question']


def needs_settling(context):
    """Is any of what this part's question asks not yet settled behind the play?"""
    questions = getattr(context, 'thread_questions', None) or []
    takes = getattr(context, 'thread_takes', None) or []
    return any(not questions[n - 1].get('settled') for n in takes) if takes else True


def settle_part(context):
    """What is newly so behind a continued part's question, in the line forms frame_episode reads;
    '' when all of it was settled in an earlier part (it is there to be found) or it could not be made."""
    from sim import director, ledger
    if not needs_settling(context):
        return ''
    told = [f"{t['who']} told {', '.join(t['to']) or 'nobody'}: {t['fact']}" for t in ledger.of(context).told]
    context.thread_struck = []
    reply = director.settle_next(context, sketch(context), context.thread_parts, context.standing_question, told, opens=context.simulation_time)
    # a question the part takes is settled when the whole answer written for it stands. With a line struck and
    # not made to fit, which of them it left bare is not known here: they stay as they were, and the director who
    # opens the next part reads every open question again against what is then so (director.CLOSE, 'still')
    # (it was: when nothing had been struck. That marked a question settled though no fact answered it, and left one
    # unsettled whose answer stood while some other line had been struck.) A question the part takes is settled when a
    # fact that answers it stands, and it is kept with the question: the one link every later reader goes by
    # (Context.answering: the planner's note, the stage manager, the rule for one more act, the director who closes).
    answer = director.answering(reply or '')
    if answer:
        for n in getattr(context, 'thread_takes', None) or []:
            context.thread_questions[n - 1].update(settled=True, answer=list(answer))
    return reply


def make_place(context, place):
    """Bring the director's new place into the world, beside the listed place it names. Returns it, or None."""
    from sim import world
    from sim.trace import trace
    if not place:
        return None
    beside = world.called(context, place['beside'])
    made = None
    if beside is not None:
        try:
            made = world.make_place(context, place['name'], place['kind'], beside.id, description=place.get('description', ''))
        except Exception as e:
            trace.event('director_place_dropped', error=f'{type(e).__name__}: {e}', **place)
            return None
    if made is None:
        trace.event('director_place_dropped', **place)
    return made
