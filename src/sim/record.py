"""The world's record, the check that keeps narration to it, and what becomes of a new thing.

The mechanics (sim/survival.py) hold what is true of bodies, things, places, stock and dangers.
A narrator is told all of it and then writes freely, and what it wrote used to be shown,
remembered and built on unchecked: a pile of berries nobody held, a fever in an unhurt body, a
boar struck five times after it had fled. settle() puts one short check between the narrator
and everything downstream.

What the check finds is either a mistake (the wrong time of day, someone in the wrong place)
or a new thing. A mistake is written out. A new thing is put in the world's own terms
(admit): a prop, an item its actions use, a place of a kind it has, a change to a body. Who
brought it in decides:
- the plan and the director (a scene's set-up, a staleness intervention): it is made real, so
  the injury is in the body and the pool can be drunk from;
- a character's own deed, or the narrator: an ordinary prop is made real (a farmer's sickle);
  anything that would meet a need or change the odds has to be earned through the world's
  actions, and is refused.
A thing that cannot be put in the world's terms is refused whoever brings it.

Refusal is carried through as carefully as admission: the world states what is not so, the
narration is written again from that, the actor is told, bystanders see a try and not a deed,
and the page shows what happened instead of what was meant. Nothing is checked in a world
without mechanics: there is no record there.
"""
from __future__ import annotations

import re

from sim import ablation, stage, survival
from sim.trace import trace
from utils.Messages import UserMessage

NEARBY_TILES = 10   # places this near are in sight: they may be seen, pointed at, spoken of
BODILY = (r"hung|starv|fed\\b|thirst|quench|parch|tired|exhaust|rested|fatigue|sleep|hurt|wound|injur|pain|ache|sore|swol|swell|"
          r"sprain|twist|broke|bleed|blood|bruis|fever|sick|ill\\b|nause|cramp|limp|lame|burn|bandag|sling|faint|dizz|weak")
COMMON = {'their', 'there', 'while', 'which', 'about', 'after', 'before', 'still', 'being', 'where', 'other', 'again', 'hands', 'holds', 'takes'}
KINDS = ('item', 'condition', 'presence', 'place', 'time', 'stock', 'outcome', 'gender')

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
- item: someone has, gains, uses, eats, drinks from, drops or hands over a thing that Holdings, Lying here and Settled do not give them (a tool, weapon, food, water, medicine, container, device), or finds such a thing lying there. Picking, touching or handling what grows or lies at a place the character is at (a berry at the berry bush, a branch at a fallen branch), or something of a kind it carries, is supported.
- condition: of the body only (fear, shock, a scare, relief and calm are of the mind and never a condition): a wound, illness or pain from injury, or a need met or unmet (hungry, fed, thirsty, quenched, tired, rested), that Bodies and Settled do not show. A stronger or weaker word for what Bodies shows (exhausted for tired) is supported.
- presence: an animal or person that Present, Danger and Settled do not mention; or the danger listed under Danger gone, struck or beaten when Settled does not say so. (A danger leaving as Settled says is supported. Anyone listed under Present is there, wherever the narration puts them within the place. Places, plants, wind, noises and things in the landscape are never 'presence'.)
- place: a character standing at, or having moved to, a place other than where Present puts them, when Settled has no walk. Seeing, facing, pointing at, turning or taking a few steps toward a place in sight is not moving there.
- time: light, dark or a time of day that does not fit Time.
- stock: how much is left somewhere, other than as Stock says.
- gender: someone called he or she, him or her, sir or madam, a man or a woman, against what Cast says they are.
- outcome: only when Settled refuses the action (it says 'cannot', 'must move there first', 'still has it', 'nothing to') and the narration has it done all the same; or Settled has it done and the narration has it fail. When Settled is 'none' there is no outcome claim: one character touching, pushing, pinning, gripping or striking another, or handling a thing it holds, is allowed.
Everything the record does not keep is allowed: the ground, plants, weather that fits the time, the clothes a person wears, sounds, gestures, how a thing looks or feels, whatever the action itself says the character does with its own body, and everything of the mind: what anyone thinks, feels, fears, wants, knows, remembers or has forgotten is never a 'condition'.

Saying that something is NOT there (a hand that closes on nothing, an empty desk, 'no file lies there') is never unsupported: it claims nothing.
Most narration has nothing unsupported. When you are not sure, it is supported.

Respond with one line for each unsupported statement, giving the narration's own words exactly:
#claim <kind>: <the exact words from the narration>
or, when there is none, the single line:
#ok
</end>
"""


def cast(context, people=None):
    """The people of the play with what each is, for any prompt that writes about someone other
    than its own character: 'Hu (female), Xue (male), ...'. Each call used to see one
    character's description and everyone else as a bare name, and a name like Hu, Xue or Xiaoyu
    says nothing: one scene had Hu as 'him', Xiaoyu as 'her' and then 'his', and 'the lady' said
    to a man. `people`: those to list; the whole cast by default."""
    people = people or list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', []))
    return ', '.join(f"{c.name} ({(getattr(c, 'gender', '') or '').strip().lower() or 'not stated'})" for c in people)


def of(context, actor, people=None):
    """The record as it bears on what may be said of `actor` and those with it (or of `people`,
    a scene's cast)."""
    people = [c for c in (people or [actor] + actor.bystanders()) if getattr(c, 'mapAgent', None) is not None]
    at = stage.here(actor)
    near = sorted((stage._distance(at, res['location']), rid, res) for rid, res in context.map.resource_registry.items())
    places = [f"{rid} ({'here' if d <= stage.AT_PLACE_TILES else stage._direction(at, res['location']).lower()})"
              for d, rid, res in near if d <= NEARBY_TILES]
    return '\n'.join([
        f'Cast: {cast(context)}',
        f'Time: {stage.clock(context)}',
        'Present: ' + '; '.join(f'{c.name} {stage.where(context, stage.here(c))}' for c in people) + '. Nobody else.',
        'Places in sight: ' + (', '.join(places[:16]) or 'none named'),
        'Bodies: ' + ('; '.join(f'{c.name}: {c.body.summary(public=True)}' for c in people if getattr(c, 'body', None) is not None)
                      or 'not tracked'),
        'Holdings: ' + ('; '.join(f"{c.name}: {', '.join(c.body.inventory) or 'nothing'}" for c in people
                                   if getattr(c, 'body', None) is not None) or 'not tracked'),
        'Lying here, for anyone present to take or use: ' + (', '.join(p['name'] for p in survival.lying(context, at)) or 'nothing'),
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


DAYPARTS = {'night': r'night|dark|moon|star|midnight', 'dawn': r'dawn|sunrise|first light|early morning|morning',
            'morning': r'morning|sunlit|sunlight|daylight', 'midday': r'midday|noon|sunlit|sunlight|daylight|high sun',
            'afternoon': r'afternoon|sunlit|sunlight|daylight|long shadows', 'dusk': r'dusk|twilight|sunset|evening|fading light|dim'}


def _fits_clock(context, low):
    """Do these words name the part of the day it is? (The check called 'early morning light' at
    nine in the morning a mistake.)"""
    part = stage.clock(context).split(', ')[-1]
    return bool(re.search(DAYPARTS.get(part, r'$^'), low))


def _elsewhere(context, actor, words):
    """Do these words put someone at a named place other than where the actor is?"""
    at = stage.here(actor)
    for rid in stage.places_named(context, words):
        if stage._distance(at, context.map.resource_registry[rid]['location']) > stage.AT_PLACE_TILES:
            return True
    return bool(re.search(r'\b\d+ tiles?\b|\b(two|three|four|five|six|seven|eight|nine|ten) tiles\b', words.lower())) and \
        bool(re.search(r'\b(walk|run|sprint|cross|march|hurr|stride|flee|rush|mov)\w*', words.lower()))


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
    done = (facts or '').lower()
    if re.search(r"\bcannot\b|must move|must go|still has it|nothing to|there is no\b", done):
        done = ''   # a refusal settles that the thing was not done
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
        if kind == 'place' and not _elsewhere(context, actor, words):
            continue   # a turn, a step, a walk 'toward the trail': only arriving at another named place is a move
        if kind == 'time' and _fits_clock(context, low):
            continue
        if kind == 'item' and re.search(r"\b(no|not|nothing|none|empty|without|bare|absent|missing)\b", low):
            continue   # the narration saying a thing is not there claims nothing
        if kind == 'item' and done and any(w in done for w in _words(words) if len(w) >= 5 and w not in names and w not in COMMON):
            continue   # the thing is in what the world just settled (Joe 'bites into the unripe apple' he was settled to have eaten)
        claims.append((kind, words))
    return claims


PROPOSE = """A simulated world keeps track of certain kinds of thing. A text has brought in something the world does not hold yet. Put each new thing in the world's own terms.

<world>
Kinds of place where something can be done: {{$places}}
Things its actions use: {{$items}}
Bodily states it tracks: {{$needs}}
People present: {{$people}}
</world>

<text>
{{$text}}
</text>

New in that text:
{{$claims}}

For each numbered entry, one line, in one of these forms:
#thing <n> prop | <its name, one or two plain words> | held by <a person present> OR lying here | ordinary OR unusual
#thing <n> item | <one of the things the actions use> | held by <a person present> OR lying here
#thing <n> place | <one of the kinds of place> | <what it is, a few words: 'a rain pool in a hollow log'>
#thing <n> body | <a person present> | <one of the bodily states> | worse OR better | slightly OR clearly OR badly
#thing <n> none
A prop is a thing nobody acts on through the world's rules: a tool, a paper, a cup, a phone, a rope. It is ordinary when it would be found at such a place, or on such a person, without remark (a sickle on a farm, a phone in an office, a pen in a pocket). It is unusual when it is a find that changes what the characters can do, and always when it is a weapon, a medicine, a means of making fire or light, or a means of calling for help (a knife or a radio in a forest, a cache of supplies: one line for each thing in it).
Use 'place' for anything that gives water, food, shelter or rest, whatever it is called. Use 'none' when the entry is not a thing or a bodily state at all, or fits no form.
</end>
"""

FOUND = ('item', 'condition')     # the kinds of finding that may be a new thing, not a mistake


def admit(context, actor, claims, text, source, people=None, at=None):
    """Sort what a check found. A new thing is put in the world's terms and, if its source may
    bring it in, made real. Returns (facts, refused, remaining): the facts now true, the facts
    of what was refused, and the claims still to be written out of the text.
    source: 'director' (a plan or an intervention) or 'deed' (a character's act, the narrator)."""
    people = [c for c in (people or [actor] + actor.bystanders()) if getattr(c, 'body', None) is not None]
    new = [c for c in claims if c[0] in FOUND]
    remaining = [c for c in claims if c[0] not in FOUND]
    facts, refused = [], []
    if not new:
        return facts, refused, remaining
    places = survival.place_kinds(context)
    cfg = survival.config(context) or {}
    needs = [n for n in (people[0].body.levels if people else {})]
    reply = context.llm.ask({'places': '; '.join(f"{k} ({', '.join(v)})" for k, v in places.items()) or 'none',
                             'items': ', '.join(survival.item_kinds(context)) or 'none', 'needs': ', '.join(needs) or 'none',
                             'people': ', '.join(c.name for c in people), 'text': text,
                             'claims': '\n'.join(f'{i + 1}. {words}' for i, (_, words) in enumerate(new))},
                            [UserMessage(content=PROPOSE)], tag='record.admit', temp=0.1, stops=['</end>'], max_tokens=200) or ''
    at = tuple(at) if at else stage.here(actor)
    person = lambda name: next((c for c in people if c.name.lower() == name.strip().lower()), None)
    seen = set()
    for number, form, rest in re.findall(r'^\s*#thing\s+(\d+)\s+(\w+)\s*\|?\s*(.*?)\s*$', reply, re.M):
        index = int(number) - 1
        if not 0 <= index < len(new):
            continue
        seen.add(index)
        fields = [f.strip() for f in rest.split('|')]
        form, fact, why = form.lower(), '', ''
        if form in ('prop', 'item') and len(fields) >= 2 and fields[0]:
            name = fields[0].lower()
            holder = person(fields[1].lower().removeprefix('held by')) if 'held by' in fields[1].lower() else None
            usual = form == 'prop' and len(fields) > 2 and fields[2].lower().startswith('ordinary')
            if form == 'item' and name not in survival.item_kinds(context):
                form = 'prop'   # not one of the world's own: a prop, and from a deed an unusual one
            noun = name.split()[-1]
            if any(noun in i.lower() for c in people for i in c.body.inventory) or any(noun in p['name'] for p in survival.lying(context, at)):
                continue   # it is already here (Joe 'releases the branch' he is carrying): nothing new, nothing refused
            if source == 'director' or usual:
                fact = survival.add_thing(context, name, holder=holder, at=at)
            else:
                why = f"There is no {name} here to be had."
        elif form == 'place' and len(fields) >= 2 and source == 'director':
            fact = survival.add_place(context, next((k for k in places if k.lower() == fields[0].lower()), ''), fields[1], at)[1]
        elif form == 'place' and len(fields) >= 2:
            why = f"There is no {fields[1].lower()} here."
        elif form == 'body' and len(fields) >= 4 and person(fields[0]) is not None:
            who = person(fields[0])
            need = fields[1].lower()
            need = survival._harm(cfg)['name'] if need in ('injury', 'harm', 'hurt', 'wound') else need
            felt = who.body.levels.get(need, 0.0) >= survival.NEED_FELT
            if felt == fields[2].lower().startswith('worse'):
                continue   # the body already shows it (a 'bruised thigh' on a bruised body made it 'badly hurt')
            if source == 'director':
                fact = survival.change_body(who, fields[1].lower(), fields[2].lower().startswith('worse'), fields[3].lower())
            else:
                why = f"{who.name} is {who.body.summary(public=True)}, and nothing else."
        if fact:
            facts.append(fact)
        else:
            refused.append(why)
            if new[index] not in remaining:
                remaining.append(new[index])
    remaining += [c for i, c in enumerate(new) if i not in seen]
    refused = [r for r in refused if r]
    trace.event('admit', source=source, character=actor.name, claims=[f'{k}: {w}' for k, w in new], reply=reply.strip(),
                made=facts, refused=refused)
    return facts, refused, remaining


def rejection(claims, facts=(), refused=(), settled=''):
    """The note that sends a narration back to be written again."""
    return ("\nYour earlier answer was turned down: the world does not hold these things, so they may not be stated or implied:\n"
            + '\n'.join(f'- {words}' for _, words in claims)
            + (("\nWhat is so instead:\n" + '\n'.join(f'- {r}' for r in refused)) if refused else '')
            + (("\nThese are now true and may be used:\n" + '\n'.join(f'- {f}' for f in facts)) if facts else '')
            + (f"\nWhat the world has settled stands, and your answer must not undo it: {settled}" if settled else '')
            + "\nWrite it again without what was turned down, from what is so: say plainly what is actually done and found. "
              "If little is left to say, say little.\n")


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


class Settled(str):
    """A narration as the world admits it, with what settling it changed: .facts (things now
    real), .refused (what the world said is not so), .tainted (the deed itself assumed a thing
    that was refused, so the deed as written did not happen)."""
    facts: tuple = ()
    refused: tuple = ()
    tainted: bool = False

    def __new__(cls, text, facts=(), refused=(), tainted=False):
        self = super().__new__(cls, text or '')
        self.facts, self.refused, self.tainted = tuple(facts), tuple(refused), tainted
        return self


def strip(text, claims):
    """`text` without the sentences that carry one of `claims` (a claim found in no sentence is
    let go: this is for a plan's fields, each checked against claims found in all of them)."""
    sentences = [s for s in re.split(r'(?<=[.!?;])\s+', (text or '').strip()) if s]
    def carries(sentence):
        flat = ' '.join(sentence.lower().split())
        for _, words in claims:
            want = _words(words)
            if ' '.join(words.lower().split()).rstrip('.') in flat or (want and len(want & _words(sentence)) / len(want) >= 0.6):
                return True
        return False
    return ' '.join(s for s in sentences if not carries(s))


def settle(context, actor, action, facts, narration, rewrite, kind='do', source='deed'):
    """`narration` as the world admits it (a Settled). rewrite(note) asks the narrator again with
    a note on what was turned down and what is so. One check; new things are admitted or refused
    (admit); then at most one rewrite and re-check, and what is still unsupported is cut. Every
    finding is traced (kind 'narration_check')."""
    if (not narration or not survival.config(context) or getattr(actor, 'body', None) is None
            or not ablation.enabled('narration_gate')):
        return narration
    claims = check(context, actor, action, facts, narration)
    if not claims:
        trace.event('narration_check', character=actor.name, of=kind, outcome='ok' if claims is not None else 'unchecked', claims=[])
        return narration
    made, refused, remaining = admit(context, actor, claims, f'{action}\n{narration}', source)
    if not remaining:   # everything found was a new thing, and is now real
        trace.event('narration_check', character=actor.name, of=kind, outcome='admitted',
                    claims=[f'{k}: {w}' for k, w in claims], made=made, before=narration, after=narration)
        return Settled(narration, made)
    if any(k == 'gender' for k, _ in remaining):
        refused = refused + [f'Cast: {cast(context)}.']
    settled = ' '.join(filter(None, [facts or ''] + made + refused))
    again = rewrite(rejection(remaining, made, refused, settled=re.sub(survival.BODY_NOTE, '', facts or '').strip()))
    still = check(context, actor, action, settled, again) if again else None
    if again and still == []:
        result, outcome = again, 'rewritten'
    elif again and still:
        result, outcome = cut(again, still), 'cut'
    else:
        result, outcome = cut(narration, remaining), 'cut'
    # did the deed itself rest on what was refused? ('rotates his swollen ankle', unhurt)
    deed = _words(action)
    tainted = any(_words(w) and len(_words(w) & deed) / len(_words(w)) >= 0.5 for k, w in remaining if k in FOUND)
    trace.event('narration_check', character=actor.name, of=kind, outcome=outcome,
                claims=[f'{k}: {w}' for k, w in claims], still=[f'{k}: {w}' for k, w in (still or [])],
                made=made, refused=refused, tainted=tainted, before=narration, after=result)
    return Settled(result, made, refused, tainted)


# --------------------------------------------------------------------------- speech
# A character may be wrong about the world, and may lie. What it cannot be is mistaken about
# itself at this moment: what is in its hands, what it is doing, the state of its own body,
# where it stands, whether it is light or dark. A line that gets one of those wrong, and was
# not meant as a lie, is an error of the engine ("Here. Clean water." with empty hands, which
# the listener then took as given), and is said again without it before anyone hears it.
# Everything else a character says is its own: opinions, guesses, plans, fears, figures of speech.
SPEECH = """A character in a play is about to speak. Check the line against what the speaker cannot be mistaken about: itself, at this moment.

Speaker: {{$name}}
<about {{$name}}, now>
{{$record}}
</about>

<line>
{{$line}}
</line>

Find only statements of these kinds. The first five are made in the first person, about right now:
- holds: {{$name}} has, holds, offers or hands over a thing it does not have ("Here, clean water" with none in hand; "take my knife").
- body: {{$name}} says its own body is in a state that Bodies does not show (hurt, bleeding, fed, parched, rested).
- doing: {{$name}} says it is doing, this moment, a physical thing that it is not ("I'm climbing down" while sitting).
- where: {{$name}} says it, or someone present, is at a place other than where Present puts them.
- time: light, dark or a time of day that does not fit Time.
And one that may be about anyone in the play, present or not:
- gender: {{$name}} speaks of or to someone as he or she, him or her, sir or madam, a man or a woman, against what Cast says that person is ("Try asking him" of a woman).
Nothing else is to be found. These are the speaker's own and are never findings: what it plans, wants, fears, guesses or remembers; anything in the future or the past; orders, questions and offers of help; what it says about another person's mind; exaggeration, sarcasm and figures of speech ("we're just meat waiting to cool"); things it speaks of without claiming to hold them ("we need water").

Also say if the line hands a thing the speaker really has to someone present, in so many words ("Here, take the branch"):
#give <the thing, as Holdings names it> to <the person>

Respond with one line for each finding, giving the words from the line exactly:
#claim <kind>: <the exact words>
and the #give line if there is one; or, when there is neither, the single line:
#ok
</end>
"""

RESAY = """A character in a play was about to say the line below, but part of it is untrue of the speaker at this moment, and the character did not mean to lie.

Speaker: {{$name}}
<about {{$name}}, now>
{{$record}}
</about>

<line>
{{$line}}
</line>

Not so:
{{$claims}}
(Who is who: {{$cast}})

Say the line again as {{$name}} would, in the same voice, in the first person, and no longer than it was. Keep everything else it says and means; leave out what is not so.
- Do not announce the correction: never "It's morning", "I'm not up the tree", "I have no water". Just do not say the untrue thing.
- Do not move the untrue thing onto someone else.
- If the point of the line was the untrue part, say what {{$name}} can truly say toward the same end: an offer of water one does not have becomes an offer to fetch it; a claim to hold a branch becomes going to get one; a warning that it is nearly dark becomes a warning that dark will come.
Respond with the spoken words only.
</end>
"""

FELT_WORDS = {'thirst': r'thirst|parch|dry|water', 'hunger': r'hung|starv|empty|food|eat', 'fatigue': r'tired|exhaust|weary|worn|sleep|spent|fatigue'}


def _body_bears(speaker, low):
    """Does the speaker's body bear out these words about itself? ('My throat's dry', thirsty.)"""
    levels = speaker.body.levels
    harm = survival._harm(speaker.body.cfg)['name']
    for need, level in levels.items():
        pattern = FELT_WORDS.get(need) or (r'hurt|wound|injur|pain|ache|sore|bleed|blood|bruis|limp|swol' if need == harm else re.escape(need[:5]))
        if re.search(pattern, low):
            return level >= survival.NEED_FELT
    return False


# 'doing' is asked for but not acted on: the world does not keep what a character is doing with
# itself (scanning, holding a position), so nothing can show such a line wrong; and a spoken
# deed that is one of the world's actions ("I'm drinking") is made true instead (survival.spoken_deed)
SPEECH_KINDS = ('holds', 'body', 'where', 'time', 'gender')


ME_NOW = r"\b(i'm|i’m|i am|we're|we’re|we are)\b"
LATER = r"\b(i'll|i’ll|i will|we'll|we’ll|going to|gonna|let me|let's|let’s|should|could|would|can|if|when|until|before|after)\b"
SENSING = r"\b(see|saw|hear|heard|watch|look|notice|know|feel|felt|think|smell|sense|remember|understand|believe|hope|want|need)\w*"
HOUR_NOW = (r"\b(it'?s|it’s|it is|getting|nearly|almost|already|now)\b.{0,24}\b(dark|night|dusk|dawn|late|evening|morning|noon|midday|sunset|sunrise)\b"
            r"|\b(sun|light|day)\b.{0,16}\b(setting|fading|going|gone|rising|up|down|dying)\b")


def speech_claims(context, speaker, reply):
    """The findings in a speech check's reply that hold up: [(kind, words)]. The model finds
    more than the narrow list allows, so each kind has a test of its own. (In its first run it
    took 'Thicket3 is just a bigger room to hide in' for a claim to be there, 'the dark fruit'
    for a time of day, 'I see them' for a deed and 'I'll scout the perimeter' for one under
    way: five lines resaid, none of them wrong.)"""
    claims = []
    held = [i.split()[-1].lower() for i in speaker.body.inventory]
    for kind, words in re.findall(r'^\s*#claim\s+(\w+)\s*:\s*(.+?)\s*$', reply or '', re.M):
        kind, words, low = kind.lower(), words.strip(' "\''), words.lower()
        if kind not in SPEECH_KINDS or not words.strip('.'):
            continue
        if kind in ('holds', 'body', 'doing') and not re.search(r"\b(i|i'm|i’m|i've|i’ve|my|me|mine|here)\b", low):
            continue   # not said of the speaker ("You're sweating")
        if kind == 'holds' and (any(h in low for h in held) or (held and re.search(r'\b(this|these|it|them|that|those|one)\b', low))
                                or re.search(LATER, low)):
            continue   # it does hold one, or speaks of what it will do
        if kind == 'body' and (not re.search(BODILY, low) or _body_bears(speaker, low)):
            continue
        if kind == 'doing' and (not re.search(ME_NOW + r"\s+(\w+\s+)?\w+ing\b|" + ME_NOW + r"\s+(already\s+)?(up|in|on|inside|under|behind)\b", low)
                                or re.search(LATER, low) or re.search(ME_NOW + r"\s+(\w+\s+)?" + SENSING, low)):
            continue   # only a deed said to be under way: not sensing, not what is to come
        if kind == 'time' and (not re.search(HOUR_NOW, low) or _fits_clock(context, low)
                               or re.search(r'\b(before|until|by|when|after|tonight|soon|comes?|will|later|tomorrow)\b', low)):
            continue   # only the hour said of now ('the dark fruit' is not one)
        if kind == 'where' and not (re.search(ME_NOW + r"|\b(you're|you’re|you are|he's|she's|he is|she is)\b", low) and _elsewhere(context, speaker, words)):
            continue   # only someone said to be at another place now; naming a place is not being there
        claims.append((kind, words))
    return claims


def speech(context, speaker, line, listener=None):
    """A spoken line as its speaker can truly say it: (line, give). `give` is (thing, person)
    when the line itself hands over something the speaker holds, else None. One check; one
    rewrite when it finds something; a line the rewrite leaves no better is cut to its sound
    sentences. Traced as 'speech_check'."""
    if (not line or not survival.config(context) or getattr(speaker, 'body', None) is None
            or getattr(speaker, 'mapAgent', None) is None or not ablation.enabled('speech_gate')):
        return line, None
    people = [speaker] + [c for c in speaker.bystanders() if c is not speaker]
    if listener is not None and listener not in people and getattr(listener, 'mapAgent', None) is not None:
        people.append(listener)
    about = of(context, speaker, people)
    reply = context.llm.ask({'name': speaker.name, 'record': about, 'line': line}, [UserMessage(content=SPEECH)],
                            tag='record.speech', temp=0.1, stops=['</end>'], max_tokens=120)
    if reply is None:
        return line, None
    claims = speech_claims(context, speaker, reply)
    give = None
    handed = re.search(r'^\s*#give\s+(.+?)\s+to\s+(\w+)', reply, re.M)
    if handed and not claims:
        thing = next((i for i in speaker.body.inventory if i.split()[-1].lower() in handed.group(1).lower()), None)
        person = next((c for c in people if c is not speaker and c.name.lower() == handed.group(2).lower()), None)
        give = (thing, person) if thing and person is not None else None
    if not claims:
        trace.event('speech_check', character=speaker.name, outcome='ok', claims=[], gives=give[0] if give else None)
        return line, give
    again = context.llm.ask({'name': speaker.name, 'record': about, 'line': line, 'cast': cast(context),
                             'claims': '\n'.join(f'- {w}' for _, w in claims)}, [UserMessage(content=RESAY)],
                            tag='record.resay', temp=0.6, stops=['</end>'], max_tokens=120)
    again = (again or '').strip().strip('"“”\'').strip()
    carried = lambda text: [c for c in claims if ' '.join(c[1].lower().split()).rstrip('.') in ' '.join(text.lower().split())]
    if again and not carried(again):
        result, outcome = again, 'resaid'
    else:
        kept = strip(line, claims)
        result, outcome = (kept, 'cut') if kept else (again or line, 'unfixed')
    trace.event('speech_check', character=speaker.name, outcome=outcome, claims=[f'{k}: {w}' for k, w in claims],
                before=line, after=result)
    return result, None


# --------------------------------------------------------------------------- deeds
# An actor is told what it holds, and still writes deeds with things it does not have: Joe
# 'brandishing the branch' with empty hands, Samantha 'snapping a thick branch from the ground'
# four tiles from the nearest. Performed, such a deed can only be told as a failure ('her fingers
# closing on air'), and a scene of them is slapstick. So a deed is looked at before it is done.
# The test is the world's own and needs no model: does the deed use one of the things the
# world's actions deal in, which the actor neither holds nor can get where it stands? or treat
# its own unhurt body as hurt? A deed that fails is passed over for another the actor offered,
# or written again in the same spirit from what the actor does have.
USING = ['grip', 'swing', 'raise', 'brandish', 'thrust', 'wave', 'wield', 'hold', 'jab', 'plant', 'drive', 'brace', 'lower', 'hand',
         'give', 'pass', 'slide', 'throw', 'toss', 'drop', 'strike', 'hit', 'club', 'swat', 'eat', 'chew', 'bite', 'munch', 'offer',
         'shove', 'kick', 'wedge', 'drag', 'stack', 'crack', 'split', 'sign', 'read', 'show', 'lean on']
GETTING = ['take', 'grab', 'snatch', 'seize', 'pick up', 'snap', 'break off', 'wrench', 'gather', 'collect', 'pluck', 'pick', 'lift', 'fetch']
HURT = (r"\b(his|her|my|own)\s+(\w+\s+){0,2}(swollen|injured|wounded|bleeding|broken|sprained|twisted|bandaged|bruised|torn|hurt)\b"
        r"|\b(limp(s|ing)?|bleed(s|ing)?|nurs(es|ing))\b")


def deed_lacks(context, character, text):
    """What a deed rests on that is not so, in words for the actor: [] when the deed is sound."""
    cfg = survival.config(context)
    body = getattr(character, 'body', None)
    if not cfg or body is None or getattr(character, 'mapAgent', None) is None or not text:
        return []
    low = text.lower().replace('’', "'")
    at = stage.here(character)
    lacks = []
    others = [c for c in character.bystanders() if getattr(c, 'body', None) is not None]
    threat = survival.active_threat_for(character)
    # a verb form after 'his', 'the', ... is a thing ('Samantha's hands'), not a deed
    verbs = re.sub(r"\b(his|her|their|its|my|your|our|the|a|an|\w+'s)\s+\w+", r'\1', low)
    for kind in survival.item_kinds(context):
        noun = kind.split()[-1].lower()
        stem = noun[:-3] if noun.endswith('ies') else noun[:-1] if noun.endswith('s') and len(noun) > 4 else noun
        if not re.search(rf"\b{re.escape(stem)}\w*", low):
            continue
        using, getting = survival._matches(verbs, USING), survival._matches(verbs, GETTING)
        if not (using or getting):
            continue   # it is looked at, pointed out or spoken of, not used
        if any(stem in i.lower() for i in body.inventory) or any(stem in p['name'] for p in survival.lying(context, at)):
            continue
        if re.search(rf"\b(\w+'s|her|his|their)\s+(\w+\s+)?{re.escape(stem)}", low) and any(stem in i.lower() for o in others for i in o.body.inventory):
            continue   # it is someone else's, and they have it
        sources = sorted((stage._distance(at, res['location']), rid) for rid, res in context.map.resource_registry.items()
                         if stem in rid.lower())
        reach = cfg.get('approach_tiles', 8)
        if sources and (sources[0][0] <= stage.AT_PLACE_TILES or (getting and not using and sources[0][0] <= reach and not threat)):
            continue   # it is to be had here, or the deed is to go and get it
        where = (f"; the nearest is at {sources[0][1]}, {sources[0][0]} tiles {stage._direction(at, context.map.resource_registry[sources[0][1]]['location']).lower()}"
                 + (', and with danger this close there is no walking over to it' if threat else '')) if sources else ''
        lacks.append(f"{character.name} has no {noun}{where}")
    harm = survival._harm(cfg)['name']
    if (re.search(HURT, low) and body.levels.get(harm, 0.0) < survival.NEED_FELT
            and not re.search(r"\b\w+'s\s+(\w+\s+){0,2}(swollen|injured|wounded|bleeding|broken|bruised)", low)):
        lacks.append(f"{character.name} is unhurt")
    return lacks


def _is_do(act):
    return str(getattr(act, 'mode', '')).split('.')[-1] == 'Do'


def sound_deeds(context, character, acts):
    """The acts an actor offered, without the Dos that rest on what is not so; all of them when
    none is sound (the chosen one is then mended)."""
    if not ablation.enabled('narration_gate'):
        return acts
    sound = [a for a in acts if not _is_do(a) or not deed_lacks(context, character, a.action)]
    if sound and len(sound) < len(acts):
        trace.event('deed_check', character=character.name, outcome='passed_over',
                    deeds=[a.action[:160] for a in acts if a not in sound])
    return sound or acts


REDO = """A character in a play was about to do the deed below, but it rests on something that is not so.

{{$name}} was about to:
<deed>
{{$deed}}
</deed>

What is so:
{{$lacks}}
{{$name}} is carrying: {{$holding}}. Lying here: {{$lying}}.

Write the deed again: what {{$name}} does instead, toward the same end and in the same manner, with what it has and where it stands. If the end was to have the missing thing, the deed is to go for it. One sentence, in the same grammatical form as the original, no longer than it was. Do not mention what is missing.
Respond with the deed only.
</end>
"""


def mend_deed(context, character, act):
    """The act about to be performed, made sound: a Do that rests on what is not so is written
    again from what is. The act is changed in place; returns True when it was."""
    if not _is_do(act) or not ablation.enabled('narration_gate'):
        return False
    lacks = deed_lacks(context, character, act.action)
    if not lacks:
        return False
    again = context.llm.ask({'name': character.name, 'deed': act.action, 'lacks': '\n'.join(f'- {l}.' for l in lacks),
                             'holding': ', '.join(character.body.inventory) or 'nothing',
                             'lying': ', '.join(p['name'] for p in survival.lying(context, stage.here(character))) or 'nothing'},
                            [UserMessage(content=REDO)], tag='record.redo', temp=0.5, stops=['</end>'], max_tokens=90)
    again = (again or '').strip().strip('"“”').strip()
    left = deed_lacks(context, character, again) if again else lacks
    outcome = 'redone' if again and not left else 'unfixed'
    trace.event('deed_check', character=character.name, outcome=outcome, lacks=lacks, before=act.action, after=again)
    if outcome == 'redone':
        act.action = again
        if getattr(act, 'affordance', None):
            act.affordance = None   # the label was for the deed as it was
    return outcome == 'redone'
