"""The world's record as each reader is shown it, what a text brings in, and how a line is heard.

The mechanics (sim/survival.py) hold what is true of bodies, things and places; of() and account() say it to whoever
asks (a ruling, an actor, the planner). Nothing here refuses (2026-11-28): a deed is resolved (sim/resolver.py), a
line is said as written (speech), and the world's narration of a look is told as written (narration_noted). What
any of them has that the record does not hold is written to the run's error log (sim/errors.py) and changes nothing.
What the director brings in is put in the world's terms where it can be (admit); what cannot is logged.
"""
from __future__ import annotations

import re

from sim import ablation, keeper, ledger, ruling, stage, survival, world
from sim.trace import trace
from utils.Messages import UserMessage
from utils import json_reply

KINDS = ('item', 'holding', 'condition', 'presence', 'place', 'time', 'stock', 'outcome', 'gender', 'writing')

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
- holding: someone ends up holding or carrying off a thing that Lying here lists and Settled did not give them (picks it up, pockets it, takes it along, ties it on). Touching, examining or turning over a thing where it lies, or pointing at it, is not holding.
- condition: of the body only (fear, shock, a scare, relief and calm are of the mind and never a condition): a wound, illness or pain from injury, or a need met or unmet (hungry, fed, thirsty, quenched, tired, rested), that Bodies and Settled do not show. A stronger or weaker word for what Bodies shows (exhausted for tired) is supported.
- presence: an animal or person that Present, Danger and Settled do not mention; or the danger listed under Danger gone, struck or beaten when Settled does not say so. (A danger leaving as Settled says is supported. Anyone listed under Present is there, wherever the narration puts them within the place. Places, plants, wind, noises and things in the landscape are never 'presence'. Someone listed as on the radio is heard, not there: their voice is supported; their standing there, handling a thing there or being touched is 'presence'.)
- place: a character standing at, or having moved to, a place other than where Present puts them, when Settled has no walk. Seeing, facing, pointing at, turning or taking a few steps toward a place in sight is not moving there.
- time: light, dark or a time of day that does not fit Time. Words that do fit it are supported, in any form ('early morning light' at nine in the morning).
- stock: how much is left somewhere, other than as Stock says.
- gender: someone called he or she, him or her, sir or madam, a man or a woman, against what Cast says they are.
- writing: what a written thing says, read out, quoted or told, other than what Writings records it says (another promise, name, sum or term).
- outcome: only when Settled refuses the action (it says 'cannot', 'must move there first', 'still has it', 'nothing to') and the narration has it done all the same; or Settled has it done and the narration has it fail. When Settled is 'none' there is no outcome claim: one character touching, pushing, pinning, gripping or striking another, or handling a thing it holds, is allowed.
Everything the record does not keep is allowed: the ground, plants, weather that fits the time, the clothes a person wears, sounds, gestures, how a thing looks or feels, whatever the action itself says the character does with its own body, and everything of the mind: what anyone thinks, feels, fears, wants, knows, remembers or has forgotten is never a 'condition'.

Saying that something is NOT there (a hand that closes on nothing, an empty desk, 'no file lies there') is never unsupported: it claims nothing.
What Settled says happened is supported, in the narration's own words (he bites into the apple Settled says he ate).
Most narration has nothing unsupported. When you are not sure, it is supported.

Respond ONLY with JSON, no prose, no markdown, in this shape:
{"findings": [{"kind": "<item, holding, condition, presence, place, time, stock, outcome, gender or writing>", "words": "<the narration's own words, exactly>"}]}
with "findings": [] when nothing is unsupported.
</end>
"""


def seen_by(character):
    """The situation as `character` can know it, for that character's own prompts. A character's
    prompts read Context.current_state, the director's and narrator's text, which carries the
    scene's title and set-up: so a character was told the scene before it happened. This is the
    world as the character stands in it, with the play's opening description while the first
    scene lasts. In a world without mechanics there is no world to render and the play's own
    situation text is used, without the scene's notes."""
    context = character.context
    opening = '' if getattr(context, '_opening_done', False) else context.initial_state.strip() + '\n\n'
    if not survival.config(context):
        return getattr(context, '_state_base', '') or context.initial_state
    # the scenes this character was in (it was told, 'as you can see for yourself', what happened in scenes it was not in)
    casts = getattr(context, 'wrapup_casts', {})
    own = [w for w in getattr(context, 'act_scene_wrapups', []) if character.name in casts.get(w, {character.name})]
    lately = ' '.join(w.split(' Assessed outcome:')[0] for w in own[-2:])
    return opening + situation(context, lately, viewer=character)


def quoted_in(quote, text, share=0.8):
    """Whether `quote` is words of `text`. A quote may join pieces ('Samantha sleeps in Hut2 ...
    in shelter; "Me first"'): each piece of three words or more must be found, whole once case,
    quotation marks and spacing are set aside, or with most of its words together in one line.
    A mechanical test that the words cited are in the record, not a reading of what they mean."""
    norm = lambda t: ' '.join(re.sub(r"[\"“”‘’]", ' ', (t or '').lower()).split())
    flat = norm(text)
    lines = [set(re.findall(r"[a-z0-9']+", line.lower())) for line in (text or '').split('\n')]
    pieces = [p.strip(" .,;:-—'") for p in re.split(r"\.\.\.|…|;|—|\(|\)|\n|\"|“|”", quote or '')]
    long = [p for p in pieces if len(re.findall(r"[a-z0-9']+", p.lower())) >= 3]
    if not long:   # a short quote ('Joe leaves', "I'll go"): found whole, or not at all
        short = norm(' '.join(p for p in pieces if p))
        return len(short.split()) >= 2 and short in flat
    pieces = long
    def found(piece):
        if norm(piece) in flat:
            return True
        words = set(re.findall(r"[a-z0-9']+", piece.lower()))
        return any(len(words & line) / len(words) >= share for line in lines)
    return all(found(p) for p in pieces)


def last_lines(lines, limit):
    """The latest whole lines of `lines` (a list, or text split on newlines) that fit in `limit`
    characters, as text. A window taken as text[-limit:] began inside a line ('ha slips…'), and
    the call read a deed with no doer."""
    if isinstance(lines, str):
        lines = lines.split('\n')
    kept, size = [], 0
    for line in reversed([l for l in lines if l is not None]):
        size += len(line) + 1
        if kept and size > limit:
            break
        kept.append(line)
    return '\n'.join(reversed(kept))


def cast(context, people=None):
    """The people of the play with what each is, for any prompt that writes about someone other
    than its own character: 'Hu (female), Xue (male), ...'. Each call used to see one
    character's description and everyone else as a bare name, and a name like Hu, Xue or Xiaoyu
    says nothing: one scene had Hu as 'him', Xiaoyu as 'her' and then 'his', and 'the lady' said
    to a man. `people`: those to list; the whole cast by default."""
    people = people or list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', []))
    def what(c):
        sex = (getattr(c, 'gender', '') or '').strip().lower() or 'not stated'
        by = (getattr(c, 'remote', None) or '').strip()
        # a play's character who is in its scenes by voice (stage.remote): every prompt reads it here
        return f"{sex}; takes part {by}, not in person" if by else sex
    return ', '.join(f"{c.name} ({what(c)})" for c in people)


def cast_note(context):
    """The cast for the head of every prompt (llm_api.set_preamble): those whose sex is known."""
    try:
        known = [c for c in list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', [])) if (getattr(c, 'gender', '') or '').strip()]
        return ('The people of this play, and whether each is he or she: ' + cast(context, known) + '.') if known else ''
    except Exception:
        return ''


def of(context, actor, people=None, full=False):
    """The record as it bears on what may be said of `actor` and those with it (or of `people`,
    a scene's cast). full: with how everyone is placed, what is held out, who has gone, what the
    stage has refused `actor` in this scene and what the world offers here: the whole stage, for
    the call that says what a deed does (sim/resolver.py)."""
    people = [c for c in (people or [actor] + actor.bystanders()) if world.on(c)]
    # those in the scene by radio are heard, not there: listed apart, so that no narration puts
    # them on the spot (Novak, the Earthside flight director, swung a camera on the gantry)
    radio = [c for c in people if c is not actor and stage.remote(c)]
    heard = actor.hearers() if hasattr(actor, 'hearers') else []
    radio += [c for c in heard if c not in people and (stage.remote(c) or stage.remote(actor))]
    people = [c for c in people if c not in radio]
    at = stage.here(actor)
    near = sorted((world.span(context, at, res.xy), rid, res) for rid, res in world.listed(context))
    places = [f"{rid} ({'here' if world.at_place(context, at, res) else world.sight_words(context, at, res.xy)})"
              for d, rid, res in near if world.in_sight(context, at, res)]
    return '\n'.join([
        f'Cast: {cast(context)}',
        f'Time: {stage.clock(context)}',
        'Present: ' + '; '.join(f'{c.name} {stage.remote(c) or stage.where(context, stage.here(c))}' for c in people) + '. Nobody else.',
        *(['On the radio, not here in person: ' + '; '.join(f'{c.name} ({stage.remote(c) or stage.where(context, stage.here(c))})' for c in radio)]
          if radio else []),
        'Places in sight: ' + (', '.join(places[:16]) or 'none named'),
        'Bodies: ' + ('; '.join(f'{c.name}: {c.body.summary(public=True)}' for c in people if getattr(c, 'body', None) is not None)
                      or 'not tracked'),
        *(['What stands (so in this world, whoever knows it; each is known only to those named with it, and to those marked as told it on the stage, as far as the mark says):\n' + survival.stands_note(context)]
          if survival.stands_note(context) else []),
        'Holdings: ' + (' | '.join(f"{c.name}: {'; '.join(survival.named(context, i, reader=None) for i in c.body.inventory) or 'nothing'}" for c in people
                                   if getattr(c, 'body', None) is not None) or 'not tracked'),
        'Lying here, for anyone present to take or use: ' + ('; '.join(survival.named(context, p, reader=None) for p in survival.lying(context, at)) or 'nothing'),
        # what a deed left on the place is that deed's doing, told as such: it is not a thing, and
        # nobody else was here to leave it
        *(["Traces left here by deeds of the play (each the doing of the one named, and of nobody else): " + '; '.join(m)]
          if (m := survival.marks_note(context, at)) else []),
        *(['Writings (what these written things say; a character knows it only once it has read it or been told): '
           + w.replace('\n', '; ')] if (w := survival.writings_note(context, at, people, readers=True)) else []),
        # what a deed played here has looked for and not found (sim/ruling.py): it is not here still, until a deed makes it
        *([f"Found not to be here (looked for by a deed, and not there: a try for one finds none, unless a deed makes it from what is here): {n}"]
          if (n := ruling.absent_note(context, at)) else []),
        *(_acts(context, actor, people) if full else []),
    ])


def account(context, reader, about=None, people=None, full=False, states=False, without=()):
    """How things stand, for whoever reads it: the one way the stage is described.
    reader: a character (what lies where it is and the traces there, each as whose doing it is
    for this reader; what has been told between it and those with it; how what it carries has
    been left); 'stage' (everything at the place of `about`, for the calls that say what an act
    does or check one: of()); 'planner' (where every thing and trace of the play is, what the
    written things say, and what each person has told of their life and to whom).
    The same things and traces, by the same wording (survival.named, marks_note, whose), whoever
    reads: each prompt used to put its own list together, and one of them listed a woman's
    fingermarks as a thing lying at a spring.
    For the planner: states, each thing with how it has been left; without, standing facts the same
    prompt already gives and that are not printed again."""
    if reader == 'planner':
        cast = list(people) if people is not None else context.everyone()
        told = [f"{t['who']} has told {', '.join(t['to']) or 'nobody yet'}: {t['fact']}" for t in ledger.of(context).told][-24:]
        promised = [f"{p['who']} ({p['kind']}, said to {', '.join(p['to']) or 'nobody'}): {p['what']}" for p in ledger.of(context).promised][-16:]
        stands = survival.stands_note(context, without=without)
        parts = [survival.things_note(context, people=cast, states=states), survival.writings_note(context, people=cast),
                 ('What stands (so in this world, whoever knows it; each is known only to those named with it, and to those marked as told it on the stage, as far as the mark says):\n' + stands) if stands else '',
                 (f"Found not to be there, by a deed that looked (it is not there still): {n}") if (n := ruling.absent_note(context, places=True)) else '',
                 ("What each has told of their own life (the one told knows it from then on):\n" + '\n'.join(told)) if told else '',
                 ("What has been promised or agreed (the latest, and not every word said), oldest first:\n" + '\n'.join(promised)) if promised else '']
        return '\n'.join(p for p in parts if p)
    if reader == 'stage':
        return of(context, about, people, full)
    here, body, lines = stage.here(reader), getattr(reader, 'body', None), []
    things = [survival.named(context, p, reader=reader) for p in survival.lying(context, here)]
    if things:   # the actor saw nothing of what lay at its feet but what it carried
        lines.append('Lying here: ' + '; '.join(things))
    traces = survival.marks_note(context, here, reader=reader)
    if traces:   # told as whose doing each is: listed bare, her own fingermarks were a stranger's to him
        lines.append('Traces left here by what has been done: ' + '; '.join(traces))
    from sim import resolver
    for other in (reader.bystanders() if hasattr(reader, 'bystanders') else []):
        # what has been told between them of their lives is known to both from then on (a woman
        # told the theft of her pack in one scene and told it again as news three scenes later)
        mine, theirs = resolver.told_to(context, reader, other)[-12:], resolver.told_to(context, other, reader)[-12:]
        if mine:
            lines.append(f"You have already told {other.name} this of your life, and {other.name} knows it (it is not news to tell again): " + '; '.join(mine))
        if theirs:
            lines.append(f"{other.name} has told you this of their life: " + '; '.join(theirs))
    gone = ruling.absent_note(context, here, reader=reader)
    if gone:   # what this person has looked for here, or seen looked for, and found not there
        lines.append('Looked for here and not here, as you have seen: ' + gone)
    left = [survival.named(context, i, reader=reader) for i in (body.inventory if body is not None else []) if survival.state_of(context, i)]
    if left:   # how what it carries has been left by handling
        lines.append('Of what you carry: ' + '; '.join(left))
    # and how what the others carry has been left, as it can be seen: an actor was shown that state for its own things
    # only, and said 'Skin's full' of a skin in another's hands that the stage had as 'no longer full'
    for other in (reader.bystanders() if hasattr(reader, 'bystanders') else []):
        theirs = getattr(other, 'body', None)
        seen = [survival.named(context, i, reader=reader) for i in (theirs.inventory if theirs is not None else []) if survival.state_of(context, i)]
        if seen:
            lines.append(f'Of what {other.name} carries: ' + '; '.join(seen))
    return '\n'.join(lines)


def _acts(context, actor, people):
    """The rest of the stage (of, full): placing, things held out, exits, what is hidden here."""
    placed = [f"{c.name}: {', '.join(c.body.bearing)}" for c in people if getattr(c, 'body', None) is not None and c.body.bearing]
    held = [f'{holder} holds the {item} out to {to}' for item, holder, to in context.held_out]
    gone = [f"{e['who']}{' with ' + ', '.join(e['with']) if e.get('with') else ''}{' for ' + e['to'] if e.get('to') else ''}"
            for e in getattr(context, 'scene_exits', None) or []]
    head = ['How they are placed: ' + ('; '.join(placed) or 'all on their feet, nobody touching'),
            'Held out, for the one named to take: ' + ('; '.join(held) or 'nothing'),
            'Gone out of this scene: ' + ('; '.join(gone) or 'nobody')]
    # what is hidden here for a search to find
    hidden = [p['name'] for p in survival.lying(context, stage.here(actor), hidden=True)]
    # (said only when something is: 'Hidden here: nothing' was read as nothing to be found, and a
    # search the world's facts leave to chance was ruled empty without a draw)
    return head + (['Hidden here, to be found only by someone who searches this place: ' + ', '.join(hidden)] if hidden else [])


# ---------------------------------------------------------------------------- a look at something
# A look is aimed at a person, a thing or a place, by the name the stage gives it, and shows what
# the world holds of it that can be seen from where the looker stands: nothing hidden, nothing of
# a body's needs or anyone's thoughts. Of a person close by, also how they seem, in the two or
# three words of the stance they carry. Of a written thing at hand, its words.
def _plain(name):
    return re.sub(r'^(at|the|a|an)\s+', '', ' '.join(str(name or '').lower().strip(' .,;:!?"\'').split()))


def _seems(person):
    stance = getattr(person, 'emotionalStance', None)
    words = [str(getattr(getattr(stance, part, None), 'value', '') or '').lower() for part in ('arousal', 'tone', 'orientation')]
    return ', '.join(w for w in dict.fromkeys(words) if w and w != 'neutral')


def look_at(context, looker, what):
    """What `looker` sees of the person, thing or place named: ('person' | 'thing' | 'place', its
    name, the facts in a few lines). None when `what` names none of them in this world (the look
    is then a look around)."""
    said, here = _plain(what), stage.here(looker)
    if not said or said in ('around', 'about', 'all around'):
        return None
    for c in context.everyone():
        if c is looker or not world.on(c) or stage.remote(c) or _plain(c.name) != said:
            continue
        there, body = stage.here(c), getattr(c, 'body', None)
        if not world.sees(context, here, there):
            return 'person', c.name, f'{c.name} is not in sight from here.'
        if not world.together(context, here, there):
            carried = '; '.join(body._held_words(tuple(body.cfg.get('private_items', ())))) if body is not None else ''
            return 'person', c.name, f"{c.name}, {world.away(context, here, there)}" + (f": {'; '.join(body.bearing)}" if body is not None and body.bearing else '') \
                + (f'; {carried}' if carried else '') + '. Too far to make out more.'
        lines = [f"{c.name}: {getattr(c, 'looks', None) or 'close by'}."]
        if body is not None:
            lines.append(f'To the eye: {body.visible()}.')
        if _seems(c):
            lines.append(f'Seems: {_seems(c)}.')
        return 'person', c.name, '\n'.join(lines)
    book = ledger.of(context)
    for thing in book.things:
        if _plain(thing['name']) != said or thing.get('hidden'):
            continue
        holder = next((c for c in context.everyone() if c.name == thing.get('holder')), None)
        at = stage.here(holder) if holder is not None and world.on(holder) else thing.get('at')
        if at is None:
            continue
        mine = holder is looker
        near = mine or (world.together(context, here, at) if holder is not None else survival.at_hand(context, at, here))
        if holder is not None and not mine and str(thing['name']).startswith(tuple(holder.body.cfg.get('private_items', ())) or ('\0',)):
            continue   # what someone keeps to themselves is not looked over
        if near:
            lines = [survival.named(context, thing, reader=looker) + (f', held by {holder.name}' if holder is not None and not mine else ', in your hand' if mine else f', lying {world.where(context, at)}') + '.']
            marks = [m['words'] for m in book.marks if str(m.get('on') or '').lower() == thing['name'].lower()]
            if marks:
                lines.append('On it: ' + '; '.join(marks) + '.')
            words = survival.writing_of(context, thing['name'])
            if words:   # read where it is one's own or lies here; in another's hand only that it is written on
                lines.append(f'It says: {words}' if mine or holder is None else 'It has writing on it.')
            return 'thing', thing['name'], '\n'.join(lines)
        if world.sees(context, here, at, 1.0):
            return 'thing', thing['name'], f"{thing['name']}, " + (f'held by {holder.name}, ' if holder is not None else 'lying ') + f"{world.away(context, here, at)}. Too far to make out more."
    place = world.called(context, said.replace(' ', '_')) or world.called(context, said)
    if place is not None:
        if world.at_place(context, here, place):
            there = [c.name for c in context.everyone() if c is not looker and world.on(c) and not stage.remote(c) and world.at_place(context, stage.here(c), place)]
            lying = [survival.named(context, t, reader=looker) for t in survival.lying(context, place.xy)]
            lines = [f"{place.id}: {place.description or place.kind}" + (', walled' + (' and standing open' if place.open else '') if place.enclosed else '')
                     + (', a shelter' if place.shelter else '') + '.',
                     'Here: ' + (', '.join(there) or 'nobody else') + '.', 'Lying here: ' + ('; '.join(lying) or 'nothing') + '.']
            traces = survival.marks_note(context, place.xy, reader=looker)
            if traces:
                lines.append('Traces: ' + '; '.join(traces) + '.')
            return 'place', place.id, '\n'.join(lines)
        if not world.in_sight(context, here, place):
            return 'place', place.id, f'{place.id} is not in sight from here.'
        seen = [c.name for c in context.everyone() if c is not looker and world.on(c) and world.at_place(context, stage.here(c), place) and world.sees(context, here, stage.here(c))]
        return 'place', place.id, (f"{place.id}: {place.description or place.kind}, {world.away(context, here, place.xy)}."
                                   + (' Its inside cannot be seen from here.' if place.enclosed and not place.open else '')
                                   + (f" There: {', '.join(seen)}." if seen else ' Nobody is to be seen there.'))
    return None


def situation(context, lately='', viewer=None):
    """The Situation every prompt reads, in a world with mechanics: where and when things stand,
    from the world itself. It was free text that each narration added to and an LLM rewrote
    after every scene, so whatever a narrator once wrote was the situation from then on.
    viewer: a character. The situation is then as that character can know it: itself and those
    with it, and of another body only what shows (Body.visible), not its thirst or hunger."""
    cast = [c for c in list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', [])) if world.on(c)]
    if viewer is not None:
        in_scene = getattr(context, 'current_scene', None) is not None
        with_it = viewer.bystanders() if in_scene else [c for c in cast if c is not viewer]
        radio = [c for c in viewer.hearers() if c not in with_it] if in_scene and hasattr(viewer, 'hearers') else []
        cast = [viewer] + [c for c in with_it if world.on(c)]
    else:
        radio = []
    lines = [f'It is now {stage.clock(context, date=True)}.',
             'Where everyone is: ' + '; '.join(f'{c.name} {stage.remote(c) or stage.where(context, stage.here(c))}' for c in cast) + '.']
    if radio:   # heard over the radio, not seen: no body to see
        lines.append('On the radio with you, not here in person: ' + '; '.join(
            f'{c.name} ({stage.remote(c) or stage.where(context, stage.here(c))})' for c in radio) + '.')
    seen = lambda c: c.body.summary() if c is viewer else c.body.visible() if viewer is not None else c.body.summary(public=True)
    bodies = [f'{c.name}: {seen(c)}' for c in cast if getattr(c, 'body', None) is not None]
    if bodies:
        lines.append(('You, and what can be seen of the others: ' if viewer is not None else 'Bodies and what each carries: ') + '; '.join(bodies) + '.')
    if lately:
        lines.append(f'Lately: {lately}')
    return '\n'.join(lines)


def check(context, actor, action, facts, narration, people=None):
    """The claims in `narration` that the record does not support: [(kind, words)]. None when
    the check itself failed (the narration then stands: a broken check must not silence the play)."""
    reply = context.llm.ask({'name': actor.name, 'action': action, 'facts': facts or 'none', 'narration': narration,
                             # what was just settled is part of the record (a boar that Settled drives off is no longer under Danger)
                             'record': of(context, actor, people) + f"\nJust settled, and true: {facts or 'nothing'}"},
                            [UserMessage(content=CHECK)], tag='record.check', temp=0.1, stops=['</end>'], max_tokens=220)
    if reply is None:
        return None
    found = json_reply.parse(reply)
    if not isinstance(found, dict):
        trace.event('check_unreadable', of='narration', reply=reply[:300])
        return None   # unreadable: the narration stands (a broken check must not silence the play)
    return [(f['kind'], f['words'].strip(' "\'')) for f in json_reply.items(found, 'findings', KINDS)]


PROPOSE = """A simulated world keeps track of certain kinds of thing. A text has brought in something the world does not hold yet. Put each new thing in the world's own terms.

<world>
Kinds of place where something can be done: {{$places}}
Things its actions use: {{$items}}
Bodily states it tracks: {{$needs}}
People present: {{$people}}
Things it already has, each one of a kind, and where: {{$known}}
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
#thing <n> absent
#thing <n> none
When the text means one of the things the world already has, give that thing's name exactly as listed, wherever the text puts it.
A prop is a thing nobody acts on through the world's rules: a tool, a paper, a cup, a phone, a rope. It is ordinary when it would be found at such a place, or on such a person, without remark (a sickle on a farm, a phone in an office, a pen in a pocket). It is unusual when it is a find that changes what the characters can do, and always when it is a weapon, a medicine, a means of making fire or light, or a means of calling for help (a knife or a radio in a forest, a cache of supplies: one line for each thing in it).
Use 'place' for anything that gives water, food, shelter or rest, whatever it is called. Use 'absent' when the text does not put the thing here at all: it is only spoken of, remembered, wished for or compared to, or it belongs to someone's past or to another place (the lamp her mother kept, the knife he left at home). A thing somebody in the text holds, hands over, finds or uses with their hands is here, and is not 'absent'. Use 'none' when the entry is not a thing or a bodily state at all, or fits no form.
</end>
"""

def known_things(context):
    """The named things the world holds (props lying about, and the play's clues wherever they
    are now), with where each is: so that a text's 'brass key' is taken for the planted
    "child's house key" and not made a second time."""
    names = [p['name'] for p in getattr(context, 'props', [])]
    screenplay = getattr(context, 'screenplay', None)
    names += [re.sub(r'^(a|an|the|some)\s+', '', c['thing'].strip(), flags=re.I) for c in (getattr(screenplay, 'clues', None) or [])]
    listed = []
    for name in dict.fromkeys(names):
        where = survival.whereabouts_of(context, name)
        if where:
            listed.append(f'{name} ({where})')
    return '; '.join(listed) or 'none'


FOUND = ('item', 'condition')     # the kinds of finding that may be a new thing, not a mistake


def admit(context, actor, claims, text, source, people=None, at=None):
    """Sort what a check found. A new thing is put in the world's terms and, if its source may
    bring it in, made real. Returns (facts, refused, remaining): the facts now true, the facts
    of what was refused, and the claims still to be written out of the text.
    source: 'director' (the director's own: an intervention), 'plan' (a scene's set-up: only what
    is ordinary there comes in) or 'deed' (a character's act, the narrator: nothing new comes in)."""
    people = [c for c in (people or [actor] + actor.bystanders()) if getattr(c, 'body', None) is not None]
    facts, refused = [], []
    # a thing lying within reach that the narration has the actor take: the world takes it too
    # (Joe's search found the scarf, the narration had him pick it up, the world left it lying,
    # and from then on the actors had it in hand and the world on the floor). Out of reach, refused.
    # Only for a deed's narration, where `actor` is the doer: a set-up's check has no doer (its
    # 'actor' is the first of the cast), and a set-up moves nothing (its claim is written out).
    held = [c for c in claims if c[0] == 'holding'] if source == 'deed' else []
    claims = [c for c in claims if c not in held]
    for claim in held:
        words = claim[1].lower()
        said = set(re.findall(r"[a-z0-9']+", words))
        thing = next((p['name'] for p in survival.lying(context, stage.here(actor)) if not p.get('fixed')   # a mark is not carried off
                      and (p['name'].lower() in words or p['name'].lower().split()[-1] in said)), None)
        fact = survival._prop_do(context, actor, 'take', thing, None) if thing else None
        if fact:
            facts.append(fact)
            trace.event('holding_admitted', character=actor.name, thing=thing)
        else:
            refused.append(f"{actor.name} does not have it: it is not lying within reach.")
            claims.append(claim)
    new = [c for c in claims if c[0] in FOUND]
    remaining = [c for c in claims if c[0] not in FOUND]
    if not new:
        return facts, refused, remaining
    needs = []   # the world keeps no needs (2026-11-28)
    places = {p.kind: [] for p in world.places(context)}   # the kinds of place this world has
    reply = context.llm.ask({'places': '; '.join(places) or 'none',
                             'items': 'none', 'needs': ', '.join(needs) or 'none',
                             'people': ', '.join(c.name for c in people), 'text': text,
                             'known': known_things(context),
                             'claims': '\n'.join(f'{i + 1}. {words}' for i, (_, words) in enumerate(new))},
                            [UserMessage(content=PROPOSE)], tag='record.admit', temp=0.1, stops=['</end>'], max_tokens=200) or ''
    at = tuple(at) if at else stage.here(actor)
    person = lambda name: next((c for c in people if c.name.lower() == name.strip().lower()), None)
    seen = set()
    # one line an entry: with \s between the fields a '#thing 1 none' took the next line as its
    # own, and the entry on that line was never read (30 of 290 recorded replies)
    for number, form, rest in re.findall(r'^[ \t]*#thing[ \t]+(\d+)[ \t]+(\w+)[ \t]*\|?[ \t]*(.*?)[ \t]*$', reply, re.M):
        index = int(number) - 1
        if not 0 <= index < len(new):
            continue
        seen.add(index)
        fields = [f.strip() for f in rest.split('|')]
        form, fact, why = form.lower(), '', ''
        if form == 'absent':
            # spoken of, remembered, or somewhere else: nothing is made and nothing is written out
            # (a direction to 'use her grandfather's pink ribbons to ask' put the ribbons of her
            # childhood in Joe's hands, and two scenes were then planned around them)
            continue
        if form in ('prop', 'item') and len(fields) >= 2 and fields[0]:
            name = fields[0].lower()
            holder = person(fields[1].lower().removeprefix('held by')) if 'held by' in fields[1].lower() else None
            usual = form == 'prop' and len(fields) > 2 and fields[2].lower().startswith('ordinary')
            if form == 'item':
                form = 'prop'   # the world lists no things of its own: a prop, and from a deed an unusual one
            same = lambda held: held.lower() == name or held.lower().endswith(' ' + name) or name.endswith(' ' + held.lower())
            holders = [c for c in people if any(same(i) for i in c.body.inventory)]
            lying_here = any(same(p['name']) for p in survival.lying(context, at))
            if holder is not None and holders and holder not in holders:
                # it is here, but not where the text puts it ('the rest stays in the can with Lanya'
                # when the can had gone to Kidd): that is not a new thing, and it is not so
                refused.append(f"{', '.join(c.name for c in holders)} has the {name}; {holder.name} does not.")
                if new[index] not in remaining:
                    remaining.append(new[index])
                continue
            if holders or lying_here:
                continue   # it is already here (Joe 'releases the branch' he is carrying): nothing new, nothing refused
            elsewhere = survival.whereabouts_of(context, name) if form == 'prop' else ''
            if elsewhere:
                # the world has it, somewhere else: not a new one (a set-up put the planted key in
                # the mud at Spring2 while it lay at Spring3)
                why = f"The {name} is {elsewhere}, not here."
            elif source == 'director' or (source == 'plan' and (usual or form == 'item')):
                fact = survival.add_thing(context, name, holder=holder, at=at)
            elif source == 'plan':
                # A scene's set-up is the planner's, and the planner is not the director: it may
                # set out what such a place or person would have without remark, and no more. (A
                # set-up put the rain gauge of Joe's childhood in his hands in the forest, and
                # made 'the dry strip' a thing Mara held: lost-20261119-160400.)
                why = f"There is no {name} here: nothing has brought one onto the stage."
            else:
                # Once a scene is set, nothing new comes onto the stage because a narration says
                # so, however ordinary: what a deed needs that the stage lacks goes to the keeper
                # before the deed is played (sim/keeper.py), and is there or is not.
                why = f"There is no {name} here to be had."
        elif form == 'place' and len(fields) >= 2 and source in ('director', 'plan'):
            fact = survival.add_place(context, next((k for k in places if k.lower() == fields[0].lower()), ''), fields[1], at)[1]
        elif form == 'place' and len(fields) >= 2:
            why = f"There is no {fields[1].lower()} here."
        elif form == 'body' and len(fields) >= 4 and person(fields[0]) is not None:
            who = person(fields[0])
            # a plan does not decide bodies either (survival.PLAN_RULE): set-ups that had someone
            # 'still desperately thirsty despite this morning's drink' made it so, and the scene
            # went on another drink
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


def narration_noted(context, actor, action, narration, of='look'):
    """`narration` as it stands: what the world's narrator wrote (of a look) is told as written. What the check
    finds in it that the record does not hold is written to the run's error log (sim/errors.py): the stage is not
    changed by it, and a deed that leans on it meets the stage as it is. (It was written again and cut.)"""
    if not narration or not survival.config(context) or getattr(actor, 'body', None) is None:
        return narration
    claims = check(context, actor, action, None, narration)
    if claims:
        from sim import errors
        errors.log('record.check', 'narration against the record', of=of, character=actor.name,
                   claims=[f'{k}: {w}' for k, w in claims], narration=narration[:400])
    trace.event('narration_check', character=actor.name, of=of, outcome='noted' if claims else 'ok' if claims is not None else 'unchecked',
                claims=[f'{k}: {w}' for k, w in claims or []])
    return narration


# --------------------------------------------------------------------------- speech
# A character may be wrong, may misremember and may lie: a line is said as its actor wrote it (2026-11-28: nothing in
# a performance refuses). What the ruling of the line finds in it that the stage or the speaker's own record has
# otherwise ('Here. Clean water.' with empty hands) is an error of the actor's model, written to the run's error log
# (sim/errors.py), and changes nothing: the line is heard as said, and only what the stage holds is handed over. Words
# written into a line that are not spoken ('She snorts.', '*pats her pocket*') are taken out of it: they are not speech.
def speech(context, speaker, line, listener=None, deed=''):
    """A spoken line as it is said: (line, give). `give` is (thing, person) when the line itself hands over something
    the speaker holds, else None. Traced as 'speech_check'."""
    if (not line or not survival.config(context) or getattr(speaker, 'body', None) is None
            or not world.on(speaker)):
        return line, None
    people = [speaker] + [c for c in speaker.bystanders() if c is not speaker]
    if listener is not None and listener not in people and world.on(listener):
        people.append(listener)
    read = survival.read_by(context, speaker)
    speaker.said = None
    # the line is resolved as a deed is (sim/resolver.py resolve_words): what it hands over,
    # what it states that is not so, what it tells of the speaker's life and what it promises
    from sim import errors, resolver
    found = resolver.resolve_words(context, speaker, line, listener, deed=deed, about=of(context, speaker, people, full=True)
                                   + (f"\nWritings {speaker.name} has read: " + read.replace('\n', '; ') if read else ''))
    if found is None:
        errors.log('stage.resolve_say', 'answer could not be read', speaker=speaker.name, line=line[:300])
        return line, None   # unreadable: the line stands as written
    speaker.said = found   # Character.acts applies it once the line stands
    findings = [f for f in found.get('findings') or [] if isinstance(f, dict) and f.get('kind') in SPEECH_KINDS]
    for f in findings:
        if f['kind'] != 'aloud':
            errors.log('stage.resolve_say', 'line against the stage', speaker=speaker.name, finding=f['kind'],
                       words=str(f.get('words') or '')[:200], why=str(f.get('why') or '')[:200])
    said = line
    unspoken = [f for f in findings if f['kind'] == 'aloud']
    put_right = str(found.get('line') or '').strip().strip('"“”').strip()
    if unspoken and put_right:
        said = put_right   # the words with what is not spoken taken out
    # A name told in a line is known from then on to everyone who heard it: one the speaker itself knows, and that is
    # in the line as it is said. (Joe gave his name, the check missed it, and Samantha 'did not know' the name she used
    # all night: lost-20261119-160400.)
    named = {str(n).strip().lower() for n in (found.get('names') or []) if isinstance(n, str)}
    words = set(re.findall(r"[a-z][a-z'-]*", re.sub(r"['’]s\b", '', said.lower())))
    roster = list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', []))
    told = {c.name for c in roster if c.name.lower() in words or c.name.lower() in named}
    told -= set(getattr(speaker, 'unknown_names', None) or ())
    for hearer in people:
        if hearer is not speaker and getattr(hearer, 'unknown_names', None):
            learned = hearer.unknown_names & told
            if learned:
                hearer.unknown_names -= learned
                trace.event('name_learned', character=hearer.name, names=sorted(learned), said_by=speaker.name)
    give = None
    handed = next((e for e in found['effects'] if e.get('thing') and e.get('to')), None)
    if handed:
        name = str(handed.get('thing') or '').strip().lower()
        thing = next((i for i in speaker.body.inventory if i.lower() == name), None) \
            or next((i for i in speaker.body.inventory if name and (name in i.lower() or i.lower() in name)), None)
        person = next((c for c in people if c is not speaker and c.name.lower() == str(handed.get('to') or '').strip().lower()), None)
        give = (thing, person) if thing and person is not None else None
    trace.event('speech_check', character=speaker.name, outcome='unspoken taken out' if said != line else 'ok',
                findings=[f"{f['kind']}: {f.get('words')}" for f in findings], before=line, after=said, gives=give[0] if give else None)
    return said, give


SPEECH_KINDS = ('aloud', 'holds', 'body', 'where', 'here', 'time', 'done', 'gender', 'knows', 'writing', 'whose', 'past', 'stands')


# --------------------------------------------------------------------------- deeds
# A Do is resolved once it is chosen (sim/resolver.py): what it does when it is tried. Nothing is refused: what comes
# off is in its effects, what does not is on its page as a try. (Deeds were refused, sent back and written again; what
# a refusal left behind, in the record, in notes to planners and in the actor's prompt, was a second account of the
# world beside the stage, and the play was built on it: bellona-20261127-142348, lost-20261125-154614.)
def _mode(act):
    return str(getattr(act, 'mode', '')).split('.')[-1]


def _is_do(act):
    return _mode(act) == 'Do'


def gone(act):
    """The characters an act is aimed at who have left the scene."""
    targets = act.target if isinstance(getattr(act, 'target', None), (list, tuple)) else [getattr(act, 'target', None)]
    return [t for t in targets if t is not None and hasattr(t, 'name') and stage.off_stage(t)]


def judge(context, character, act):
    """Have a Do resolved, once: act.found is what it does when it is tried (sim/resolver.py), act.resolved
    'resolved'. Acts of other modes, and actors with no body, are not resolved."""
    if not _is_do(act) or getattr(act, 'resolved', None) is not None:
        return
    if not survival.config(context) or getattr(character, 'body', None) is None:
        return
    from sim import resolver
    act.found = resolver.resolve(context, character, act)
    act.resolved = 'resolved'
    act.leaves, act.leaves_to = resolver.leaving(context, character, act.found)


def sound_deeds(context, character, acts):
    """The acts an actor offered, without those aimed at someone who has left; all of them when every one is
    (a choice among the actor's own alternatives: none of them is refused)."""
    return [a for a in acts if not (_mode(a) in ('Do', 'Say') and gone(a))] or acts


def ready(context, character, act):
    """The act about to be performed, made ready to play. Who it is aimed at that has left is taken off it: the words
    are said and the deed done where they stand, and nobody gone hears or is reached (logged: the actor wrote it for
    someone the stage no longer has here). A Do is resolved. One whose ruling could not be read at all has nothing to
    show: it stays its actor's own thought."""
    away = gone(act) if _mode(act) in ('Do', 'Say') else []
    if away:
        from sim import errors
        errors.log('act', 'aimed at someone gone', doer=character.name, mode=_mode(act), at=[t.name for t in away], act=str(act.action)[:200])
        act.target = [t for t in act.target if t not in away] if isinstance(act.target, list) else []
    judge(context, character, act)
    found = getattr(act, 'found', None)
    if _is_do(act) and found is not None and not found['page'] and not found['effects']:
        act.mode, act.target, act.unshown, act.unrecorded, act.found = 'Think', [], True, True, None
