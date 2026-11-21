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

from sim import ablation, keeper, ledger, stage, survival
from sim.trace import trace
from utils.Messages import UserMessage
from utils import json_reply

NEARBY_TILES = 10   # places this near are in sight: they may be seen, pointed at, spoken of
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
    people = [c for c in (people or [actor] + actor.bystanders()) if getattr(c, 'mapAgent', None) is not None]
    # those in the scene by radio are heard, not there: listed apart, so that no narration puts
    # them on the spot (Novak, the Earthside flight director, swung a camera on the gantry)
    radio = [c for c in people if c is not actor and stage.remote(c)]
    heard = actor.hearers() if hasattr(actor, 'hearers') else []
    radio += [c for c in heard if c not in people and (stage.remote(c) or stage.remote(actor))]
    people = [c for c in people if c not in radio]
    at = stage.here(actor)
    near = sorted((stage._distance(at, res['location']), rid, res) for rid, res in context.map.resource_registry.items())
    places = [f"{rid} ({'here' if d <= stage.AT_PLACE_TILES else stage._direction(at, res['location']).lower()})"
              for d, rid, res in near if d <= NEARBY_TILES]
    return '\n'.join([
        f'Cast: {cast(context)}',
        f'Time: {stage.clock(context)}',
        'Present: ' + '; '.join(f'{c.name} {stage.remote(c) or stage.where(context, stage.here(c))}' for c in people) + '. Nobody else.',
        *(['On the radio, not here in person: ' + '; '.join(f'{c.name} ({stage.remote(c) or stage.where(context, stage.here(c))})' for c in radio)]
          if radio else []),
        'Places in sight: ' + (', '.join(places[:16]) or 'none named'),
        'Bodies: ' + ('; '.join(f'{c.name}: {c.body.summary(public=True)}' for c in people if getattr(c, 'body', None) is not None)
                      or 'not tracked'),
        'Holdings: ' + ('; '.join(f"{c.name}: {', '.join(survival.named(context, i, reader=None) for i in c.body.inventory) or 'nothing'}" for c in people
                                   if getattr(c, 'body', None) is not None) or 'not tracked'),
        'Lying here, for anyone present to take or use: ' + (', '.join(survival.named(context, p, reader=None) for p in survival.lying(context, at)) or 'nothing'),
        # what a deed left on the place is that deed's doing, told as such: it is not a thing, and
        # nobody else was here to leave it
        *(["Traces left here by deeds of the play (each the doing of the one named, and of nobody else): " + '; '.join(m)]
          if (m := survival.marks_note(context, at)) else []),
        *(['Writings (what these written things say; a character knows it only once it has read it or been told): '
           + w.replace('\n', '; ')] if (w := survival.writings_note(context, at, people)) else []),
        'Stock: ' + (survival.stock(actor) or 'nothing has been used up'),
        'Danger: ' + (danger(actor) or 'none'),
        *(_acts(context, actor, people) if full else []),
    ])


def account(context, reader, about=None, people=None, full=False):
    """How things stand, for whoever reads it: the one way the stage is described.
    reader: a character (what lies where it is and the traces there, each as whose doing it is
    for this reader; what has been told between it and those with it; how what it carries has
    been left); 'stage' (everything at the place of `about`, for the calls that say what an act
    does or check one: of()); 'planner' (where every thing and trace of the play is, what the
    written things say, and what each person has told of their life and to whom).
    The same things and traces, by the same wording (survival.named, marks_note, whose), whoever
    reads: each prompt used to put its own list together, and one of them listed a woman's
    fingermarks as a thing lying at a spring."""
    if reader == 'planner':
        cast = list(people) if people is not None else context.everyone()
        told = [f"{t['who']} has told {', '.join(t['to']) or 'nobody yet'}: {t['fact']}" for t in ledger.of(context).told][-24:]
        parts = [survival.things_note(context, people=cast), survival.writings_note(context, people=cast),
                 ("What each has told of their own life (the one told knows it from then on):\n" + '\n'.join(told)) if told else '']
        return '\n'.join(p for p in parts if p)
    if reader == 'stage':
        return of(context, about, people, full)
    here, body, lines = stage.here(reader), getattr(reader, 'body', None), []
    things = [survival.named(context, p, reader=reader) for p in survival.lying(context, here)]
    if things:   # the actor saw nothing of what lay at its feet but what it carried
        lines.append('Lying here: ' + ', '.join(things))
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
    left = [survival.named(context, i, reader=reader) for i in (body.inventory if body is not None else []) if survival.state_of(context, i)]
    if left:   # how what it carries has been left by handling
        lines.append('Of what you carry: ' + ', '.join(left))
    return '\n'.join(lines)


def _acts(context, actor, people):
    """The rest of the stage (of, full): placing, things held out, exits, refusals, the world's actions."""
    placed = [f"{c.name}: {', '.join(c.body.bearing)}" for c in people if getattr(c, 'body', None) is not None and c.body.bearing]
    held = [f'{holder} holds the {item} out to {to}' for item, holder, to in context.held_out]
    gone = [f"{e['who']}{' with ' + ', '.join(e['with']) if e.get('with') else ''}{' for ' + e['to'] if e.get('to') else ''}"
            for e in getattr(context, 'scene_exits', None) or []]
    refused = [f"{r['deed']} ({'; '.join(r['why']) if isinstance(r['why'], (list, tuple)) else r['why']})"
               for r in getattr(context, 'scene_refusals', None) or [] if r.get('who') == actor.name][-4:]
    offered = [o.strip() for o in survival.offered_actions(actor, cut=None) if o.strip()]
    head = ['How they are placed: ' + ('; '.join(placed) or 'all on their feet, nobody touching'),
            'Held out, for the one named to take: ' + ('; '.join(held) or 'nothing'),
            'Gone out of this scene: ' + ('; '.join(gone) or 'nobody'),
            *([f'Tried by {actor.name} in this scene and not possible here: ' + ' | '.join(refused)] if refused else [])]
    if survival.free(context):
        # no catalogue of actions: what is hidden here for a search to find, and, with a danger
        # on the stage, the answers this world has to it
        hidden = [p['name'] for p in survival.lying(context, stage.here(actor), hidden=True)]
        answers = [o for o in offered if o.startswith('THREAT')]
        return head + ['Hidden here, to be found only by someone who searches this place: ' + (', '.join(hidden) or 'nothing'),
                       *([f'Answers to the danger for {actor.name}:', *[f'- {o}' for o in answers]] if answers else [])]
    return head + [f"The world's actions for {actor.name} (each is 'name@where' or 'name@who', with how far off it is or what it lacks):",
                   *([f'- {o}' for o in offered] or ['- none here'])]


def situation(context, lately='', viewer=None):
    """The Situation every prompt reads, in a world with mechanics: where and when things stand,
    from the world itself. It was free text that each narration added to and an LLM rewrote
    after every scene, so whatever a narrator once wrote was the situation from then on.
    viewer: a character. The situation is then as that character can know it: itself and those
    with it, and of another body only what shows (Body.visible), not its thirst or hunger."""
    cast = [c for c in list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', [])) if getattr(c, 'mapAgent', None) is not None]
    if viewer is not None:
        in_scene = getattr(context, 'current_scene', None) is not None
        with_it = viewer.bystanders() if in_scene else [c for c in cast if c is not viewer]
        radio = [c for c in viewer.hearers() if c not in with_it] if in_scene and hasattr(viewer, 'hearers') else []
        cast = [viewer] + [c for c in with_it if getattr(c, 'mapAgent', None) is not None]
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
    threat = getattr(context, 'active_threat', None)
    if viewer is None or threat is None or viewer.name in threat['targets']:
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
    places = survival.place_kinds(context)
    cfg = survival.config(context) or {}
    needs = [n for n in (people[0].body.levels if people else {})]
    reply = context.llm.ask({'places': '; '.join(f"{k} ({', '.join(v)})" for k, v in places.items()) or 'none',
                             'items': ', '.join(survival.item_kinds(context)) or 'none', 'needs': ', '.join(needs) or 'none',
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
            if form == 'item' and name not in survival.item_kinds(context):
                form = 'prop'   # not one of the world's own: a prop, and from a deed an unusual one
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
            need = fields[1].lower()
            need = survival._harm(cfg)['name'] if need in ('injury', 'harm', 'hurt', 'wound') else need
            felt = who.body.levels.get(need, 0.0) >= survival.NEED_FELT
            if felt == fields[2].lower().startswith('worse'):
                continue   # the body already shows it (a 'bruised thigh' on a bruised body made it 'badly hurt')
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


SETUP_DEEDS = """A scene of an improvised play opens with a set-up; then the actors play it. The set-up says only how the scene opens: where the cast is, what is there, what presses on them.

<setup>
{{$setup}}
</setup>

<what_the_scene_is_for>
{{$purpose}}
</what_the_scene_is_for>

Which words of the set-up have a character already do something, so that it is played before the actors begin? A deed is taking, finding, giving, opening, building, barring, eating, deciding, agreeing, or saying something. Being somewhere, wanting, needing, carrying what one carries, watching, keeping a distance, or a thing lying there to be found are not deeds; nor is what was done before the scene opens ('they have drunk their fill', 'having walked here').
Answer in JSON only: {"deeds": ["<the set-up's own words for each deed, exactly as written>"]}, or {"deeds": []} if there are none.
</end>
"""


def setup_deeds(context, setup, purpose):
    """The words of a scene's set-up that already play part of the scene, [str]; [] for none or
    when the check could not be read. (Told the set-up is the opening only, the stage manager
    still wrote 'she takes the scarf' and 'Joe kneels to build the fire' into it, and the scene
    then began with them done.)"""
    if not (setup or '').strip():
        return []
    from utils import json_reply
    reply = context.llm.ask({'setup': setup, 'purpose': purpose or 'not said'}, [UserMessage(content=SETUP_DEEDS)],
                            tag='record.setup_deeds', temp=0.1, stops=['</end>'], max_tokens=200)
    found = json_reply.parse(reply) if reply else None
    deeds = found.get('deeds') if isinstance(found, dict) else None
    return [d.strip() for d in deeds if isinstance(d, str) and d.strip()] if isinstance(deeds, list) else []


def strip(text, claims):
    """`text` without the sentences that carry one of `claims` (a claim found in no sentence is
    let go: this is for a plan's fields, each checked against claims found in all of them)."""
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if s]
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
    made, refused, remaining = ([], [], claims) if kind == 'look' else admit(context, actor, claims, f'{action}\n{narration}', source)
    if not remaining:   # everything found was a new thing, and is now real
        trace.event('narration_check', character=actor.name, of=kind, outcome='admitted',
                    claims=[f'{k}: {w}' for k, w in claims], made=made, before=narration, after=narration)
        return Settled(narration, made)
    settled = ' '.join(filter(None, [facts or ''] + made + refused))
    # who is he and who is she is for the rewrite only: it is not a fact about what happened
    so = refused + ([f'Cast: {cast(context)}.'] if any(k == 'gender' for k, _ in remaining) else [])
    again = rewrite(rejection(remaining, made, so, settled=re.sub(survival.BODY_NOTE, '', facts or '').strip()))
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
RESAY = """A character in a play was about to say the line below, but part of it is untrue of the speaker at this moment, and the character did not mean to lie.

Speaker: {{$name}}
<about {{$name}}, now>
{{$record}}
</about>

<line>
{{$line}}
</line>

Not so, or not {{$name}}'s to know:
{{$claims}}
(Who is who: {{$cast}})
What {{$name}} does not know:
{{$unknown}}
A thing {{$name}} does not know may be asked about, or guessed at as a guess; a name not yet heard is not used ('you', or nothing).

Say the line again as {{$name}} would, in the same voice, in the first person, and no longer than it was. Keep everything else it says and means; leave out what is not so.
- Do not announce the correction: never "It's morning", "I'm not up the tree", "I have no water". Just do not say the untrue thing.
- Do not move the untrue thing onto someone else.
- If the point of the line was the untrue part, say what {{$name}} can truly say toward the same end: an offer of water one does not have becomes an offer to fetch it; a claim to hold a branch becomes going to get one; a warning that it is nearly dark becomes a warning that dark will come.
Respond with the spoken words only.
</end>
"""

# 'doing' is asked for but not acted on: the world does not keep what a character is doing with
# itself (scanning, holding a position), so nothing can show such a line wrong; and a spoken
# deed that is one of the world's actions ("I'm drinking") is made true instead (survival.spoken_deed)
SPEECH_KINDS = ('aloud', 'holds', 'body', 'where', 'time', 'done', 'gender', 'knows', 'writing', 'whose', 'past')


def speech_claims(context, speaker, found):
    """The findings of a speech check: [(kind, words)]. The model decides; nothing here second-
    guesses it by the words it quoted. (Word tests here threw out correct findings: "Here, take
    one" was kept because the line said "one" and the speaker held a notebook; "Morning" said
    at 02:26 because a greeting matched no phrase for the hour.)"""
    return [(f['kind'], f['words'].strip(' "\'')) for f in json_reply.items(found, 'findings', SPEECH_KINDS)]


def speech(context, speaker, line, listener=None, deed=''):
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
    missing = speaker.does_not_know() if hasattr(speaker, 'does_not_know') else []
    unknown = '\n'.join(f'- {m}' for m in missing) or 'nothing listed'
    read = survival.read_by(context, speaker)
    if read and 'Writings' not in about:
        about += '\nWritings (what the written things the speaker has read say): ' + read.replace('\n', '; ')
    speaker.said = None
    # the line is resolved as a deed is (sim/resolver.py resolve_words): what it hands over,
    # what it states that is not so, what it tells of the speaker's life and what it promises
    from sim import resolver
    found = resolver.resolve_words(context, speaker, line, listener, deed=deed, about=of(context, speaker, people, full=True)
                                   + (f"\nWritings {speaker.name} has read: " + read.replace('\n', '; ') if read else ''))
    if found is None:
        trace.event('check_unreadable', of='speech', reply='')
        return line, None   # unreadable: the line stands as written
    handed = next((e for e in found['effects'] if e.get('thing') and e.get('to')), None)
    found['give'] = {'thing': handed['thing'], 'to': handed['to']} if handed else None
    speaker.said = found   # Character.acts applies it once the line stands
    # A name told in a line is known from then on to everyone who heard it: one the speaker itself
    # knows, and that is in the line as it is finally said. (Joe gave his name, the check missed
    # it, and Samantha 'did not know' the name she used all night, until one use of it was cut to
    # 'You.'; and Joe's use of a name he had not heard, cut from his line, still taught it to
    # Samantha: lost-20261119-160400.)
    said = {str(n).strip().lower() for n in (found.get('names') or []) if isinstance(n, str)}

    def heard_names(final):
        words = set(re.findall(r"[a-z][a-z'-]*", re.sub(r"['’]s\b", '', (final or '').lower())))
        roster = list(context.actors) + list(getattr(context, 'extras', [])) + list(getattr(context, 'npcs', []))
        told = {c.name for c in roster if c.name.lower() in words or (c.name.lower() in said and final == line)}
        told -= set(getattr(speaker, 'unknown_names', None) or ())
        for hearer in people:
            if hearer is not speaker and getattr(hearer, 'unknown_names', None):
                learned = hearer.unknown_names & told
                if learned:
                    hearer.unknown_names -= learned
                    trace.event('name_learned', character=hearer.name, names=sorted(learned), said_by=speaker.name)
    claims = speech_claims(context, speaker, found)
    give = None
    handed = found.get('give') if isinstance(found.get('give'), dict) else None
    if handed and not claims:
        named = str(handed.get('thing') or '').strip().lower()
        thing = next((i for i in speaker.body.inventory if i.lower() == named), None) \
            or next((i for i in speaker.body.inventory if named and (named in i.lower() or i.lower() in named)), None)
        person = next((c for c in people if c is not speaker and c.name.lower() == str(handed.get('to') or '').strip().lower()), None)
        give = (thing, person) if thing and person is not None else None
    if not claims:
        heard_names(line)
        trace.event('speech_check', character=speaker.name, outcome='ok', claims=[], gives=give[0] if give else None)
        return line, give
    # the line will not stand as written: what was read of its words is not applied; a deed done
    # beside it is, whatever becomes of the words. A line that was only written with what is not
    # spoken in it ('She snorts.') says what it said: what it told and promised stands.
    if any(kind != 'aloud' for kind, _ in claims):
        speaker.said = {**found, 'tells': [], 'promises': [], 'retold': ''} if deed and isinstance(speaker.said, dict) else None
    put_right = str(found.get('line') or '').strip().strip('"“”').strip()
    if put_right and put_right != line:
        # the call that found what is not so, with the stage before it, also says the line as it
        # can truly be said (a second, smaller call told only the words to avoid kept 'Somebody
        # knelt here before us' two times in three, having dropped 'a stranger')
        trace.event('speech_check', character=speaker.name, outcome='resaid', claims=[f'{k}: {w}' for k, w in claims], before=line, after=put_right)
        heard_names(put_right)
        return put_right, None
    # with what the stage shows instead, where the finding gave it ('Samantha left the fingermarks at 08:01')
    whys = {str(f.get('words') or '').strip(' "\''): str(f.get('why') or '').strip() for f in found.get('findings') or [] if isinstance(f, dict)}
    again = context.llm.ask({'name': speaker.name, 'record': about, 'line': line, 'cast': cast(context), 'unknown': unknown,
                             'claims': '\n'.join(f'- {w}' + (f' (in fact: {whys[w]})' if whys.get(w) else '') for _, w in claims)}, [UserMessage(content=RESAY)],
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
    heard_names(result)
    return result, None



# --------------------------------------------------------------------------- deeds
# A Do is judged before it can be chosen (survival.classify_deed): it is one of the actions the
# world or the stage offers, or it is not done. One that is not is written again once from what
# can be done (mend_deed), and failing that is not played. (Deeds were checked by lists of verbs
# against the things the world names: 'Joe has no rope' turned a rope handed over into 'pulls the
# branch from his belt', and what no list caught was the narrator's to make true.)
def _mode(act):
    return str(getattr(act, 'mode', '')).split('.')[-1]


def _is_do(act):
    return _mode(act) == 'Do'


def gone(act):
    """The characters an act is aimed at who have left the scene."""
    targets = act.target if isinstance(getattr(act, 'target', None), (list, tuple)) else [getattr(act, 'target', None)]
    return [t for t in targets if t is not None and hasattr(t, 'name') and stage.off_stage(t)]


def judge(context, character, act):
    """Have a Do judged, once, before it can be chosen: act.resolved is what it is made of
    ('drink@Spring1', 'kneel + take the arm of@Joe', 'passing'), or '' when it cannot be done,
    with act.refusal saying why. Acts of other modes, and worlds with no rules, are not judged."""
    if not _is_do(act) or getattr(act, 'resolved', None) is not None:
        return
    if not survival.config(context) or getattr(character, 'body', None) is None:
        return
    # one call, shown the whole stage, says what the deed does (sim/resolver.py); it is played
    # as found (Context.do), and whoever it takes out of the scene is the resolver's to say
    from sim import resolver
    act.found = resolver.resolve(context, character, act)
    act.resolved, act.refusal = ('resolved', '') if act.found['can'] else ('', act.found['why'] or 'it cannot be done here')
    if act.found['can']:
        act.leaves, act.leaves_to = resolver.leaving(context, character, act.found)
    else:
        act.found = None


def _unsound(context, character, act):
    """Why this act cannot be done as written: [] when it can."""
    lacks = [f'{t.name} has left and is not here' for t in gone(act)] if _mode(act) in ('Do', 'Say') else []
    if _is_do(act):
        judge(context, character, act)
        if getattr(act, 'resolved', None) == '':
            lacks.append(act.refusal or 'no action here covers it')
    return lacks


def sound_deeds(context, character, acts):
    """The acts an actor offered, without the Dos that rest on what is not so; all of them when
    none is sound (the chosen one is then mended)."""
    if not ablation.enabled('narration_gate'):
        return acts
    # a deed is resolved once it is chosen, not before (Character.step_task): resolved as a
    # candidate, eleven of nineteen answers in one act were for deeds never played
    return [a for a in acts if not (_mode(a) in ('Do', 'Say') and gone(a))] or acts


# What the stage has refused a character in the scene being played: kept so that the actor is
# told and does not try the same thing in other words, and so that a beat whose deed is refused
# twice is given up. (Each try was judged alone: one man tried five wordings of banking a fire
# in one scene, and the scene could not end while he did.)
BEAT_REFUSALS = 2   # chosen deeds refused in one beat before the beat is given up


def refused(context, character, deed, why, chosen=False):
    task = character.focus_task.peek() if getattr(character, 'focus_task', None) is not None else None
    context.__dict__.setdefault('scene_refusals', []).append(
        {'who': character.name, 'deed': ' '.join(str(deed).split())[:200], 'why': '; '.join(why) if isinstance(why, (list, tuple)) else str(why),
         'task': getattr(task, 'id', None), 'chosen': chosen})


def refusals_note(character):
    """For the actor: what it has tried in this scene that the stage does not allow."""
    mine = [r for r in getattr(character.context, 'scene_refusals', []) if r['who'] == character.name]
    if not mine:
        return ''
    return ("Tried by you in this scene and not possible here: " + ' | '.join(f"{r['deed']} ({r['why']})" for r in mine[-4:])
            + ". None of these can be done here in any wording: do not try them again or work toward them; do something else, or say it in words.")


def beat_refused(character, task):
    """Has the stage refused the deed this character chose, twice, in this beat?"""
    return len([r for r in getattr(character.context, 'scene_refusals', [])
                if r['chosen'] and r['who'] == character.name and r['task'] == getattr(task, 'id', None)]) >= BEAT_REFUSALS


def mend_deed(context, character, act):
    """The act about to be performed, made sound: a Do that rests on what is not so is written
    again from what is. The act is changed in place; returns True when it was."""
    if not ablation.enabled('narration_gate'):
        return False
    lacks = _unsound(context, character, act)
    if not lacks:
        return False
    if _mode(act) == 'Say':
        # words for someone who has gone are not said to an empty chair: they are thought
        trace.event('deed_check', character=character.name, outcome='unsaid', lacks=lacks, before=act.action)
        # kept as the character's own thought, not shown: a line in quotes as a thought read as
        # speech nobody heard (..."Aight, I'm down"...)
        act.mode, act.target, act.unshown = 'Think', [], True
        return True
    trace.event('deed_refused', character=character.name, deed=act.action[:300], why=lacks, label=getattr(act, 'affordance', None), again=False)
    refused(context, character, act.action, lacks, chosen=True)
    # the actor is told why and goes another way with its next act (nothing is written again for it)
    again = ''
    again = (again or '').strip().strip('"“”').strip()
    before = act.action
    if again and gone(act):
        act.target = [t for t in act.target if t not in gone(act)] if isinstance(act.target, list) else []
    if again:
        act.action, act.affordance, act.resolved = again, None, None
        left = _unsound(context, character, act)
    else:
        left = lacks
    outcome = 'redone' if again and not left else 'unfixed'
    trace.event('deed_check', character=character.name, outcome=outcome, lacks=lacks, before=before, after=again)
    if outcome == 'unfixed':
        # Sent back once and still not something this world lets a body do: it is not done, and
        # not shown. The actor is told, once, so that the next act goes another way. (A deed
        # that could not be done was played, then taken back on the page: a shirt held out,
        # then 'her fist closes on air'.)
        trace.event('deed_refused', character=character.name, deed=(again or before)[:300], why=left, label=None, again=True)
        character.add_perceptual_input(f"You meant to do this and found you could not, here: {before} ({'; '.join(left)}). "
                                       f"Go at it another way: in words, or by one of the acts open to you.", percept=False, mode='internal')
        act.mode, act.target, act.unshown = 'Think', [], True
        act.action = f"I cannot do that here: {before}"
    return True
