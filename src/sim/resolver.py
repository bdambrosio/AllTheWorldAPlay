"""What a deed does, said once by one call that is shown the whole stage (2026-11-20).

A deed used to exist twice: as the actor's words, which the audience was shown, and as a list of
labels a judge matched to it, which the world applied. The two were written by different calls and
never set side by side: a judge's 'show leaf@Kidd' matched no listed label and a plain pick-up was
refused; a deed that picked up a strip lying at another place was passed as 'kneel + examine' and
shown as written; 'leaves with Joe' was read by one of three code paths that dropped Joe. Every
one of those was a model that had the situation right and code that lost it at a label, a field or
a pattern.

Here one call is given the stage as it stands (record.of), the scene so far and the deed, and says
what the deed does as effects in the ledger's own terms, with the line the audience sees. Code
does not read the deed's words, and does not judge the deed a second time. It checks only that each
name in the effects is the name of something (a thing on the stage, a person in the scene, one of
the world's actions) and applies them; what
changes a body's needs or finds what a place holds is still the world's own action, applied by
survival._apply. The page is written from what was done, so the two cannot differ.

Tested offline on 225 recorded deeds (logs/review-20261120/resolver): Claude Opus 5.5 gave 409 of
409 well-formed answers, no thing from nothing, no thing taken from another's hands, about two
seconds a call."""
from __future__ import annotations

import re

from sim import ablation, ledger, stage, survival
from sim.trace import trace
from utils import json_reply
from utils.Messages import UserMessage

RESOLVE = """You keep the physical world of an improvised play. A character is about to do a deed. Say what the deed does in this world as it now stands.

<world>
{world}
</world>

<stage>
{stage}
</stage>

<the_scene_so_far>
{scene}
</the_scene_so_far>

{name} is about to:
<deed>
{deed}
</deed>
{leaves}
How this world works:
- It is the world the setting describes. What an ordinary person could do there with their body and what is at hand can be done, though it take time and effort (a fire worked up from dry wood, a shelter of boughs); what they could not, cannot, however the deed is worded.
- The stage above is the whole truth of who holds what and what lies here. A deed can use only what its doer holds, what lies here, what another person here gives or holds out, and what this place gives. A thing that lies at another place, however near, is not here: nobody picks it up from where they stand.
- Ordinary handling of a thing that is here needs no leave: pick it up, put it down, hand it over, hold it out or up to be seen, pocket it, open, shut, fold, tear, wipe or turn it, set it on something, read it. Each is an effect: say it.
- What changes a body's needs (eating, drinking, resting, tending a wound) and what finds things a place holds (searching it, picking from it) happens only by one of the world's actions listed on the stage, named exactly as listed. An action marked as lacking something cannot be done. With a danger on the stage, what is done about it is one of the answers listed for it.
- Putting a thing on a person (a coat round someone, a strip of cloth bound round a wound) leaves it on them, worn: say it as on. Taking off what is worn puts it in a hand.
- A touch is done as it is written: taking a hand or an arm, a hand on a shoulder, pressing a thing into someone's hand (the thing is then theirs to keep or give back). The one touched may draw away afterwards; that is theirs to play.
- Nobody is moved, held back, stopped or searched against their will. To lead, draw, pull or beckon someone along is to ask: they answer for themselves, and where they stand or go is no effect of this deed. A thing another person holds comes away only if they give it or have held it out.
- Nothing comes from nothing: a new thing is one this place gives (by a listed action), or is made from things that are here.
- A look, a gesture, a step or two, a change of posture changes nothing that lasts, except the posture.
- A deed is about the things it names: where the thing it means is not here, that part is not done, and no other thing is used in its place.
- A deed meets this place as the stage has it. A thing, or a state or trace of a thing (a clean-cut limb, a fresh footprint), that the stage does not hold and a place of this kind would not ordinarily have is not there: the page does not show it, and what of the deed needs it is left out.
- Where one part of the deed cannot be done and the rest still makes sense without it, the rest is done and that part is left out. Where nothing of it can be done, it cannot be done.

Answer in JSON only:
{{"can": true or false,
  "why": "<if it cannot: the world's reason, ten words at most; else empty>",
  "effects": [<one entry for each lasting thing the deed does, in order; empty if it changes nothing that lasts>
     {{"thing": "<a thing, by its exact name on the stage>", "from": "<who holds it now, by name, or the word here if it lies here>", "to": "<the person who now holds it, by name, or the word here for lying here>"}},
     {{"thing": "<exact name>", "state": "<how the thing itself is left, two to five words>"}},
     {{"thing": "<exact name>", "held_out_to": "<the person it is held out or shown to>"}},
     {{"thing": "<exact name>", "on": "<the person it is now worn by or bound on, by name>"}},
     {{"thing": "<exact name of a written thing>", "read_by": "{name}"}},
     {{"world": "<one of the world's actions, the whole of the part before the bracket, with its @place or @person, exactly as listed>"}},
     {{"who": "{name}", "posture": "<standing, sitting, kneeling, lying down>"}},
     {{"who": "{name}", "touches": "<the person touched>", "how": "<two to four words>"}},
     {{"mark": "<a lasting change the deed makes to the place itself, which someone coming here tomorrow would see, two to five words. A touch, a brush of the hand, a footstep, a thing looked at or traced leaves none>"}},
     {{"who": "{name}", "goes": "<a place named on the stage, or the words out of the scene>", "with": ["<those {name} asks along>"]}},
     {{"new": "<a thing now made or picked>", "from": "<the listed action or the things here it comes from>"}}],
  "left_out": "<the part of the deed that is not done, in a few words; empty if all of it is done>",
  "page": "<what the audience sees happen: {name}'s deed in the third person and present tense, only what is done, no words spoken, 30 words at most>"}}
</end>"""

AGAIN = """

Your answer could not be applied to the stage:
{faults}
Answer again, in the same form, from the stage as it is given."""


def _low(value):
    """A name as the stage has it: no state in brackets, no article, lower case."""
    return re.sub(r'^(a|an|the|some)\s+', '', re.sub(r'\s*\(.*', '', str(value or '')).strip().lower())


def people(context, character):
    """Those in the scene with `character`, itself first."""
    scene = getattr(context, 'current_scene', None)
    if scene is not None:
        cast = [c for c in context.actors + context.extras + context.npcs
                if getattr(c, 'current_scene', None) is scene and not stage.off_stage(c)]
    else:
        cast = [character] + list(character.bystanders())
    return [character] + [c for c in cast if c is not character]


def find(context, character, name, frm=None):
    """Where a thing named on the stage is: ('mine' | 'here' | 'theirs', the thing's own name, its
    holder) or None. frm: whose it is as the resolver gave it (a person's name, or 'here'): two
    people carried berries, one held his out, and code that guessed which was meant guessed the
    other. With no holder given: the doer's own, then what lies here, then another's (one held
    out to the doer first)."""
    low, frm = _low(name), _low(frm)
    mine = next((str(i) for i in character.body.inventory if _low(i) == low), None)
    here = next((p['name'] for p in survival.lying(context, stage.here(character)) if _low(p['name']) == low), None)
    theirs = [(str(item), other) for other in people(context, character)[1:]
              for item in (other.body.inventory if getattr(other, 'body', None) is not None else []) if _low(item) == low]
    if frm == 'here' and here is not None:
        return 'here', here, None
    if frm == character.name.lower() and mine is not None:
        return 'mine', mine, character
    named = [(item, other) for item, other in theirs if other.name.lower() == frm]
    if named:
        return 'theirs', named[0][0], named[0][1]
    if mine is not None:
        return 'mine', mine, character
    if here is not None:
        return 'here', here, None
    offered = [(item, other) for item, other in theirs if (item, other.name, character.name) in context.held_out]
    if offered or theirs:
        item, other = (offered or theirs)[0]
        return 'theirs', item, other
    return None


def _person(context, character, name):
    low = _low(name)
    return next((c for c in people(context, character) if c.name.lower() == low), None)


def faults(context, character, found):
    """Why these effects cannot be applied: [] when they can. Only whether each name is the name
    of something: a thing on the stage (or made earlier in the deed), a person in the scene, one
    of the world's actions. Whether the deed is allowed (was the thing held out, does the action
    lack something, is the place too far) is the resolver's judgement, made with the whole stage
    before it, and is not judged again here: a second, cruder reading of the same rules refused
    a woman berries held out to her, twice, because another person carried berries too."""
    out, made = [], set()
    for e in found.get('effects') or []:
        if not isinstance(e, dict):
            out.append('an effect is not one of the forms')
        elif e.get('new'):
            made.add(_low(e['new']))
        elif e.get('world'):
            chosen = survival._chosen(context, character, str(e['world']))
            if chosen is None or (survival.free(context) and chosen[0] != 'threat'):
                out.append(f"'{e['world']}' is not one of the answers listed" if survival.free(context) else f"'{e['world']}' is not one of the world's actions listed")
        elif e.get('need'):
            who = _person(context, character, e.get('who') or character.name)
            if who is None or getattr(who, 'body', None) is None or (survival.need_called(who.body, e['need']) is None
                                                                      and survival.flag_called(who.body, e['need']) is None):
                out.append(f"'{e['need']}' is not one of the needs the stage shows for {e.get('who') or character.name}")
        elif e.get('found'):
            if not any(_low(p['name']) == _low(e['found']) for p in survival.lying(context, stage.here(character), hidden=True)):
                out.append(f"there is no '{e['found']}' hidden here")
        elif e.get('mark'):
            pass   # a trace on the place: nothing it names has to be there
        elif e.get('thing'):
            if find(context, character, e['thing'], e.get('from')) is None and _low(e['thing']) not in made:
                out.append(f"there is no '{e['thing']}' held by anyone here or lying here")
            for key in ('to', 'held_out_to', 'on'):
                if e.get(key) and _low(e[key]) != 'here' and _person(context, character, e[key]) is None:
                    out.append(f"nobody called '{e[key]}' is here")
        elif e.get('touches'):
            if _person(context, character, e['touches']) is None:
                out.append(f"nobody called '{e['touches']}' is here")
        elif e.get('goes'):
            for name in e.get('with') or []:
                if _person(context, character, name) is None:
                    out.append(f"nobody called '{name}' is here")
        elif not (e.get('posture') or e.get('mark')):
            out.append('an effect is not one of the forms')
    return out


def _ask(context, character, act, note=''):
    from sim import keeper, record
    direction = getattr(character, 'direction', None) or {}
    scene = '\n'.join(filter(None, [f"It opened: {direction['setup']}" if direction.get('setup') else '',
                                    *context.witnessed(character, n=14)])) or 'Nothing yet.'
    party = act.leaves if isinstance(getattr(act, 'leaves', None), list) else []
    to = getattr(act, 'leaves_to', None)
    leaves = '' if not getattr(act, 'leaves', None) else \
        f"\nIts actor marks it as going out of the scene{' with ' + ', '.join(c.name for c in party) if party else ' alone'}{', for ' + to if to else ''}.\n"
    prompt = _for(context, RESOLVE).format(world=f"{' '.join((context.initial_state or '').split())}\nThis world keeps count of: {keeper._tracks(context)}.",
                            stage=record.of(context, character, people(context, character), full=True), scene=scene,
                            name=character.name, deed=act.action, leaves=leaves) + note
    reply = context.llm.ask({}, [UserMessage(content=prompt)], tag='stage.resolve_deed', temp=0.2, stops=['</end>'], max_tokens=700)
    found = json_reply.parse(reply) if reply else None
    return found if isinstance(found, dict) and 'can' in found else None


def resolve(context, character, act):
    """What this deed does, decided before it is played: {'can', 'why', 'effects', 'left_out',
    'page'}. An answer that cannot be applied is sent back once with what is wrong with it; one
    that still cannot be, or cannot be read, is a deed that cannot be done. A deed of which
    nothing is done and something is left out cannot be done either: the actor is told, and the
    page is not given a line about nothing."""
    found = _ask(context, character, act)
    wrong = faults(context, character, found) if found else ['the answer could not be read']
    if found and wrong:
        trace.event('resolve_again', character=character.name, deed=act.action[:200], faults=wrong)
        again = _ask(context, character, act, AGAIN.format(faults='\n'.join(f'- {w}.' for w in wrong)))
        if again is not None:
            found, wrong = again, faults(context, character, again)
    if not found or wrong:
        found = {'can': False, 'why': wrong[0], 'effects': [], 'left_out': '', 'page': ''}
    found['effects'] = [e for e in found.get('effects') or [] if isinstance(e, dict)]
    found['left_out'], found['page'] = str(found.get('left_out') or '').strip(), str(found.get('page') or '').strip()
    if found.get('can') and not found['effects'] and found['left_out']:
        found.update(can=False, why=str(found.get('why') or '').strip() or f"not here: {found['left_out']}")
    if found.get('can') and not found['page']:
        found.update(can=False, why='the deed came to nothing that could be shown')
    found['can'], found['why'] = bool(found.get('can')), str(found.get('why') or '').strip()
    trace.event('resolved', character=character.name, deed=act.action[:300], can=found['can'], why=found['why'],
                effects=found['effects'], left_out=found['left_out'], page=found['page'])
    return found


# ---------------------------------------------------------------------------- words
# A line is resolved as a deed is: one call, shown the whole stage, what its speaker has lived,
# what it has already told its hearer of its own life, and what it does not know, says what the
# words do (a thing handed over in so many words), what they state that is not so, what they
# newly tell of the speaker's life, and what they promise. It takes the place of the speech
# check, the guess that words hand a thing over, the reading of commitments at a talk's end, and
# the planner's reading of what was told. (Tested on 209 recorded lines: logs/review-20261120/act.)
_head, _rest = RESOLVE.split('How this world works:')
_rules, _answer = _rest.split('Answer in JSON only:')
_effects = _answer[_answer.index('  "effects"'):_answer.index('  "left_out"')]
ACT = """You keep the world of an improvised play. A character is about to act: to say something, to do something, or both at once. Say what the act does in this world as it now stands, and what of its words may stand.

<world>
{world}
</world>

The stage is the whole truth of now: who is here, how each body is, who holds what, what lies here, and whose doing each trace and each state of a thing is.
<stage>
{stage}
</stage>

What {name} has seen, done and heard, oldest first:
<lived>
{lived}
</lived>

What {name} has already told {listener} of {name}'s own life before the play:
<told>
{told}
</told>

What {name} does not know:
<does_not_know>
{unknown}
</does_not_know>

{name} is about to:
<words>
{words}
</words>
<deed>
{deed}
</deed>
{leaves}
How this world works:""" + _rules + """
The words:
- The words are what {name} says aloud, and only that. Anything else written into them, what is done or how it is said ('She snorts.', '*pats her pocket*', '(whispered)'), is a finding (aloud). The mark (lie) is not one: it stays.
- Words do nothing physical, with one exception: words that hand a thing the speaker holds to someone present, in so many words ("here, take half"), are that hand-over. Give it among the effects, and say in "page" what is seen.
- A person cannot be mistaken about itself at this moment. A finding is something the words state as true now that the stage or <lived> contradicts: what {name} holds or hands over (holds); how its own body is (body); where it or someone present is (where); a thing at this place, or a state or trace of one, that the stage does not hold and a place of this kind would not ordinarily have (here: 'a clean-cut limb' where the stage has only torn bark); the danger on the stage, said to be gone, going or held off, or a danger said to be upon them that the stage does not show (danger); the hour or the light (time); a physical act said to be done or being done, here and now, that was not done (done); who made a trace or left a thing as it is, when the stage says whose doing it is and the words give it to someone else or to a stranger (whose); what a written thing says (writing); he or she against the cast (gender); something under does_not_know stated as known, unless <lived> shows {name} has since seen or been told it (knows); and what was said or done earlier in the play, told wrongly (past: 'the berry theory' when it was the mushrooms). <lived> holds only the latest of the play: what is not in it may have been said or done before it, and is no finding.
- These are never findings: a plan, a wish, a fear, a memory of life before the play, a guess said as a guess, what someone makes of what is here (who did it, what it means: 'somebody pruned this tree'), a question, an order, an offer, what is said of another's mind, a figure of speech, a deliberate lie marked (lie). When you are not sure, it is no finding.
- What {name} tells of its own life before the play is new to the world unless it is in <told>: list each new fact plainly, in the third person. If the words tell {listener} as news something already in <told>, give those words under "retold".
- A promise is something the speaker says it will do, or agrees that they will do. Its kind: now, when it is to be done here in this scene (eat this, go in, set off); standing, when it is a rule or a condition to keep with nothing to do now ("no mushrooms", "wake me for my watch"); later, when it is for another time or place. A question, a proposal not yet agreed, or a demand on someone else is not a promise.

Answer in JSON only:
{{"can": true or false,
  "why": "<if a deed cannot be done: the world's reason, ten words at most; else empty>",
""" + _effects + """  "left_out": "<the part of the deed that is not done, in a few words; empty if all of it is done or there is no deed>",
  "page": "<what the audience sees {name} do: third person, present tense, only what is done, no words spoken, 30 words at most; empty when nothing is done but speaking>",
  "findings": [{{"kind": "<aloud, holds, body, where, here, danger, time, done, whose, writing, gender, knows or past>", "words": "<the words that say it, exactly>", "why": "<what the stage or the record shows instead, ten words at most>"}}],
  "line": "<only when there is a finding: the words as {name} can truly say them, the same line in the same voice with each finding put right from what the stage shows or left out, and nothing else changed; else empty>",
  "tells": ["<each new fact {name} tells of its own life before the play>"],
  "retold": "<the words that tell again as news what is already in told; else empty>",
  "promises": [{{"who": "<who is to do it>", "what": "<what, in a few words>", "kind": "<now, standing or later>"}}],
  "names": ["<each person the words tell the hearers the name of, as the cast names them>"]}}
</end>"""


# ---------------------------------------------------------------------------- a world without its catalogue
# In a world played without its catalogue of named actions (survival.free) a body's needs are
# facts about the person, and what an act does to them is the resolver's to say; what a kind of
# place gives (berries on a bush, water at a spring) is the model's to know, not a list. The
# rules and the effect forms that spoke of 'the world's actions' are put this way instead.
# (Untested for quality when written: 2026-11-20.)
_SWAPS = (
    ("- What changes a body's needs (eating, drinking, resting, tending a wound) and what finds things a place holds (searching it, picking from it) happens only by one of the world's actions listed on the stage, named exactly as listed. An action marked as lacking something cannot be done. With a danger on the stage, what is done about it is one of the answers listed for it.",
     "- A body's needs are on the stage (Bodies). Eating, drinking, resting or tending a wound eases the need it answers when what it takes is really here: food in hand or growing at this place, water at this place, somewhere to lie down, something to bind a wound with. Say it as a need effect; it is done whether or not the body was in want (a person may drink when not thirsty). What a place of its kind ordinarily gives (berries on a bush, an apple on a tree, water from a spring, a limb off a fallen branch) can be picked, drawn or taken from it while standing at it: say that as a new thing. A thing listed as hidden here comes to light only by a deed that searches where it is. With a danger on the stage, what is done about it is one of the answers listed for it."),
    ("- Nothing comes from nothing: a new thing is one this place gives (by a listed action), or is made from things that are here.",
     "- Nothing comes from nothing: a new thing is one a place of this kind ordinarily gives, taken at that place, or is made from things that are here."),
    ("""     {{"world": "<one of the world's actions, the whole of the part before the bracket, with its @place or @person, exactly as listed>"}},""",
     """     {{"who": "<the person whose need it is>", "need": "<one of the needs this world keeps count of, by that name>", "change": "<eased or worsened>"}},
     {{"found": "<a thing listed as hidden here, by its exact name, that a search brings to light>"}},
     {{"world": "<only with a danger on the stage: one of the answers listed for it, exactly as listed>"}},"""),
    ("""{{"new": "<a thing now made or picked>", "from": "<the listed action or the things here it comes from>"}}""",
     """{{"new": "<a thing now made or picked>", "from": "<the place, or the things here, it comes from>"}}"""))


def _for(context, prompt):
    """A resolver prompt as this world is played: with its catalogue of actions, or without."""
    if not survival.free(context):
        return prompt
    for with_catalogue, without in _SWAPS:
        prompt = prompt.replace(with_catalogue, without)
    return prompt


def told_to(context, speaker, listener=None):
    """What `speaker` has told of its own life before the play (to `listener`, or to anyone): the
    facts, oldest first. Kept as told, with the line that told each (ledger.told): what a person
    has said of itself is the same for every later reader, and is not told again as news."""
    name, to = speaker.name, getattr(listener, 'name', None)
    return [t['fact'] for t in ledger.of(context).told if t['who'] == name and (to is None or to in t['to'])]


def tell(context, speaker, hearers, facts):
    book = ledger.of(context)
    event = book.open[-1] if book.open else None
    for fact in facts:
        fact = ' '.join(str(fact).split())
        if fact and not any(t['who'] == speaker.name and t['fact'].lower() == fact.lower() for t in book.told):
            book.told.append({'who': speaker.name, 'to': [h.name for h in hearers if h is not speaker], 'fact': fact, 'event': event})


def resolve_words(context, speaker, words, listener=None, about=None, deed=''):
    """What a spoken line does and states: {'effects', 'page', 'findings', 'line', 'tells',
    'retold', 'promises', 'names', 'deed_refused'}, or None when the answer could not be read (the
    line then stands as written). about: the stage as already rendered for the speaker
    (record.speech). deed: what the speaker does as it speaks, resolved with the words as one
    act; 'deed_refused' is then why it cannot be done, or ''."""
    from sim import keeper, record
    others = people(context, speaker)
    listener = listener if listener is not None else next((c for c in others if c is not speaker), None)
    direction = getattr(speaker, 'direction', None) or {}
    lived = '\n'.join(filter(None, [f"It opened: {direction['setup']}" if direction.get('setup') else '', *context.witnessed(speaker, n=16)])) or 'Nothing yet.'
    missing = speaker.does_not_know() if hasattr(speaker, 'does_not_know') else []
    prompt = _for(context, ACT).format(world=f"{' '.join((context.initial_state or '').split())}\nThis world keeps count of: {keeper._tracks(context)}.",
                        stage=about or record.of(context, speaker, others, full=True), name=speaker.name,
                        listener=getattr(listener, 'name', 'the others'), lived=lived,
                        told='\n'.join(f'- {t}' for t in told_to(context, speaker, listener)) or 'Nothing yet.',
                        unknown='\n'.join(f'- {m}' for m in missing) or 'Nothing listed.', words=words, deed=deed or '(does nothing; only speaks)', leaves='')
    def ask(note=''):
        reply = context.llm.ask({}, [UserMessage(content=prompt + note)], tag='stage.resolve_say', temp=0.2, stops=['</end>'], max_tokens=700)
        found = json_reply.parse(reply) if reply else None
        if not isinstance(found, dict) or 'can' not in found:
            return None, []
        found['effects'] = [e for e in found.get('effects') or [] if isinstance(e, dict) and not e.get('goes')]   # leaving is the actor's mark
        return found, faults(context, speaker, found)
    found, wrong = ask()
    if found is None:
        return None
    if wrong:   # sent back once with what is wrong with it, as a deed is (resolve): a thing's name cut short lost a deed
        trace.event('resolve_again', character=speaker.name, deed=deed[:200], faults=wrong)
        again, still = ask(AGAIN.format(faults='\n'.join(f'- {w}.' for w in wrong)))
        if again is not None:
            found, wrong = again, still
    left = str(found.get('left_out') or '').strip()
    refused = ''
    if wrong:   # what names something that is not here does nothing
        found['effects'], refused = [], wrong[0]
    elif deed and not found.get('can'):
        found['effects'], refused = [], str(found.get('why') or '').strip() or 'it cannot be done here'
    elif deed and not found['effects'] and left:
        refused = f'not here: {left}'
    found['deed_refused'] = refused if deed else ''
    # a deed done is shown though it changes nothing that lasts; words alone are shown doing
    # something only when they hand a thing over
    found['page'] = str(found.get('page') or '').strip() if (found['effects'] or (deed and not refused)) else ''
    found['left_out'] = left if deed and not refused else ''
    found['line'] = str(found.get('line') or '').strip()
    found['findings'] = [f for f in found.get('findings') or [] if isinstance(f, dict) and str(f.get('words') or '').strip()]
    found['tells'] = [str(t).strip() for t in found.get('tells') or [] if isinstance(t, str) and t.strip()]
    found['retold'] = str(found.get('retold') or '').strip()
    found['promises'] = [p for p in found.get('promises') or [] if isinstance(p, dict) and str(p.get('what') or '').strip()]
    trace.event('resolved_words', character=speaker.name, words=words[:300], deed=deed[:300], refused=found['deed_refused'], effects=found['effects'], page=found['page'],
                findings=found['findings'], tells=found['tells'], retold=found['retold'], promises=found['promises'])
    return found


# ---------------------------------------------------------------------------- a scene's set-up
# The planner's set-up is a proposal. One call, shown the stage, every thing and trace of the
# play and the places of the world, says where the scene is set, what the set-up brings in that
# such a place would have, what it says that is not so, and the set-up as it may stand; what it
# brings is then in the world, by one event. It takes the place of the check of the set-up's
# claims, the admitting of them one by one, the reading of the set-up for deeds already played,
# and the matching of place names in its text. It also puts the title and the cast's goals right, says
# which danger the set-up brings on, and is shown what has been promised (paired replay:
# logs/review-20261121/setup).
SETUP = """You keep the world of an improvised play. A scene is about to open with the set-up below, written by the planner. Say how it stands against the world as it is, before the audience is shown it.

<world>
{world}
</world>

The stage as it now is, where the cast stands:
<stage>
{stage}
</stage>

Every thing and trace of the play and where it is, what the written things say, and what has been told:
<things>
{things}
</things>

The cast as the play gives them:
<cast>
{cast}
</cast>

The places of this world (a scene is set at one of them):
<places>
{places}
</places>

Dangers this world may bring on: {hazards}

The scene, '{title}', planned at: {location}
<setup>
{setup}
</setup>
It is to end:
<ending>
{ending}
</ending>
What each of the cast is to go after in it:
<goals>
{goals}
</goals>

How a set-up stands:
- It is the opening only: where the cast is, what is there, what presses on them. It may bring in, without remark, what such a place would ordinarily have (a flat stone by a spring, dust in a hut), and one of the dangers listed, upon the cast from the scene's first moment.
- It may not move a thing the world has elsewhere, put in anyone's hands what they do not hold, say of a body what the stage does not show, bring in a person who is not listed, or give a trace or a state of a thing to anyone but the one whose doing the stage says it is.
- It may not play the scene: what the cast says, decides, finds out or does to one another is theirs to play, and is taken out of the set-up.
- Time and walking between scenes are the world's to tell: the set-up does not say that needs were met on the way unless the stage shows them met.
- The title, the ending and each goal are held to the same: one that rests on something not so is put toward the same end with what is so. A person may still want, fear or suspect what is not so; a goal that is to show, find or use it may not.
- Not_so is for what the world shows otherwise. What it does not show either way (who once lived here, what is past the trees) is not for you to settle; nor is what the cast have said or promised to one another, which <things> holds only in part. What <cast> says of a person and of what they carry is so (pages in a notebook they do not remember writing), though the world does not hold it yet.
- Change as little as you can: keep the planner's words wherever they are true.

Answer in JSON only:
{{"place": "<the one listed place this scene is set at, by its exact name; or the word same when it is where the cast now stands>",
  "brings": [{{"thing": "<a thing the set-up says is there, which the world does not yet have and such a place would ordinarily have, two to five words>", "fixed": true or false}}],
  "danger": "<one of the dangers listed, by its name, when the set-up as it may stand has it upon the cast now, near enough to strike; else empty. One that is heard from afar, feared, spoken of or somewhere off is not upon them>",
  "not_so": [{{"words": "<the words of the title, set-up, ending or a goal that cannot stand>", "why": "<what the world shows instead, ten words at most>"}}],
  "title": "<the title as it may stand>",
  "setup": "<the set-up as it may stand: the same words if nothing is wrong; else the same with each of not_so put right or left out>",
  "ending": "<the ending as it may stand; it may not need what nobody could do in this world>",
  "goals": {{"<a name from goals>": "<that goal as it may stand>"}}}}
</end>"""


def resolve_setup(context, scene, cast, hazards=()):
    """How a scene's set-up stands against the world: {'place', 'brings', 'danger', 'not_so',
    'title', 'setup', 'ending', 'goals'}, or None when the answer could not be read (the set-up then
    stands as written). hazards: the names of the dangers the scene may bring on. The goals are
    the planner's for each of the cast (scene['characters']), put right as the set-up is: Samantha
    was to 'show Joe clean tool cuts on the torn limbs' after the cuts were struck from the set-up,
    and showed them."""
    from sim import keeper, record
    first = cast[0]
    planned = {name: str((entry or {}).get('goal') or '').strip() for name, entry in (scene.get('characters') or {}).items() if isinstance(entry, dict)}
    prompt = SETUP.format(world=f"{' '.join((context.initial_state or '').split())}\nThis world keeps count of: {keeper._tracks(context)}.",
                          stage=record.account(context, 'stage', about=first, people=cast),
                          things=record.account(context, 'planner', people=cast) or 'nothing yet',
                          cast='\n'.join(f"{c.name}: {' '.join(str(getattr(c, 'original_character', '') or getattr(c, 'character', '')).split())[:700]}" for c in cast) or 'none listed',
                          places=stage.gazetteer(context) or 'none listed', hazards=', '.join(hazards) or 'none',
                          title=scene.get('scene_title', ''), location=scene.get('location') or 'same',
                          setup=scene.get('pre_narrative', ''), ending=scene.get('post_narrative', ''),
                          goals='\n'.join(f'{n}: {g}' for n, g in planned.items() if g) or 'none given')
    reply = context.llm.ask({}, [UserMessage(content=prompt)], tag='stage.resolve_setup', temp=0.2, stops=['</end>'], max_tokens=900)
    found = json_reply.parse(reply) if reply else None
    if not isinstance(found, dict) or not str(found.get('setup') or '').strip():
        return None
    registry = context.map.resource_registry
    place = str(found.get('place') or '').strip()
    found['place'] = next((rid for rid in registry if rid.lower() == place.lower()), '')
    found['brings'] = [b for b in found.get('brings') or [] if isinstance(b, dict) and str(b.get('thing') or '').strip()][:4]
    found['not_so'] = [n for n in found.get('not_so') or [] if isinstance(n, dict) and str(n.get('words') or '').strip()]
    found['setup'], found['ending'] = ' '.join(str(found['setup']).split()), ' '.join(str(found.get('ending') or '').split())
    found['title'] = ' '.join(str(found.get('title') or '').split()).strip('"\'')
    danger = _low(found.get('danger'))
    found['danger'] = next((h for h in hazards if h.lower() == danger), '')
    goals = found.get('goals') if isinstance(found.get('goals'), dict) else {}
    found['goals'] = {name: ' '.join(str(g).split()) for name in planned for k, g in goals.items()
                      if _low(k) == name.lower() and str(g or '').strip() and ' '.join(str(g).split()) != planned[name]}
    trace.event('resolved_setup', scene=scene.get('scene_title'), place=found['place'], brings=found['brings'], danger=found['danger'],
                not_so=found['not_so'], title=found['title'], setup=found['setup'][:600], ending=found['ending'][:300], goals=found['goals'])
    return found


def apply_setup(context, scene, cast, found):
    """Put what a set-up brings into the world, at the scene's place (inside the scene's 'set'
    event). Returns the names of the things made."""
    registry = context.map.resource_registry
    at = tuple(registry[found['place']]['location']) if found['place'] in registry else stage.here(cast[0])
    made = []
    for b in found['brings']:
        if survival.add_thing(context, str(b['thing']), at=at, fixed=bool(b.get('fixed'))):
            made.append(str(b['thing']))
    return made


def effects_note(effects):
    """What an act changed, in a few words, for whoever reads the page to see what was done and
    not only said: 'berries: Joe to Samantha; Samantha kneeling'. '' when it changed nothing."""
    out = []
    for e in effects or []:
        if not isinstance(e, dict):
            continue
        if e.get('thing') and e.get('to'):
            out.append(f"{e['thing']}: {e.get('from') or '?'} to {e['to']}")
        elif e.get('thing') and e.get('held_out_to'):
            out.append(f"{e['thing']} held out to {e['held_out_to']}")
        elif e.get('thing') and e.get('on'):
            out.append(f"{e['thing']} on {e['on']}")
        elif e.get('thing') and e.get('state'):
            out.append(f"{e['thing']} {e['state']}")
        elif e.get('thing') and e.get('read_by'):
            out.append(f"{e['thing']} read by {e['read_by']}")
        elif e.get('need'):
            out.append(f"{e.get('who', '')} {e['need']} {e.get('change', 'eased')}".strip())
        elif e.get('world'):
            out.append(str(e['world']))
        elif e.get('posture'):
            out.append(f"{e.get('who', '')} {e['posture']}".strip())
        elif e.get('mark'):
            out.append(f"left: {e['mark']}")
        elif e.get('new'):
            out.append(f"now there: {e['new']}")
        elif e.get('goes'):
            out.append(f"{e.get('who', '')} goes {e['goes']}".strip())
    return '; '.join(out)


def leaving(context, character, found):
    """Who goes out of the scene by this deed, and where to: (None | 'alone' | [characters], place or None).
    Going to another place is going out of the scene: a scene has one place."""
    goes = next((e for e in found.get('effects') or [] if e.get('goes')), None)
    if goes is None:
        return None, None
    party = [c for c in (_person(context, character, n) for n in goes.get('with') or []) if c is not None and c is not character]
    to = str(goes['goes']).strip()
    return (party or 'alone'), (None if 'scene' in to.lower() else to)


POSTURES = {'sitting': 'sitting', 'kneeling': 'kneeling', 'crouching': 'kneeling', 'lying down': 'lying down', 'lying': 'lying down'}


def apply(context, character, act):
    """Do what the deed was found to do. Returns the world's own sentences for what it settled
    (what a world action found or eased, what a written thing says): the rest is on the page
    already, in the resolver's line."""
    from sim import keeper
    found, facts, me = act.found, [], character.name
    opened = ledger.of(context).open
    if opened:   # the deed's event keeps what it was found to do, beside what that changes and the line shown
        opened[-1]['effects'] = found['effects']
    worlded = False
    for e in found['effects']:
        if e.get('world'):
            chosen = survival._chosen(context, character, str(e['world']))
            if chosen and chosen[0] == 'threat':
                # what came of it is told by the danger itself, after the act's own line (survival._told)
                survival.answer_danger(context, character, survival.active_threat_for(character), chosen[1])
                fact = None
            elif chosen:
                fact = survival._apply(context, character, chosen[1], chosen[2], chosen[3], 'model', act.action)
            else:
                fact = None
            worlded = True
            if fact:
                facts.append(fact)
            if (character.last_affordance or {}).get('applied') is False:
                # the world's own rules would not have it (its sentence says why, after the page line)
                trace.event('effect_refused', character=me, world=str(e['world']), fact=fact)
        elif e.get('need'):
            # a need eased or made worse, as the resolver found it: a fact about the person, in
            # the body's own terms (eased: one step down in its words, survival.eased; worsened: a
            # step harder). A state that is or is not (body.flags) is cleared when eased.
            who = _person(context, character, e.get('who') or me) or character
            need = survival.need_called(who.body, e['need'])
            worse = _low(e.get('change')).startswith('wors')
            if need is not None:
                level = who.body.level(need)
                who.body.adjust(need, 0.2) if worse else survival.eased(who.body, need)
                trace.event('need_changed', character=who.name, need=need, change='worsened' if worse else 'eased', by=me, was=round(level, 2))
            elif not worse and survival.flag_called(who.body, e['need']):
                flag = survival.flag_called(who.body, e['need'])
                who.body.flags.discard(flag)
                ledger.of(context).wrote('needs')
                trace.event('need_changed', character=who.name, need=flag, change='cleared', by=me)
        elif e.get('found'):
            thing = next((p for p in survival.lying(context, stage.here(character), hidden=True) if _low(p['name']) == _low(e['found'])), None)
            if thing is not None:
                thing.pop('hidden', None)   # in plain sight from now on
                trace.event('thing_found', character=me, thing=thing['name'])
        elif e.get('mark'):
            # a trace of this deed on the place, kept as this deed's doing and never as a thing
            # lying here (survival.leave_mark)
            survival.leave_mark(context, character, e['mark'], on=e.get('thing'))
        elif e.get('thing'):
            where = find(context, character, e['thing'], e.get('from'))
            if where is None:
                continue   # made earlier in this deed and not come by: nothing to move
            kind, name, holder = where
            if e.get('on'):
                wearer = _person(context, character, e['on'])
                if wearer is not None and (kind in ('mine', 'here') or holder is wearer):
                    thing = survival._thing(context, name, holder or character)
                    if survival.wear(context, thing, wearer):
                        trace.event('prop_deed', character=me, deed='put on', item=name, other=wearer.name)
                continue
            to = _low(e.get('to'))
            other = _person(context, character, e.get('to') or e.get('held_out_to') or '')
            if e.get('to'):
                if kind == 'here' and to != 'here':
                    survival._prop_do(context, character, 'take', name, None)
                    kind = 'mine'
                if kind == 'theirs' and to == me.lower():
                    # as the resolver found it: whether it was given or held out was its to judge
                    character.body.inventory.append(holder.body.inventory.remove(name))
                    context.held_out[:] = [h for h in context.held_out if not (h[0] == name and h[1] == holder.name)]
                    trace.event('prop_deed', character=me, deed='take from', item=name, other=holder.name)
                elif kind == 'mine' and to == 'here':
                    survival._prop_do(context, character, 'put down', name, None)
                elif kind == 'mine' and other is not None and other is not character:
                    survival._prop_do(context, character, 'give', name, other)
            elif e.get('held_out_to') and other is not None and other is not character:
                if kind == 'here':
                    survival._prop_do(context, character, 'take', name, None)
                    kind = 'mine'
                if kind == 'mine':
                    survival._prop_do(context, character, 'offer', name, other)
            elif e.get('read_by'):
                if survival.writing_of(context, name):
                    survival._prop_do(context, character, 'read', name, None)
                    facts.append(f"The {name} says: {survival.writing_of(context, name)}")
            elif e.get('state') and kind in ('mine', 'here'):
                survival._handle(context, character, name, e['state'])
        elif e.get('posture'):
            poses, _ = survival._arrangement(context)
            word = POSTURES.get(_low(e['posture']))
            poses.pop(me, None) if word is None else poses.__setitem__(me, word)
            survival._bear(context)
        elif e.get('touches'):
            other = _person(context, character, e['touches'])
            if other is not None and other is not character:
                _, between = survival._arrangement(context)
                # the stage keeps that they are touching ('touching Lanya'); how is on the page
                between[:] = [b for b in between if not (b[0] == me and b[2] == other.name and b[3] == 'contact')] + [(me, 'touching', other.name, 'contact')]
                survival._bear(context)
        elif e.get('new') and not worlded:
            # made from what is here, not given by a world action (whose own gain put it in hand): the
            # keeper says what kind of thing it is and how it behaves in time, where that matters
            # (a fire); a thing it does not model is simply there
            kept = keeper.consider(context, character, act.action) if ablation.enabled('keeper') else {'can': False}
            fact, done = keeper.perform(context, character, act.action) if kept.get('can') else ('', False)
            if not done:   # it is here; a later effect of the deed puts it in a hand
                survival.add_thing(context, str(e['new']), at=stage.here(character))
            elif fact:
                facts.append(fact)
    character.last_affordance = None   # what was done is told by the page line, not by the last part's own sentence
    if found['left_out']:
        character.add_perceptual_input(f"Of what you meant to do, this was not done, because it cannot be here: {found['left_out']}.",
                                       percept=False, mode='internal')
    return ' '.join(facts)
