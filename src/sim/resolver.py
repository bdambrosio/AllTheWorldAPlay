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
name in the effects is the name of something (a thing on the stage, a person in the scene) and
applies them. What a world's file says of itself (its facts, what threatens people there, what is
held) is on the stage as sentences, and nothing in it is matched by code
(docs/world-facts-design.md). The page is written from what was done, so the two cannot differ.

Tested offline on 225 recorded deeds (logs/review-20261120/resolver): Claude Opus 5.5 gave 409 of
409 well-formed answers, no thing from nothing, no thing taken from another's hands, about two
seconds a call."""
from __future__ import annotations

import re

from sim import ablation, ledger, ruling, stage, survival, world
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
- The stage above is the truth of who holds what, and of every thing the play has so far had to do with. Beside those, a place has to hand what a place of its kind plainly has (a kitchen its cups, plates and forks, a bar its glasses and bottles, a wood its sticks and stones): such a thing is there, and the first deed to use it says it as a new effect, from the place. What is particular is only where the stage has it: a thing someone owns or has named (the grey suit, the car keys, a letter), money, food and drink in a world where they are hard come by, and anything the stage shows elsewhere.
- A deed can use only what its doer holds, what lies here, what another person here gives or holds out, and what this place gives or plainly has. A thing that lies at another place, however near, is not here: nobody picks it up from where they stand.
- What stands (on the stage, where anything does) is so in this world, whoever knows it: a deed meets the world as it says, and what it says is at a place is there though the stage does not list it yet (the first deed to use it says it as a new effect, from the place). Each is known only to those named with it. What only some know you never show or tell: the page says what is seen done, never what it means, whose doing an old thing was, or what lies behind it.
- Ordinary handling of a thing that is here needs no leave: pick it up, put it down, hand it over, hold it out or up to be seen, pocket it, open, shut, fold, tear, wipe or turn it, set it on something, read it. Each is an effect: say it. Reading a writing aloud gives what it says to everyone here who hears it; read silently, it is the reader's alone.
- The facts of this world are on the stage. They are true here, with their sums and their times, and are read with sense: what they say of one thing is a guide to things like it. What they do not speak of goes as it ordinarily would in such a world.
- What is held (Held, on the stage) no deed or word gets past, however it is put and whoever asks. Where a deed would break one, that part is not done, and why says which. In "held", name each one that bears on this deed and how it stands.
- A deed that settles something others will later have to reckon with (a contract signed, a debt agreed, a thing sold, land made over) is a stands effect: one plain sentence, with the day. A thing simply bought, sold, eaten or handed over is not: the stage shows it already. Where it ends something that stood, give that sentence, word for word, in ends.
- A body's needs are on the stage (Bodies). Eating, drinking, resting or tending a wound eases the need it answers when what it takes is really here. Say it as a need effect, and how far: a little, well or fully, as the facts and ordinary sense have it (a mouthful a little, a meal well). Half of a thing eases less than the whole and leaves the other half: say the state the thing is left in. It is done whether or not the body was in want.
- Money is on the stage (Bodies). It moves only as a money effect, from the one who pays to the one paid, in the sum the facts give or the people have agreed. Someone outside the scene who is paid, or pays, in the ordinary way of this world (the bar, the baker, an employer) is named in plain words. Nobody pays what they do not have, and no money comes from nowhere.
- A thing that leaves the play by the deed (eaten, drunk, burnt, sold to someone outside the scene) is a gone effect: it is then nowhere. A place that gives a thing which runs out (a bush, a shelf, a woodpile) is left with a mark saying how much is left, and a later deed is ruled by that mark. A place and its parts take marks, never a state: a state is for a thing held or lying here. A part of a place that a deed changes back (a bolted door unbolted, a shut window opened) is a new mark that ends the old one.
- A deed takes the time it would take: give it in minutes when it is more than a moment.
- Where a deed may or may not come off, or may go badly (a search, a doubtful mouthful, a repair), do not decide it. Say how likely it is to go as the doer means, from the facts and from how things stand here, and what follows either way: the draw is made for you.
- Words reach someone who is not here only by a means the facts give (a phone, a radio): say it as a sent effect.
- A danger upon someone is on the stage (Danger), and what this world says of such dangers is among its facts. A deed against it (to drive it off, to hide from it, to bar a door on it, to buy it off) is put to chance unless it could hardly fail or hardly work: how likely, from the facts and from what the doer has to hand, and what follows either way. If so: the danger over. If not: the danger still there, and whoever it reaches hurt, as a need effect on the harm this world counts, a little, well or fully. A deed that does not touch the danger leaves it as it is, and it is not for you to end it out of kindness.
- Putting a thing on a person (a coat round someone, a strip of cloth bound round a wound) leaves it on them, worn: say it as on. Taking off what is worn puts it in a hand.
- A touch is done as it is written: taking a hand or an arm, a hand on a shoulder, pressing a thing into someone's hand (the thing is then theirs to keep or give back). The one touched may draw away afterwards; that is theirs to play.
- Nobody is moved, held back, stopped or searched against their will. To lead, draw, pull or beckon someone along is to ask: they answer for themselves, and where they stand or go is no effect of this deed. A thing another person holds comes away only if they give it or have held it out.
- Nothing comes from nothing: a new thing is one a place of this kind ordinarily gives, taken at that place, or is made from things that are here.
- A look, a gesture or a step or two changes nothing that lasts; a change of posture changes only the posture.
- What a deed leaves on a person that others can see and that lasts beyond the moment (hands inked or washed clean, a sleeve torn, a face wet; up on the rock, out of sight behind the shelves) is a state of that person: say it, and when a later deed undoes it, end it. Needs, harm this world counts, posture and what is worn have their own forms.
- Whatever can be put one way and then another and back again (opened and shut, fastened and loosed, lit and put out, filled and emptied) is a thing with a state, whether it can be carried or is part of the place. If the stage does not list it yet, give it first as a new entry, fixed when it is part of the place, and then its state. A thing's state takes the place of its earlier one, so the stage always shows how it now is; the same change written as a trace would stand beside the trace of its opposite, and nothing would say which is so now. A mark is only for what a deed adds to the place that no later deed undoes by doing the opposite.
- What a person plainly has on (clothes, shoes) is theirs to take off and use though the stage does not list it: give it first as a new entry, from that person.
- A deed is about the things it names: where the thing it means is not here, that part is not done, and no other thing is used in its place.
- A deed meets this place as the stage has it. A thing, or a state or trace of a thing (a clean-cut limb, a fresh footprint), that the stage does not hold and a place of this kind would not ordinarily have is not there: the page does not show it, and what of the deed needs it is left out. What the stage lists as found not to be here was looked for and is not here, whatever such a place might ordinarily have: a deed that needs it cannot be done with it, and only a deed that itself makes it from what is here brings it.
- A deed may go to, or make, a place the stage does not name only where this world would have such a place inside or beside a named one (the shed behind the hut, the loft over the stable, a lean-to built here of branches): say it as a place effect, and it is there from then on; going there is a goes effect to the name you gave it. A place that belongs to no named place is not made, and what of the deed needs it is left out.
- Where one part of the deed cannot be done and the rest still makes sense without it, the rest is done and that part is left out. Where nothing of it can be done, it cannot be done.
- The page and the effects are one account of the deed. When you have written both, read the page again: whatever it shows that is still so once the deed is over (a thing set down, moved, opened, shut, emptied or written in; a door bolted; a lamp put out; a writing read out; hands washed clean) has its entry among the effects, in the form that fits. A later deed is ruled from the effects, never from the page: what the page shows and no effect carries will not be so.

Answer in JSON only:
{{"can": true or false,
  "why": "<if it cannot: the world's reason, ten words at most; else empty>",
  "effects": [<one entry for each lasting thing the deed does, in order; empty if it changes nothing that lasts>
     {{"thing": "<a thing, by its exact name on the stage>", "from": "<who holds it now, by name, or the word here if it lies here>", "to": "<the person who now holds it, by name, or the word here for lying here>"}},
     {{"thing": "<exact name>", "state": "<how the thing now is, in two to eight words. It takes the place of what the stage says of the thing: carry over what is still so, drop what no longer is>"}},
     {{"thing": "<exact name>", "held_out_to": "<the person it is held out or shown to>"}},
     {{"thing": "<exact name>", "on": "<the person it is now worn by or bound on, by name>"}},
     {{"thing": "<exact name of a written thing>", "read_by": ["<each person who now knows what it says: {name} when {name} reads it, and each person here it is read out to, aloud>"]}},
     {{"who": "<the person whose need it is>", "need": "<one of the needs this world keeps count of, by that name>", "change": "<eased or worsened>", "how_far": "<a little, well or fully>"}},
     {{"thing": "<exact name>", "gone": "<how it leaves the play: eaten, drunk, burnt, sold to the dealer>"}},
     {{"money": <the sum, a number>, "from": "<who pays: a person here by name, or in plain words someone outside the scene>", "to": "<who is paid, likewise>"}},
     {{"stands": "<what is now settled, one plain sentence with the day>", "about": ["<each person it concerns, by name>"], "ends": "<a sentence under What stands that this ends, word for word; else empty>"}},
     {{"sent": "<the words sent>", "to": "<the person they reach, by name, wherever they are>", "by": "<the means the facts give>"}},
     {{"chance": "<how likely it goes as the doer means: rare, unlikely, as likely as not, likely or almost certain>", "if_so": [<the effects if it does, in these same forms>], "so_page": "<what is then seen, one short sentence>", "if_not": [<the effects if it does not>], "not_page": "<what is then seen>"}},
     {{"found": "<a thing listed as hidden here, by its exact name, that a search brings to light>"}},
     {{"danger": "<only with a danger on the stage: the word over, when this ends it or puts those it is upon beyond it for good>", "how": "<what is seen of its going, one short sentence>"}},
     {{"who": "<a person here, by name>", "state": "<how the deed leaves that person, to the eye, as it now is, two to six words>", "placed": <true when it is how or where the person is placed here (up on the rock, behind the shelves), over when they move off; false when it is on the body or clothes and goes with them>, "ends": "<a state the stage shows on that person that this undoes or takes the place of, word for word; else empty>"}},
     {{"who": "{name}", "posture": "<standing, sitting, kneeling, lying down>"}},
     {{"who": "{name}", "touches": "<the person touched>", "how": "<two to four words>"}},
     {{"mark": "<a trace the deed adds to the place itself and that stays (what is cut, pressed, spilt, scraped, strewn or moved aside there), as it now is, two to six words. A touch, a footstep, a thing looked at leaves none; and what can be put back by doing the opposite is no mark but a thing with a state>", "ends": "<a trace the stage lists here that this undoes or takes the place of, word for word; else empty>"}},
     {{"place": "<a place this deed reaches or makes, which the stage does not name, two or three plain words>", "kind": "<what kind of place, one or two words>", "at": "<the named place it is inside or beside, by its exact name>", "inside": true or false, "metres": <how far from that place, when beside it>, "enclosed": true or false}},
     {{"who": "{name}", "goes": "<a place named on the stage (one a few steps off is stepped over to, and the scene goes on), a place this deed has made, or the words out of the scene>", "with": ["<those {name} asks along>"]}},
     {{"new": "<a thing now made or picked, or taken up from what this place or a person here plainly has; or a part of the place whose state this deed changes and the stage does not list yet>", "from": "<the place, the person, or the things here, it comes from>", "fixed": <true when it is part of the place and cannot be carried off; else false>, "was_absent": "<what the stage lists as found not to be here that this deed has now made, word for word; else empty>"}}],
  "minutes": <about how long the deed takes; 0 for a moment>,
  "held": "<each thing held that bears on this deed, and how it stands; empty if none bears>",
  "left_out": "<the part of the deed that is not done, in a few words; empty if all of it is done>",
  "seen": "<only when the deed cannot be done, or a part of it is left out: what those present see instead, one plain sentence in the third person and present tense that names nothing which is not here (Joe crouches by the cap and finds only bare soil); empty when there is nothing to see>",
  "absent": "<only when it cannot be done, or a part is left out, because a thing, mark or trace of the place itself is not there (on the ground, a wall or a tree, lying or standing here): that thing, two to six words (a boot print by the cap; a stack of firewood). Empty when the reason is anything else: what a person does or does not hold, wear or have on them, which the stage shows already; a thing another person holds; a thing done already or somewhere else>",
  "page": "<what the audience sees happen: {name}'s deed in the third person and present tense, only what is done, no words spoken, 30 words at most>"}}
</end>"""

# A ruling that named a thing the stage does not list was sent back with the bare fact, named the same thing again, and
# the deed was lost: a shoe taken off, an apple off its tree, a door (9 recorded: logs/review-20261126/hardest/again.py).
# It is sent back with how such a thing is given. The fault itself stays short: it is the reason the actor and the
# record are given when the deed is still lost.
NOT_LISTED = "there is no '{thing}' held by anyone here or lying here"
HOW_GIVEN = ("A thing entry is for a thing the stage lists, by the name it is listed under. What this deed makes, picks, or takes up from a place, "
             "from a thing here or from what a person here plainly has on them is given first, as a new entry with where it comes from; "
             "what it does to the place itself or to a part of it is a mark; and what is not here at all cannot be used, which is said in can or left_out.")


def again_note(wrong):
    """What an answer is sent back with: each fault, and for a thing not listed, how such a thing is given."""
    lost = NOT_LISTED.split('{thing}')[1]
    return AGAIN.format(faults='\n'.join(f'- {w}.' for w in wrong) + ('\n' + HOW_GIVEN if any(w.endswith(lost) for w in wrong) else ''))


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


def _names(value):
    """The names a field gives: one, or a list of them."""
    return [str(n) for n in (value if isinstance(value, (list, tuple)) else [value]) if n]


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
        elif e.get('money') is not None:
            payer = _person(context, character, e.get('from'))
            if not isinstance(e['money'], (int, float)) or isinstance(e['money'], bool) or e['money'] <= 0:
                out.append('a sum of money is a number above nothing')
            elif payer is None and _person(context, character, e.get('to')) is None:
                out.append('money moves from or to someone who is here')
            elif payer is not None and getattr(payer, 'body', None) is not None and payer.body.money < e['money']:
                out.append(f"{payer.name} has only {payer.body.money:g} {payer.body.cfg.get('currency', '')}".strip())
        elif e.get('danger'):
            if not survival.active_threat_for(character):
                out.append('no danger is on the stage')
        elif e.get('stands'):
            if len(str(e['stands']).split()) < 3:
                out.append('what stands is said in a sentence')
        elif e.get('sent'):
            if next((c for c in context.everyone() if c.name.lower() == _low(e.get('to'))), None) is None:
                out.append(f"nobody called '{e.get('to')}' is in the play")
        elif e.get('chance'):
            if _low(e['chance']) not in survival.LIKELY:
                out.append(f"how likely is one of: {', '.join(survival.LIKELY)}")
            for branch in ('if_so', 'if_not'):
                out += faults(context, character, {'effects': e.get(branch) or []})
        elif e.get('need'):
            who = _person(context, character, e.get('who') or character.name)
            if who is None or getattr(who, 'body', None) is None or survival.need_called(who.body, e['need']) is None:
                out.append(f"'{e['need']}' is not one of the needs the stage shows for {e.get('who') or character.name}")
        elif e.get('found'):
            if not any(_low(p['name']) == _low(e['found']) for p in survival.lying(context, stage.here(character), hidden=True)):
                out.append(f"there is no '{e['found']}' hidden here")
            else:   # found by this ruling, it is there for the rest of it: a page found and pocketed in one answer was
                made.add(_low(e['found']))   # sent back twice as 'not held or lying here' (bellona-20261126-115753)
        elif e.get('mark'):
            pass   # a trace on the place: nothing it names has to be there
        elif e.get('who') and e.get('state'):
            if _person(context, character, e['who']) is None:
                out.append(f"nobody called '{e['who']}' is here")
        elif e.get('place'):
            if not world.makes_places(context):
                out.append('no place can be added to this world')
            elif world.called(context, e.get('at')) is None:
                out.append(f"there is no place called '{e.get('at')}' on the stage for '{e['place']}' to be inside or beside")
        elif e.get('thing'):
            if find(context, character, e['thing'], e.get('from')) is None and _low(e['thing']) not in made:
                out.append(NOT_LISTED.format(thing=e['thing']))
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


def loose_ends(context, character, found):
    """What an answer says it ends that the stage does not list, word for word: a trace at this place, or a state of
    a person. Asked about once with the answer's other faults (resolve, resolve_words); never a reason to refuse
    the deed. It went by with a trace event only, and what was to be ended stayed on the stage beside what replaced
    it: 'back room door bolted' and 'back room door unbolted, open' together, for every later ruling to choose from."""
    out, at = [], stage.here(character)
    for e in found.get('effects') or []:
        ends = survival._said(e.get('ends')) if isinstance(e, dict) else ''
        if not ends:
            continue
        if e.get('mark'):
            here = [m['words'] for m in ledger.of(context).marks if survival.at_hand(context, m['at'], at)]
            if ends.lower() not in [w.lower() for w in here]:
                out.append(f"'{ends}' is not a trace the stage lists here, so nothing was ended. The traces listed here: {'; '.join(here) or 'none'}. "
                           "Give the one this takes the place of word for word, or leave ends empty if it takes the place of none")
        elif e.get('who') and e.get('state'):
            who = _person(context, character, e['who'])
            have = [s['words'] for s in ledger.of(context).states if who is not None and s['who'] == who.name]
            if who is not None and ends.lower() not in [w.lower() for w in have]:
                out.append(f"'{ends}' is not a state the stage shows on {who.name}, so nothing was ended. Those it shows: {'; '.join(have) or 'none'}. "
                           "Give the one this takes the place of word for word, or leave ends empty if it takes the place of none")
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
    prompt = RESOLVE.format(world=f"{' '.join((context.initial_state or '').split())}\nThis world keeps count of: {keeper._tracks(context)}.",
                            stage=record.of(context, character, people(context, character), full=True), scene=scene,
                            name=character.name, deed=act.action, leaves=leaves) + note
    reply = context.llm.ask({}, [UserMessage(content=prompt)], tag='stage.resolve_deed', temp=0.2, stops=['</end>'], max_tokens=700)
    found = json_reply.parse(reply) if reply else None
    return found if isinstance(found, dict) and 'can' in found else None


def _draw(context, character, found):
    """Where the resolver left something to chance, the draw: each chance effect gives place to
    the effects of the way it fell, and what is then seen joins the page. The resolver says how
    likely; it does not choose (docs/world-facts-design.md)."""
    out, seen = [], []
    for e in found.get('effects') or []:
        if not (isinstance(e, dict) and e.get('chance')):
            out.append(e)
            continue
        p = survival.LIKELY.get(_low(e['chance']), 0.5)
        so = survival._rng.random() < p
        trace.event('chance', character=character.name, likely=_low(e['chance']), p=p, so=so, so_page=str(e.get('so_page') or '')[:200], not_page=str(e.get('not_page') or '')[:200])
        out += [x for x in (e.get('if_so') if so else e.get('if_not')) or [] if isinstance(x, dict) and not x.get('chance')]
        seen.append(' '.join(str(e.get('so_page' if so else 'not_page') or '').split()))
    found['effects'] = out
    if any(seen):
        found['page'] = ' '.join(filter(None, [str(found.get('page') or '').strip(), *seen]))
    return found


def resolve(context, character, act):
    """What this deed does, decided before it is played: {'can', 'why', 'effects', 'left_out',
    'page'}. An answer that cannot be applied is sent back once with what is wrong with it; one
    that still cannot be, or cannot be read, is a deed that cannot be done. A deed of which
    nothing is done and something is left out cannot be done either: the actor is told, and the
    page is not given a line about nothing."""
    found = _ask(context, character, act)
    wrong = faults(context, character, found) if found else ['the answer could not be read']
    loose = loose_ends(context, character, found) if found else []
    if found and (wrong or loose):
        trace.event('resolve_again', character=character.name, deed=act.action[:200], faults=wrong + loose)
        again = _ask(context, character, act, again_note(wrong + loose))
        if again is not None:
            found, wrong = again, faults(context, character, again)
    if not found or wrong:
        found = {'can': False, 'why': wrong[0], 'effects': [], 'left_out': '', 'page': ''}
    found['effects'] = [e for e in found.get('effects') or [] if isinstance(e, dict)]
    _draw(context, character, found)
    found['left_out'], found['page'] = str(found.get('left_out') or '').strip(), str(found.get('page') or '').strip()
    found['seen'], found['absent'] = str(found.get('seen') or '').strip(), str(found.get('absent') or '').strip()
    if found.get('can') and not found['effects'] and found['left_out']:
        found.update(can=False, why=str(found.get('why') or '').strip() or f"not here: {found['left_out']}")
    if found.get('can') and not found['page']:
        found.update(can=False, why='the deed came to nothing that could be shown')
    found['can'], found['why'] = bool(found.get('can')), str(found.get('why') or '').strip()
    trace.event('resolved', character=character.name, deed=act.action[:300], can=found['can'], why=found['why'],
                effects=found['effects'], left_out=found['left_out'], page=found['page'], minutes=found.get('minutes'), held=str(found.get('held') or '')[:300],
                seen=found['seen'][:300], absent=found['absent'][:120])
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
# The prompt is in two parts. ACT_RULES is the same for every line a speaker says to a hearer (the world, its rules,
# the form of the answer) and is sent first, so that a server that keeps what it is asked to (utils/llm_api.py
# CACHED_LEAD) reads it once; it was the last three fifths of every prompt, and
# read again on each of some 370 lines of a play. ACT_NOW is what changes with each line. Paired replay on 24 recorded
# lines, 2026-11-23: logs/review-20261123/cost.
_act_rules = _rules.replace('The stage above is the truth', 'The stage is the truth')   # the stage now comes after the rules
assert _act_rules != _rules
ACT_RULES = """You keep the world of an improvised play. A character is about to act: to say something, to do something, or both at once. Say what the act does in this world as it now stands, and what of its words may stand. The world, how it works and the form of your answer come first; after them the stage as it now stands, what {name} has lived and told, and the act itself.

<world>
{world}
</world>

How this world works:""" + _act_rules + """
The words:
- The words are what {name} says aloud, and only that. Anything else written into them, what is done or how it is said ('She snorts.', '*pats her pocket*', '(whispered)'), is a finding (aloud). The mark (lie) is not one: it stays. Words that are sung, recited or read out are spoken: marks set round them (asterisks, quotation marks) are no finding, only what is not spoken is.
- Words do nothing physical, with one exception: words that hand a thing the speaker holds to someone present, in so many words ("here, take half"), are that hand-over. Give it among the effects, and say in "page" what is seen.
- A person cannot be mistaken about itself at this moment, or about what is in plain sight before it. A finding is something the words state as true now that the stage or <lived> contradicts: what {name} holds or hands over (holds); how its own body is (body); where it or someone present is (where); a thing in plain sight at this place, or a state or trace of one, that the stage does not hold and a place of this kind would not ordinarily have (here: 'a clean-cut limb' where the stage has only torn bark); the danger on the stage, said to be gone, going or held off, or a danger said to be upon them that the stage does not show (danger); the hour or the light (time); a physical act said to be done or being done, here and now, that was not done (done); who made a trace or left a thing as it is, when the stage says it is {name}'s own doing or <lived> shows {name} saw it done, and the words give it to someone else or to a stranger (whose); what a written thing says (writing); he or she against the cast (gender); something under does_not_know stated as known, unless <lived> shows {name} has since seen or been told it (knows); something under What stands that {name} is named as knowing, said otherwise (stands: 'I signed him this morning' when nothing is signed; what {name} is not named as knowing, {name} may get wrong, and that is no finding); and what was said or done earlier in the play, told wrongly (past: 'the berry theory' when it was the mushrooms). A past finding needs <lived> to show what was said or done and the words to have it otherwise. <lived> holds only the latest of the play: what is not in it may have been said or done before it, and is no finding.
- What stands and {name} is not named as knowing is not {name}'s to know. {name} may be wrong about it, deny it or guess at it, and that is no finding of any kind: never put a line right from it. What {name} is marked as told on the stage, {name} has heard said, as far as the mark goes: {name} may say it, doubt it or deny it, and none of that is a finding.
- A person can be mistaken about what it cannot see for itself at this moment, and saying so is no finding: what is under, inside or behind a thing, or out in the dark; what something felt, heard or half seen is; who made a trace it did not see made. The words stay as they are, the world is not changed by them, and nothing is given as absent: whether it is so is settled when someone looks, by that deed.
- These are never findings: a plan, a wish, a fear, a memory of life before the play, a guess said as a guess, what someone makes of what is here (who did it, what it means: 'somebody pruned this tree'), a question, an order, an offer, what is said of another's mind, a figure of speech or a loose way of putting a thing ('it's nowhere now' of a thing put away, 'a whole minute' of a wait), a deliberate lie marked (lie). When you are not sure, it is no finding.
- What {name} tells of its own life before the play is new to the world unless it is in <told>: list each new fact plainly, in the third person. If the words tell {listener} as news something already in <told>, give those words under "retold". What {name} sings, recites or reads out is not a telling of {name}'s life: it is no new fact, and sung or recited again it is not retold.
- A promise is something the speaker says it will do, or agrees that they will do. Its kind: now, when it is to be done here in this scene (eat this, go in, set off); standing, when it is a rule or a condition to keep with nothing to do now ("no mushrooms", "wake me for my watch"); later, when it is for another time or place. A question, a proposal not yet agreed, or a demand on someone else is not a promise.

Answer in JSON only, in this form:
{{"can": true or false,
  "why": "<if a deed cannot be done: the world's reason, ten words at most; else empty>",
""" + _effects + """  "left_out": "<the part of the deed that is not done, in a few words; empty if all of it is done or there is no deed>",
  "seen": "<only when there is a deed and it cannot be done, or a part of it is left out: what those present see instead, one plain sentence in the third person and present tense that names nothing which is not here (Joe crouches by the cap and finds only bare soil); empty when there is nothing to see>",
  "absent": "<a thing, mark or trace of the place itself (on the ground, a wall or a tree, lying or standing here) that is not there, two to six words (a boot print by the cap; a stack of firewood): when that is why a deed cannot be done or a part of it is left out. Empty for anything else: what is only said to be here; what a person does or does not hold, wear or have on them, and how a thing that is here is (full, dry, cut), which the stage shows already; a thing another person holds; a thing done already or somewhere else>",
  "page": "<what the audience sees {name} do: third person, present tense, only what is done, no words spoken, 30 words at most; empty when nothing is done but speaking>",
  "findings": [{{"kind": "<aloud, holds, body, where, here, danger, time, done, whose, writing, gender, knows, stands or past>", "words": "<the words that say it, exactly>", "why": "<what the stage or the record shows instead, ten words at most>"}}],
  "line": "<only when there is a finding: the words as {name} can truly say them, the same line in the same voice with each finding put right from what the stage shows or left out, and nothing else changed; else empty>",
  "tells": ["<each new fact {name} tells of its own life before the play>"],
  "retold": "<the words that tell again as news what is already in told; else empty>",
  "promises": [{{"who": "<who is to do it>", "what": "<what, in a few words>", "kind": "<now, standing or later>"}}],
  "names": ["<each person the words tell the hearers the name of, as the cast names them>"]}}"""
ACT_NOW = """The stage is the truth of now: who is here, how each body is, who holds what, what lies here, and whose doing each trace and each state of a thing is.
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

Answer in JSON only, in the form given above.
</end>"""


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
    hearer = getattr(listener, 'name', 'the others')
    rules = ACT_RULES.format(world=f"{' '.join((context.initial_state or '').split())}\nThis world keeps count of: {keeper._tracks(context)}.",
                             name=speaker.name, listener=hearer)
    now = ACT_NOW.format(stage=about or record.of(context, speaker, others, full=True), name=speaker.name, listener=hearer, lived=lived,
                         told='\n'.join(f'- {t}' for t in told_to(context, speaker, listener)) or 'Nothing yet.',
                         unknown='\n'.join(f'- {m}' for m in missing) or 'Nothing listed.', words=words, deed=deed or '(does nothing; only speaks)', leaves='').replace('\n\n\nAnswer in JSON', '\n\nAnswer in JSON')
    def ask(note=''):
        reply = context.llm.ask({}, [UserMessage(content=rules), UserMessage(content=now + note)], tag='stage.resolve_say', temp=0.2, stops=['</end>'], max_tokens=700)
        found = json_reply.parse(reply) if reply else None
        if not isinstance(found, dict) or 'can' not in found:
            return None, []
        found['effects'] = [e for e in found.get('effects') or [] if isinstance(e, dict) and (not e.get('goes') or _step(context, speaker, e))]   # leaving is the actor's mark; a step within the scene is done
        return found, faults(context, speaker, found)
    found, wrong = ask()
    if found is None:
        return None
    loose = loose_ends(context, speaker, found)
    if wrong or loose:   # sent back once with what is wrong with it, as a deed is (resolve): a thing's name cut short lost a deed
        trace.event('resolve_again', character=speaker.name, deed=deed[:200], faults=wrong + loose)
        again, still = ask(again_note(wrong + loose))
        if again is not None:
            found, wrong = again, still
    if not wrong:
        _draw(context, speaker, found)
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
    found['seen'] = str(found.get('seen') or '').strip() if deed and (refused or left) else ''
    here = any(isinstance(f, dict) and f.get('kind') == 'here' for f in found.get('findings') or [])
    found['absent'] = str(found.get('absent') or '').strip() if (deed and (refused or left)) or here else ''
    found['line'] = str(found.get('line') or '').strip()
    found['findings'] = [f for f in found.get('findings') or [] if isinstance(f, dict) and str(f.get('words') or '').strip()]
    found['tells'] = [str(t).strip() for t in found.get('tells') or [] if isinstance(t, str) and t.strip()]
    found['retold'] = str(found.get('retold') or '').strip()
    found['promises'] = [p for p in found.get('promises') or [] if isinstance(p, dict) and str(p.get('what') or '').strip()]
    trace.event('resolved_words', character=speaker.name, words=words[:300], deed=deed[:300], refused=found['deed_refused'], effects=found['effects'], page=found['page'],
                findings=found['findings'], tells=found['tells'], retold=found['retold'], promises=found['promises'], seen=found['seen'][:300], absent=found['absent'][:120],
                minutes=found.get('minutes'), held=str(found.get('held') or '')[:300])
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

What has happened lately, as the record has it (the latest only: what is not in it may have happened before it):
<lately>
{lately}
</lately>

The cast as the play gives them:
<cast>
{cast}
</cast>

The places of this world (a scene is set at one of them, or at a new place inside or beside one):
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
- A trace of what someone did (a print, a cut, a stain, ash, a thing dropped) is not what a place ordinarily has: it is brought only where <things> holds it, or something under What stands names it or would plainly have left it at this same place. And what <stage> or <things> lists as found not to be there was looked for and is not there: the set-up may not have it, and it is not brought. Where the cast has only spoken of such a thing, the set-up may say that they look for it; that it is there is not so.
- It may also give a person what such a person would ordinarily have with them for what they have come to do (a student her essay and her bag, a clerk his pen and papers, a traveller a coat): say each as a brings entry with its holder, and it is in their hands. Not so given: a thing the world has elsewhere or in another's hands, money, food or drink in a world where they are hard come by, and a thing that would settle what the scene or the play is about (a signed contract, a deed to land, a key or a weapon someone lacks, a letter that tells a secret). A set-up that needs one of those is not so in that.
- It may say how a person looks, to the eye, where that lasts through the scene: what the stage shows of them; what <cast> says of them; a trace on them of what stands, as it is seen and never how it came; and how such a person would ordinarily look for what they have been doing (a writer's inked fingers, a walker's muddy boots). Say each that the stage does not show yet as a shows entry, and it is so of them from then on. Not so given: a wound or a hurt, or a mark that would settle what the scene or the play is about, unless What stands gives it.
- A scene is at one place. Where the set-up is at a place <places> does not list, which such a world would have inside or beside a listed one (the back room of a listed shop, a reception desk in a listed office, a shed behind a listed hut), say so as new_place, and it is made. A place that belongs to no listed place, or that this world would not have, is not made: the scene is set at the listed place nearest to what is meant.
- It may not move a thing the world has elsewhere, put in anyone's hands a thing barred above, say of a body's needs or hurts what the stage does not show, bring in a person who is not listed, or give a trace or a state of a thing to anyone but the one whose doing the stage says it is.
- What stands (in <things>) is so in this world, whoever knows it. A set-up may rest on it, may not have it otherwise, and may show what a place or a person would show of it (a pruned limb, a form on a desk: bring it, at the place that thing names, where the world does not yet hold it; a scabbed knuckle: a shows entry). What is to be seen of it, on a place or on a person, is seen by everyone there, the person too; how it came is known only to those named with it and those marked as told it on the stage: a set-up may not have anyone else know that or act on it. The set-up is read out to the audience and to everyone in the scene, so it puts into words only as much of what stands as everyone in the scene knows or is marked as told: where you put a set-up right, leave the wrong part out, and write nothing from What stands in its place.
- A thing <things> lists as not yet found is out of sight until a search turns it up. A set-up or a title that shows it, or says it is there, is given under not_so with the why 'hidden until found' and with absent left empty (the thing is there, only unseen), and the set-up as it may stand leaves it out. An ending may turn on whether it is found.
- It may not play the scene: what the cast says, decides, finds out or does to one another is theirs to play, and is taken out of the set-up.
- A danger the set-up has upon the cast is either given under danger, and is then on the stage from the scene's first moment, or is not so and is taken out: no set-up is read out with a danger in it that the world has not started.
- Time and walking between scenes are the world's to tell: the set-up does not say that needs were met on the way unless the stage shows them met.
- The title, the ending and each goal are held to the same: one that rests on something not so is put toward the same end with what is so. A person may still want, fear or suspect what is not so; a goal that is to show, find or use it may not. A goal is a want its character already has as the scene opens and could own to at that moment, from what it knows then. One that waits on a thing not yet found, on a writing its character has not read, or on what another character will do in the scene (to see what he does with what he finds) is written again as the want behind it (to miss nothing that turns up), with nothing in it that has yet to happen. So is an ending: it may ask whether a person does a thing, but one that takes as done what someone in the scene has yet to do, or that a thing not yet found has been found (whether Tom sees what Ann slips into her coat), is written again as the matter it would settle (whether anything Ann comes on is kept from Tom). Such an ending is put right in ending alone, not under not_so: nothing in the world is against it.
- Not_so is for what the world shows otherwise. What it does not show either way (who once lived here, what is past the trees) is not for you to settle; nor is what the cast have said, agreed or done earlier, which <things> and <lately> hold only in part: a set-up that speaks of it is not so only where one of them shows it went otherwise. What <cast> says of a person and of what they carry is so (pages in a notebook they do not remember writing), though the world does not hold it yet.
- The set-up and its entries are one account: whatever the set-up as it may stand says of how a person looks that lasts is on the stage already or has its shows entry.
- Change as little as you can: keep the planner's words wherever they are true.

Answer in JSON only:
{{"place": "<the one listed place this scene is set at, by its exact name; the word same when it is where the cast now stands; or the word new, with new_place>",
  "new_place": <null, or when place is new: {{"name": "<what it is called, two or three plain words>", "kind": "<what kind of place, one or two words>", "at": "<the listed place it is inside or beside, by its exact name>", "inside": true or false, "metres": <how far from that place, when beside it>, "enclosed": true or false, "description": "<one plain sentence, as the places above are described>"}}>,
  "brings": [{{"thing": "<a thing the set-up says is there, which the world does not yet have and such a place, or such a person, would ordinarily have or What stands says is there; never a thing listed as found not to be there; two to five words>", "fixed": true or false, "holder": "<the person who has it with them, by name; empty when it is at the place>"}}],
  "shows": [{{"who": "<a person of the cast, by name>", "state": "<how they look, to the eye, as it will still be so at the scene's end, two to six words (ink-stained fingers)>"}}],
  "danger": "<what is upon the cast, in three to eight words (a gaunt grey wolf at the clearing's edge), when the set-up as it may stand has one of the dangers above upon them now, near enough to strike, at an hour and in a place where this world says it comes; else empty. A danger the set-up has that cannot come now (the wrong hour, the wrong ground, a thing already mended) is not so: it is left out, and no other is put in its place. One that is heard from afar, feared, spoken of or somewhere off is not upon them>",
  "not_so": [{{"words": "<the words of the title, set-up, ending or a goal that cannot stand>", "why": "<what the world shows instead, ten words at most>", "absent": "<where the words cannot stand because a thing, mark or trace of the place itself is not there: that thing, two to six words (a boot print in the earth); else empty>"}}],
  "title": "<the title as it may stand>",
  "setup": "<the set-up as it may stand: the same words if nothing is wrong; else the same with each of not_so put right or left out>",
  "ending": "<the ending as it may stand; it may not need what nobody could do in this world>",
  "goals": {{"<a name from goals>": "<that goal as it may stand>"}}}}
</end>"""


def befalls(context, character, threat):
    """A danger left to itself has come to a head: the world says what it now does (how likely it
    is to reach someone, whom, how hard, or that it passes), the draw is made, and it is over.
    Ruled as a deed is, with nobody's deed in it; told in the ruling's own line."""
    from sim.agh import Act
    what = threat['hazard']['description']
    act = Act(mode='Do', action=f"(Nobody's deed.) {what[0].upper() + what[1:]}, upon {', '.join(threat['targets'])}, has been left to itself and now comes to a head.",
              actors=[character], reason='', duration=1, source=None, target=[])
    note = (f"\nThis is not a deed of {character.name}'s or of anyone's: nobody has dealt with the danger, and its time has come. Say what it now does, as this world's facts "
            f"have such a danger do: in a chance effect, how likely it is to reach someone (the chance is of its passing them by; if_not is its reaching them: name whom, and the harm as a need effect), "
            f"and in page and the two page lines what the audience sees. Either way it is then over: end both branches with the danger over.\n")
    found = _ask(context, character, act, note)
    wrong = faults(context, character, found) if found else ['unread']
    if not found or wrong:
        trace.event('befalls', what=what, unread=True, faults=wrong)
        survival.danger_over(context, f"{what[0].upper() + what[1:]} passes.")
        context.to_page('', f"    {what[0].upper() + what[1:]} passes.", kind=None, transcript=False)
        return None
    found['effects'] = [e for e in found.get('effects') or [] if isinstance(e, dict)]
    _draw(context, character, found)
    if not any(e.get('danger') for e in found['effects']):
        found['effects'].append({'danger': 'over'})
    page = ' '.join(str(found.get('page') or '').split())
    trace.event('befalls', what=what, effects=found['effects'], page=page)
    act.found = found
    apply(context, character, act)
    if page:
        context.add_event(page)
        context.to_page('', f'    {page}', kind=None, transcript=False)
    return found


# The rules a set-up is held to, as the check has them: the stage manager is shown the same words (Context.reground_scene).
# It had rules of its own for what a set-up may bring ('an object, a source of water, food or shelter'), which allowed what
# these strike (food where it is scarce, a thing in someone's hands that settles the scene).
SETUP_RULES = '\n'.join(l for l in SETUP[SETUP.index('How a set-up stands:') + len('How a set-up stands:'):SETUP.index('\nAnswer in JSON only:')].strip().split('\n')
                        if not l.startswith('- Change as little as you can')).replace('{{', '{').replace('}}', '}')


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
                          lately=(' '.join(getattr(context, 'act_scene_wrapups', [])[-2:]) + '\n' + context.recent_events_text(6)).strip() or 'Nothing yet.',   # as the stage manager is shown it (context.reground_scene): the check had no record, and struck 'the handwriting sample he promised' that the scene before had ended on
                          cast='\n'.join(f"{c.name}: {' '.join(str(getattr(c, 'original_character', '') or getattr(c, 'character', '')).split())}" for c in cast) or 'none listed',   # whole: cut at 700 characters, a sketch lost its end
                          places=stage.gazetteer(context, bearings=True) or 'none listed', hazards=('\n' + '\n'.join(f'- {h}' for h in hazards) if hazards else 'none'),
                          title=scene.get('scene_title', ''), location=scene.get('location') or 'same',
                          setup=scene.get('pre_narrative', ''), ending=scene.get('post_narrative', ''),
                          goals='\n'.join(f'{n}: {g}' for n, g in planned.items() if g) or 'none given')
    reply = context.llm.ask({}, [UserMessage(content=prompt)], tag='stage.resolve_setup', temp=0.2, stops=['</end>'], max_tokens=900)
    found = json_reply.parse(reply) if reply else None
    if not isinstance(found, dict) or not str(found.get('setup') or '').strip():
        return None
    place = world.called(context, str(found.get('place') or '').strip())
    found['place'] = place.id if place is not None else ''
    new = found.get('new_place') if place is None and world.makes_places(context) else None
    # a place the set-up is at and the world does not list: made when the scene is set (apply_setup)
    found['new_place'] = new if (isinstance(new, dict) and str(new.get('name') or '').strip() and world.called(context, new.get('at')) is not None) else None
    found['brings'] = [b for b in found.get('brings') or [] if isinstance(b, dict) and str(b.get('thing') or '').strip()][:4]
    names = {c.name.lower() for c in cast}
    found['shows'] = [s for s in found.get('shows') or [] if isinstance(s, dict) and _low(s.get('who')) in names and str(s.get('state') or '').strip()][:4]
    found['not_so'] = [n for n in found.get('not_so') or [] if isinstance(n, dict) and str(n.get('words') or '').strip()]
    found['setup'], found['ending'] = ' '.join(str(found['setup']).split()), ' '.join(str(found.get('ending') or '').split())
    found['title'] = ' '.join(str(found.get('title') or '').split()).strip('"\'')
    danger = _low(found.get('danger'))
    found['danger'] = ' '.join(str(found.get('danger') or '').split()).strip(' .')[:90] if hazards and len(danger.split()) > 1 else ''
    goals = found.get('goals') if isinstance(found.get('goals'), dict) else {}
    found['goals'] = {name: ' '.join(str(g).split()) for name in planned for k, g in goals.items()
                      if _low(k) == name.lower() and str(g or '').strip() and ' '.join(str(g).split()) != planned[name]}
    trace.event('resolved_setup', scene=scene.get('scene_title'), place=found['place'], new_place=found['new_place'], brings=found['brings'], shows=found['shows'], danger=found['danger'],
                not_so=found['not_so'], title=found['title'], setup=found['setup'][:600], ending=found['ending'][:300], goals=found['goals'])
    return found


def apply_setup(context, scene, cast, found):
    """Put what a set-up brings into the world, at the scene's place (inside the scene's 'set'
    event). Returns the names of the things made."""
    new = found.get('new_place')
    if new:
        made_place = world.make_place(context, str(new['name']), str(new.get('kind') or ''), str(new['at']), inside=bool(new.get('inside')),
                                      metres=new.get('metres') if isinstance(new.get('metres'), (int, float)) else None,
                                      enclosed=bool(new.get('enclosed')), description=' '.join(str(new.get('description') or '').split()))
        if made_place is not None:
            found['place'], found['made_place'] = made_place.id, True
    place = world.place(context, found['place'])
    at = place.xy if place is not None else stage.here(cast[0])
    made = []
    for n in found.get('not_so') or []:   # a thing the set-up had there and the check found is not: kept for every later ruling (sim/ruling.py)
        if str(n.get('absent') or '').strip():
            ruling.not_there(context, n['absent'], at, by='setup')
    for b in found['brings']:
        # in the hands of the one the set-up has it with (a student's essay), or at the place
        holder = next((c for c in cast if c.name.lower() == _low(b.get('holder')) and getattr(c, 'body', None) is not None), None)
        if (survival.add_thing(context, str(b['thing']), holder=holder) if holder is not None
                else survival.add_thing(context, str(b['thing']), at=at, fixed=bool(b.get('fixed')))):
            made.append(str(b['thing']))
    # how the set-up has a person look is so of them from here on, as a deed's leaving them so is: the audience was
    # told 'ink wet on his fingers', the stage did not hold it, and his own word of it was later ruled not so
    for s in found.get('shows') or []:
        who = next((c for c in cast if c.name.lower() == _low(s.get('who'))), None)
        if who is not None:
            survival.leave_state(context, who, str(s['state']))
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
        elif e.get('thing') and e.get('gone'):
            out.append(f"{e['thing']} gone ({e['gone']})")
        elif e.get('thing') and e.get('state'):
            out.append(f"{e['thing']} {e['state']}")
        elif e.get('thing') and e.get('read_by'):
            out.append(f"{e['thing']} read by {', '.join(_names(e['read_by']))}")
        elif e.get('who') and e.get('state'):
            out.append(f"{e['who']} {e['state']}")
        elif e.get('need'):
            out.append(f"{e.get('who', '')} {e['need']} {e.get('change', 'eased')}".strip())
        elif e.get('money') is not None:
            out.append(f"{e['money']} from {e.get('from') or '?'} to {e.get('to') or '?'}")
        elif e.get('danger'):
            out.append(f"the danger {e['danger']}")
        elif e.get('stands'):
            out.append(f"now stands: {e['stands']}")
        elif e.get('sent'):
            out.append(f"sent to {e.get('to') or '?'}")
        elif e.get('posture'):
            out.append(f"{e.get('who', '')} {e['posture']}".strip())
        elif e.get('mark'):
            out.append(f"left: {e['mark']}")
        elif e.get('place'):
            out.append(f"a place now there: {e['place']}")
        elif e.get('new'):
            out.append(f"now there: {e['new']}")
        elif e.get('goes'):
            out.append(f"{e.get('who', '')} goes {e['goes']}".strip())
    return '; '.join(out)


def _step(context, character, e):
    """The place a goes effect names, when it is only a step from where the doer stands: in sight
    and near enough to be walked over to in the course of a deed (across a room, to the tree a few
    yards off). The doer is then there, and still in the scene."""
    p = world.called(context, str(e.get('goes') or '').strip())
    return p if p is not None and world.can_walk_over(context, stage.here(character), p.xy) else None


def leaving(context, character, found):
    """Who goes out of the scene by this deed, and where to: (None | 'alone' | [characters], place or None).
    Going to another place is going out of the scene: a scene has one place."""
    goes = next((e for e in found.get('effects') or [] if e.get('goes')), None)
    if goes is None or _step(context, character, goes) is not None:
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
    minutes = found.get('minutes')
    if isinstance(minutes, (int, float)) and not isinstance(minutes, bool) and minutes > 0:
        # the time the deed takes, as the resolver gave it
        context.simulation_time += survival.timedelta(minutes=min(float(minutes), 480))
    for e in found['effects']:
        if e.get('money') is not None:
            # money moves between people here, or to or from someone outside the scene (a bar, an
            # employer): it leaves one purse and comes into another, and is made nowhere
            payer, paid = _person(context, character, e.get('from')), _person(context, character, e.get('to'))
            for who, sign in ((payer, -1), (paid, 1)):
                if who is not None and getattr(who, 'body', None) is not None:
                    who.body.money += sign * e['money']
            ledger.of(context).wrote('needs')
            trace.event('money', character=me, sum=e['money'], frm=str(e.get('from') or ''), to=str(e.get('to') or ''))
        elif e.get('stands'):
            # settled, and kept in the words it was settled in: known to those who were there
            if e.get('ends'):
                survival.no_longer_stands(context, e['ends'])
            survival.stands(context, e['stands'], about=[a for a in e.get('about') or [] if isinstance(a, str)],
                            known=[c.name for c in [character] + list(character.hearers())])
        elif e.get('danger'):
            if _low(e['danger']).startswith('over'):
                survival.danger_over(context, e.get('how'))
        elif e.get('sent'):
            # words that reach someone who is not here, by a means the world's facts give: they
            # are with that person from then on (on a phone, to be found by whoever looks)
            other = next((c for c in context.everyone() if c.name.lower() == _low(e.get('to'))), None)
            if other is not None and getattr(other, 'body', None) is not None:
                words = ' '.join(str(e['sent']).split())
                other.body.inventory.append(f"message from {me}: {words}"[:160])
                other.add_perceptual_input(f"A message from {me} reaches you: {words}", mode='internal')
                trace.event('sent', character=me, to=other.name, words=words[:300])
        elif e.get('need'):
            # a need eased or made worse, and how far, as the resolver judged it from the world's
            # facts: a fact about the person, in the body's own terms
            who = _person(context, character, e.get('who') or me) or character
            need = survival.need_called(who.body, e['need'])
            worse = _low(e.get('change')).startswith('wors')
            if need is not None:
                level = who.body.level(need)
                step = survival.HOW_FAR.get(_low(e.get('how_far')), survival.HOW_FAR['a little'])
                who.body.adjust(need, min(step, 0.3) if worse else -min(step, level))
                trace.event('need_changed', character=who.name, need=need, change='worsened' if worse else 'eased', by=me, was=round(level, 2),
                            now=round(who.body.level(need), 2), way=str(e.get('how_far') or ''))
        elif e.get('found'):
            thing = next((p for p in survival.lying(context, stage.here(character), hidden=True) if _low(p['name']) == _low(e['found'])), None)
            if thing is not None:
                thing.pop('hidden', None)   # in plain sight from now on
                trace.event('thing_found', character=me, thing=thing['name'])
        elif e.get('mark'):
            # a trace of this deed on the place, kept as this deed's doing and never as a thing
            # lying here (survival.leave_mark)
            survival.leave_mark(context, character, e['mark'], on=e.get('thing'), ends=e.get('ends'))
        elif e.get('who') and e.get('state'):
            # how the deed leaves a person, to the eye: on the stage with the body's bearing from now on
            who = _person(context, character, e['who'])
            if who is not None:
                survival.leave_state(context, who, e['state'], placed=bool(e.get('placed')), ends=e.get('ends'), by=character)
        elif e.get('goes'):
            near = _step(context, character, e)   # a few steps, within the scene: the doer is there (going further is leaving: `leaving`)
            if near is not None:
                survival.travel(character, near.xy)
        elif e.get('place'):
            # a place the deed reaches or makes, inside or beside one the stage names: in the world from now on
            world.make_place(context, str(e['place']), str(e.get('kind') or ''), str(e.get('at') or ''), inside=bool(e.get('inside')),
                             metres=e.get('metres') if isinstance(e.get('metres'), (int, float)) else None, enclosed=bool(e.get('enclosed')),
                             description=f"{str(e['place']).strip()}, {'in' if e.get('inside') else 'by'} {e.get('at')}")
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
            if e.get('gone'):
                # out of the play by this deed (eaten, sold to someone outside the scene): nowhere from now on
                thing = survival._thing(context, name, holder or character) if kind != 'here' else \
                    next((p for p in survival.lying(context, stage.here(character)) if _low(p['name']) == _low(name)), None)
                if thing is not None:
                    ledger.of(context).remove(thing)
                    trace.event('thing_gone', character=me, thing=name, how=str(e['gone'])[:60])
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
                    # read out: each hearer the ruling names knows what it says from now on
                    survival.read_out(context, character, name, [o for o in (_person(context, character, n) for n in _names(e['read_by'])) if o is not None and o is not character])
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
        elif e.get('new'):
            # made from what is here, not given by a world action (whose own gain put it in hand): the
            # keeper says what kind of thing it is and how it behaves in time, where that matters
            # (a fire); a thing it does not model is simply there
            kept = keeper.consider(context, character, act.action) if ablation.enabled('keeper') else {'can': False}
            fact, done = keeper.perform(context, character, act.action) if kept.get('can') else ('', False)
            if not done:   # it is here; a later effect of the deed puts it in a hand
                # fixed: a part of the place that a deed puts one way or another (how it is, is then its state, which takes the
                # place of the last: as traces, a door 'bolted' and 'unbolted, open' stood side by side for eight rulings)
                survival.add_thing(context, str(e['new']), at=stage.here(character), fixed=e.get('fixed') is True)
            if str(e.get('was_absent') or '').strip():   # what was found not to be here is here now, by this deed (sim/ruling.py)
                ruling.there_again(context, e['was_absent'], at=stage.here(character))
            elif fact:
                facts.append(fact)
    character.last_affordance = None   # what was done is told by the page line, not by the last part's own sentence
    if found['left_out']:
        character.add_perceptual_input(f"Of what you meant to do, this was not done, because it cannot be here: {found['left_out']}.",
                                       percept=False, mode='internal')
        if found.get('absent'):   # the part was left out for want of a thing: that it is not here is kept (sim/ruling.py),
            # and said in the record those present have, as it is for a deed refused whole (record.refusal_line)
            if ruling.not_there(context, found['absent'], stage.here(character), seen_by=people(context, character), by='deed') is not None:
                context.add_event(f"{character.name} looked for it here and it was not there: {' '.join(str(found['absent']).split()).rstrip('. ')}.", doer=character)
    return ' '.join(facts)
