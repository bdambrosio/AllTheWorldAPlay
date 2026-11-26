"""The director of a play that stands on a question (docs/director-design.md 1.6 and 1.7). Every
play has one unless its file sets `W.directed = False`; it holds no question of its own.

The question is written here for each performance, from the sketch (Context.standing_question is
tonight's). Part of it asks what is so: that part is settled here, afresh for each performance,
before the first act. The rest asks what the people of the play do about it: that part is theirs,
in play, and nothing here plans it.

This is the light director. Three rounds of offline plans (logs/review-20261118/standing) showed
that the more a director settles, the more it writes what the characters did before the play and
the terms of their choice. So it settles little:

  question  the question the play stands on tonight, from what the sketch leaves unexplained
  answers   several answers the question could have tonight, written by one or more planner
            models; one is picked by code, never by a model's preference
  behind    the picked answer as a few facts (mostly about the place), the things that show
            them, and at most one person who knows a part and wants something of their own
  read      after each act, what of it the characters have now come to, by their own doing

No steer and no planned choice: the act planners are given the question and told not to answer it
for anyone (Context.episode_note). What is written here comes back in the line forms
Context.frame_episode already reads (#fact, #clue, #past), with #person added.

Played once (lost-20261119-084100), with the question still written in the play file.
"""
import os, re, json, random

SKETCH = """<setting>
{setting}
</setting>

<people>
{people}
</people>

<places>
{places}
</places>"""

QUESTION = """You are the director of an improvised play that is performed many times from the same sketch. The people and the place are the same every night. Before each performance you write, from the sketch, the question the play stands on that night. It is read to the audience before the first act.

""" + SKETCH + """

<questions_of_earlier_nights>
{earlier}
</questions_of_earlier_nights>

Write the question tonight's play stands on.

- It comes from what the sketch leaves unexplained for these people: what they do not know or cannot remember, how they come to be where they are, what this place or another person is to them. It takes up the largest of these, the one these people themselves most want answered, and need not take up all of it.
- It adds nothing to the sketch and supposes nothing the sketch does not say: no person, thing, place or event of your own, and no hint of an answer.
- It has two parts. The first asks what is so as the play begins: something that has already happened or is already the case, that these people do not know, and that can be found out during the play. It never asks what will happen tonight (whether they will reach a place, win a thing, be caught by the dark), and never what a person at the centre of the play feels or wants: those are the play's to show. The second part asks what these people do once they know, or if they never do: that is theirs to answer by what they do, and it is asked openly, not as a choice between two ways you name.
- It asks and does not answer, and leans toward no answer. Nothing in it says how the play comes out.
- It is two to four short questions in plain words, sixty words at most, naming the people as an audience would ('these two', or by name).
- It is not a question of an earlier night with the words changed: it comes at what the sketch leaves unexplained from another side, or asks after what these people do in another way.

Respond with one line only:
#question <the question>
</end>"""

WHOSE = """<standing_question>
{standing}
</standing_question>

Part of the standing question asks what is so. That part you settle tonight, as fact, and the people of the play do not know it. The rest asks what these people do about it. That part is theirs: they answer it in the performance, by what they do, and nothing you write says how, leans toward one way, or sets the terms of it."""

ANSWERS = """You are the director of an improvised play that is performed many times from the same sketch. The people and the place are the same every night. The question the play stands on tonight was written from the sketch and is given below. What is so behind it you settle afresh each night, and the performance is a different play.

""" + SKETCH + """

""" + WHOSE + """

Write five answers the standing question could have tonight.

- An answer is short: what is so, in two or three plain sentences. It settles the part of the question that asks what is so, and stops.
- It is mostly about this place and what stands in it now: what it is, who else is here, what can and cannot be done from here. It may say one thing about how these people came to be in it. It does not say what they wanted, agreed to or signed before the play, and it gives them no kin, debts, jobs or appointments the sketch does not: who they are is theirs to play.
- It contradicts nothing in the sketch: what each person knows and remembers, who is a stranger to whom, what the setting says is and is not here. Where the sketch says a person does not remember a stretch of time, the answer may settle one thing that happened in it, and no more time than that is lost.
- What these people can do about it is done here: with these places, what can be done at them, the dangers and needs this world has, and each other. Nothing turns on a gate, a hearing, money or a person beyond the places listed.
- It can be shown here: it needs nothing but the places listed, what can be done at them, things small enough to be handled, and people who can be met. Where the setting is itself out of the ordinary, the answer may be out of the ordinary in that setting's own way, and only as something a person standing there sees or hears. No other worlds, copies of people, or time that runs differently.
- It can be found a part at a time, from things and traces at the places listed, and not from one paper or one telling. {person_rule}
- The five are five different plays, not one told five ways: they differ in what this place is to these people and in what they would have to live with. If two could be the same play with a detail changed, replace one.

Respond with five lines only, each in this form:
#answer <number> | <what is so, two or three plain sentences> | found by: <two to four things or traces and the listed place of each> | person: <who knows a part, where they are, what they want, and what they need from these people; or none> | short: <the answer in one sentence of at most 25 words>
</end>"""

PERSON = ("One person besides those at the centre of the play may know a part of it: the owner of one of the places listed as having an owner nobody has yet been cast as, or another person the sketch lists. "
          "They are alive, awake and at a listed place as the play begins. They want something of their own that would be harder to have if these people knew all of it: "
          "so they are not there to explain, and nothing says what they will tell. They also need one thing these people could give them or do for them here, "
          "and for that they would part with a piece of what they know.")
NO_PERSON = "No person other than those at the centre of the play knows any of it."

REPEAT = """A play is performed many times from the same sketch, and each performance is to be a different play. Below are plays already performed, and candidates for tonight.

<performed>
{history}
</performed>

<candidates>
{candidates}
</candidates>

A candidate is the same play as one performed if what is so is the same with details changed. Sharing the people, the place, or a thing the sketch gives them does not make two plays the same.

Respond with one line for each candidate that is the same play as one performed, and nothing for the others:
#same <candidate number>
If none is, respond with the single line:
#none
</end>"""

BEHIND = """You are the director of an improvised play. The play stands on a question, written for tonight from its sketch. What is so behind that question tonight has been chosen and is given below. Settle it as a few facts and the things that show them.

""" + SKETCH + """

""" + WHOSE + """

<tonight>
{answer}
</tonight>

- Facts: three or four, each ONE thing that is so: a standing condition of this place, or a single thing that happened before the play began. Never what anyone will do or feel in the play, and never what the people at the centre of it wanted, agreed to or chose before it. Use only these people, these places (by their exact names) and the kinds of thing this world has. Say who knows each as the play begins: names from the people, or nobody. The people at the centre do not know what the sketch says they do not remember. One of the facts, and one only (two only where the first half of the question asks two things), answers the first half of the question as it is asked, in its own terms: asked whose print it is, that fact says whose foot made that print, and where. It says that one thing and nothing else: whatever else that person saw or did is another fact. Mark it answers: yes, and the others answers: no.
- Things: two or three small, ordinary things that really lie in this world, each at one of the places listed, each a trace of one fact. Each is an object that can be picked up, or a mark on the ground, a wall or a tree at that spot: never a view, a sound, a smell or a stretch of country. A thing shows what a person standing there would see; it does not explain. Its wording and its question name no other object that is not itself one of these things or a listed place: if marks lie under a sling, the sling is one of the things, or there is no sling. At most one is written, by a person, by hand, before the play, to someone and for a reason of their own: it says what that person would have written then and no more, and never the whole of what is so. Choose places these people can reach today. Where the first half of the question asks about a thing or a mark that the play does not already have (whose print, who wrote the note), one of two is so, and you say which by what you write: it is one of these things, at one of the places, and how it is to the eye is given (hobnail dots in the heel), with no word of whose; or there is no such thing, and the fact that answers says so plainly.
- A mark on a person: one at most, and only where one of the facts would have left on one of these people something still to be seen as the play begins (a scabbed knuckle, a stained cuff, a torn sleeve). It is what anyone looking at them sees, and they see it themselves; it does not explain, and how it came is known only to whoever knows the fact. It is a trace, not a hurt that hinders them. Do not also write it as a fact.
{person_rule}
- Then one thing from the own past of each of: {leads}. A single thing that happened to that person, or that they did, before the play and apart from all this: plain and particular (what, where, with whom), somewhere else in their life. It gives a reason for something their description says of them and contradicts nothing in it. Write none for a person whose description says they do not remember their past.
- Nothing here says what anyone will do in the play, or how it comes out.

Respond with these lines only, one sentence each:
#fact <one plain sentence> | known to: <names, or nobody> | answers: <yes or no>   (three or four lines)
#clue <the thing, by a name of two to four plain words with no comma> | <exact name of a place listed> | <what finding it makes someone wonder> | <if it is the written one: its words, in one or two plain sentences; else leave empty> | <loose if it can be picked up and carried; fixed if it is a mark or a part of the place> | <how it is, as someone standing there sees it, where that is what makes it a trace: three to eight words (struck through in heavy pen; one corner torn off); else leave empty>   (two or three lines)
#mark <exact name of one of the people> | <the mark as anyone looking at them sees it, three to eight words> | <what seeing it makes someone wonder>   (one line, or none)
{person_format}#past <name> | <one plain sentence>   (one line for each of: {leads}; none where the description rules it out)
</end>"""

# What would bring the one who knows a part to tell a piece of it. The person was written only as wanting
# what these people's knowing would cost them, and was told in every scene to hold it back: Agnes kept what
# took their day through twelve scenes and both parts of a thread, and three people guarding filled a Bellona
# scene of fifty lines. The need is theirs alone to know (it stands, known to them: Context.frame_episode), and
# what they tell for it is still theirs to play. (Bruce, 2026-11-25.) Replay: logs/review-20261125/needs.
NEEDS = ("Say also the one thing they need from the people at the centre, which those people could give them or do for them at these places "
         "(a hand with a job, a thing fetched, carried or mended, a watch kept): for that they would part with what they know of what the question asks, never a piece these people already have. "
         "It is something they cannot well do alone and would not ask of a stranger at once; it is not these people's silence or their going away, "
         "which the person wants already. It says what they need; whether and when they tell is theirs.")

PERSON_BEHIND = ("- A person who knows a part: the one named in what is so tonight, if it names one. Say who they are in one sentence as a description of a person (age, manner, what they do here), "
                 "the listed place where they are as the play begins, and the one or two things they want. They know only the facts you mark them as knowing. Nothing says what they will tell or do: they are played, like the others. "
                 + NEEDS)
PERSON_FORMAT = "#person <their own first name, one word: a person's name, never a place's or a role's> | <exact name of the listed place where they are> | <who they are, one sentence> | wants: <one or two things, separated by ';'> | needs: <the one thing they need from the people at the centre, ten words at most>   (one line, or none)\n"

READ = """You are the director of an improvised play. The play stands on a question, written for tonight from its sketch. What is so behind it tonight was settled before the play and is given below; the people of the play did not know it. An act has just been played.

""" + WHOSE + """

<what_is_so>
{facts}
</what_is_so>

<the_act_as_played>
{act}
</the_act_as_played>

<known_before_this_act>
{known}
</known_before_this_act>

Say what, of what is so, the people at the centre of the play have now come to, by what was done, found or said on the stage in this act or before it. Only what the account of the act shows them seeing, finding, reading or being told; nothing they could only have guessed. Say it as they would know it, which may be less than the fact. Then say whether this act brought them nearer to what is so, or went by without it.

Respond with these lines only:
#knows <name> | <what this person has now come to, one sentence; or nothing yet>   (one line for each of: {leads})
#nearer <yes or no>
</end>"""


# ------------------------------------------------------------------ reading the replies
def _lines(tag, text):
    return [m.strip() for m in re.findall(rf'^[ \t]*#{tag}[ \t]+(\S.*?)[ \t]*$', text or '', re.M)]


def _fields(line):
    return [f.strip() for f in line.split('|')]


def _labelled(parts, label):
    return next((p.split(':', 1)[1].strip() for p in parts if p.lower().startswith(label.lower() + ':')), '')


def read_answers(text, by=''):
    out = []
    for line in _lines('answer', text):
        f = _fields(line)
        if len(f) > 2 and f[1]:
            person = _labelled(f, 'person')
            out.append({'by': by, 'so': f[1], 'found': _labelled(f, 'found by'), 'short': _labelled(f, 'short'),
                        'person': '' if person.lower().strip(' .') in ('', 'none', 'nobody') else person})
    return out


def read_person(text):
    """The '#person' line of a behind reply as {'name', 'place', 'description', 'wants': [..]}, or None."""
    for line in _lines('person', text):
        f = _fields(line)
        if len(f) >= 3 and f[0] and f[0].lower().strip(' .') not in ('none', 'nobody'):
            wants = [w.strip() for w in _labelled(f, 'wants').split(';') if w.strip()]
            return {'name': f[0].strip(' ."\''), 'place': f[1], 'description': f[2], 'wants': wants[:2], 'needs': _labelled(f, 'needs').strip(' .')}
    return None


def read_needs(text):
    """What the one who knows a part needs from the people at the centre: [(name, the need)], from the
    '#person' line of someone new to the play or the '#needs' line of someone already in it."""
    out = []
    person = read_person(text)
    if person and person.get('needs') and person['needs'].lower() not in ('none', 'nothing'):
        out.append((person['name'], person['needs']))
    for line in _lines('needs', text):
        f = _fields(line)
        if len(f) > 1 and f[0] and f[1].strip(' .') and f[0].lower().strip(' .') not in ('none', 'nobody') and f[1].lower().strip(' .') not in ('none', 'nothing'):
            out.append((f[0].strip(' ."\''), f[1].strip(' .')))
    return out


def pick(candidates, same, rng):
    """One of the candidates that is not a repeat, by chance: not the model's favourite."""
    left = [c for n, c in enumerate(candidates, 1) if n not in same] or candidates
    return rng.choice(left) if left else None


# ------------------------------------------------------------------ the record of performances
def _history_path(play):
    return os.path.expanduser(f'~/.local/share/alltheworldaplay/performances/{os.path.splitext(os.path.basename(play or "play"))[0]}.jsonl')


def history(play, n=3, key='short'):
    try:
        return [e for e in (json.loads(l).get(key, '') for l in open(_history_path(play)) if l.strip()) if e][-n:]
    except OSError:
        return []


def remember(play, entry):
    try:
        path = _history_path(play)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except OSError:
        pass


# ------------------------------------------------------------------ in the engine
_pool = {}


def _backends(context):
    """[(name, llm, tag)]: the play's own planner, and any others named in ATWAP_DIRECTOR_LLMS
    (server names as utils/llm_api.LLM takes them, e.g. 'fireworks,haiku,sol'). Several models
    write the candidates so that no one model's favourite reading is every night's play."""
    from utils import llm_api
    out = [('planner', context.llm, 'director.answers')]
    for name in [n.strip() for n in os.environ.get('ATWAP_DIRECTOR_LLMS', 'fireworks,haiku').split(',') if n.strip()]:
        try:
            if name not in _pool:
                _pool[name] = llm_api.LLM(name)
            out.append((name, _pool[name], 'director.answers.pool'))   # not a planner tag: it goes to the model named
        except Exception as e:
            print(f'director: no backend {name}: {e}')
    return out


def _ask(llm, prompt, tag, max_tokens):
    from utils.Messages import UserMessage
    return (llm.ask({}, [UserMessage(content=prompt)], tag=tag, temp=0.7, stops=['</end>'], max_tokens=max_tokens) or '').replace('</end>', '').strip()


def question(context, setting, people, places):
    """The question the play stands on tonight, written from the sketch by one of the backends
    (which one is by chance); '' when it could not be made."""
    from sim.trace import trace
    name, llm, tag = random.Random(os.environ.get('ATWAP_SEED') or None).choice(_backends(context))
    earlier = history(context.play_file, key='question')
    reply = _ask(llm, QUESTION.format(setting=setting, people=people, places=places,
                                      earlier='\n'.join(f'- {q}' for q in earlier) or 'None: this is the first night.'), tag, 300)
    found = _lines('question', reply) or ([' '.join(reply.split())] if reply.strip().endswith('?') and '#' not in reply else [])   # the tag left off
    trace.event('director_question', by=name, question=found[0] if found else '', earlier=earlier)
    return found[0] if found else ''


def settle(context, setting, people, places, leads):
    """Settle what is so behind the standing question tonight. Returns the reply in the line forms
    frame_episode reads (#fact, #clue, #past, and #person), '' when it could not be made."""
    from sim.trace import trace
    standing = context.standing_question
    base = dict(setting=setting, people=people, places=places, standing=standing)
    seed = os.environ.get('ATWAP_SEED') or str(random.random())
    rng = random.Random(seed)
    with_person = rng.random() < 0.75
    prompt = ANSWERS.format(**base, person_rule=PERSON if with_person else NO_PERSON)
    candidates = []
    for name, llm, tag in _backends(context):
        candidates += read_answers(_ask(llm, prompt, tag, 1100), by=name)
    if not candidates:
        trace.event('director_failed', step='answers')
        return ''
    same, past = [], history(context.play_file)
    if past:
        reply = _ask(context.llm, REPEAT.format(history='\n'.join(f'{k + 1}. {h}' for k, h in enumerate(past)),
                                                candidates='\n'.join(f"{n}. {c['so']}" for n, c in enumerate(candidates, 1))), 'director.answers', 200)
        same = [int(m) for m in re.findall(r'^\s*#same\s+(\d+)', reply, re.M)]
    chosen = pick(candidates, same, rng)
    person = bool(chosen['person']) and with_person
    reply = _ask(context.llm, BEHIND.format(**base, leads=leads,
                                            answer=f"{chosen['so']} It can be found by: {chosen['found']}." + (f" A person who knows a part: {chosen['person']}." if person else ''),
                                            person_rule=PERSON_BEHIND if person else '- No person other than those at the centre of the play knows any of it.',
                                            person_format=PERSON_FORMAT if person else ''), 'director.behind', 900)
    trace.event('director', seed=seed, candidates=[{'by': c['by'], 'short': c['short'] or c['so']} for c in candidates], dropped=same,
                chosen=chosen, behind=reply)
    if '#fact' not in reply:
        trace.event('director_failed', step='behind')
        return ''
    remember(context.play_file, {'seed': seed, 'question': standing, 'short': chosen['short'] or chosen['so'], 'by': chosen['by']})
    return reply


def read(context, act_text, leads):
    """After an act: what of what is so each leading person has now come to, and whether the act
    came nearer to it. Kept on the screenplay (known, walked_past); {} when it could not be read."""
    from sim.trace import trace
    sp = context.screenplay
    facts = '\n'.join(f"- {f['text']}" for f in sp.facts if not f.get('of'))
    if not facts or not act_text:
        return {}
    known = getattr(sp, 'known', {}) or {}
    reply = _ask(context.llm, READ.format(standing=context.standing_question, facts=facts, act=act_text, leads=', '.join(leads),
                                          known='\n'.join(f'{n}: {k}' for n, k in known.items()) or 'Nothing yet.'), 'director.read', 300)
    for line in _lines('knows', reply):
        f = _fields(line)
        if len(f) > 1 and f[0] in leads and f[1].lower().strip(' .') not in ('nothing yet', 'nothing', 'none'):
            known[f[0]] = f[1]
    nearer = (_lines('nearer', reply) or ['yes'])[0].lower().startswith('y')
    sp.known = known
    sp.walked_past = 0 if nearer else getattr(sp, 'walked_past', 0) + 1
    trace.event('director_read', known=known, nearer=nearer, walked_past=sp.walked_past)
    return {'known': known, 'nearer': nearer}


# ------------------------------------------------------------------ what has been told
# Who knows a thing that stands is written when it comes to stand. After each scene the scene is
# read once for what of it someone was told or saw there, and that is kept with the thing
# (survival.tell): the one account every later reader has of who has it. (Nothing kept it: see
# survival.told_to.) Replayed on the scenes of three recorded plays: logs/review-20261125/told.
TOLD = """You keep the record of an improvised play. Some things are so in its world that not all of its people know. A scene has just been played. Say which of these things someone in the scene was told, or saw for themselves, in this scene.

<what_is_so>
{facts}
</what_is_so>

<the_scene_as_played>
{scene}
</the_scene_as_played>

- A person has been told a thing when someone said it to them or in their hearing, as so, in so many words: the thing itself, in the fact's words or in others. A hint at it, a question about it, a guess, or their own suspicion is not a telling. Whether they believe what they were told is theirs: told is told.
- A person has seen a thing when the scene shows them looking at what plainly is that thing (a store behind a mat that is pulled aside is the store, seen). A trace that leaves the thing to be worked out (bitten caps, a print) is not the thing seen; but a trace set beside what made it and plainly matching, before their eyes (a sole laid in its own print), is: they have seen whose it is.
- Where the whole of what a fact means was told or seen, say all, though a detail of it (the exact spot, the order, the hour) went unsaid. Where only a part was, say which part, in eight words at most, as that person now has it. A part is something the fact itself says (that the cave is stocked, of 'she keeps the cave stocked and hidden'); what only lies near it (that there is a cave, that it is hers) is no part and gets no line.
- Not told: a thing denied, refused ("not for telling") or put off; a thing said by someone who does not know it and is guessing; a thing said after the person had left the scene.
- Only the people named with each fact as not yet having it, and only what this scene shows. What a person was told before this scene is already given with the fact and needs no line.

Respond with one line for each telling or seeing, and nothing for the others:
#told <the fact's number> | <who was told or saw it: names, separated by commas> | <who said or showed it; or seen> | <the one word all; or the part, eight words at most> | <the words of the scene that show it, copied exactly, fifteen words at most>
If nobody was told or saw any of it, respond with the single line:
#none
</end>"""


def told_facts(entries, present, told_to, has_it):
    """(the lines for TOLD, the entries they number): the standing things someone present does not
    wholly have, each with who knows it, what each of the others has of it so far, and who is still without it."""
    rows, lines = [], []
    for s in entries:
        without = [n for n in present if not has_it(s, n)]
        if not without:
            continue
        rows.append(s)
        part = [f'{n} has been told: {told_to(s, n)}' for n in without if told_to(s, n)]
        lines.append(f"{len(rows)}. {s['fact']} [known to: {', '.join(s.get('known') or []) or 'nobody'}; not yet having it: {', '.join(without)}"
                     + (f"; {'; '.join(part)}" if part else '') + ']')
    return '\n'.join(lines), rows


def read_told(reply, rows, present):
    """The '#told' lines of a reply as [(entry, [names], by, what, words)], for the facts and people asked about."""
    by_lower = {n.lower(): n for n in present}
    out = []
    for line in _lines('told', reply):
        f = _fields(line)
        if len(f) < 4 or not f[0].strip().isdigit() or not 0 < int(f[0]) <= len(rows):
            continue
        names = [by_lower[n.strip().lower()] for n in re.split(r',|\band\b', f[1]) if n.strip().lower() in by_lower]
        if names:
            what = f[3].strip(' .') or 'all'
            out.append((rows[int(f[0]) - 1], names, f[2].strip(' .'), 'all' if re.match(r'all\b', what.lower()) else what, f[4].strip(' "\'') if len(f) > 4 else ''))
    return out


def told_in(context, cast, played, scene=None):
    """After a scene: what of the things that stand someone in it was told or saw, kept with each
    (survival.tell). The words quoted must be in the scene as played. Returns [(fact, names, what)]."""
    from sim import survival, ledger, record
    from sim.trace import trace
    present = [c.name for c in cast]
    lines, rows = told_facts([s for s in ledger.of(context).standing if not s['ended']], present, survival.told_to, survival.has_it)
    if not rows or not played:
        return []
    reply = _ask(context.llm, TOLD.format(facts=lines, scene=played), 'director.told', 500)
    out = []
    for entry, names, by, what, words in read_told(reply, rows, present):
        if not words or not record.quoted_in(words, played):
            trace.event('told_unbacked', fact=entry['fact'], who=names, words=words[:200])
            continue
        new = survival.tell(context, entry, names, by=by, what=what, words=words, scene=scene)
        if new:
            out.append((entry['fact'], new, what))
    return out


# ------------------------------------------------------------------ what is still promised
# The stage keeps what is promised so that later scenes are held to it, and nothing took a promise
# off it again: a play ended with 46, 'wait under cover for an hour' and 'step where Agnes steps'
# among them, hours after the storm and the walk. After each act the promises are read against the
# act as played, and those that bind nobody now are dropped. (A thing promised for now, in a scene,
# goes by rule: sim/ledger.py promised.)
SPENT = """An improvised play is being performed, and an act of it has just ended. The stage keeps what its people have promised or agreed, so that later scenes are held to it. You read the act as it was played and say which of these promises no longer binds anyone.

<the_act_as_played>
{played}
</the_act_as_played>

<promises>
{promises}
</promises>

It is now {now}.

- A promise is spent when the act shows that it binds nobody now. It was kept, broken or given up; or what happened overtook it: the thing it was about is done, gone or out of reach, the occasion it was for is over (a wait that has ended, a walk that has been made, a place they have left), or a later agreement took its place.
- It is not spent while it is still to be kept (a thing for later whose time has not come, a thing owed), nor while it is a condition still in force (what not to ask, where not to go, keeping together).
- Where two of them promise the same thing in other words, they are one promise: the earlier is spent.
- Judge only from what the act shows. Where it does not show that a promise was kept or is over, the promise is not spent.

Respond with one line for each spent promise, and nothing for the others:
#spent <its number> | <what in the act ended it, in a few words>
If none is spent, respond with the single line:
#none
</end>"""


def promise_line(p):
    at = p.get('since')
    at = at.strftime('%H:%M') if hasattr(at, 'strftime') else str(at or '')[11:16]
    return (f"{p['who']}" + (f" to {', '.join(p['to'])}" if p.get('to') else '') + f" ({p.get('kind', 'later')}" + (f", said at {at}" if at else ', said in an earlier part')
            + f"): {p['what']}")


def spent(llm, promises, played, now):
    """Which of `promises` (lines) the act as played has left binding nobody: [(index from 0, why)]."""
    from utils.Messages import UserMessage
    if not promises or not played:
        return []
    reply = llm.ask({}, [UserMessage(content=SPENT.format(played=played, now=now, promises='\n'.join(f'{n}. {p}' for n, p in enumerate(promises, 1))))],
                    tag='director.spent', temp=0.2, stops=['</end>'], max_tokens=700) or ''
    found = [(f[0], f[1] if len(f) > 1 else '') for f in (_fields(l) for l in _lines('spent', reply))]
    return [(int(n) - 1, why) for n, why in found if n.isdigit() and 0 < int(n) <= len(promises)]



# ====================================================================== the thread director
# A performance may be continued in further parts (sim/thread.py). The director of a first night
# writes its question from the sketch; the director of a later part writes it from what the play
# has left open. Five calls, before the part's first act:
#
#   close    reads the part just played: which questions it closed, what it raised and left open,
#            which promises are spent
#   next     the question the next part stands on, taken from the open ones; when the part opens
#            and what the time between did; a recap for the audience
#   answers  where the question's answer is not yet settled: five answers that go behind all that
#            already stands, one picked by chance (as on a first night)
#   behind   the picked answer as new facts and what shows them; it may bring in one person from
#            outside and one new place
#   check    each new line read against all that stands, and struck if it does not fit; an answer
#            left with no fact is dropped for another, once
#
# Nothing is planned ahead: no question is given a part to be answered in. The questions are a
# list, each with the part that raised it; a part takes one up and is to close it, and may open
# others. A question the whole performance stands on ('thread') is brought nearer in any part and
# stood on only in the last, or when no smaller one is left.
PARTS = """You are the director of an improvised play that is performed in parts, each part on another day. Its people and its place come from a sketch; what has happened in the parts already played is the play's own, and stands."""

CLOSE = PARTS + """ A part has just been played. Before the next is planned you read this one: which of the play's questions it closed, what it raised and left open, and which promises are spent.

""" + SKETCH + """

<questions_so_far>
{questions}
</questions_so_far>

<what_is_so>
{facts}
</what_is_so>

<the_play_so_far>
{story}
</the_play_so_far>

<the_last_part_as_played>
{transcript}
</the_last_part_as_played>

<what_each_person_still_holds>
{holds}
</what_each_person_still_holds>

<promises>
{promises}
</promises>

- Closed: a question so far is closed when the people at the centre of the play now know what is so behind it, by what they saw, found or were told on the stage. Knowing a part of it does not close it: where a fact of what_is_so is marked as answering the question, it is closed only when that fact is marked as told on the stage, all of it, to the people at the centre, or the part as played shows them seeing it. Only the half of a question that asks what is so can be closed; the half that asks what these people do never is.
- Open: the questions this part raised and did not answer, and anything the sketch leaves unexplained that no question so far has taken up. Three to five: the ones these people themselves most want answered. Each asks what is so (something already the case, or that has already happened, and that can be found out), never what will happen or what someone will choose. Each is in plain words, 25 at most, as the audience would ask it, and is not a question so far said again. It supposes nothing that what_is_so rules out, and nothing that what_is_so lists as looked for and found not to be there (people speaking of a print that was not there leaves no question of whose print it was): where it rests on something a person only said of themselves or guessed (that she woke knowing a name), it asks whether that is so, never why.
- For each open question, whose it is: 'thread' if it asks how these people come to be here or what this place is, so that the whole performance stands on it; 'part' if a single part could close it. And what lies behind it: 'settled' only if one numbered fact of what_is_so already gives the answer itself, though these people do not know it, and then with that fact's number (settled 3); 'unsettled' if what_is_so says nothing of it, or says only that there is something (that a person hides from a life left behind, and not what that life was).
- Still: each question so far that is not closed, read again against what_is_so as it now is. 'settled' and 'unsettled' as above. 'void' when what_is_so rules out what the question supposes, so that it can have no answer (it asks why a thing was so that was not so): say what rules it out.
- Spent: a promise is spent when it has been kept, broken, or overtaken by what happened, so that it binds nobody now. One still to be kept, and a standing condition still in force, are not spent.
- Add nothing: no event, thing, person or answer that the play has not shown.

Respond with these lines only:
#closed <number of a question so far> | <what these people now know, one sentence>   (one line for each closed, or no line)
#open <the question> | <thread or part> | <settled and the number of the fact that answers it (settled 3); or unsettled> | <what in the play raised it, a few words>   (three to five lines)
#still <number of a question so far that is not closed> | <settled and the number of the fact that answers it (settled 3); unsettled; or void> | <for void: what rules it out; else leave empty>   (one line for each, or no line)
#spent <the numbers of the spent promises, separated by commas; or none>
</end>"""

NEXT = PARTS + """ {played} been played. You open the next: the question it stands on, and when it begins.{final}

""" + SKETCH + """

<open_questions>
{questions}
</open_questions>

<what_is_so>
{facts}
</what_is_so>

<the_last_part_as_played>
{transcript}
</the_last_part_as_played>

<how_the_last_part_ended>
{ending}
</how_the_last_part_ended>

<how_people_look>
{looks}
</how_people_look>

- The question is read to the audience before this part's first act. Its first half asks what is so: take it from the open questions, one of them or two that belong together, in words an audience would use. {which} Its second half asks what these people do once they know, or if they never do: asked openly, not as a choice between two ways you name. It is two to four short questions, sixty words at most, and it asks without answering or leaning toward an answer.
- When the part opens: the same hour, later that day, the next morning, or days on. Go on from the same hour when the last part stopped in the middle of something. Let time pass when it ended at rest, or when what the question needs takes time to come (dark, weather, a habit to be noticed, someone's return). No more time than that: these people must plainly still be here, as they are.
- Between: if time passes, what it did that anyone could see, in one sentence: the hours gone, the light and the weather now. That these people slept or ate may be said; where, how, and with whom is said only as far as the last part had already settled it. Nothing found, said, done, decided, promised or learned: whatever matters happens on the stage.
- Gone: of the ways people look, those that would not have lasted through that time.
- Recap: for an audience that may not have seen the earlier parts, what has happened and what these people now know, as the last part as played shows it (what was only offered or promised is said as offered or promised, not as done), in 60 to 90 words, in the plain past tense, with no hint of what is to come and nothing these people do not know: a person whose name they have not heard is named as they would call them.

Respond with these lines only:
#question <the question this part stands on>
#takes <the numbers of the open questions it takes up, separated by commas>
#opens <YYYY-MM-DD HH:MM> | <why then, one sentence>
#between <one sentence; or nothing, when the part goes on from the same hour>
#gone <the numbers of the looks that are gone, separated by commas; or none>
#recap <60 to 90 words>
</end>"""

WHICH = "Prefer a 'part' question, and of those the one the last part's ending presses hardest; stand on a 'thread' question only when no 'part' question is left."
WHICH_FINAL = "This is the last part: it stands on the largest question still open, a 'thread' question where there is one, and may take a smaller one with it."

STANDS = """<what_already_stands>
{stands}
</what_already_stands>

<the_play_so_far>
{story}
</the_play_so_far>

<the_last_part_as_played>
{transcript}
</the_last_part_as_played>"""

ANSWERS_NEXT = PARTS + """ The next part stands on the question below. What is so behind it is not yet settled, or not all of it: you settle that now, behind everything that already stands.

""" + SKETCH + """

""" + STANDS + """

""" + WHOSE + """

Write five answers the standing question could have.

- An answer is short: what is so, in two or three plain sentences. It settles what the question asks that is not yet settled, and stops.
- It contradicts nothing that already stands, nothing in the sketch and nothing the play has shown, down to what was said and seen in the last part as played: it lies behind them, and makes sense of what was seen. What a person has told of their own life on the stage is theirs: an answer may go further back than it, and does not make it a lie.
- It is about this place and the people now in it: what it is, who else comes here, what can and cannot be done from here. It does not say what the people at the centre wanted, agreed to or signed before the play, and gives them no kin, debts, jobs, appointments or belongings that neither the sketch nor the play has given them.
- Where the sketch says a person does not remember a stretch of time, the answer may settle one thing that happened in it, and no more time than that is lost. It leaves these people as the play has shown them: nothing on them, about them or done to them that would have been seen or felt in the parts already played.
- It may bring one person into the play who has not been in it: someone from outside who has a reason of their own to come here now, or someone who has been here unseen. And it may need one place this world does not yet list, of one of the kinds given with the places, lying beside a listed place (where a trail comes out, the far side of a wall). Nothing turns on anyone or anywhere further off than that.
- What these people can do about it is done here: with these places, what can be done at them, the dangers and needs this world has, and each other.
- It can be found a part at a time, from things and traces at the places, and from people who can be met: not from one paper or one telling. A person who knows a part of it wants something of their own that would be harder to have if these people knew all of it, so they are not there to explain; and needs one thing these people could give or do here, for which they would part with a piece of it.
- The five are five different plays, not one told five ways: they differ in what this place is to these people and in what they would have to live with. If two could be the same play with a detail changed, replace one.

Respond with five lines only, each in this form:
#answer <number> | <what is so, two or three plain sentences> | found by: <two to four things or traces and the place of each> | person: <who knows a part, whether they are already in the play, where they are, what they want, and what they need from these people; or none> | place: <a place not yet listed: its kind and the listed place it lies beside; or none> | short: <the answer in one sentence of at most 25 words>
</end>"""

BEHIND_NEXT = PARTS + """ The next part stands on the question below. What is so behind it has been chosen and is given below. Settle it as a few new facts and the things that show them, behind everything that already stands.

""" + SKETCH + """

""" + STANDS + """

""" + WHOSE + """

<newly_so>
{answer}
</newly_so>

- Facts: two or three new ones, each ONE thing that is so: a standing condition of this place, or a single thing that happened before now and off the stage, its time said so that it fits the clock given with what already stands (the night before the play began, not 'last night'). None says again or contradicts what already stands or what was said and seen in the last part as played. Never what anyone will do or feel in the play, and never what the people at the centre wanted, agreed to or chose. Use these people, these places (by their exact names), the one new person and the one new place if there are any, and the kinds of thing this world has. Say who knows each as the part begins: names, or nobody. One of the facts, and one only (two only where the first half of the question asks two things), answers the first half of the question as it is asked, in its own terms: asked whose print it is, that fact says whose foot made that print, and where. It says that one thing and nothing else: whatever else that person saw or did is another fact. Mark it answers: yes, and the others answers: no.
- A new place, if what is newly so needs one: its name in two to four plain words, its kind (one of the kinds given with the places), the listed place it lies beside, and what it is, in one sentence. One at most.
- Things: one or two small, ordinary things that really lie in this world, each at a listed place or the new one, each a trace of one new fact. Each is an object that can be picked up, or a mark on the ground, a wall or a tree at that spot: never a view, a sound, a smell or a stretch of country. A thing shows what a person standing there would see; it does not explain. None is a thing the play already has, and none lies where these people have already looked and would have seen it: it is somewhere they have not been, was not to be seen from where they stood, or has come since. Choose places these people can reach in this part. Where the first half of the question asks about a thing or a mark that the play does not already have (whose print, who wrote the note), one of two is so, and you say which by what you write: it is one of these things, lying where a thing may lie by this rule, and how it is to the eye is given (hobnail dots in the heel), with no word of whose; or there is no such thing, and the fact that answers says so plainly.
{person_rule}
- Nothing here says what anyone will do in the play, or how it comes out.

Respond with these lines only, one sentence each:
#fact <one plain sentence> | known to: <names, or nobody> | answers: <yes or no>   (two or three lines)
#place <its name, two to four plain words> | <its kind> | <exact name of the listed place it lies beside> | <what it is, one sentence>   (one line, or none)
#clue <the thing, by a name of two to four plain words with no comma> | <exact name of a place, listed or new> | <what finding it makes someone wonder> | <if it is a written thing: its words, in one or two plain sentences; else leave empty> | <loose if it can be picked up and carried; fixed if it is a mark or a part of the place> | <how it is, as someone standing there sees it, where that is what makes it a trace: three to eight words; else leave empty>   (one or two lines)
{person_format}</end>"""

CHECK_NEXT = PARTS + """ New facts and things have been written for the next part. Before they are made so, you check each line against all that already stands, and strike what does not fit.

""" + SKETCH + """

""" + STANDS + """

<newly_written>
{written}
</newly_written>

Strike a line when:
- it says the contrary of something that already stands, of the sketch, or of what was said and seen in the last part as played;
- it puts on a person, or at a spot where these people have already been and looked, something they would have seen, felt or been told before now, and were not;
- it gives a time that does not fit the clock given with what already stands;
- it says what a person at the centre of the play wanted, chose or agreed to.
A line that only adds something behind what stands is not struck: a new person, a new place, a thing somewhere these people have not looked, one thing that happened in a stretch of time a person does not remember.

Respond with one line for each line struck, and nothing for the others:
#strike <number> | <what it does not fit, in a few words>
If none is struck, respond with the single line:
#none
</end>"""

AGAIN_NEXT = """
Of the lines you wrote, these were struck, because they do not fit what already stands:
{struck}
Write all your lines again. A fact that was struck is written so that it fits what stands and still settles the part of the question it was for. The fact that answers the question is never left out: where what you first wrote for it cannot fit what stands, the answer is another one that does (that there was no such thing, where that is what fits). Any other fact that can have no form that fits is left out, with nothing in its place.
</end>"""

PERSON_NEXT = ("- A person who knows a part: the one named in what is newly so, if it names one. If they are new to the play: their name, the place where they are as the part begins (listed, or the new one), "
               "who they are in one sentence as a description of a person (age, manner, what they do), and the one or two things they want. They know only the facts you mark them as knowing. Nothing says what they will tell or do: they are played, like the others. "
               + NEEDS + " If they are already in the play, write no #person line for them, only the #needs line: what they need now, as the last part left them.")
NEEDS_FORMAT = "#needs <the name of the person who knows a part, if they are already in the play> | <the one thing they need from the people at the centre, ten words at most>   (one line, or none)\n"


def _numbers(text):
    return [int(n) for n in re.findall(r'\d+', text or '')]


def question_lines(questions, open_only=False):
    """The thread's questions, numbered as they are kept (the numbers a reply gives back)."""
    out = []
    for n, q in enumerate(questions, 1):
        if open_only and q.get('closed'):
            continue
        came = '; '.join(f'{who}: {what}' for who, what in (q.get('come_to') or {}).items())
        out.append(f"{n}. [{q.get('whose', 'part')}; raised in part {q.get('raised', 1)}; what is so behind it: {'settled' if q.get('settled') else 'unsettled'}"
                   + (f"; closed in part {q['closed']}" if q.get('closed') else '') + f"] {q['text']}"
                   + (f"\n   What the people at the centre have come to: {came}" if came else ''))
    return '\n'.join(out) or 'None.'


def story_text(parts):
    """The parts already played, as they came out: each act's record and the choices it turned on."""
    out = []
    for p in parts:
        for a in (p.get('story') or {}).get('acts') or []:
            choices = '; '.join(f"{c.get('character')} chose: {c.get('choice')}" for c in a.get('choices') or [])
            out.append(f"Part {p.get('part', 1)}, act {a['number']}, '{a.get('title', '')}': {a.get('summary', '')}" + (f" ({choices})" if choices else ''))
    return '\n'.join(out) or 'Nothing has been played.'


def transcript_text(parts):
    """The last part as it was played, line by line ('' kept as 'Not kept.' for a thread written without it)."""
    return '\n'.join((parts[-1].get('transcript') or [])) or 'Not kept.'


def answering(text):
    """The '#fact' lines of a behind reply marked as answering the question: their sentences."""
    out = []
    for line in _lines('fact', text):
        f = _fields(line)
        if f and f[0] and _labelled(f, 'answers').lower().startswith('y'):
            out.append(f[0].strip(' "'))
    return out


def standing_facts(facts):
    """The facts that stand behind the play (nobody's own past), in the order they are numbered for CLOSE."""
    return [f for f in facts if not f.get('of')]


def facts_text(facts, numbered=False, questions=()):
    """numbered: the facts that stand are numbered (CLOSE names the one that answers a question by its number).
    questions: the thread's questions, so that a fact one of them is answered by says so."""
    def answers(f):
        ns = [str(n) for n, q in enumerate(questions or (), 1) if f['text'] in (q.get('answer') or [])]
        return f"; answers question {', '.join(ns)}" if ns else ''
    order = {id(f): n for n, f in enumerate(standing_facts(facts), 1)}
    def told(f):   # who has been told it on the stage (thread.so): all of it, or the part named
        said = f.get('told') or {}
        whole, part = [n for n, w in said.items() if w == 'all'], [f'{n}: {w}' for n, w in said.items() if w != 'all']
        return (f"; told on the stage, all of it, to {', '.join(whole)}" if whole else '') + (f"; told on the stage in part to {'; '.join(part)}" if part else '')
    return '\n'.join(f"{(str(order[id(f)]) + '.') if numbered and id(f) in order else '-'} {f['text']} [{(f['of'] + chr(39) + 's own past; ') if f.get('of') else ''}known to: {', '.join(f.get('known_to') or []) or 'nobody'}{told(f)}{answers(f)}]" for f in facts) or 'Nothing is settled.'


def read_close(text):
    closed = [(int(f[0]), f[1]) for f in (_fields(l) for l in _lines('closed', text)) if len(f) > 1 and f[0].strip().isdigit()]
    opened = []
    for line in _lines('open', text):
        f = _fields(line)
        if f and f[0]:
            num = _numbers(f[2])[:1] if len(f) > 2 and f[2].lower().startswith('settled') else []
            opened.append({'text': f[0], 'whose': 'thread' if len(f) > 1 and f[1].lower().startswith('thread') else 'part',
                           'settled': bool(num), 'fact': num[0] if num else None, 'raised_by': f[3] if len(f) > 3 else ''})
    spent = _numbers((_lines('spent', text) or [''])[0])
    # (settled without the number of a fact is not settled: 'settled' stood on a question nothing answered)
    def status(word):
        first = word.lower().split()[:1]
        return ('unsettled' if first == ['settled'] and not _numbers(word) else first[0]) if first and first[0] in ('settled', 'unsettled', 'void') else ''
    still = [(int(f[0]), status(f[1]), f[2] if len(f) > 2 else '', (_numbers(f[1]) or [None])[0] if status(f[1]) == 'settled' else None)
             for f in (_fields(l) for l in _lines('still', text)) if len(f) > 1 and f[0].strip().isdigit() and status(f[1])]
    return {'closed': closed, 'open': opened[:5], 'spent': spent, 'still': still}


def read_next(text):
    first = lambda tag: (_lines(tag, text) or [''])[0]
    opens = _fields(first('opens'))
    when = re.search(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}', opens[0] if opens else '')
    between = first('between')
    return {'question': first('question'), 'takes': _numbers(first('takes')), 'opens': when.group(0).replace(' ', 'T') if when else '',
            'why': opens[1] if len(opens) > 1 else '', 'between': '' if between.lower().strip(' .') in ('nothing', 'none', '') else between,
            'gone': _numbers(first('gone')), 'recap': first('recap')}


def read_place(text):
    """The '#place' line of a behind reply as {'name', 'kind', 'beside', 'description'}, or None."""
    for line in _lines('place', text):
        f = _fields(line)
        if len(f) >= 3 and f[0] and f[0].lower().strip(' .') not in ('none', 'nobody'):
            return {'name': f[0].strip(' ."\''), 'kind': f[1], 'beside': f[2], 'description': f[3] if len(f) > 3 else ''}
    return None


def close_part(context, sketch, parts, promises):
    """Read the part just played. Returns {'closed': [(number, how)], 'open': [...], 'spent': [numbers]}
    against the numbering of the last part's questions and of `promises` (lines); {} when it could not be read."""
    from sim.trace import trace
    last = parts[-1]
    holds = '\n'.join(f"{p['name']}: " + ' / '.join((p.get('carries_forward') or {}).get('holds') or ['nothing']) for p in last['people'] if p.get('carries_forward'))
    gone = absent_text(last)
    reply = _ask(context.llm, CLOSE.format(**sketch, questions=question_lines(last.get('questions') or []),
                                           facts=facts_text(last['so']['facts'], numbered=True, questions=last.get('questions') or []) + (f"\nLooked for on the stage and found not to be there: {gone}" if gone else ''),
                                           story=story_text(parts), transcript=transcript_text(parts), holds=holds or 'Not kept.',
                                           promises='\n'.join(f'{n}. {p}' for n, p in enumerate(promises, 1)) or 'None.'), 'director.close', 700)
    found = read_close(reply)
    trace.event('director_close', closed=found['closed'], open=found['open'], spent=found['spent'], still=found['still'], reply=reply)
    return found if found['open'] or found['closed'] else {}


def next_part(context, sketch, parts, questions, ending, looks, final=False):
    """Open the next part: {'question', 'takes', 'opens', 'why', 'between', 'gone', 'recap'}; {} when it could not be made."""
    from sim.trace import trace
    n = len(parts)
    reply = _ask(context.llm, NEXT.format(**sketch, played='One part has' if n == 1 else f'{n} parts have',
                                          final=' It is to be the last part of this performance.' if final else '',
                                          which=WHICH_FINAL if final else WHICH, questions=question_lines(questions, open_only=True),
                                          facts=facts_text(parts[-1]['so']['facts']) + (f"\nLooked for on the stage and found not to be there: {g}" if (g := absent_text(parts[-1])) else ''),
                                          ending=ending, transcript=transcript_text(parts),
                                          looks='\n'.join(f'{k}. {l}' for k, l in enumerate(looks, 1)) or 'Nothing is kept of how anyone looks.'), 'director.next', 700)
    found = read_next(reply)
    trace.event('director_next', final=final, reply=reply, **found)
    return found if found['question'] else {}


def absent_text(last):
    """What the stage has found not to be there, from a part's record (sim/ruling.py): '' when nothing."""
    gone = [a for a in (last.get('stage') or {}).get('absent') or [] if not a.get('ended')]
    return '; '.join(f"{a['what']} (at {a.get('place') or 'a place'})" for a in gone[-16:])


def held_text(last):
    """What the play already holds, by place, and what was looked for and found not to be there: the director was
    told that a new thing is none the play has, and lies where nobody has looked, and was not shown either."""
    stage = last.get('stage') or {}
    by = {}
    for t in stage.get('things') or []:
        if t.get('place'):
            by.setdefault(t['place'], []).append(str(t.get('name')))
    for m in stage.get('traces') or []:
        if m.get('place'):
            by.setdefault(m['place'], []).append(f"(a trace) {m.get('words')}")
    gone = absent_text(last)
    return '\n'.join(filter(None, [("What the play already has lying at its places:\n" + '\n'.join(f"- {place}: {'; '.join(names[:12])}" for place, names in by.items())) if by else '',
                                   f"Looked for on the stage and found not to be there (it is not there; nothing newly settled has it there, or rests on it): {gone}" if gone else '']))


def stands_text(parts, told=(), opens=None):
    """All that already stands, for the calls that settle more: the facts, what the people at the
    centre have come to, and what has been told on the stage of anyone's life."""
    last = parts[-1]
    came = {}
    for q in last.get('questions') or []:
        came.update(q.get('come_to') or {})
    told = list(told) or [f"{t['who']} told: {t['what']}" for t in (last.get('story') or {}).get('planted') or [] if t.get('kind') == 'told']
    lived = [f"- {p['name']}: {p['carries_forward']['lived']}" for p in last['people'] if (p.get('carries_forward') or {}).get('lived')]
    holds = held_text(last)
    clock = f"The last part ended at {str(last.get('clock') or '')[:16].replace('T', ' ')}" + (f"; the next opens at {opens.strftime('%Y-%m-%d %H:%M')}." if opens is not None else '.')
    return '\n'.join(filter(None, [clock, "So, behind the play (each known only to those named):\n" + facts_text(last['so']['facts']), holds,
                                   ("What the people at the centre have come to:\n" + '\n'.join(f'- {n}: {k}' for n, k in came.items())) if came else '',
                                   ("The last part as each person lived it, in their own telling:\n" + '\n'.join(lived)) if lived else '',
                                   ("Told on the stage, of their own lives:\n" + '\n'.join(f'- {t}' for t in told)) if told else '']))


def check_next(context, base, reply):
    """Strike from a behind reply the lines that do not fit what already stands. Returns (the reply
    without them, [(line, why)])."""
    lines = [l.strip() for l in reply.splitlines() if l.strip().startswith('#')]
    if not lines:
        return reply, []
    found = _ask(context.llm, CHECK_NEXT.format(**base, written='\n'.join(f'{n}. {l}' for n, l in enumerate(lines, 1))), 'director.check', 400)
    struck = {int(f[0]): (f[1] if len(f) > 1 else '') for f in (_fields(l) for l in _lines('strike', found)) if f and f[0].strip().isdigit()}
    struck = {n: why for n, why in struck.items() if 0 < n <= len(lines)}
    # What rests on a struck line goes with it, by the lines' own fields: a thing at a place that was struck (it was
    # moved to the nearest place of that kind and placed), the need of a person who was struck (sim/ruling.py).
    first = lambda l: (_fields(l.split(None, 1)[1]) if len(l.split(None, 1)) > 1 else [''])
    gone_places = {first(lines[n - 1])[0].strip(' ."\'').lower() for n in struck if lines[n - 1].startswith('#place')}
    gone_people = {first(lines[n - 1])[0].strip(' ."\'').lower() for n in struck if lines[n - 1].startswith('#person')}
    for n, l in enumerate(lines, 1):
        f = first(l)
        if n not in struck and ((l.startswith('#clue') and len(f) > 1 and f[1].strip().lower() in gone_places)
                                or (l.startswith('#needs') and f[0].strip().lower() in gone_people)):
            struck[n] = 'it rests on a line that was struck'
    return '\n'.join(l for n, l in enumerate(lines, 1) if n not in struck), [(lines[n - 1], why) for n, why in sorted(struck.items())]


def settle_next(context, sketch, parts, standing, told=(), opens=None):
    """Settle what is not yet settled behind the next part's question, behind all that stands.
    Returns the reply in the line forms frame_episode reads (#fact, #clue, #person, #place), each
    line checked against what stands (check_next); '' when it could not be made."""
    from sim.trace import trace
    base = dict(**sketch, standing=standing, stands=stands_text(parts, told, opens), story=story_text(parts), transcript=transcript_text(parts))
    seed = os.environ.get('ATWAP_SEED') or str(random.random())
    rng = random.Random(seed)
    prompt = ANSWERS_NEXT.format(**base)
    candidates = []
    for name, llm, tag in _backends(context):
        for line in _lines('answer', _ask(llm, prompt, tag, 1300)):
            f = _fields(line)
            if len(f) > 2 and f[1]:
                none = lambda v: '' if v.lower().strip(' .') in ('', 'none', 'nobody') else v
                candidates.append({'by': name, 'so': f[1], 'found': _labelled(f, 'found by'), 'short': _labelled(f, 'short'),
                                   'person': none(_labelled(f, 'person')), 'place': none(_labelled(f, 'place'))})
    if not candidates:
        trace.event('director_failed', step='answers_next')
        return ''
    left, last = list(candidates), ''
    for attempt in range(1, min(len(candidates), 4) + 1):   # an answer whose answering fact cannot be made to fit what stands is dropped for another
        chosen = rng.choice(left)
        left = [c for c in left if c is not chosen] or left
        answer = (f"{chosen['so']} It can be found by: {chosen['found']}." + (f" A person who knows a part: {chosen['person']}." if chosen['person'] else '')
                  + (f" A place not yet listed: {chosen['place']}." if chosen['place'] else ''))
        behind = BEHIND_NEXT.format(**base, answer=answer, person_rule=PERSON_NEXT if chosen['person'] else '- No person is new to the play.',
                                    person_format=(PERSON_FORMAT + NEEDS_FORMAT) if chosen['person'] else '')
        written = _ask(context.llm, behind, 'director.behind', 900)
        reply, struck = check_next(context, base, written) if '#fact' in written else (written, [])
        if struck:
            # A struck line was only dropped: the half of a part's question it answered then had nothing behind
            # it, and the question was still marked settled (lost-20261123-171920: 'he mumbled his own name
            # and she heard it', struck against 'never met or heard of each other'). It is sent back once with
            # the reason, as a deed is (resolver.AGAIN); the writing with fewer lines struck is kept.
            again = _ask(context.llm, behind + AGAIN_NEXT.format(struck='\n'.join(f'- {line} ({why})' for line, why in struck)), 'director.behind', 900)
            reply2, struck2 = check_next(context, base, again) if '#fact' in again else (again, struck)
            trace.event('director_behind_again', struck=struck, again=again, struck_again=struck2)
            # (the writing that keeps the fact which answers is the one wanted, before the one with fewer strikes:
            # a writing that had simply dropped the answer won, and the question stood with nothing behind it)
            if '#fact' in reply2 and (bool(answering(reply2)), -len(struck2)) > (bool(answering(reply)), -len(struck)):
                written, reply, struck = again, reply2, struck2
        trace.event('director_next_settled', seed=seed, attempt=attempt, candidates=[{'by': c['by'], 'short': c['short'] or c['so']} for c in candidates],
                    chosen=chosen, behind=written, struck=struck, kept=reply)
        if '#fact' in reply:
            last, last_struck = reply, struck
            if answering(reply):
                context.thread_struck = struck   # what of the answer could not be made to fit (thread.settle_part)
                return reply
    if last:   # no writing kept a fact that answers: what was settled is so, and the question is not marked settled (settle_part)
        trace.event('part_unanswered', question=standing[:300])
        context.thread_struck = last_struck
        return last
    trace.event('director_failed', step='behind_next')
    return ''
