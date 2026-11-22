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
#answer <number> | <what is so, two or three plain sentences> | found by: <two to four things or traces and the listed place of each> | person: <who knows a part, where they are, and what they want; or none> | short: <the answer in one sentence of at most 25 words>
</end>"""

PERSON = ("One person besides those at the centre of the play may know a part of it: the owner of one of the places listed as having an owner nobody has yet been cast as, or another person the sketch lists. "
          "They are alive, awake and at a listed place as the play begins. They want something of their own that would be harder to have if these people knew all of it: "
          "so they are not there to explain, and nothing says what they will tell.")
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

- Facts: three or four, each ONE thing that is so: a standing condition of this place, or a single thing that happened before the play began. Never what anyone will do or feel in the play, and never what the people at the centre of it wanted, agreed to or chose before it. Use only these people, these places (by their exact names) and the kinds of thing this world has. Say who knows each as the play begins: names from the people, or nobody. The people at the centre do not know what the sketch says they do not remember.
- Things: two or three small, ordinary things that really lie in this world, each at one of the places listed, each a trace of one fact. Each is an object that can be picked up, or a mark on the ground, a wall or a tree at that spot: never a view, a sound, a smell or a stretch of country. A thing shows what a person standing there would see; it does not explain. Its wording and its question name no other object that is not itself one of these things or a listed place: if marks lie under a sling, the sling is one of the things, or there is no sling. At most one is written, by a person, by hand, before the play, to someone and for a reason of their own: it says what that person would have written then and no more, and never the whole of what is so. Choose places these people can reach today.
{person_rule}
- Then one thing from the own past of each of: {leads}. A single thing that happened to that person, or that they did, before the play and apart from all this: plain and particular (what, where, with whom), somewhere else in their life. It gives a reason for something their description says of them and contradicts nothing in it. Write none for a person whose description says they do not remember their past.
- Nothing here says what anyone will do in the play, or how it comes out.

Respond with these lines only, one sentence each:
#fact <one plain sentence> | known to: <names, or nobody>   (three or four lines)
#clue <the thing, by a name of two to four plain words with no comma> | <exact name of a place listed> | <what finding it makes someone wonder> | <if it is the written one: its words, in one or two plain sentences; else leave empty> | <loose if it can be picked up and carried; fixed if it is a mark or a part of the place> | <how it is, as someone standing there sees it, where that is what makes it a trace: three to eight words (struck through in heavy pen; one corner torn off); else leave empty>   (two or three lines)
{person_format}#past <name> | <one plain sentence>   (one line for each of: {leads}; none where the description rules it out)
</end>"""

PERSON_BEHIND = ("- A person who knows a part: the one named in what is so tonight, if it names one. Say who they are in one sentence as a description of a person (age, manner, what they do here), "
                 "the listed place where they are as the play begins, and the one or two things they want. They know only the facts you mark them as knowing. Nothing says what they will tell or do: they are played, like the others.")
PERSON_FORMAT = "#person <their own first name, one word: a person's name, never a place's or a role's> | <exact name of the listed place where they are> | <who they are, one sentence> | wants: <one or two things, separated by ';'>   (one line, or none)\n"

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
    return [m.strip() for m in re.findall(rf'^\s*#{tag}\s+(.+?)\s*$', text or '', re.M)]


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
            return {'name': f[0].strip(' ."\''), 'place': f[1], 'description': f[2], 'wants': wants[:2]}
    return None


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
