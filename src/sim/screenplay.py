"""The screenplay of a play in progress: one record of what the company has settled above the
level of a scene.

Its parts used to live in loose attributes of the Context, each set by a different function at
a different time (central_narrative, episode, act_frame, act_central_narrative, previous_acts,
previous_scenes, and a 'narrative' that held the last act). The results: plans written before
the episode was framed and at odds with it, a closing act played with the act before's
direction, and prompts about 'previous acts' that were given what had been planned, not what
had happened.

It is settled in this order (Context.create_character_narratives, run_integrated_narrative):

  question   the central dramatic question, as the cast negotiated and the playwright merged it
  frame      what this episode settles, what it leaves open, the image it ends on
  clues      things placed in the world that touch the open question without answering it
  acts       one ActRecord for each act as it comes up: its focus, the act as integrated and
             performed, and what it came to

Nothing here calls a model. The Context fills it in and reads it back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

KINDS = ('opening', 'middle', 'resolving', 'closing')
QUESTION_TAGS = ('question', 'stakes', 'tension', 'success', 'failure')


@dataclass
class ActRecord:
    number: int                        # 1-based, as shown to the audience
    kind: str                          # one of KINDS
    focus: dict = field(default_factory=dict)   # {'focus', 'stakes', 'roles': {name: role}}
    framework: str = ''                # the act's dramatic framework as the playwright wrote it
    plan: Optional[dict] = None        # the integrated act that is performed
    summary: str = ''                  # what happened: scene wrap-ups and assessed outcomes
    post_state: str = ''               # what the next act starts from
    choices: list = field(default_factory=list)   # choices its scenes turned on: {'scene', 'character', 'decision', 'choice'}

    @property
    def title(self):
        return (self.plan or {}).get('act_title') or f'Act {self.number}'

    @property
    def performed(self):
        return bool(self.summary)


@dataclass
class Screenplay:
    question: dict = field(default_factory=dict)   # 'text' (as written), QUESTION_TAGS, 'roles': {name: role}
    frame: dict = field(default_factory=dict)      # 'settles', 'open', 'last'
    clues: list = field(default_factory=list)      # [{'thing', 'place', 'touches'}]
    marks: list = field(default_factory=list)      # [{'who', 'looks', 'touches'}]: a trace of a fact on a person, seen by anyone (director's #mark)
    facts: list = field(default_factory=list)      # [{'text', 'known_to': [names], 'open': bool}]: what lies behind the play (set_facts)
    answers: list = field(default_factory=list)    # the texts of the facts that answer the first half of the question the play stands on (Context.frame_episode)
    standing: bool = False                         # the play stands on a question from its sketch (sim/director.py): its facts are to be found
    known: dict = field(default_factory=dict)      # {name: what that person has come to, of what is so} (director.read)
    walked_past: int = 0                           # acts in a row that came no nearer to what is so
    threads: list = field(default_factory=list)    # [{'kind': 'found'|'told', 'who', 'what', 'scene'}]: what the play has planted
    earlier: str = ''                              # the parts of this performance already played, as they came out (sim/thread.py)
    act_count: int = 3                             # acts that carry the story; a closing act follows
    acts: list = field(default_factory=list)       # [ActRecord], in the order played

    # ----------------------------------------------------------------- the question
    def set_question(self, reply):
        """Take the merged central question from the playwright's reply: a paragraph, followed
        by its elements on tagged lines (#question, #stakes, #tension, #role <name>: ...,
        #success, #failure). The paragraph is kept as written for the planning prompts; the
        elements are what an actor's notes are built from (story_for)."""
        # the tags may come on one line ('...costs something. #question ... #stakes ...'): read at
        # line starts only, the paragraph kept them all, and every actor's notes carried the other's
        # role and how the play was to succeed and fail
        reply = re.sub(r'\s+#(question|stakes|tension|success|failure|role)\b', r'\n#\1', reply or '')
        find = lambda tag: next((m.strip() for m in re.findall(rf'^\s*#{tag}\s+(.+)$', reply or '', re.M)), '')
        text = re.sub(r'^\s*#(question|stakes|tension|success|failure|role)\b.*$', '', reply or '', flags=re.M).strip()
        self.question = {'text': text, **{tag: find(tag) for tag in QUESTION_TAGS},
                         'roles': {n.strip().strip('*'): r.strip()
                                   for n, r in re.findall(r'^\s*#role\s+([^:\n]+):\s*(.+)$', reply or '', re.M)}}
        return self.question

    @property
    def question_text(self):
        return self.question.get('text', '')

    def story_for(self, name):
        """The story as an actor is told it: the question, what rides on it, what pulls against
        what, and this actor's own part. Not the other parts, and not how it comes out
        (an actor given 'Joe: Antagonist' and the success scenario played both at once).
        Falls back to the paragraph when the playwright gave no tagged elements."""
        q = self.question
        if not q.get('question'):
            return q.get('text', '')
        role = next((r for n, r in (q.get('roles') or {}).items() if n.lower() == name.lower()), '')
        return '\n'.join(filter(None, [q['question'],
                                       f"What is at stake: {q['stakes']}" if q.get('stakes') else '',
                                       f"What pulls against what: {q['tension']}" if q.get('tension') else '',
                                       f"Your part in the story: {role}" if role else '']))

    # ----------------------------------------------------------------- the acts
    def kind(self, number):
        """What sort of act a number is: the first opens, the last planned act resolves, the
        one after it closes, the rest are the middle."""
        if number > self.act_count:
            return 'closing'
        if number == 1 and self.act_count > 1:
            return 'opening'
        return 'resolving' if number == self.act_count else 'middle'

    def begin_act(self, number):
        act = ActRecord(number=number, kind=self.kind(number))
        self.acts.append(act)
        return act

    @property
    def act(self):
        """The act now being planned or played, or None before the first."""
        return self.acts[-1] if self.acts else None

    def performed(self):
        return [a for a in self.acts if a.performed]

    def so_far(self):
        """What has happened in the acts already played, for whoever plans the next: each act's
        record of events and assessed outcomes, not what was planned for it."""
        now = '\n\n'.join(f"Act {a.number}, '{a.title}': {a.summary}" for a in self.performed())
        if not self.earlier:
            return now
        return f"In the parts of this performance already played:\n{self.earlier}" + (f"\n\nIn this part:\n{now}" if now else '\n\nThis part has not begun.')

    # ----------------------------------------------------------------- the plan, for the director's chair
    def view(self):
        """The record as plain data for the Director's Chair: what is settled for this
        performance and how far it has been played. Acts are written one at a time, so an act
        not yet reached has no entry; one whose plan is still being written has its focus only."""
        def scene(s):
            people = s.get('characters') or {}
            return {'number': s.get('scene_number'), 'title': s.get('scene_title', ''),
                    'place': s.get('location', ''), 'when': s.get('when') or '',
                    'cast': [{'name': str(name), 'goal': entry.get('goal', '') if isinstance(entry, dict) else '',
                              'keeps': list(entry.get('keeps') or []) if isinstance(entry, dict) else []}
                             for name, entry in (people.items() if isinstance(people, dict) else [])],
                    'setup': s.get('pre_narrative', ''), 'ending': s.get('post_narrative', ''),
                    'played': bool(s.get('wrapup')), 'wrapup': s.get('wrapup', ''),
                    'outcomes': [{'text': o.get('outcome', ''), 'happened': o.get('test') is True}
                                 for o in s.get('outcomes') or [] if isinstance(o, dict)]}
        def act(a):
            plan = a.plan or {}
            return {'number': a.number, 'kind': a.kind, 'title': a.title, 'written': a.plan is not None,
                    'focus': a.focus.get('focus', ''), 'stakes': a.focus.get('stakes', ''),
                    'roles': dict(a.focus.get('roles') or {}),
                    'opens': plan.get('act_pre_state', ''), 'ending': plan.get('act_post_state', ''),
                    'scenes': [scene(s) for s in plan.get('scenes') or [] if isinstance(s, dict)],
                    'choices': list(a.choices), 'played': a.performed, 'summary': a.summary}
        return {'question': dict(self.question), 'frame': dict(self.frame),
                'facts': [dict(f) for f in self.facts], 'clues': [dict(c) for c in self.clues], 'marks': [dict(m) for m in self.marks],
                'threads': [dict(t) for t in self.threads], 'known': dict(self.known),
                'act_count': self.act_count, 'acts': [act(a) for a in self.acts]}

    # ----------------------------------------------------------------- the threads
    MAX_THREADS = 12

    def add_thread(self, kind, who, what, scene):
        """Something the play has planted: a thing found, or a piece of someone's past told."""
        what = ' '.join(str(what).split()).strip().rstrip('.')
        if what and not any(t['kind'] == kind and t['what'].lower() == what.lower() for t in self.threads):
            self.threads.append({'kind': kind, 'who': who, 'what': what, 'scene': scene})
            del self.threads[:-self.MAX_THREADS]

    def thread_note(self):
        """The threads as a note for whoever plans an act. (A play raised a hospital, a steering
        wheel in fog, a ticket with a name, toys in a hut and a cave, and picked none of them up;
        the memories the two told each other were exchanged and never set side by side.)"""
        if not self.threads:
            return ''
        found = [t for t in self.threads if t['kind'] == 'found']
        told = [t for t in self.threads if t['kind'] == 'told']
        lines = ["What the play has planted so far, and not yet used:"]
        lines += [f"- found: {t['what']} ({t['who']}, in '{t['scene']}')" for t in found]
        lines += [f"- told by {t['who']}: \"{t['what']}\" (in '{t['scene']}')" for t in told]
        lines.append("Pick up at least two: a found thing used, questioned or compared; pieces of the past set beside each other "
                     "so that they meet (two memories that fit, or that cannot both be true). Picking up is not explaining: "
                     + ("what these people make of it is theirs." if self.standing else "the open question stays open."))
        return '\n'.join(lines)

    # ----------------------------------------------------------------- the clues
    # ----------------------------------------------------------------- what lies behind the play
    # The director settles, once, two or three things that are so behind the play: each a single
    # thing that was so before it began, or a standing condition of its world, with who knows it.
    # Clues are traces of them and the director's own plans are written from them; a character who
    # knows one is told it privately; nobody else is told. (Nothing was settled behind the clues:
    # seven piled up in one play with no meaning between them, and in another a father and a
    # daughter gave opposite terms of the same bargain.)
    def set_facts(self, reply, names):
        """Read '#fact <sentence> | known to: <names or nobody> | open: yes/no' lines. names: who
        may know one (cast and owners); other names are dropped. Three at most."""
        by_lower = {n.lower(): n for n in names}
        facts = []
        for line in re.findall(r'^[ \t]*#fact[ \t]+(\S.*?)[ \t]*$', reply or '', re.M):
            fields = [f.strip() for f in line.split('|')]
            text = fields[0].strip(' "')
            if not text:
                continue
            known, is_open, answers = [], False, False
            for f in fields[1:]:
                key, _, value = f.partition(':')
                if key.strip().lower().startswith('known'):
                    known = [by_lower[n.strip().lower()] for n in re.split(r',|\band\b', value) if n.strip().lower() in by_lower]
                elif key.strip().lower() == 'open':
                    is_open = value.strip().lower().startswith('y')
                elif key.strip().lower().startswith('answers'):
                    answers = value.strip().lower().startswith('y')
            facts.append({'text': text, 'known_to': known, 'open': is_open, **({'answers': True} if answers else {})})
        facts.sort(key=lambda f: not f.get('answers'))   # the fact that answers the question is never the one the cap leaves out
        self.facts = facts[:4 if self.standing else 3]
        if len(facts) > len(self.facts):   # said, not dropped in silence (sim/ruling.py)
            from sim.trace import trace
            trace.event('director_capped', what='facts', kept=len(self.facts), left_out=[f['text'] for f in facts[len(self.facts):]])
        return self.facts

    def set_past(self, reply, name):
        """Read '#past <name> | <sentence>' for one person and keep it among the facts, known to
        that person alone ('of': whose past it is). Returns the sentence, '' when there is none."""
        for who, text in re.findall(r'^[ \t]*#past[ \t]+([^|\n]+?)[ \t]*\|[ \t]*(\S.*?)[ \t]*$', reply or '', re.M):
            text = text.strip(' "')
            if who.strip().lower() == name.lower() and text and text.lower().rstrip('.') not in ('none', 'nothing', 'n/a', '-'):
                self.facts = [f for f in self.facts if f.get('of') != name] + [{'text': text, 'known_to': [name], 'open': False, 'of': name}]
                return text
        return ''

    def facts_note(self, pasts=False, knowing=None, answers=()):
        """The facts, for the director's own calls (the act as performed, a scene's set-up): never
        for an actor's prompt or a character's own plan. pasts: with each person's own past, for
        the calls that say how a person changes or what they keep back. Whoever writes the scenes
        is not given them: a past is its owner's to tell or keep. (Given Joe's, the planner wrote
        'use the story of the two rattling fans' into his goal in four scenes and then into
        Samantha's; it was told three times as if new.) knowing(fact text): who knows it and who has
        been told it on the stage, as the stage keeps it (Context.knowing), in place of whom the
        director first marked."""
        facts = [f for f in self.facts if pasts or not f.get('of')]
        if not facts:
            return ''
        who = lambda f: (knowing(f['text']) if knowing and not f.get('of') else None) or f"known to: {', '.join(f['known_to']) or 'nobody'}"
        return ("What lies behind this play (the director's own; no character knows any of it except as marked, and "
                "none of it says what anyone will do):\n"
                + '\n'.join(f"- {f['text']} [{(f['of'] + chr(39) + 's own past; ') if f.get('of') else ''}{who(f)}"
                             + ('; behind the open question: not revealed in this episode' if f['open'] else '')
                             + ("; this is what answers the first half of the play's question" if f['text'] in (answers or ()) else '') + ']' for f in facts)
                + "\nA scene may bring a character nearer to one of these, or show a trace of it; a character acts on one only if it knows it or has been told it on the stage, and then only on as much as it was told."
                + ("\nThese are there to be found in this performance, from the things and people that show them, a part at a time. No one scene has all of it told, and no character has any of it before a thing or a person has shown or told it." if self.standing else ''))

    def clue_note(self, now=None):
        """The clues and where each is, for whoever writes a plan. now(clue): where it is at this
        point of the play in a few words ('held by Samantha'), or '' when it lies where it was
        placed. (Each was given as lying where it was planted for the whole play: a set-up laid
        the wristwatch at the spring again while Samantha held it.)"""
        seen = ('' if not self.marks else "To be seen on a person by anyone who looks, and by themselves, with nothing to say how it came:\n"
                + '\n'.join(f"- on {m['who']}: {m['looks']}: {m['touches']}" for m in self.marks))
        if not self.clues:
            return seen
        def where(c):
            moved = now(c) if now else ''
            return moved or f"at {c['place']}" + (', not yet found' if c.get('hidden') else '')
        return ("Things that are really there to be found, where each is now, and what each touches on:\n"
                + '\n'.join(f"- {c['thing']}, {where(c)}: {c['touches']}" for c in self.clues)
                + "\nA scene may lead a character to one and turn on its being found. "
                + ("Each shows a part of what is so and none shows all of it; nothing else is hidden in this world." if self.standing
                   else "None of them answers the open question; nothing else is hidden in this world.")
                + (f'\n{seen}' if seen else ''))
