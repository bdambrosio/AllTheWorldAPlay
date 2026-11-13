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
    act_count: int = 3                             # acts that carry the story; a closing act follows
    acts: list = field(default_factory=list)       # [ActRecord], in the order played

    # ----------------------------------------------------------------- the question
    def set_question(self, reply):
        """Take the merged central question from the playwright's reply: a paragraph, followed
        by its elements on tagged lines (#question, #stakes, #tension, #role <name>: ...,
        #success, #failure). The paragraph is kept as written for the planning prompts; the
        elements are what an actor's notes are built from (story_for)."""
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
        return '\n\n'.join(f"Act {a.number}, '{a.title}': {a.summary}" for a in self.performed())

    # ----------------------------------------------------------------- the clues
    def clue_note(self, now=None):
        """The clues and where each is, for whoever writes a plan. now(clue): where it is at this
        point of the play in a few words ('held by Samantha'), or '' when it lies where it was
        placed. (Each was given as lying where it was planted for the whole play: a set-up laid
        the wristwatch at the spring again while Samantha held it.)"""
        if not self.clues:
            return ''
        def where(c):
            moved = now(c) if now else ''
            return moved or f"at {c['place']}" + (', not yet found' if c.get('hidden') else '')
        return ("Things that are really there to be found, where each is now, and what each touches on:\n"
                + '\n'.join(f"- {c['thing']}, {where(c)}: {c['touches']}" for c in self.clues)
                + "\nA scene may lead a character to one and turn on its being found. "
                  "None of them answers the open question; nothing else is hidden in this world.")
