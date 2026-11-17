"""The owner of a scene while it is played.

A scene was assembled from parts that did not know about each other: a schedule of tasks, each
with an end test of its own (Character.test_termination); conversations that ended on a rating
and a line count (Character.natural_dialog_end); exits held by a count of tasks
(Context.exit_allowed); and a planned ending that ended nothing, and was only judged afterwards.
So a scene that had arrived went on (rubbish bins searched twice after the page had been read),
and one that had not could empty on a spoken line.

The owner holds the scene's sheet and is asked, as the scene is played, where it stands:

  sheet      the planned ending as two to four outcomes, written before the first line
             ('happens': a thing said or done at one moment; 'holds': a thing that must not
             happen, or a state that must not change, before the scene ends)
  arrange()  the scene's beats: what each character's actor proposed (its task plan), read
             together beside the sheet, with repeats dropped, a beat added for an outcome nobody
             proposed to bring about, and a leaving put last. The beats are Tasks, so all that
             reads a task (the page, the trace, a character's memory of what it did) is unchanged.
  check()    after an act and before a reply: which outcomes are now reached, which can no
             longer be ('gone': refused and the refusal accepted, come about another way, or
             tried, failed and left behind), whether the beat being played is done, whether
             the last line waits for an answer
  converse() a talk, one actor's turn at a time, in a loop here (it was calls inside calls)
  closed     every 'happens' outcome is reached: the scene has arrived, and what is left of its
             schedule is not played
  winding    the scene has had its lines, or every 'happens' outcome is reached or gone and one
             at least is gone (it has nowhere left to arrive): no beat is begun but a leaving,
             a talk ends at the next line that asks nothing, and anyone may go

What reads it: Context.run_scene (the beats; the scene is over), Character.clear_task_if_satisfied
(the beat is done), Character.natural_dialog_end (the talk is over), Character.acts (a line said
to someone is handed to converse) and Context.exit_allowed. Where the owner has nothing to say
(no outcomes could be written, a reply that cannot be read) each of those does what it did
before. The record of what the scene came to is still the after-scene check
(Context.check_post_state_ambiguity); the sheet's final state is traced beside it.

Not done here: the actor's notes are still put together in Character.director_notes, and an act
and a reply are still written from two prompts (generate_acts, generate_dialog_turn).

Tested offline on ten recorded scenes before it was built (logs/review-20261115/tracker): the
outcomes were judged as a reader judged them 28 times in 30, and no scene was closed early.
Whether the beat is 'stalled' is asked for and traced, and decides nothing: the acting model does
not tell circling from talk.

'gone' was added after four first scenes in six could not close, none of them through a refused
deed: an outcome the characters turned down ("Split up? No"), one overtaken (a name given before
it was asked for), one tried that failed (a strip of cloth reached for and not let go), and a
state written as an event ('neither leaves as the light fades': now a 'holds'). Tested offline on
eleven recorded scenes, old wording against new (logs/review-20261116/gone): all four outcomes
that could no longer happen were called gone, none that later happened was, and outcomes were read
as a reader read them 32 times in 33 (31 before). A 'gone' counts once two checks running say it:
a single one was twice taken back at the next look.

Off with the ablation switch 'scene_owner'. An unplanned play has no scenes and no owner.
"""
from __future__ import annotations

from sim import ablation
from sim.trace import trace
from utils.Messages import UserMessage
from utils import json_reply

OUTCOMES = """A scene of an improvised play is about to be played. Someone will watch it turn by turn and must be able to tell, from what is said and done on stage, when the scene has arrived where it was meant to. Turn the planned ending into what that watcher looks for.

Scene: '{{$title}}', at {{$place}}
How it opens: {{$setup}}
Planned ending: {{$ending}}
What each is after: {{$goals}}

Write two to four outcomes. Each is one thing the planned ending needs, small enough to be seen in a single line or deed.
- "kind": "happens" for something that is said or done on stage at one moment (a thing told, admitted, refused, handed over, a deed done). Write it so that one could point at the line or deed that does it.
- "kind": "holds" for something that is so only if it lasts until the scene ends: a thing that must NOT happen before then (a secret not told, a question not asked, a thing not opened), or a state that must not change (nobody leaves, it stays shut). Write what would break it. Only where the planned ending's own words say a thing is not done, not said, kept back or stays as it is; most endings have none.
- Only what the planned ending says. Nothing from how it opens, and nothing the ending does not ask for.
- One outcome is one thing. Leave out what merely goes with it (the light, what is on someone's hands, where they stand), unless the ending turns on that.
- What someone feels, notices, decides or understands is not an outcome unless the ending has it said or shown; then write the saying or the showing.
- If the ending has two things happen, they are two outcomes. At least one outcome is "happens".

Answer with JSON only:
{"outcomes": [{"kind": "happens" or "holds", "text": "..."}]}
End your response with: </end>"""

TRACK = """You are keeping track of one scene of an improvised play while it is being played. You do not write it and you do not judge whether it is good. Every few turns you say where it stands.

Scene: '{{$title}}', at {{$place}}
How it opened: {{$setup}}
Planned ending: {{$ending}}

What the planned ending needs:
{{$outcomes}}

The beat being played now: {{$beat}}

The scene so far. A line with a name and quoted words was spoken; any other line is what was done, and it says how far the deed got. The turns since you last looked are marked NEW.
{{$turns}}

Answer four things.

1. For each outcome above not yet REACHED or BROKEN: has it happened now?
- Judge only from what the turns show said or done. A thing offered, promised, proposed, intended, asked for, talked about or about to happen has not happened.
- A deed happened only as far as its line says. If it says the thing stayed where it was, or that nothing passed, it did not happen, whatever is said afterwards.
- An outcome in several parts has happened only when every part has.
- For a 'happens' outcome that has not happened, say also whether it still can. It is "gone" when the turns show it no longer can in this scene: the one who was to do it refused and the other gave way; or what it was for has come about another way (she was to be asked her name, and gave it unasked); or it was tried, it failed, and the two have since gone on to other things. Not yet tried, put off, or refused and still being pressed is not gone.
- A 'holds' outcome stays open until the turns show the very thing it forbids; then it is broken.
- Give the turn number and quote the words that show it.

2. The beat being played now:
- "done": what the beat was for has been got, or refused for good.
- "stalled": the NEW turns ask for, offer or refuse again what was already asked for, offered or refused, and nothing new came of it.
- "going": otherwise.

3. "awaiting": true if the last turn is a question, an offer or a demand put to someone who has not yet answered it; false if it is an answer, an agreement, a refusal, a remark that asks nothing, or a deed.

4. What should happen next:
- "close": every 'happens' outcome has been reached or is gone. The scene has arrived, or has nowhere left to arrive.
- "next_beat": the beat is done or stalled and the scene has not arrived.
- "continue": otherwise.

JSON only:
{"happens": [{"n": 1, "reached": true or false, "gone": true or false, "turn": 0, "by": "the words that show it"}], "holds": [{"n": 4, "broken": true or false, "turn": 0, "by": "the words that break it"}], "beat": "going" or "done" or "stalled", "awaiting": true or false, "why": "12 words at most", "next": "continue" or "next_beat" or "close"}
List only outcomes not yet REACHED or BROKEN, each under its own kind. "broken" and "gone": true only when you can quote the turn that shows it; if you cannot, it is false.
End your response with: </end>"""

BEATS = """A scene of an improvised play is about to be played. The actor of each character has proposed, without seeing the other's, what that character does in it. Arrange the proposals into the scene's beats.

Scene: '{{$title}}', at {{$place}}{{$kind}}
How it opens: {{$setup}}
Planned ending: {{$ending}}
What the planned ending needs:
{{$outcomes}}
What each is after: {{$goals}}
{{$keeps}}
The proposals:
{{$proposals}}

Write three to six beats, in the order they are to be played. A beat is one thing one character sets out to do; the other answers it as they will.
- Take a proposal as it stands wherever you can: give its number as "from" and leave "name", "what" and "done_when" empty.
- Two beats never go after the same thing. Where two proposals ask for, offer or do the same thing (in the same words or in others, by the same character or by both), keep one and drop the other. A proposal that is a later stage of the same demand ("ask", then "press", then "hold to it") is the same thing.
- Each 'happens' outcome needs a beat that can bring it about. If no proposal can, write a new beat for the character who would do it: "from" is null, and it is written from what that character wants and knows, never from what another keeps back.
- Keep the beats that are a character's own business and ask nothing of the other (work in hand, a joke, something offered, shown or told): the people are seen in those. A scene is not only its ending.
- Order: what a later beat needs comes before it; the beats that bring about the ending come late; a beat that takes anyone away from this place is the last of all. Both characters have beats, and neither has more than two running.
- "done_when": something the character can bring about alone, or that either answer from the other satisfies (told, or refused). Never one that waits for the other to give in.

JSON only:
{"beats": [{"who": "<character>", "from": 1 or null, "name": "4 to 6 words, or empty", "what": "6 to 10 words: what is done, offered, shown or asked; or empty", "done_when": "5 to 8 words, or empty", "leaves": true or false}], "dropped": [{"p": 2, "why": "8 words at most"}]}
End your response with: </end>"""

PAGE_TURNS = 80   # the latest turns shown to the tracker; a scene rarely has more
# How many spoken lines a scene has before it is wound up, whether or not it has arrived: a scene
# at ease is an interlude, and one that has not got where it was going by then is not going to.
# (With talks held open while a question waited and leaving held while beats remained, a first
# scene ran past eighty lines, the last thirty of them about to walk to a spring.)
LINES = {'easy': 40, 'other': 32}


class Owner:
    def __init__(self, context, scene):
        self.context, self.scene = context, scene
        self.start = len(context.record_entries)   # the scene's page is the record from here on
        self.outcomes = []        # [{'kind', 'text', 'state': 'open' | 'reached' | 'gone' | 'broken', 'turn', 'by'}]
        self.closed = False
        self.checks = 0
        self._seen = 0            # turns of the page already looked at
        self._verdict = None      # the last check, kept while nothing new has happened
        self._beat = None

    # ------------------------------------------------------------------ the sheet
    def write_sheet(self):
        """The planned ending as outcomes, before the first line. False when none could be
        written: the scene is then played as it was before there was an owner."""
        scene = self.scene
        ending = (scene.get('post_narrative') or '').strip()
        if not ending:
            return False
        goals = '; '.join(f"{name}: {(entry or {}).get('goal', '')}" for name, entry in (scene.get('characters') or {}).items()
                          if isinstance(entry, dict))
        found = None
        for _ in range(2):
            reply = self.context.llm.ask({'title': scene.get('scene_title', ''), 'place': scene.get('location', ''),
                                          'setup': scene.get('pre_narrative', ''), 'ending': ending, 'goals': goals or 'not given'},
                                         [UserMessage(content=OUTCOMES)], tag='scene.outcomes', temp=0.1, stops=['</end>'], max_tokens=400)
            found = json_reply.parse(reply) if reply else None
            if isinstance(found, dict) and found.get('outcomes'):
                break
        rows = [o for o in (found.get('outcomes') if isinstance(found, dict) else None) or []
                if isinstance(o, dict) and str(o.get('text') or '').strip() and o.get('kind') in ('happens', 'holds')][:4]
        if not any(o['kind'] == 'happens' for o in rows):
            trace.event('sheet', outcomes=[], why='none could be written')
            return False
        self.outcomes = [{'kind': o['kind'], 'text': str(o['text']).strip(), 'state': 'open', 'turn': None, 'by': ''} for o in rows]
        trace.event('sheet', outcomes=[{'kind': o['kind'], 'text': o['text']} for o in self.outcomes])
        return True

    @property
    def active(self):
        return bool(self.outcomes)

    def spoken(self):
        return len([e for e in self.context.record_entries[self.start:] if e.get('kind') == 'speech'])

    @property
    def spent(self):
        """No 'happens' outcome is still open and one at least is gone: the scene has not
        arrived and has nowhere left to arrive."""
        states = [o['state'] for o in self.outcomes if o['kind'] == 'happens']
        return 'gone' in states and 'open' not in states

    @property
    def winding(self):
        """The scene has had its lines, or is spent: no more beats are begun here but a leaving,
        a talk ends at the next line that asks nothing, and anyone may go."""
        kind = ' '.join(str(self.scene.get('kind') or '').strip().lower().replace('_', ' ').split())
        lines = self.spoken() >= LINES['easy' if kind in ('shared job', 'lull', 'confession', 'play') else 'other']
        over = lines or self.spent
        if over and not getattr(self, '_wound', False):
            self._wound = True
            trace.event('scene_winding', spoken=self.spoken(), why='lines' if lines else 'outcomes gone')
        return over

    def page(self):
        return [self.context.entry_line(e) for e in self.context.record_entries[self.start:] if e.get('text')]

    # ------------------------------------------------------------------ the beats
    def arrange(self, proposals):
        """The scene's beats, from what each character's actor proposed: one reading of all of
        them beside the sheet, in place of taking each character's list in turn. proposals:
        [{'actor', 'goal', 'task'}], the schedule as it would have been played. Returns the new
        schedule, or None when the arrangement cannot be read (the old one then stands).

        (Each character planned three tasks toward its own goal without seeing the other's, and
        the lists were dealt out alternately: three stages of one demand from her, three ways of
        dodging it from him, and nothing that saw the six as one scene.)"""
        if len(proposals) < 2:
            return None
        scene, context = self.scene, self.context
        cast = {e['actor'].name: e['actor'] for e in proposals}
        goals = '; '.join(f"{name}: {(entry or {}).get('goal', '')}" for name, entry in (scene.get('characters') or {}).items()
                          if isinstance(entry, dict))
        keeps = '\n'.join(f"{name} keeps back in this scene: {'; '.join(entry['keeps'])}"
                          for name, entry in (scene.get('characters') or {}).items() if isinstance(entry, dict) and entry.get('keeps'))
        lines = []
        for i, e in enumerate(proposals):
            t = e['task']
            lines.append(f"P{i + 1}. {e['actor'].name}: {t.name}. {t.description} (wants: {str(t.reason).strip()})"
                         + (f" [done when: {t.termination}]" if t.termination else '') + (' [takes them away from this place]' if getattr(t, 'leaves', False) else ''))
        outcomes = '\n'.join(f"{i + 1}. [{o['kind']}] {o['text']}" for i, o in enumerate(self.outcomes))
        kind = str(scene.get('kind') or '').strip()
        reply = context.llm.ask({'title': scene.get('scene_title', ''), 'place': scene.get('location', ''), 'kind': f' (a {kind})' if kind else '',
                                 'setup': scene.get('pre_narrative', ''), 'ending': scene.get('post_narrative', ''), 'outcomes': outcomes,
                                 'goals': goals or 'not given', 'keeps': (keeps + '\n') if keeps else '', 'proposals': '\n'.join(lines)},
                                [UserMessage(content=BEATS)], tag='scene.beats', temp=0.2, stops=['</end>'], max_tokens=700)
        found = json_reply.parse(reply) if reply else None
        rows = found.get('beats') if isinstance(found, dict) else None
        if not isinstance(rows, list):
            trace.event('check_unreadable', of='scene.beats', reply=(reply or '')[:300])
            return None
        schedule, used = [], set()
        for row in rows[:6]:
            if not isinstance(row, dict):
                continue
            actor = cast.get(str(row.get('who') or '').strip()) or next((a for n, a in cast.items() if n.lower() == str(row.get('who') or '').strip().lower()), None)
            try:
                n = int(str(row.get('from')).lstrip('Pp')) - 1 if row.get('from') not in (None, '', 'null') else None
            except (TypeError, ValueError):
                n = None
            name, what, done = (str(row.get(k) or '').strip() for k in ('name', 'what', 'done_when'))
            if n is not None and 0 <= n < len(proposals) and n not in used:
                entry = dict(proposals[n]); used.add(n); task = entry['task']
                if actor is not None and actor is not entry['actor']:
                    continue   # a proposal is its own character's to play
                if name and what:   # reworded: the actor's reason for it stands
                    task.name, task.description = name, what
                    task.termination = done or task.termination
            elif actor is not None and name and what and getattr(actor, 'focus_goal', None) is not None:
                other = next((a.name for a in cast.values() if a is not actor), 'None')
                task = actor.validate_and_create_task(f"#name {name}\n#description {what}\n#reason the scene needs it of {actor.name}\n#actors {other}\n"
                                                      f"#start_time now\n#duration 5 minutes\n#termination {done or what}\n##", actor.focus_goal)
                if task is None:
                    continue
                entry = {'actor': actor, 'goal': actor.focus_goal, 'task': task}
            else:
                continue
            entry['task'].leaves = bool(row.get('leaves')) or bool(getattr(entry['task'], 'leaves', False))
            schedule.append(entry)
        if len(schedule) < 2 or len({e['actor'].name for e in schedule}) < min(2, len(cast)):
            trace.event('beats', kept=0, why='too few beats, or one character left without any')
            return None
        schedule.sort(key=lambda e: bool(getattr(e['task'], 'leaves', False)))   # a leaving is played last
        trace.event('beats', kept=len(schedule), proposed=len(proposals),
                    beats=[{'who': e['actor'].name, 'task': e['task'].name, 'new': e not in proposals and e['task'] not in [p['task'] for p in proposals]} for e in schedule],
                    dropped=[d for d in (found.get('dropped') or []) if isinstance(d, dict)][:6])
        return schedule

    # ------------------------------------------------------------------ the talk
    async def converse(self, speaker, target, message, source):
        """A line has been said to `target`: it is answered, and the answer answered, one turn
        at a time, until Character.hear lets one stand unanswered (natural_dialog_end, which
        asks check()). Each turn is one actor's; who has the next is settled here.

        (A talk was a chain of calls inside calls: a line called the listener's hear, whose
        reply called the speaker's hear, and so on down, each line's own aftermath waiting until
        the whole talk had unwound. The same turns are now taken in a loop the scene owns.)"""
        if getattr(self, '_talking', False):
            self._pending = (speaker, target, message, source)   # a reply inside the loop: the loop takes it up
            return
        self._talking, turn = True, (speaker, target, message, source)
        try:
            while turn is not None:
                self._pending = None
                said_by, said_to, words, task = turn
                await said_to.hear(said_by, words, task)
                turn = self._pending
        finally:
            self._talking, self._pending = False, None

    # ------------------------------------------------------------------ the tracker
    def check(self, task=None):
        """Where the scene stands now: {'beat': 'going' | 'done' | 'stalled', 'awaiting': bool,
        'next': ...}, or None when there is nothing to go on. Asked again with nothing new on the
        page and the same beat, it gives the answer it gave."""
        if not self.active:
            return None
        page = self.page()
        beat = self._beat_line(task)
        if self._verdict is not None and len(page) == self._seen and beat == self._beat:
            return self._verdict
        if not page:
            return None
        shown = page[-PAGE_TURNS:]
        first = len(page) - len(shown)
        turns = '\n'.join(f"{first + i + 1}{' NEW' if first + i >= self._seen else ''}. {line}" for i, line in enumerate(shown))
        outcomes = '\n'.join(f"{i + 1}. [{o['kind']}] {o['text']} -- "
                             + ('still open' if o['state'] == 'open' else f"GONE at turn {o['turn']}, unless it has happened since"
                                if o['state'] == 'gone' else f"{o['state'].upper()} at turn {o['turn']}")
                             for i, o in enumerate(self.outcomes))
        scene = self.scene
        reply = self.context.llm.ask({'title': scene.get('scene_title', ''), 'place': scene.get('location', ''),
                                      'setup': scene.get('pre_narrative', ''), 'ending': scene.get('post_narrative', ''),
                                      'outcomes': outcomes, 'beat': beat or 'none in particular', 'turns': turns},
                                     [UserMessage(content=TRACK)], tag='scene.track', temp=0.1, stops=['</end>'], max_tokens=500)
        found = json_reply.parse(reply) if reply else None
        self.checks += 1
        if not isinstance(found, dict) or found.get('beat') not in ('going', 'done', 'stalled'):
            trace.event('check_unreadable', of='scene.track', reply=(reply or '')[:300])
            return None
        said = {}
        for field in ('happens', 'holds'):
            for row in found.get(field) or []:
                try:
                    n = int(row.get('n')) - 1
                except (TypeError, ValueError, AttributeError):
                    continue
                if 0 <= n < len(self.outcomes) and self.outcomes[n]['kind'] == field:
                    said[n] = row
        for n, o in enumerate(self.outcomes):
            row = said.get(n) or {}
            flag = 'reached' if o['kind'] == 'happens' else 'broken'
            if o['state'] in ('open', 'gone') and row.get(flag) is True:   # a gone outcome may still happen
                o.update(state=flag, turn=row.get('turn'), by=str(row.get('by') or '')[:200])
                trace.event('outcome_' + flag, n=n + 1, text=o['text'], turn=row.get('turn'), by=o['by'])
            elif o['kind'] == 'happens' and o['state'] == 'open':
                # gone once two checks running say so: a single one has been taken back at the next look
                o['_gone'] = o.get('_gone', 0) + 1 if row.get('gone') is True else 0
                if o['_gone'] >= 2:
                    o.update(state='gone', turn=row.get('turn'), by=str(row.get('by') or '')[:200])
                    trace.event('outcome_gone', n=n + 1, text=o['text'], turn=row.get('turn'), by=o['by'])
        was_closed = self.closed
        self.closed = all(o['state'] == 'reached' for o in self.outcomes if o['kind'] == 'happens')
        self._seen, self._beat = len(page), beat
        self._verdict = {'beat': found['beat'], 'awaiting': found.get('awaiting') is True, 'next': found.get('next'), 'why': found.get('why')}
        trace.event('track', at=len(page), beat_now=beat, beat=found['beat'], awaiting=self._verdict['awaiting'], next=found.get('next'),
                    why=found.get('why'), closed=self.closed)
        if self.closed and not was_closed:
            trace.event('scene_arrived', at=len(page))
        return self._verdict

    @staticmethod
    def _beat_line(task):
        if task is None:
            return ''
        who = ', '.join(a.name for a in (getattr(task, 'actors', None) or [])[:1])
        line = f"{who}: {task.name}. {task.description}".strip()
        return line + (f" It is done when: {task.termination}" if getattr(task, 'termination', '') else '')

    def current_task(self):
        """The task of the scene's schedule now being played (as Context.run_scene refreshed it)."""
        if getattr(self, 'now', None) is not None:
            return self.now
        context = self.context
        plan = context.scene_integrated_task_plan or []
        return plan[context.scene_integrated_task_plan_index]['task'] if context.scene_integrated_task_plan_index < len(plan) else None

    def final(self):
        """The sheet as the scene leaves it, for the trace (one last look at the page)."""
        self.check(self.current_task())
        return [{'kind': o['kind'], 'text': o['text'], 'state': o['state'], 'turn': o['turn'], 'by': o['by']} for o in self.outcomes]


def owner_of(context):
    """The owner of the scene being played, or None: no scene, no sheet, or switched off."""
    owner = getattr(context, 'owner', None)
    return owner if owner is not None and owner.active and context.current_scene is owner.scene else None


def open_scene(context, scene):
    """Give the scene an owner and write its sheet. Not while a danger is on the cast: such a
    scene is about meeting the danger, and ends as the world ends it."""
    context.owner = None
    if not ablation.enabled('scene_owner') or getattr(context, 'active_threat', None):
        return None
    owner = Owner(context, scene)
    if owner.write_sheet():
        context.owner = owner
    return context.owner
