"""Character drives: what a character wants, and how pressing each want is right now.

A play gives each character a few drives, most important first (Drive("find food and water")).
That order is the character's standing priority and does not change. A drive's activation
(0..1) is how much it presses at this moment, and it moves with what happens:

  appraise()        before a character acts, one LLM call reads what it has seen, heard and
                    thought since the last appraisal and says which drives that touched: a
                    threat or a chance raises a drive, a satisfaction eases it. The noticed
                    event is kept with the drive, so prompts say what is on the character's mind.
  settle()          between appraisals an activation drifts back to its baseline as play time
                    passes (HALF_LIFE_MINUTES), and the body holds the bodily drives up: a
                    thirsty character's drive for water is at least as active as the thirst.
  goal_satisfied()  a goal names the drive it serves (#drive in the goal prompts); achieving it
                    eases that drive, and after EVOLVE_AFTER achievements the drive is rewritten.

Prompts get drive_lines(): the drives, most pressing first, in words.

History: until 2025-07 every percept got its own LLM pass for 'signals', which were clustered
and scored for goal prompts; activation itself was only ever set by a per-drive LLM check at
goal time. That per-drive check was all that remained (one call per drive per scene, no link to
the percepts). Sentence embeddings were tried for the percept-to-drive link (2026-11) and do
not discriminate, so the link is one small LLM call per turn for all drives together.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, List, Optional
from weakref import WeakValueDictionary

import numpy as np

from sim import ablation
from sim.cognitive.perceptualState import SensoryMode
from sim.trace import trace
from utils.Messages import UserMessage

if TYPE_CHECKING:
    from sim.agh import Character, Goal

BASELINES = (0.6, 0.5, 0.45, 0.4)   # resting activation by the play's drive order; later drives rest at the last
HALF_LIFE_MINUTES = 60              # play time for an activation to drift halfway back to its baseline
PRESSING, PRESENT = 0.7, 0.4        # activation at or above: 'pressing', 'present'; below: 'quiet'
RAISE, EASE = 0.6, 0.6              # how far a full-strength threat/chance raises, or a satisfaction eases
FLOOR = 0.1
EVOLVE_AFTER = 3                    # achieved goals before a drive (other than the first) is rewritten
THOUGHT = f'{SensoryMode.THOUGHT.value}: '   # how a memory of the character's own thinking begins (PerceptualState.add_input)
MAX_PERCEPTS = 12                   # most recent new memories read by one appraisal
NOTICES_KEPT = 3
NOTICE_MINUTES = 90                 # a noticed event stays 'on the mind' this long

# which drives the body holds up (sim/survival.py Body.levels): bodily needs, and harm
_BODILY = re.compile(r'physiolog|surviv|food|water|hunger|thirst|shelter|\brest\b|sleep', re.I)
_SAFETY = re.compile(r'safe|threat|health|injur|danger|harm', re.I)
_APPRAISAL = re.compile(r'#\s*(d\d+)\W+(threat|chance|eased)\W+([01](?:\.\d+)?)\W*(.*)', re.I)


@dataclass
class Notice:
    """Something the character noticed that touched a drive."""
    text: str
    kind: str        # threat, chance or eased
    strength: float
    time: datetime


@dataclass(eq=False)
class Drive:
    _id_counter: ClassVar[int] = 0
    _instances: ClassVar[WeakValueDictionary] = WeakValueDictionary()
    text: str
    activation: float = BASELINES[1]
    baseline: Optional[float] = None       # set from the drive's place in the character's list
    embedding: Optional[np.ndarray] = None  # filled on demand by memory retrieval
    attempted_goals: List['Goal'] = field(default_factory=list)
    satisfied_goals: List['Goal'] = field(default_factory=list)
    notices: List[Notice] = field(default_factory=list)
    updated: Optional[datetime] = None
    id: str = field(init=False)

    def __post_init__(self):
        Drive._id_counter += 1
        self.id = f'd{Drive._id_counter}'
        Drive._instances[self.id] = self

    @property
    def state(self):
        return 'pressing' if self.activation >= PRESSING else 'present' if self.activation >= PRESENT else 'quiet'

    def touch(self, kind, strength, text, now):
        """A noticed event moves the drive: a threat or chance raises it, a satisfaction eases it."""
        strength = min(1.0, max(0.0, strength))
        if kind in ('eased', 'given_up'):
            self.activation -= EASE * strength * max(0.0, self.activation - FLOOR)
        else:
            self.activation += RAISE * strength * (1.0 - self.activation)
        if text:
            self.notices = (self.notices + [Notice(text, kind, strength, now)])[-NOTICES_KEPT:]

    def on_mind(self, now):
        """The latest thing noticed for this drive, while it is fresh: 'threat: the boar circles back'."""
        if not self.notices or self.activation < PRESENT:
            return ''
        latest = self.notices[-1]
        if now is not None and (now - latest.time).total_seconds() > NOTICE_MINUTES * 60:
            return ''
        return f"{latest.kind.replace('_', ' ')}: {latest.text}"

    def line(self, now=None, ids=False):
        mind = self.on_mind(now)
        return (f'{self.id}: ' if ids else '- ') + f'{self.text.strip().rstrip(".")} [{self.state}' + (f'; {mind}' if mind else '') + ']'

    def evolved(self, character: 'Character', goal: 'Goal') -> Optional['Drive']:
        """The drive rewritten after repeated success (or failure): broader, deeper, more modest,
        or turned against itself. None when the LLM gives nothing."""
        prompt = [UserMessage(content="""A character in a play has pursued one of their drives through several goals. Rewrite the drive as it now stands for them.

Drive:
{{$drive}}

Goal just achieved:
{{$goal}}

Goals attempted for this drive:
{{$attempted_goals}}

Goals achieved for this drive:
{{$satisfied_goals}}

Several goals attempted and few achieved may leave the character more determined, or settling for something more modest or more specific, or turning against the drive altogether.
Several goals achieved may broaden the drive (from 'my' to 'my family'), or deepen it (from 'to be accepted' to 'to be loved').
Some drives simply recur, and are best left nearly as they are.

The character:
{{$character}}

Respond with the rewritten drive only: 8-12 words, no other text.
End your response with:
</end>
""")]
        describe = lambda g: f'{g.name}: {g.description}; termination criterion: {g.termination}'
        result = character.llm.ask({'goal': describe(goal), 'drive': self.text,
                                    'attempted_goals': '\n'.join(describe(g) for g in self.attempted_goals),
                                    'satisfied_goals': '\n'.join(describe(g) for g in self.satisfied_goals),
                                    'character': character.character},
                                   prompt, tag='Drive.evolved', temp=0.1, stops=['</end>'], max_tokens=30)
        text = (result or '').strip().strip('"')
        if not text:
            return None
        return Drive(text, activation=self.activation, baseline=self.baseline, updated=self.updated)


def _now(character):
    return character.context.simulation_time if getattr(character, 'context', None) else None


def settle(character):
    """Bring a character's drives up to the present: resting levels by the play's order (once),
    drift back toward rest over the play time since the last update, and the body's hold on the
    bodily drives."""
    now = _now(character)
    body = getattr(character, 'body', None)
    harm = (body.cfg.get('harm') or {}).get('name') if body is not None else None
    need = max((v for n, v in body.levels.items() if n != harm), default=0.0) if body is not None else 0.0
    hurt = body.level(harm) if body is not None and harm else 0.0
    for i, drive in enumerate(character.drives):
        if drive.baseline is None:
            drive.baseline = BASELINES[min(i, len(BASELINES) - 1)]
            drive.activation = drive.baseline
        if not ablation.enabled('drive_appraisal'):
            continue
        if now is not None and drive.updated is not None and now > drive.updated:
            minutes = (now - drive.updated).total_seconds() / 60
            drive.activation = drive.baseline + (drive.activation - drive.baseline) * 0.5 ** (minutes / HALF_LIFE_MINUTES)
        drive.updated = now
        if _BODILY.search(drive.text):
            drive.activation = max(drive.activation, need)
        if _SAFETY.search(drive.text):
            drive.activation = max(drive.activation, hurt)
        drive.activation = min(1.0, max(0.0, drive.activation))


def ranked(character) -> List[Drive]:
    """The character's drives, most pressing first (the play's order breaks ties)."""
    settle(character)
    return sorted(character.drives, key=lambda d: -d.activation)


def drive_lines(character, ids=False):
    """The drives for a prompt, most pressing first:
    '- find a way out of the forest [pressing; threat: the boar circles back]'."""
    now = _now(character)
    return '\n'.join(d.line(now, ids) for d in ranked(character))


def standing(character, index):
    """The drive at a place in the play's order (0 the primary), in words; '' when there is none."""
    drives = character.drives
    return drives[index].text if index < len(drives) else ''


def find(character, ref):
    """A character's drive from an id ('d7'), its text, or a model's paraphrase of it
    ('love_and_belonging'); None when nothing matches."""
    ref = str(ref or '').strip()
    lead = re.match(r'(d\d+)\b', ref.lower())
    if lead:
        ref = lead.group(1)
    words = lambda s: set(re.findall(r'[a-z]{3,}', s.lower().replace('_', ' ')))
    wanted = words(ref)
    best, overlap = None, 0
    for drive in character.drives:
        if ref == drive.id or ref.lower() == drive.text.lower():
            return drive
        shared = len(wanted & words(drive.text))
        if shared > overlap:
            best, overlap = drive, shared
    return best if wanted and overlap >= max(1, len(wanted) // 2) else None


def appraise(character):
    """Read what the character has taken in since the last appraisal against its drives, and move
    the drives it touched. One LLM call for all the drives; none when nothing new has happened."""
    settle(character)
    if not ablation.enabled('drive_appraisal') or not character.drives:
        return
    memories = character.structured_memory.concrete_memories
    seen = getattr(character, '_appraised_upto', -1)
    # What the character has thought is not an event. (Its own guesses were listed with what
    # it saw and heard, and came back as things noticed: 'Lanya walked off, leaving Kidd
    # alone', of a woman who had left with him.)
    # What it saw and heard is in the record (Context.witnessed_entries); what it was told or
    # made of things is in its memory. Both since the last appraisal, in the order they came.
    lived = [(m.timestamp, m.text) for m in memories if m.memory_id > seen and not m.text.startswith(THOUGHT)]
    if memories:
        character._appraised_upto = memories[-1].memory_id
    context = getattr(character, 'context', None)
    if context is not None and hasattr(context, 'witnessed_entries'):
        entries, character._appraised_record = context.witnessed_entries(character, getattr(character, '_appraised_record', 0))
        lived += [(e['time'], context.entry_line(e)) for e in entries]
    new = [text for _, text in sorted(lived, key=lambda x: x[0])][-MAX_PERCEPTS:]
    if not new:
        return
    now = _now(character)
    body = getattr(character, 'body', None)
    prompt = [UserMessage(content="""You are tracking what presses on {{$name}}, a character in a play, as events reach them.

{{$name}}'s drives:
{{$drives}}
{{$body}}
What {{$name}} has just seen, heard and lived through, oldest first:
{{$events}}

For each drive these events touch, write one line; leave out the drives they do not touch.
#<drive id> <threat|chance|eased> <strength> <what {{$name}} noticed: 5-12 words, a concrete event from the list above>

threat: something endangers or blocks the drive.
chance: an opening to advance the drive.
eased: the drive has just been satisfied, in part or in full, by something that happened. Still having a thing is not eased.
strength: 0.2 a passing matter, 0.5 it matters, 0.9 it cannot be ignored.

Example:
#d2 threat 0.7 the boar is circling back toward the hollow

If the events touch none of the drives, respond with #none
End your response with </end>
""")]
    reply = character.llm.ask({'name': character.name,
                               'drives': '\n'.join(f'{d.id}: {d.text}' for d in character.drives),
                               'body': f"\n{character.name}'s body: {body.summary()}\n" if body is not None else '',
                               'events': '\n'.join(f'{i}. {text}' for i, text in enumerate(new, 1))},
                              prompt, tag='drives.appraise', temp=0.1, stops=['</end>'], max_tokens=40 * len(character.drives)) or ''
    by_id = {d.id: d for d in character.drives}
    noticed = []
    for line in reply.splitlines():
        match = _APPRAISAL.match(line.strip())
        drive = by_id.get(match.group(1).lower()) if match else None
        if drive is None or sum(1 for d, _ in noticed if d is drive) >= 2:
            continue
        notice_text = match.group(4).strip().strip('"').rstrip('.')
        drive.touch(match.group(2).lower(), float(match.group(3)), notice_text, now)
        noticed.append((drive, drive.notices[-1] if notice_text else Notice('', match.group(2).lower(), float(match.group(3)), now)))
    trace.event('drives', character=character.name,
                activations={d.text: round(d.activation, 3) for d in character.drives},
                signals=[{'drive': d.text, 'text': n.text, 'type': n.kind, 'strength': round(n.strength, 2)}
                         for d, n in noticed])


def goal_attempted(goal):
    for drive in goal.drives or []:
        if goal not in drive.attempted_goals:
            drive.attempted_goals.append(goal)


def goal_satisfied(character, goal):
    """An achieved goal eases the drive it served. A drive other than the character's first that
    has been served EVOLVE_AFTER times is rewritten in place (Drive.evolved)."""
    now = _now(character)
    for drive in goal.drives or []:
        if drive not in character.drives:
            continue
        drive.satisfied_goals.append(goal)
        drive.touch('eased', 0.8, f'achieved: {goal.name}', now)
        if drive is not character.drives[0] and len(drive.satisfied_goals) >= EVOLVE_AFTER and ablation.enabled('drive_appraisal'):
            new = drive.evolved(character, goal)
            if new is not None:
                character.drives[character.drives.index(drive)] = new
                trace.event('drive_evolved', character=character.name, was=drive.text, now=new.text)


def denied(character, ref, choice):
    """The character has made a choice that gives up a drive: it recedes as a standing want."""
    drive = find(character, ref)
    if drive is None:
        return
    settle(character)
    drive.baseline *= 0.5
    drive.touch('given_up', 0.5, f'by choosing: {choice}'[:120], _now(character))


def nudge(character, ref, up):
    """A director's intervention raises or lowers a drive. Returns the drive, or None."""
    drive = find(character, ref)
    if drive is not None:
        settle(character)
        drive.touch('threat' if up else 'eased', 0.8, 'a sudden turn of events', _now(character))
    return drive


def for_ui(character):
    """Recent notices as the Explore panel's 'signals' rows."""
    return [{'text': n.text, 'drives': [d.text], 'type': n.kind, 'signals': [f'strength {n.strength:.2f}'],
             'score': n.strength, 'last_seen': n.time.isoformat() if n.time else ''}
            for d in character.drives for n in d.notices]
