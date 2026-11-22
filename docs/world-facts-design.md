# The world holds what is so, not what may be done

Agreed 2026-11-22. This replaces the catalogue of actions in the world files (the "rules"), the
tables of ways a need is eased, the states, and the code that offers, matches and applies them.

## Why

A list of named deeds with fixed results has failed in every form it has taken here.

1. It is never complete. A deed nobody listed is impossible or free of consequence (Bellona: nobody could eat).
2. It is wrong in context. Alex pours coffee for Susan and the world says he drank it. Kidd eats half a tin and the whole tin is gone.
3. It refuses for reasons nobody in the play could state: "Xiaoyu is alert: there is nothing to stand."
4. It needs recognising: words matched to entries.
5. It puts the engine's sentences and sums on the page: a payment of exactly 500.
6. It never converges: every repair is another key.
7. It makes the author script the world in advance.

In nine plays of 21 and 22 November there were 31 refusals. Five came straight from a table,
about ten were knock-on from a table's fixed result or vocabulary, about nine were a stage too
bare (no fork in a kitchen), two were a name not matched, and about five were right. No deed
was wrongly allowed. Over-easy acceptance was found only in words (a meeting invented, a clue
claimed), which is the words check's matter and not this design's.

## Goal

1. **Open.** Anything a person could do in that world can be attempted. No deed needs to have been foreseen.
2. **Judged in its particulars.** What a deed does follows from this deed, this doer, these things and people, and what the world says is true of itself. Nothing maps a named action to an outcome.
3. **What is so stays so.** A deed's results are changes to the state of things, people and places, and every later ruling sees them.
4. **The engine keeps only what is true of any world:** a thing is in one place and one hand; money and goods are not made by saying so; time passes and bodies change with it; only those present see and hear; a person knows what they witnessed or were told.
5. **A world or sketch states facts in plain words,** including what an act that matters to the play requires, settles or is limited by. These are seeds, as places are. Code never matches on them.
6. **A refusal is one a person there could give,** and says what would work.
7. **The page tells the deed in its own terms.** No engine sentence reaches it.
8. **How likely a thing is** comes from what is known and how things stand, judged each time. The engine only draws.

The test for any proposal: does the engine, or a world file, name an action for code to act on?
Then it is out. Would the code change for a new kind of world? Then it does not belong in the engine.

## Who rules

The world: the fourth party of `docs/performance-model.md`. It says what each set-up, line and
deed does and keeps the one account; it wants nothing and steers nothing. It rules through the
set-up check and the resolver (`sim/resolver.py`). "Judge" stays the word for the models that
score a finished play.

## Where each thing lives

| Place | Holds |
|---|---|
| World file | Places; bodies' needs (names, how fast they rise, their words); starting money; `facts`, in sentences; `held`, the few restrictions the world may not be argued past |
| Sketch | People, and the standing facts the play opens on |
| Engine | Bookkeeping: where things are, who holds what, money conserved, time, who witnessed what, the draw of a chance |
| Record | What has happened, and what now stands between people |

## What the world is asked

One call for a deed and one for a line, as before. It is shown the world's facts, what is
held, and what stands, beside the stage. It answers with changes of state. Beside the kinds it
had (a thing moves or is left in a state, a mark, a place, someone goes, a thing made or taken
from a place), it has:

| Kind | The resolver says | The engine does |
|---|---|---|
| Money | how much, from whom, to whom; either may be someone outside the play (a bar, an employer) | checks the payer has it, moves it |
| A need | whose, eased or worsened, how far: a little, well, fully | maps the three words to amounts |
| Time | about how many minutes the deed takes | moves the clock |
| A standing fact | a sentence, whom it is about, an earlier fact it ends | keeps it, shows it to those who know it |
| A chance | how likely (rare, unlikely, as likely as not, likely, almost certain), what follows if so and if not | draws |
| Word sent | the words and whom they reach, where a fact of the world gives the means | puts them with that person |

It no longer has: "one of the world's rules", "eased by a listed way", "a state this world keeps".

A thing that a place of its kind plainly has (a fork in a kitchen, a glass at a bar) is there,
and comes into the record when a deed first uses it.

The resolver says which held restriction bore on a deed, so that a wrong ruling shows in the trace.
A held restriction is enforced by that instruction and the recorded fact, and by nothing harder.

## Consequences

- No extra model calls. Each ruling's prompt grows by the world's facts.
- The cast and the planner are shown the facts where they were shown a list of rules.
- Until every world is converted the engine carries both ways; a world file with `facts` is ruled this way, one without is ruled by its tables. The tables' code goes when the last world is converted.
- Three words for how far a need is eased is coarse on purpose.
- Not covered: dangers (the hazard tables), and what the keeper replays by code. Both need their own design before the worlds that lean on them are finished. An outcome that arrives later (cramps in an hour) is not held yet.

## Order of work

1. Acceptance cases, written first (`logs/review-20261122/facts/`): past refusals, wrong fixed results, and deeds that try to get past each held restriction, each with the outcome expected. City, rural and forest.
2. Standing facts in the record.
3. The resolver's new kinds of change and revised instructions, with the bare stage. Whole-prompt review; replay on the cases and on recorded deeds.
4. The world files as facts; the sketch's opening facts as standing facts; the cast and planner shown facts.
5. Play *A Better Life*, act 1, and review before any other world is played.
6. The other worlds; dangers designed first; then the tables' code is deleted.
