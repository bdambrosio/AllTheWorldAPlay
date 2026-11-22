# How a performance is made: director, planner, actors and the world

A note on the model the engine has arrived at (2026-11-21, after the resolver and world overhauls), how it compares with other ways of staging a whole play, where it is weak, and what to do about the weakest part. It sits beside `director-design.md` (what lies behind a play) and `physical-world-design.md` (what is so on the stage).

## The model

Four parties make a performance. What marks the model is what each may decide.

| Party | What it does | What it may not do |
|---|---|---|
| **Director** (`sim/director.py`) | Before the play, settles what is secretly so behind the play's question, the things that are traces of it, and a person who knows a part | Say what anyone will do, or how it comes out |
| **Planner** (acts and scenes; the scene owner) | Lays the play out as matters to be settled: what is at stake in each act and scene, never the result | Decide outcomes, bodies, or what is in the world |
| **Actors** (`sim/agh.py`) | Speak and act from what they have seen, been told, want and remember | Make something so by saying it |
| **The world** (set-up check, resolver, keeper, the places, the record) | Says what each proposed set-up, line and deed does, and keeps the one account of it | Want anything, or steer the story |

Two rules hold it together.

1. **Everyone proposes; only the world disposes.** A planner's set-up, an actor's deed and a spoken claim are proposals, held against one record (`record.account`).
2. **What the engine will hold nobody to is simply played.** A world file declares needs, states something depends on, and exchanges and odds; all else is said, done, shown and remembered.

In theatre's words the fourth party is the stage manager, the props master and the fight director; in a game's, the referee. In one phrase the model is **refereed improvisation from a scenario of questions**.

## Beside other ways of staging a play

| Way | Who holds the plot | Who says what is so | Where this differs |
|---|---|---|---|
| Scripted theatre | The playwright, wholly, beforehand | The script | Nothing planned here is a line or an outcome |
| Commedia dell'arte | A scenario: the order of scenes and how each ends | Convention, stock business | Nearest in spirit; but a scenario says how a scene ends |
| Long-form improvisation | Nobody; a form (the Harold) gives shape | Whatever is said ("yes, and") | Here saying does not make it so: the largest departure |
| Role-play with a referee | The referee's prepared secrets; "play to find out" | The referee and the rules | The nearest relative. The referee is split in three: director (secrets), planner (pace), world (rulings) |
| Drama management (Façade, the Oz project) | Authored beats, chosen to follow a curve of tension | Authored content | They measure tension and author every beat; this authors nothing and measures no tension |
| Simulation of characters (Tale-Spin, emergent narrative, generative agents) | Nobody; story is hoped for | A world model or shared memory | Believable people, shapeless story. The director and planner are the answer to that |
| Generation from an author's plan (Universe, Minstrel, intent planners, outline-to-script) | One plan, refined downward | The plan | Coherent plots; characters are the plan's puppets. Here they can refuse it |
| Director and actor language models | A director model steering actors to plot points | Usually nothing: the text is the world | The closest recent work; it lacks a world that can say no |

No one piece is new. The combination may be: a truth settled for each performance, a plan made of matters and not outcomes, actors who answer for themselves, and a referee with a world of its own.

## Level by level

**What is so.** The strongest level. Most generated drama lets the text be the world, so things multiply and people are in two places. Here a deed is refused for a missing key. The risk is the other one: the world was right that nobody held the car keys, and the scene died of it (alex-20261121-173215). A stage manager prevents that by asking, before the scene, what it will need. The set-up check does not yet ask.

**Plot and tension.** The weakest beside the best earlier work. Tension is managed by a few rules of planning ("each scene's matter is harder than the one before") and an intervention when a scene goes stale. Nothing reads what was played for whether it rose.

**Character coherence.** Good. Memory, drives, what stands between two people, what each has told whom, names not known until learned; and, which simulations of characters lack, a check that a line claims only what its speaker has seen or been told.

**Character development.** Largely unowned. See below.

**Dialogue.** Good line by line, unmanaged over a scene. Susan said "it keeps till ten" some eight times in two scenes. See below.

**The audience.** Nobody stands for it during a performance. The judges read afterwards.

## Whose job are repetition and change?

An earlier draft of this note proposed a fifth party, a dramaturg, to read each scene for its shape: what moved, what repeated, whose turn it is to change. Bruce's objection (2026-11-21) is right and changes the proposal:

> A real Susan would not say "it keeps till ten" eight times. She would not need someone to monitor that for her. Likewise a real character would be affected by events and would change as a result. The director might want input there, as might the scene; but to maximise character autonomy the director might provide an impetus, the scene might be more specific, and within guidelines the actor should be the one to effect change in the character model itself.

So these are faults in the actor and the character model, not gaps to fill with a watcher. A watcher that corrects an actor's repetition, or decides that a character has changed, takes from the actor exactly what the model gives it: its own answer for itself.

### What the actor has today

- **Its description**, as its author wrote it, kept whole; and one paragraph the actor writes itself after a scene, "Lately": what it now believes, wants or has committed to, and how it stands with the others (`Agh.update`). This is already the actor changing its own model, and the only place it does.
- **Drives**, whose pull moves with what happens (`drives.appraise`). What they are does not change.
- **What stands between it and each other person** (shown, seen, owed, wanted, shared, stands), added to after each exchange.
- **A stance** of three words, re-read as it goes.
- **Its goal in the scene**, given by the plan and put right by the set-up check.
- **The talk so far**, as text.

### Why Susan repeated herself

Her own prompt, late in the scene, held the phrase "after ten" fourteen times, six of them her own "keeps". The words were in front of her. What was not:

- **Anything of her own doing.** The talk so far is a transcript. Nothing says, as her own knowledge, "you have put him off four times with the same words, and each time he has pushed harder."
- **More than one way to hold her ground.** Her goal was to keep the secret until after ten. One goal, one tactic, and no cost for using it again.
- **Anything that her holding out was costing.** Alex's pressure rose on the page and in his prompts, not in anything she carries.

A person in her place knows what she has already tried, feels it wearing thin, and shifts: gives a little, changes the subject, gets angry, or tells. She repeats only when she means to, and knows she is doing it.

### Why nobody changes

"Lately" records what a character now believes and has committed to. It does not record being changed: nothing in the model says that something a person held to has given way, or hardened, or what it took. And nothing asks. A scene's matter is about the situation; a drive's text is fixed for the play.

### Proposal: the actor keeps its own account of itself

Three small things, each the actor's own, none a watcher.

**1. What I have tried, and what came of it.** Within a talk, the actor keeps a short running note in its own voice of its moves on the matter in hand and how each was met: "put him off till after ten, twice; he has stopped accepting it." It is written by the actor at its turn (one line, replacing the last), and shown to it at the next. Not a rule against repeating: a person may repeat, knowingly. It gives the actor what a person has and it lacks, the knowledge of what it has already done.

**2. What holding to this is costing me.** Where a character is keeping something back or refusing something, the note says what it has cost so far (his trust; the morning; her own ease), again in the actor's words. This is what makes a position wear thin from the inside, and lets it give way for a reason the character owns.

**3. How I have been changed.** The "Lately" paragraph gains a second part, written only when it is true: one thing the character held at the start of the play that has given way, hardened or turned, and what did it. It is the actor's to write, and the description above it stays as the author wrote it.

**Who may ask for change, and how far.**

| Party | What it may give | What it may not |
|---|---|---|
| Director | An impetus, before or between acts: what this play presses on in this person ("her patience with waiting") | Say that they change, or how |
| Scene (planner, set-up) | The occasion: what in this scene bears on it ("he finds the list") | Say what they make of it |
| Actor | The change itself, in its own account, when what has happened to it warrants | Change what the record shows it did, saw or was told |
| World | Nothing here: it checks facts, not character | |

**Guidelines the actor's own account is held to.** A language model asked whether it has changed will too readily say yes. So: a change names what in the record did it (a thing seen, said or done, not a mood); one change at a time; nothing of the author's description is unwritten; and "unchanged" is an answer, as it is now.

### What this leaves for a reader of shape

Less than the earlier draft gave it, and none of it over the actors. Whether tension rose across an act, and whether the next scene's matter is harder than the last, is still nobody's reading. That belongs with the planner, as a reading of the record between scenes, and is a separate question from this one.

## Open questions

1. Is the running note (1 and 2) one more model call for each turn, or part of the turn's own answer? The second is cheaper and keeps the note in the same breath as the line.
2. Does the director's impetus belong in what it settles before the play (`director.settle`), beside each person's past, or in its reading after each act?
3. Should the other actor ever see any of this? A person's sense that the other is wearing thin comes from what they do and say, which argues no.
4. The set-up check asking what a scene will need that is not there (the car keys): a small change, and separate from all of the above.
