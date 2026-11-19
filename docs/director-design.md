# The director: a design

Draft for mark-up, 2026-11-17. No code has been written for any of this.

Decided by Bruce, 2026-11-17 (worked into the text below):
- The order before the play is premise and facts first, then each character's side, then the
  question (1.5). The process matters more than who takes part in it.
- A person from outside the cast may be drawn. One or two such persons at most in a performance;
  one who has come on may be drawn to come on again (2.4).
- The number of acts is not fixed in advance: up to five, ending when the play has reached its
  final test (2.5).
- The draws and the premise are not shown for approval before act 1.
- The pace draw and the ending-when-ready rule are both to be tried, as written in 2.5.
- An outside person does not draft scenes; the director places them.
- A play file does not weight the draws.
- The repeat check reads the last three performances and is one quick call.

Decided by Bruce, 2026-11-18, after `lost-20261118-112635` dropped its mystery (1.6):
- A play may stand on a question from its sketch, the same every night. For lost: why these two
  are here and how they came, whether it is a place to escape or of safety, whether there is a
  way out; and if there is, whether they take it, and if not, how they live there.
- The director settles what is so behind that question, afresh each night. It does not resolve
  it: what the characters do about it is theirs, in play. The director's work is to steer the
  performance along this thread and bring them face to face with it.
- The answer varies in kind from night to night. What it shows of the characters matters more
  than how strange it is.
- A night may end with it only partly found.
- Persons from outside the cast: yes (the hut's owner is the first candidate in lost).

Part 1 is the director's record and the director's duties. Part 2 is how a performance is kept
from being the same play every night, and from being a script. Part 3 is the prop model, in
outline only: it depends on Part 1. Part 4 is build order and testing. Part 5 is the questions
I need you to answer.

## 0. What this is fixing

From `lost-20261117-104840` and `bellona-20261117-134621`:

- The whole play is held by one call at the start (`frame_episode`). After that its result is a
  note pasted into the act-planning prompts. Nobody reads what was played against it.
- Each act's plot is a merge of the characters' own drafts. The frame is a reminder to the merge.
- The play's closing record, and every act's end state, are the planned text repeated back.
- The frame may not say who, how or why, may name no person outside the cast, and may invent
  nothing but two or three clues. So nothing lies behind a clue to be found out. This is the
  "flat" of the lost run.
- The thing a play turns on (bellona's notebook page) can exist with no text, no states and no
  uses, because no one is responsible for it.
- A talk-made beat can send the next act somewhere the scene's ending ruled out (bellona act 4
  went to the shop; act 3 ended heading for the bus stop).

## 1. The director's record

### 1.1 Who, where, when

A new `Director` (proposed `src/sim/director.py`) owns the `Screenplay` record that exists today
and is the only writer of it. It runs on the planner model. It acts at five fixed points:

| when | call | what it does |
|---|---|---|
| before the play | `premises` | writes several different plays this sketch could be tonight (Part 2) |
| before the play | `behind` | for the one that was drawn: what lies behind it, the things, who knows what |
| before the play | (each character) `side` | knowing only what it knows, says what it wants and what it would have to choose between |
| before the play | `question` | writes the question from the facts and the sides, and lists what is missing (1.5) |
| before the play | `line` | writes the rest of the record (1.2): the line, reveals, final test, deck |
| before each act | `brief` | says what this act must settle, its hinge, and what it may not do |
| after each act | `read` | reads the act as played; updates the record; says whether the final test is ready; may revise the acts to come |
| after the play | `verdict` | says, from the record of what was played, what was settled and how |

That is 4 director calls and one per character before the play, 2 per act, and one at the end:
about 15 calls for a four-act play with two characters. At this run's planner speed that is
about four minutes in a 90-minute play.

`brief` replaces `generate_act_central_narrative` plus `episode_note`. `read` replaces
`close_act`'s copied end state. `premises`, `behind` and `line` replace `frame_episode`. `side`
and `question` replace `negotiate_central_narrative` and `merge_central_narratives`.

### 1.2 What the record holds

Written before act 1, in the order of 1.5:

- **Behind**: what is so behind this play, in full. A short paragraph and the separate facts
  drawn from it, each with who knows it. Unlike today's facts it may say who, how and why, and it
  may include a person outside the cast if the draw allows one (Part 2). It still may not
  contradict the sketch, and it still uses only this world's places.
- **Outside persons**: none, one or two (2.4). For each: who they are, what they want, what they
  know of the facts, and where they are when the play begins.
- **Sides**: for each character, the choice it faces, in its own terms.
- **Question**: the dramatic question, written from the facts and the sides, with both ways it
  can come out.
- **Things**: every thing the play turns on. For each: who has it or where it lies, what it says
  if it is written, what it is evidence of (which fact), and its states and uses (Part 3). A
  thing named in Behind or in the line with nothing here is an error the director must fix
  before the play starts. This is the check that would have caught the notebook page.
- **The line**: one entry per act. Not scenes. Each entry says what must be different by the
  act's end (what is known, what has changed between them, where the things are), and names the
  act's hinge (1.4).
- **Reveals**: for each fact, the earliest act in which it may come out, and whether it stays
  hidden in this play.
- **The final test**: the situation in which the question gets answered, and who decides. Not
  the answer (1.4).
- **Deck**: three or four things that could happen from outside (Part 2).
- **Draws**: the seed and everything that was drawn, so a performance can be explained.

Kept by `read`, after each act, from the transcript and the world's record:

- **Played**: what happened, in the act's own events. This exists today as the act summary.
- **Settled**: which entries on the line are now done, with the line or deed that did each.
- **Owed**: what the line or the characters have promised and not delivered (a story owed, a
  walk agreed, a clue found and not used). Today's threads fold into this.
- **Known**: which facts each character now knows, and how they came to know it.
- **Drift**: where the play has left the line. For each, the director rules: follow it (rewrite
  the acts to come) or bring it back (say so in the next brief).

### 1.3 How the rest of the engine uses it

- **Characters' act drafts** are kept. Each character is given the brief with anything it does
  not know taken out, and proposes scenes that serve its own wants. This is where the scenes come
  from.
- **The merge** (`integrate_narratives`) becomes the director's. It still builds the act from the
  drafts, but it now has a duty: the act as merged must be able to deliver the brief. If the
  drafts cannot, it says what is missing and adds one scene, marked as its own.
- **The scene check** (`reground_scene`) gets the record. Today it checks a scene against the
  world's facts only. With the record it can also see that a scene repeats what is settled, or
  spends something the line needs later. This is the act-level review you proposed after the lost
  run, done scene by scene.
- **The scene owner** is unchanged. It is given the ending as now.
- **Talk-made commitments** are placed against the record. One that fits the scene's ending
  becomes a beat as now. One that contradicts it is not played in this scene: it goes to Owed,
  and the director rules on it at the act boundary.
- **Act and play end states** are written by `read` and `verdict` from what was played.

### 1.4 What the director may not do

This is the line between the director's part and the characters'.

- It writes no speech and no scene beats.
- It tells no character a fact that character has not come to know.
- It does not change a fact once written. It may change which acts reveal it.
- **It does not decide a hinge.** A hinge is a point in an act where a named character chooses
  between two things, each of which the director has thought through. The scene is planned up to
  the choice, not past it. The character decides in play, and the next brief is written from what
  was chosen. The engine has a decision step today, but the ending is written with the choice
  already made (bellona scene 2: the decision and the ending both said "one stop, then the
  shop" before a line was spoken).
- **It does not decide the answer.** The final test is the last hinge.

### 1.5 The order before the play

What the bellona run did (`bellona-20261117-134621`): Kidd proposed a question with nothing on
the table, and its premise ("the page says I left her behind") was an aside in an eight-word
slot. Lanya, shown his proposal, took the premise whole, including what her character cannot
know. A merge call wrote the question. The facts were settled after that, and none of them said
that he left her, when, or what the page says. The question was a good one. Nothing stood behind
it.

What was worth keeping: each character put the question as its own choice between two things
("show her, or follow it alone"; "follow him, or leave him first"). Those are nearly hinges.

The order proposed:

1. **Draws** (2.1).
2. **Premises**: five written to differ; one picked by code.
3. **Behind**: the drawn premise settled in full: facts, who knows each, the things with their
   text, any outside person.
4. **Sides**: each character, told only what it knows at the start, says what it wants here and
   what it would have to choose between. One call each, in character, with no sight of the
   other's.
5. **Question**: the director writes it from the facts and the sides.
6. **Missing**: anything the question needs to be so that the facts do not say goes back to
   step 3 and is settled. One pass back at most; a question that still needs something unsettled
   is rewritten without it.
7. **Line**: the rest of the record.
8. The characters write their own plans, as now.

Draft instruction for step 5 (untested; under the prompt-change rule it is to be run offline
against today's negotiate-and-merge on both plays before it replaces anything):

> You are the director of an improvised play. What lies behind tonight's performance has been
> settled and is given below, with the things it turns on and who knows what. Each character has
> said, knowing only what they know, what they want and what they would have to choose between.
>
> Write the dramatic question of this performance.
>
> - It is a question about what these people will do to and with each other, here, in this play.
>   It is not a question about what happened before: that is already settled, and is only found
>   out.
> - It has a side for each character: a choice that character must make between two things they
>   want, where taking one costs the other. Take each side from what that character said, in
>   their terms. A side may not depend on a fact its character does not know at the start; say
>   instead what would bring them to it.
> - It can come out either way. Write what the play looks like if it is answered yes and if it is
>   answered no, each as something an audience would accept as an ending. Do not say which is
>   better.
> - It rests on the facts given. If the question needs something to be so that the facts do not
>   say (when, where, who was there, what a written thing says), do not assume it: list it under
>   #missing.
> - Name the final test: the moment where the question is answered, where it happens, and which
>   thing is in whose hands. The one who decides there is given below.
>
> Respond with these lines only:
> #question / #side \<name\>: \<the choice\> (one for each character) / #stakes / #yes / #no /
> #test / #missing (one line each, or none)

For the bellona run, `#missing` would have had to list the words on the page and when and where
he left her.

### 1.6 The standing question

`lost-20261118-112635` was a play about trust between two people. Why they were there was touched
in act 1 and never again. This was by construction: `frame_episode` says the larger question "is
not answered", clues "never say who, how or why", and the sequence of 1.5 as tested on 2026-11-17
gave "will they stay together or go alone" in 20 of 20 lost plans, because a side written by a
character who knows nothing can say nothing else and the question rule ruled out what happened
before.

The sketch already holds the question: it is the first want of each character in `lost.py`. It is
now written there once, as `W.standing_question`. With it, 1.5 becomes:

1. **Draws** (2.1), without the decider: each character chooses for themself at the end.
2. **Answers**: five answers to the part of the standing question that asks what is so; the
   repeat check; one picked by code. This is the premises step, pointed at the question.
3. **Behind**: as before, and every fact names what carries it (a thing, a person, a trace, or
   what a place does when tried). A fact nothing carries stays hidden. No one carrier holds all
   of it. A place writes and records nothing; a written thing was written by a person for their
   own reasons.
4. **Thread**: for each act, what is in front of the characters at its end that was not, and from
   which carrier at which place. The last act puts the choice to them: where, what is in whose
   hands, and what pulls each of them each way. Nothing says what either chooses.

**Changed 2026-11-19 (Bruce, watching `lost-20261119-084100`):** the question is not written in the play file. A play file
asks for a director with `W.directed = True`, and the director writes the question for each performance from the
sketch (`director.QUESTION`), before the answers step; the last three nights' questions are given to it so that
tonight's is another. Offline test: `logs/review-20261119/question/`.

As first built, there was no sides step and no question step: the question was the sketch's. This settles the two
decisions of Part 5 for a play with a standing question (the sides are the pulls, written by the
director who knows what is so; both characters choose, separately). A play without a standing
question keeps 1.5 as it was, and those two decisions stay open for it.

Harness and results: `logs/review-20261118/standing/` (`run.py`, `RESULTS.md`).

### 1.7 What three offline rounds showed, and the light director that was built

`logs/review-20261118/standing/RESULTS.md`. Lost 20 plans twice, bellona, laTerre and ABetterLife
8 each, and a reading of the seven sketches not run (`fit_other_sketches.md`).

- Held: ordinary answers that this world can show; a person with a stake; more different plays
  when several planner models write the candidates and code picks.
- Not fixed: the answer is told by one keeper or one paper (13 to 14 of 20), and interest stayed
  at about 3 of 5 through all three rounds.
- Against character autonomy: the more the director settles, the more it writes what the
  characters did and agreed to before the play and the terms and price of their choice
  (ABetterLife 7 of 8, bellona 5 of 8). Most plans contradicted their sketch to do it.
- The last choice was usually about something off the stage, so it could be said and not done.
- Not every sketch has a standing question (cafe, kittens, sage), and where one fits, what is so
  may be a present condition and not a hidden past (VaccumLine), or already known to some of
  the cast (ABetterLife, dilemma).

So what is built (`src/sim/director.py`, Bruce's go-ahead 2026-11-18) is lighter than 1.6:

- **answers**: five short answers from each planner model named (`ATWAP_DIRECTOR_LLMS`, beside the
  play's own planner), mostly about the place; a repeat check against the last three
  performances (`~/.local/share/alltheworldaplay/performances/<play>.jsonl`); one picked by code.
  `ATWAP_SEED` replays the draws.
- **behind**: three or four facts, two or three things placed in the world, at most one person
  who knows a part and wants something of their own. The person is made a played character at
  their place. Nothing says what they will tell.
- No steer and no planned choice. The act planners are given the question and told to plan no
  scene that answers it for anyone (`Context.episode_note`).
- **read**: after each act, what each leading person has now come to, and whether the act came
  nearer; the next act's planners are told.

A play file with no `standing_question` is played as before. Not built: pressure cards, hinges,
the closing verdict, per-sketch declarations beyond the question itself. No live run yet.

## 2. Keeping the plot from being fixed

Two different failures: the same play every night, and a play that runs on rails once it starts.
They need different mechanisms. One principle for both: **chance is drawn by code, and the model
writes what the draw means.** Model temperature is not a source of variety. Given the same
sketch, a model returns its favourite reading nearly every time (both plays to date turned on
trust and a walk to the next place).

### 2.1 Before the play: what is drawn

One seed per performance, written to the trace. `ATWAP_SEED` replays the draws (not the wording).

| draw | from | how |
|---|---|---|
| Lead want | each character's drives in the play file | one per character, weighted toward the first (proposed 50/30/20 over the first three) |
| Premise | candidates the director writes | see below |
| Where the main reveal falls | early, middle, late | even odds. Early makes a play about living with it; late, a play about getting to it |
| Who decides the final test | the cast | even odds |
| Outside pressure | none; a danger this world has; a person from outside the cast | proposed 40/40/20; see 2.4 for persons |
| Pace | 3, 4 or 5 acts as the length the line is first written for | proposed 30/50/20; the play may end sooner or later than this (2.5) |

**The premise draw.** `premises` is given the sketch and the lead wants and writes five plays
this could be tonight. One call, because a model varies far more within one list than across
five separate calls. Each candidate says what happened before the play, who is in it, what the
main thing is evidence of, and what question it raises. The five must differ in what happened,
not in wording. Then:

1. Each is checked against the world as plans are today (places exist, things can be held).
2. Any that repeats one of this play's last three performances is dropped. One quick call: a
   model is given the five candidates and the three one-sentence histories and says which
   candidates are the same play as one already performed; no word matching.
3. Code picks one of the rest at random. Not the model's favourite, and the model is never asked
   which is best.

**History.** One line per performance in
`~/.local/share/alltheworldaplay/performances/<play>.jsonl`: seed, draws, the premise in one
sentence, how it ended. The last three are what step 2 reads.

For bellona, the lead-want draw alone gives different plays: Kidd led by "what happens in the
hours you lose" against Lanya led by "not the one left behind" is roughly today's run; Kidd led
by writing against Lanya led by the song she wants sung after she is gone is another play with
the same people.

### 2.2 During the play: what stays open

- **Hinges** (1.4). One per act, the character's to decide. The answer to the play's question is
  the last of them. This is the main guard against rails, and it is not chance: it is the
  characters owning whether, as well as how.
- **The deck.** At `line` the director writes three or four things that could happen from
  outside: what happens, when it could (a place, or after which act), and what it would change.
  At each act boundary code draws: proposed even odds that one card is played, at most one per
  act. The director writes it into the next brief. A card never settles the question; it changes
  what settling it costs. This replaces the canned lists in the staleness intervention ("a
  distant sound catches everyone's attention").
- **The world's own chance** stays as it is: dangers, and what a search turns up.
- **Revision.** After each act the director may rewrite the acts to come to follow where the
  characters went. Facts stand; the line does not.

### 2.3 What is fixed, so that it is still a play

After the draws: what lies behind, the things, and the question. These do not change during a
performance. Everything else can.

### 2.4 Persons from outside the cast

A performance has at most two. They are written at `behind`, as part of what lies behind the
play, not made up when a card is drawn: an outside person is someone the facts already contain
(the one who left the boot prints, the lover who walked out), so that when they come on they
fit.

- The pre-play draw says whether tonight has none, one or two (proposed 50/35/15, where the
  world can hold one).
- A person comes on by a deck card. A person who has come on stays in the deck: a later draw
  may bring them on again. No draw creates a third.
- An outside person has wants and knows some of the facts. They do not decide a hinge and they
  do not answer the question: they change what it costs.
- They are played as the cast are, with lines of their own, for the scenes they are in.

What exists today and what has to be reviewed before this is built (read from the code, not
tested):

- The engine can make a person outside the cast on demand (`get_npc_by_name`) and the old
  intervention step adds such a person to the play's extras. These have a description and
  drives made from one sentence.
- To review: whether such a person gets what a cast member gets at the start (a body in the
  world, a place on the map, a voice, a picture, a card of what it knows); how it enters and
  leaves a scene, and whether the exit rule and the scene owner count it; whether the scene
  owner's sheet and beats can include it; what it is told of the play (its own wants and facts
  only); whether the others' models of other people pick it up; and what the audience is shown
  when it arrives.
- An outside person does not draft scenes for an act as the cast do. The director places them,
  by the card.

### 2.5 How many acts

Not fixed. The pace draw (3, 4 or 5) is what the line is first written for. After each act,
`read` says whether the play is ready for its final test: the facts the test needs are known to
those who need them, and the things it needs are where they must be.

- Ready: the next act is the one the test is played in, whatever its number.
- Not ready, and fewer than four acts played: another act, and the line is rewritten for it.
- Four acts played: the fifth is the test, ready or not, and its brief is written to get there.

The closing act follows the test as now. No play ends before its second act.

Today the count is the largest number of acts in any character's own plan, settled before the
play. With this the characters' plans no longer set it.

## 3. Things, in outline

Detailed design to follow once Part 1 is agreed. The shape:

- A thing has a state as well as a holder: open or shut, whole or torn, lit or not, and what is
  inside it or tucked into it.
- Whoever brings a thing into the play says its states and uses. For things the play turns on
  that is the director, in the record. For things in the play file it is the play file, or the
  director at `behind` where the file is silent.
- Ordinary handling of a thing a character holds is always allowed and the world keeps the
  result: open, shut, pocket, take out, fold, hand over, put inside another thing, read.
- The existing guard stays for what it was built for: a deed that makes a thing exist, defeats
  an obstacle, or uses a thing for something its uses do not cover.
- A deed that moves another person becomes an offer the other takes or declines.
- A written thing the play turns on has its text in the record before the play starts.

## 4. Build order and testing

1. **Everything before act 1 (1.5), offline only.** Nothing is played. Run it twenty times each
   on lost and bellona. Metrics, by blind reading: how many distinct premises in twenty (same or
   different play, judged in pairs); every thing in Behind present in Things with text where
   written; nothing contradicting the sketch; each side resting only on what its character
   knows; the question answerable either way; nothing left under missing; every line entry
   something that can be seen done. Compare against twenty runs of today's negotiate, merge and
   `frame_episode`. This is the cheap step and it tells us whether the draws produce real
   variety before anything else is built.
2. **`read` and `verdict`, offline**, on the two recorded plays: does `read` catch what we found
   by hand (the page with no text, act 4 going to the shop, the last image never played)?
3. **`brief` and the director's merge**, live: one play each of lost and bellona.
4. **Hinges**, live.
5. **The deck**, live, without outside persons.
6. **Acts not fixed** (2.5), live.
7. **Outside persons** (2.4), after the review listed there.
8. **Things** (Part 3), after its own design.

Every prompt replaced here falls under the prompt-change rule: whole-prompt review and an
offline A/B on past instances.

## 5. First test and what it leaves to decide

Part 4 step 1 was run on 2026-11-17: `logs/review-20261117/director/RESULTS.md` (harness `pre.py` there; no
engine code). Twenty runs each of lost and bellona against today's negotiate, merge and frame.

Held up: the draws and the premise pick. Blind readers found 15 to 18 distinct plays in 20 against 4 to 6
for today's chain; something settled behind every plan against none; interest about 3.3 against 2.2 of 5.
The repeat check dropped 13 and 18 candidates, so it is needed.

Did not hold up: the question and the line. Of 20 records a play, only 2 (bellona) and 1 (lost) were scored
4 of 5 for playability. Found:

- Half the question is not open: the yes and the no ending share one character's act (15 and 20 of 20).
- The question is stale against the premise, because a side written knowing almost nothing can only say
  "stay with this person, or go and look" (9 and 10).
- Hinges that change nothing, that the act's own entry decides, or that a later act needs one way.
- Characters stating what nobody knows, where no thing or person could have told them (6 and 14).
- Acts opening at places nothing leads to, and clock times ignored (2 and 11).
- Smaller: a carried thing of the sketch used and not listed; outside events that are not outside; an
  outside person's arrival as the only source of a fact the acts need.

To decide (Bruce), since these change 1.5:

1. Sides. Either the director writes the question from the premise and each character then states its side
   of that question, or the sides stay first and the question is required to turn on what lies behind.
2. One decider against a two-sided question. Either both halves are real choices at two different moments,
   or the question is one person's choice and the other's part is to make it costly.

To add and retest without a decision: every fact that is to come out names its carrier (a thing whose words
say it, a person who knows it, a trace that shows it), and a fact with none stays hidden; the line is
written against the map and the clock; the sketch's carried things are always among the things; the
premises call is given the last three performances itself.

## 6. Questions answered

Answered: the order before the play (1.5); outside persons allowed, two at most, placed by the
director (2.4); acts not fixed, up to five, with a pace draw (2.5); no showing of the draws
before act 1; no play-file weighting; the repeat check reads three performances.

Still open:

1. **The weights in 2.1, 2.2 and 2.4** are my proposals with nothing behind them but judgement.
   The offline test in Part 4 step 1 is where they first get looked at.
