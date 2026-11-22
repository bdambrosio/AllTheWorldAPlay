# What befalls a character, and who saw it

A note (2026-11-21), not built. It replaces an earlier proposal for a "lost time" event in the engine, which was wrong in kind: lost time is one play's matter, and it belongs in that play's sketch, where it already is. What the case exposed is three general gaps in what the existing parties can do. Lost time is the example; nothing here names it in code.

## The example

Bellona's sketch says of Kidd that he loses time, comes back somewhere else, is afraid of what he did in those hours and hides the gaps; his first drive is to find out what happens in them. In the act played on 2026-11-21 he talked about it and it never happened. The critique of that act asked for the rule to be shown once, early: lights down on him in one place, lights up somewhere else, later, with no idea what he did, and someone who saw.

## What the engine can already do

- **Open a scene later and elsewhere.** The planner gives a scene its place and its "when"; the cast is put there and the clock moved.
- **Leave someone with no memory of a stretch.** What is not played is in nobody's memory. A character brought on after a jump recalls nothing of it, by construction.
- **Hold a fact that only some people know.** The director settles facts for each performance, each with who knows it (`director.settle`, the screenplay's facts).
- **Make what was done so.** A page newly written, a thing taken, a mark left: the resolver applies them as it does any deed.

A lost hour is these together: a scene that opens after a jump, and a fact about what was done in it that the one it happened to is not among the knowers of.

## The three gaps

1. **Nothing tells the planner to stage what a sketch says befalls someone.** It takes "you lose time" as a trait to speak of. A rule for the prompts that plan acts and scenes: what a person's description says happens to them (a fit, a blackout, sleepwalking, hours lost) is shown happening, at least once and early, by a scene that opens after it; what they did in it is not the plan's to say.

2. **The director settles facts only before the play.** Between acts it reads what the cast has come to know and whether the act came nearer to what is so (`director.read`). It cannot add a fact about something that happened off the stage between two scenes. It should be able to: one fact, in one sentence, with who knows it, when a scene opens after a stretch in which something was done.

3. **A set-up is one text, shown to everyone in the scene.** If it says that Lanya watched him walk here without a word, he reads that too. A scene needs to be able to open on what each person there knows. The fact mechanism already does this for facts; a set-up does not use it.

None of the three mentions lost time, and each serves any play in which something happens off the stage that not everyone saw.

## What a sketch would add

Possibly nothing: with the first gap closed, Kidd's description says enough. An author who wants to steer it (that the first should come in the first act; that someone should be there to see) says so in the play file, in their own words.

## Order

The first is a line in the planning prompts, with a paired replay on Bellona's plans. The second and third are small changes to the director's reading and to how a scene opens; they want a design of their own before they are built.
