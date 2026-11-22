# Dangers: proposal

Proposed 2026-11-22, not yet agreed. It applies `docs/world-facts-design.md` to what threatens
people: the last part of the world that is still a table.

## What is there now

Each world file lists hazards. A hazard is a name, a description, when and on what ground it
comes, a chance per hour, minutes until it strikes, how much it harms, three or four named
answers each with a chance of working and things that add to that chance, and the sentences it
is told in. About 300 lines of `sim/survival.py` roll for it, start it, match a deed to one of
its named answers, apply the chance, and say the canned sentence.

Faults seen: the strange light in Bellona came on, harmed nobody, and was told as "The light
fades as suddenly as it came" thirty-one minutes later, the same sentence every time. A deed
against a danger counts only if the resolver can call it one of the listed answers. The lunar
base's rupture reads a flag that only a table rule sets. And the play's danger is whichever the
dice or a random pick chose, not one that follows from where the story stands.

## Proposal

1. **A world states its dangers as facts**, in sentences, among its other facts: what threatens
   people there, when and where, what keeps it off, what may drive it off, what it does to
   someone it reaches. A sketch may state one too (the time a character loses).
2. **A danger comes on because the play brings it**: a scene's set-up has it upon the cast
   (allowed by the set-up check only where the facts have such a danger, at such an hour and
   place), or the director brings one on between scenes. The hourly dice go.
3. **A danger upon someone is a state, kept in words**: what it is, where, whom it is upon,
   since when, and by when it comes to a head. It is shown on the stage like any other state.
4. **What a deed does about it is ruled like any deed.** The resolver says how likely the deed
   is to end it, from the facts and how things stand (a branch in hand, a fire burning, a door
   to shut), and what follows either way: the danger over, or still there, and who is hurt and
   how far. The engine draws. There are no named answers.
5. **Left to itself, it comes to a head.** When its time runs out, or the scene would end with
   it still there, the world is asked once what it now does: how likely it is to reach someone,
   whom, how hard; or that it passes. The engine draws. One extra ruling for each danger.
6. **Getting clear is the engine's**: out of its reach and out of its sight, by the measures
   the place model already has, ends it. Whether a shelter or a fire keeps this danger off is a
   fact of the world, read by the resolver from the stage.
7. **It is told in the ruling's own page line.** No stock sentences.

## What stays as engine policy

- Dangers are rare: at most one in a play (Bruce's ruling), and the director's chance of
  bringing one on for a scene.
- A scene with a danger upon the cast is urgent: short lines, the danger first.
- When one ends in a scene, the scene's set-up carries a note that it has.

## What goes

The hazard tables in every world file; the code that rolls, matches and applies them; the
`world` kind of change in the resolver (its last table hook); the keeper's `keeps off` lookup.
With the lunar base converted (a repaired line is a standing fact), the table code can then be
deleted whole.

## Consequences and risks

- One extra model call for each danger, at most one a play.
- A model asked what a danger does may be soft: nobody ever hurt. The draw is the engine's and
  both outcomes must be given, and the acceptance cases must show a wolf more likely driven off
  with a branch than with bare hands, and someone hurt when the draw goes against them.
- Without dice, a play in the forest at night has no wolf unless the plan or the director brings
  one. That is already nearly so: the limit is one danger a play.

## Acceptance cases to write first

A wolf at night: bare hands, a branch, beside a fire, running for the hut. A boar left to
itself. Hail with a barn a minute off. A gang, and two people who hide. The strange light.
The lunar rupture before and after the line is sealed. The neighbour's dog and a kitten up a tree.
