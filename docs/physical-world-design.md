# Physical things and deeds: review and design proposal

2026-11-19. Research only: no engine code was changed for this. Tests are replays of recorded
prompts; scripts and outputs are in `logs/review-20261119/physical/`. Nothing here has been
played. Code references are to commit `a546d6e`.

## 1. What goes wrong today

### 1.1 Numbers

From the 44 runs of lost and bellona since 2026-11-15:

| | |
|---|---|
| Deeds judged before play (`survival.classify_deed`) | 721 |
| Refused by the judge | 140 (19%) |
| Do acts played | 467 |
| Lines that claimed a deed or a holding the world did not have (`record.speech`) | about 100 of 2,122 |
| Scenes whose planned ending needed something the stage cannot do (`ending_beyond`) | 28 of 138 |
| Exits held because the scene was "not finished" | 103 |

In the one full run measured (`lost-20261119-084100`, 1,820 model calls, 6,831 s) the checks on
the physical path took 24% of the calls and 23% of the time. The largest are the line check
(222 calls, 825 s), the deed judge (92 calls for 45 deeds, 317 s) and the scene fitter with its
checks (46 calls, 201 s). The outcome tracker, which reads deeds as well as talk, took another
286 calls and 1,535 s.

### 1.2 What the 140 refused deeds were (my reading of each)

| Class | About | Examples |
|---|---|---|
| Putting a thing in or taking it out of another; open and shut | 35 | page into notebook, notebook shut, can opened, leaf pocketed, bag zipped |
| Moving another person | 19 | pulls her up, draws him along, tugs her hand |
| Taking what a place plainly gives | 16 | picks an apple under the apple tree, a twig, a leaf, a stone from the cairn |
| Moving a thing a little where it is | 12 | slides the harmonica across, sets the core down, nudges the page |
| A thing that exists in the story's text and not in the world | 14 | the canvas sling, the pile of cans, the chair, the cairn's stones |
| Fire | 9 | snapping kindling, banking, putting out |
| Breaking, folding, unfolding | 9 | apple in two, branch snapped, map unfolded |
| Moving about within the place | 7 | circles him, a few steps down the trail, takes up a post by the door |
| Looking closely (judge error: these are allowed) | 8 | runs two fingers along the flattened stems |
| Writing | 3 | writes a line in the notebook |
| Other | 8 | too many acts, wedges a chair against a door |

Almost none is an attempt at something a person could not do. They are ordinary handling that
the world has no way to represent.

### 1.3 Structural faults (from the code inventory)

1. **State is in about twelve stores with different lifetimes**: resource `properties`,
   `context.props`, `Body`, `thing_states`, `writings`, `readers`, `held_out`, `shared`, `poses`,
   `between`, `survival_fire`, `active_threat`, plus free text (`_state_notes`, set-ups,
   wrap-ups). No one function shows all of it.
2. **Checkers and authors see different worlds.** The fire is shown to nobody. The record the
   narration and line checks read lists bare names with no state, spot or fixedness. Things held
   by a supporting actor count as "not in the world". The actor is told to label from a list cut
   to 14, while the judge accepts the full list and labels the actor is never shown.
3. **Things are names, not objects.** Every "branch" shares one state, and the state outlives
   the branch. Two apples cannot be told apart.
4. **Authors write physical facts without the constraint that later refuses them.** The act
   planner sees what places offer with no requirements, stock or flags. The beat arranger, the
   outcome writer, the commitment extractor and the bystander cut-in see nothing physical. The
   scene fitter may bring things in but cannot revise the planned business. A director's clue
   named a sling the world never held. A scene plan demanded "quench signal fire", which does
   not exist, and the actor tried it five times.
5. **Possibility is decided late, by a model, once per attempt.** Asked the same question twice
   with the same prompt, the judge gave the same parts for 64 of 74 deeds.
6. **Paths round the gate.** A Move that names nowhere becomes an unjudged Do; a Say with a
   stage direction records an unjudged deed; the narration check can pick up a fixed thing;
   closing business applies planned actions unseen.
7. **No life cycle.** Nothing regrows. Fire is one global flag for three hours. A thing comes
   into being only by a world action's `gain` or by a model's text being admitted, and leaves
   only by `consumes`. Needs met off stage use up nothing.

## 2. What was tested

All on recorded prompts, acting model (GLM-5.3-Flash) unless said.

| Test | Result |
|---|---|
| **A. The committed judge** (`handle`, a546d6e) on the 140 refused deeds | places 57; of 140 placed before, still places 131 |
| **C. Typed things** (below) on the same deeds | places 98 to 103 depending on how strictly my test validator matches names; of 140 placed before, 124 to 135 |
| Prompt size, A vs C | 7,064 vs 4,457 characters (37% smaller) |
| **Typing a thing** with a fixed trait list, asked three times | 28 of 46 things identical; portable or fixed the same for 43 |
| **A menu of what each thing allows, in the actor's prompt** (60 recorded act prompts, old vs with menu) | no gain: design C places 52 of 60 Dos from the old prompt and 49 of 58 with the menu |
| The actor's own `#affordance` label is already a valid list of parts | 40 of 60 |
| Telling the actor what was refused, and not demanding invented business (earlier today) | fire retried in 9 of 12 replies before, 0 of 12 after |
| Director's clues must be objects or marks (earlier today) | view-clues 4 of 4 before, 0 of 4 after |

What these say:

- **The gain is in what the world can represent, not in wording.** Giving things traits and a
  small grammar of handling places most of what is refused, with a shorter prompt.
- **The actor is not the problem.** Without any new instruction, 87% of its deeds are placeable
  once the grammar exists, and a menu did not raise that. What is left after C is nearly all
  things that are in the story and not in the world, and fire.
- **So constraints belong where things and tasks enter the story** (the director, the act plan,
  the scene set-up), not in the actor's prompt. The business and refusal results show the same:
  the retries came from a plan demanding the impossible.
- **Typing must be done once and stored.** A model's traits vary a little between askings
  (one asking dropped "portable" from every thing in bellona), so a thing is typed when it
  enters and never again.

Limits of the tests: my validator for C is a test script, not an engine; it mis-matched names in
both directions, which is why the counts are ranges. Things held by the other person were not
visible to the menu test. Only two worlds are in the data. Nothing was played.

## 3. Proposal

### 3.1 Two layers

**The map is static.** Terrain, places, what kind each place is, who owns it. Nothing in a
performance changes it. Each kind of place says what it gives for the taking (an apple tree:
apples, twigs, leaves) and how fast that comes back.

**A ledger holds everything that changes**, in one module with one store:

- bodies: needs, harm, position on the map;
- things, as separate objects: kind, traits, where (in a hand, in another thing, at a spot in a
  place, attached to the place), state, amount, any text;
- how people stand to each other: posture, position, contact, a thing held out, an offer made;
- dangers on stage.

It replaces `props`, inventories as name lists, `thing_states`, `writings`, `readers`,
`held_out`, `shared`, `poses`, `between`, `survival_fire`. Two functions matter:
`view(reader)` gives the one account of the stage, cut for an actor, a judge, a narrator or a
planner, so that every author and checker reads the same facts; `apply(parts)` is the only way
state changes, and it either does all of a deed or none.

### 3.2 Things

A thing has **traits** from a short fixed list, set once when it enters:

| Trait | Allows |
|---|---|
| portable | take, put down, place (move a little, it stays here), stow, take out, give, hold out, show |
| fixed (a mark, a part of the place) | examine only |
| attached (a snagged thread, a ribbon on a branch) | examine; take it off once, then portable |
| container | put a thing in, take a thing out |
| openable | open, shut |
| foldable | fold, unfold |
| breakable | break into its parts |
| edible, drinkable | eat, drink: the thing is used up, the body is fed by the kind's amount |
| writable, with something that writes in hand | write on: the words are kept |
| readable | read |
| fuel | may be fed to a fire |

**State** is a few named facets with fixed values (open or shut, folded, whole or broken, how
much is left), not free text. A facet that a world action depends on is read by that action.

**Coming into existence:**
1. given by the world or play file at the start;
2. taken from a place that gives it (stock, with regrowth);
3. made from parts by a recipe a kind declares (kindling from a branch; a fire from fuel and
   whatever the world requires to light it);
4. brought in by an author at the two points where that is allowed: the director before the
   play, and the stage manager at a scene's set-up. It is typed in that same call.

**Leaving existence:** eaten, drunk, burned, used up, or broken into its parts. Marks and parts
of a place do not leave.

**Fire** is a thing at a spot with a state (burning, banked, out) and fuel left, made and tended
by the world's actions. What it does (keeps a wolf off) reads that state.

### 3.3 What is free and what is the world's

The consequence rule, as agreed: handling that only changes how a thing is, or where it lies
within reach, is free and comes from the traits. Anything that changes what bodies need, what a
danger does, where people can go, or what is known stays a world action with its own
requirements. Each world's file therefore says: its kinds and their traits, what its places
give, its recipes, and its consequence-bearing actions. That is where "start a fire" is allowed
or not, once, before any performance.

**Moving another person** becomes an offer: `lead@who` records that one has taken the other's
hand or arm and started off. The other goes if their next act agrees. No judgement of force.

**Moving about within a place** is a passing act.

### 3.4 The cycle for one deed

1. The actor writes the deed and its parts (`#affordance`, as now).
2. **Code checks the parts** against the ledger: each is a world action on offer, a stage
   action, a trait verb on a thing that is here, an offer, or passing. In the test this already
   holds for two deeds in three.
3. Only when the label is missing or wrong, a model **parses** the deed into parts (a smaller
   prompt than today's judge), and code checks those. The model never decides what is possible.
4. `apply` changes the ledger. A refusal is worded by code from the failing part ("the notebook
   is in Kidd's hands"), remembered for the scene, and a beat gives up after two.
5. The narrator is called only for a world action with a result to tell, as now.

The side doors are closed by sending every physical act through step 2: a Move that names
nowhere, a stage direction inside a Say, closing business, and anything the narration check
wants to make real.

### 3.5 Constraints, earliest first

| When | Who | What is fixed there |
|---|---|---|
| World file, checked at load | code | kinds, traits, what places give, recipes, actions; an action whose requirement nothing can supply is an error |
| Before the play | director | every object a fact or clue mentions is an entry in the ledger with traits, or the clue is rewritten; one check in the call that writes them |
| Act plan | planner, then code | business and endings are written as parts and checked by code for existence and for requirements that can be met; one rewrite; today this is found at scene start or never |
| Scene set-up | stage manager | the only point where set dressing enters; each new thing is typed in the call that admits it; business is restated for the scene as it now is |
| Scene sheet | outcome writer, then code | an outcome that is a deed carries its parts; one that cannot be done is rewritten before the scene, not held open during it |
| In the scene | actor | sees the same view the checker uses; told what was refused |

After the set-up, no new kind of thing enters a scene except from what the place gives or a
recipe makes. That is the main new constraint, and it is what removes the largest class of
refusals left after typing.

### 3.6 What this constrains

- Only things in the ledger can be acted on. A noun in a line of dialog is not a thing.
- Handling is the trait verbs; anything else is a world action or is not done.
- One level of containing (a page in a notebook, a notebook in a bag; not deeper).
- Nobody is moved by force. A place is not altered (dug, barricaded) unless the world has an
  action for it.
- Traits and kinds are fixed for the performance.
- Amounts are whole portions.

### 3.7 Cost

Not measured live; estimated from the recorded run.

- Deed judging: today one model call per Do alternative (92 calls for 45 deeds). With labels
  checked by code first, about one deed in three needs a parse call, with a prompt 37% shorter.
- Typing: no added call. Things are typed inside calls that already exist (the director's, the
  set-up's admit step).
- Plan checks by code replace part of `ending_beyond` and the set-up rewrite loop.
- The narration and line checks get one consistent record; their count does not change.
- Fewer refusals also means fewer rewrite calls and fewer held exits.

No new call is added on the path of an ordinary deed.

## 4. Risks and what is not shown

- **Coverage is from two worlds.** City, rural and lunar things (contracts, deeds to land,
  sealant) were not in the data. The trait list will need two or three additions.
- **Typing errors are permanent for a night.** A thing wrongly typed fixed cannot be picked up.
  The director's and stage manager's entries should be shown in the Plan tab.
- **Parts labels from the actor** were valid two times in three under today's wording; the
  parse call covers the rest, so this is a cost question, not a correctness one.
- **The no-new-things rule** will make some set-ups plainer. It should be watched in the first
  runs for scenes that feel under-furnished.
- **Said for done** ("fire's banked" spoken, not done) is not fixed by this; it is the line
  check's and the outcome tracker's job, and they will read the ledger.
- **Position within a place** (under this tree, not that one three tiles off) caused about nine
  refusals. The proposal treats a place as one spot; a scene set at a place should gather the
  cast onto it, which `set_scene` already does for the named place.

## 5. Order of work, each step testable by replay before the next

1. The ledger as one store behind the present functions, no behaviour change. Test: the
   recorded judgements give the same answers.
2. Things as objects with traits, containers, open and shut, spots; code validation of parts;
   the judge reduced to parsing. Test: the 140 refused and 140 placed deeds.
3. What places give, regrowth, recipes; fire as a thing. Test: the fire and picking deeds.
4. Plan-time checks of business, endings and outcomes; typing at the director and at set-up.
   Test: the 28 recorded endings beyond the stage and the recorded set-ups.
5. Offers for moving a person. Test: the 19 recorded deeds.
6. The side doors closed.

## 6. Decisions (Bruce, 2026-11-19)

Agreed: the two-layer split with the ledger as the single store; a trait list; no new things
after a scene's set-up except from a place or by being made; moving a person as an offer.

Rejected: declaring each world's kinds, recipes and actions ahead of time (3.2 "enters by",
3.3 "each world's file says", 3.5 first row, and step 3 of section 5 as written). Bruce: that is
still pre-scripting; decide how a thing works when a deed first needs it, and let that change
as the performance goes on. Section 7 replaces those parts.

## 7. Revision: the world is kept, not declared

**A world-keeper models a thing when a deed first needs it.** Nothing about fire, baskets,
contracts or ploughs is written in a world's file. When code cannot place a deed with the
models it has, one call gives the keeper the setting, the stage as the ledger has it, the
models so far and the deed. It answers one of three ways:

1. the deed as parts, with a new model of a kind of thing (traits, named states, what it is
   made from, what must be at hand, how it changes by a deed and in time, how long, what chance,
   and what it does to anything the world tracks);
2. the same with an extension to a model that exists (a state, a change or a need added);
3. it cannot happen in this world, with the world's reason, or what is lacking.

The model is stored in the ledger. From then on code applies it with no model call, until a deed
goes beyond it and it is extended.

**What stays fixed is a few laws, not content**, given to the keeper every time: it is the world
the setting describes; nothing comes from nothing; everything takes time and the uncertain may
fail; what is settled stays settled (models are added to, never taken back); a model says what a
thing is like, never what anyone does; an effect on people is said only in what the world tracks.
Code holds the structural ones: inputs must be on the stage, time is spent, a stored state is not
rewritten.

**A lack is a goal, not a dead end.** "No plough or beast is on the stage" tells the actor and
the planner what would make the deed possible.

**Early constraints, by the same call.** The act plan's business, a scene's planned ending and a
sheet's deed-outcomes are put to the keeper when they are written. What they need is then
modelled before the scene, and what they lack is known to the planner while it can still plan
for it.

**Test** (`t4_keeper.py`, acting model, twelve deeds in four worlds, each asked twice, nothing
declared ahead): 24 of 24 replies well formed.

- Fire, first need: modelled both times as lit by friction, 35 to 40 minutes, chance 0.3, made
  from the snapped branch, keeping the wolf off while burning.
- Fire fed, when only burning down had been modelled: one reply extended the fire with "laying a
  branch on"; the other modelled the branch and left the fire's time unchanged.
- Fire banked: both extended it with a smouldering state and kept the earlier states.
- Writing in a notebook, a chair wedged under a door (tied to the gang at night), a contract
  drafted and signed, a basket woven: all modelled.
- Flying home, digging through rock with bare hands, a rope nobody had mentioned: refused, each
  for a reason of the world's.
- Ploughing: refused for lack of a plough and a draft animal.
- The canvas sling: refused both times as not there. The keeper was not shown the story's text,
  which says the grooves lie beneath it. It needs the record, and a rule that what the record
  has said is there can be given a model.

Seen in the test and still to solve: two askings give different state names and numbers, so a
model is made once and stored; the keeper invents chances (a fire that fails seven times in ten)
that shape the drama and that nobody chose; an effect on a danger comes back as words and needs
a form code can apply; one extension landed on the wrong thing.

**Cost.** The keeper is called only when code cannot place a deed. In the recorded run that
judge was called 92 times; that run used about ten kinds of thing.

**Decided (Bruce, 2026-11-19, corrected the same day):** nothing is remembered across performances. A model
lasts for the performance it was made in and no longer; each night the world is found out afresh.
(He first agreed to keeping models between nights, and that was built and then removed.)

## 8. What is built (uncommitted, not played)

- `src/sim/keeper.py`: the keeper as in section 7. A deed the judge cannot place goes to it
  (`record.judge`); its models are kept in `context.models` for the performance; a deed it places is played by
  code (`perform`); what a modelled thing allows in its present state is offered to the judge and
  the actor (`offers`), and done with no keeper call; things move on in time (`age`); a thing
  whose model keeps a danger off blocks that danger (`keeps_off`). Ablation switch `keeper`.
- The forest's file no longer has `build a fire`.
- Moving a person is the stage action `lead`, an offer.

Checked on the built lost world (`keeper_check.py`, five keeper calls): a fire made from a branch
(20 minutes, chance 0.3), a thing at the place that keeps the wolf off; fed by a second branch,
which is used up; banked to smouldering; gone nine hours on; flying and an unmentioned rope
refused for the world's reasons. `lead` on the recorded
deeds that moved a person: 16 placed as lead, none wrongly (`lead_ab.py`).

## 9. Built after section 8 (2026-11-19, later)

- **The ledger** (`src/sim/ledger.py`): each thing an object of its own, held or lying at a spot,
  with its state, since when, and its model's kind. `body.inventory` and `context.props` are
  views of it, as are `held_out`, `shared`, `poses`, `between`, `writings` and `readers`. A
  thing keeps what is known of it as it changes hands. The fire flag is gone. The record the
  narration and line checks read shows states and fixedness.
- **No new thing after a scene's set-up from a narration.** `record.admit` makes a thing real
  only for the director; what a deed needs goes to the keeper before it is played.
- **Planned endings go to the keeper.** A part of an ending that the stage has no action for is
  put to the keeper while the scene is being fitted; only what it says cannot happen is written
  out.
- **The keeper may name an action the stage already has**, and a place counts as a source for
  what is made from it (found in the first live scene: a branch "made from the fallen branch"
  at Fallen_Branch5 failed for lack of a thing called fallen branch).
- **A kept deed that does not come off** is the doer's to know, as any refusal is, not an event.
- **Effects on needs**: a thing near a body eases or worsens a need a little each hour where its
  model says so.
- **Side doors**: a Move that names nowhere is a passing act; a stage direction inside a Say is
  judged like any deed; a fixed thing cannot be picked up by a narration.
- **Plan tab**: "What this world has come to hold" lists the kept models.
- **Supporting actors**: twenty places that left people cast during the play out of what the
  cast is told, shown or checked against now include them.

Dropped: code accepting the actor's own label without a model call. A valid label does not show
that the deed's words do no more than the label says, and checking that is the judge's job;
skipping it would let a line show more than the world did.

Not done: names are still capitalised by `Character`; things with the same name held by one
person are still looked up first-match in a few places; outcomes on a scene's sheet are not put
to the keeper (only endings are); the keeper has no say yet over what a place gives beyond the
world's own actions.

Crash checks, two scenes of lost each, read for errors only: the code of `eff887e` (662 s, no
error) and the code of this section (602 s, no error; no deed was refused and the keeper was
not needed).

## 10. One way in, one way out (2026-11-20)

What stands after the overhaul of 2026-11-20. Written the evening the code went in; none of
sections D to F below had been played when this was written.

**Why.** The physical faults of November all sat where prose became state: an act existed twice,
as the text shown and as the labels applied, written by different calls. Fixing that seam with
one resolver call left a second fault visible: the store kept what was true now and nothing of
how it came to be, and some twenty code paths wrote to it while sixteen described it. A woman's
own fingermarks in the mud were listed as a thing lying at a spring, and the cast spent an act
tracking a stranger.

**One way in.** Whatever changes the stage is one event (`Context.committing`): a deed, a line,
a walk, a leaving, a scene being set, time passing, the director's placing, a hazard. The event
keeps what it changed, the line shown for it and its entry in the record together. A thing knows
the event that made it and the one that last changed it (`ledger.Thing.made`, `.last`,
`.state_by`). A change or a shown deed outside any event is traced (`uncommitted_write`,
`unrecorded_line`).

**One way out.** `record.account(context, reader)` describes the stage for a character, for the
stage's own calls, or for the planner, from the same things, traces and told facts by the same
wording. What a deed made or left is told as whose doing it is, as the reader may know it
(`survival.whose`): by you, by the one named, or not seen by whom. A trace of a deed is never a
thing lying there (`ledger.marks`).

**One act, resolved once** (`sim/resolver.py`). The actor writes one act: words, a deed, or both,
and a walk after them. One call, shown the stage, what the doer has lived, what it has told and
what it does not know, says what the deed does (effects in the ledger's terms and the line for
the page), what the words hand over, what they state that is not so and how the line can truly
be said, what they tell of the speaker's life, and what they promise. What a person tells of
itself is kept (`ledger.told`) and shown to both. A deed is resolved when chosen, not before.

**Scenes and time.** A set-up is resolved against the world in one call (`resolve_setup`) and
what it brings is in the world by one event. A scene is where its cast stands and walks on with
it. Nothing is met off the page between scenes.

**Needs and the catalogue.** In a world without money (`survival.free`) a body's needs are facts
about the person, eased or worsened by what an act is found to do, and what a kind of place gives
is taken from it without a list of actions. Worlds with money (rural, coastal, suburban, city)
keep their catalogue: paying and owning have not been worked out without it.

**Dangers** (`survival`, the hazards section). A danger comes on in one of three ways: a scene's
set-up has it upon the cast (`resolve_setup` names it, `stage_hazard` starts it), the director
brings it on (`director_hazard`), or the hours bring it (`hazard_tick`); at most one in a play
(`may_come_on`). It is answered only by one of its own answers, as the resolver found the deed to
be (`answer_danger`); no answer is read from a deed's words, and no danger from a text's verbs.
It is kept in the ledger (`Context.active_threat`), so each change to it is part of an event.
What it does is told once (`_told`): to those present, and on the page after the act it belongs
to. A line that says the danger on the stage is gone, or one is upon them that the stage does not
show, is a finding (`danger`). The world's listed answers and their chances are unchanged.

**Things on a person, and easing.** A thing can be worn or bound on its holder (`Thing['worn']`,
`survival.wear`; the resolver's `on` effect, the keeper's `ends: on <name>`); moved again, it is
no longer worn. An eased need goes one step down in its words (`survival.eased`); a state that is
or is not (`body.flags`) is cleared.

**Still standing from before, on purpose.** `survival._apply`, `_chosen`, `classify_deed`,
`resolve_do` and the narrator path of `Context.do` remain for the money worlds and for worlds
with no mechanics. `record.situation` is not yet routed through `record.account`; in a world
without its catalogue `keeper._stage` is.

**Evidence** (in `logs/review-20261120/`): `resolver/` (225 recorded deeds), `render/` (whose
doing, replayed), `act/` (209 recorded lines; the one-act form), `engine/smoke.py` (one small
check per stage).

## Space (sim/world.py)

One module answers every question of space: what places there are (`places`, `place`, `called`, `nearest`), where a person stands (`at`, `put`, `enter`), how far and which way (`tiles`, `bearing`), what ground a position is on (`region`, `pace`), a position in words (`where`), a name into a place (`resolve`, `find`), and what is in view (`surroundings`, `view_report`). A place is a `world.Place`: id, kind, description, position, and props (its owner, by name or as a part nobody is cast in; what is left of what it gives; whatever a rule sets). Nothing outside this module reads a person's tile, the table of places or the ground; a position is handed back to it and not looked into.

A world file that says how its places sit (`kinds`, `layout`: plays/scenarios/forest.py, apocalypse.py, suburban.py) is laid out in metres, anew for each performance (sim/land.py). Places hold other places: a region (the forest, a clearing, a rocky rise; the street, a plaza) holds the springs and shops that stand in it, and a house holds its rooms' things. A position is (x, y) in metres. The rules of nearness are stated once, at the head of sim/land.py: what lies at hand, who is together, earshot, a step against a walk, what is in sight (by the size of the thing, the cover overhead, high ground, walls, and places that block the view or hide what is in them), shelter by being inside a place that is one, and how near a danger comes. Walking takes the metres at the pace of the ground. A vehicle (a kind the world file calls one) is gone by for a way too long to walk, when one is within reach, in its own time, and ends where its riders do.

Places come into the world while it is played, by the two calls that decide what is so: the set-up check may set a scene at a new place inside or beside a listed one (`new_place`), and the resolver may have a deed or a walk reach or make one (the `place` effect). Each is made by `world.make_place`, inside an event, and is listed from then on. A place that belongs to no listed place is not made. (Paired replays and live cases: logs/review-20261121/places.)

Every world file says how its places sit; the old grid (sim/map.py) is gone, and sim/world.py is the front of sim/land.py and nothing else. Checks: logs/review-20261121/world/land_check.py (layout rules over twenty seeds; the rules of nearness and sight), drive_check.py, golden.py (a fixed script over each play, no model).

## What a world file rules on, and what is the resolver's

A world file declares only what the engine must hold someone to; whatever nothing would ever check is not declared, and is simply played: said, done, shown and kept in the record. Below its places a world file holds three kinds of thing (`affordances`):

- **Needs**, each with how it rises, what neglect does, and the ways it is eased (`'eased'`: by what, how far, and what that takes: a kind of place, a thing used up, minutes, a stock that runs out, a risk). The resolver says which way a deed was (the need effect's `by`); the amount, the time and the cost are the world's (`survival.ease`). Eased any other way, a need falls no further than by the least of its listed ways.
- **States something depends on** (`'states'`: dressed, because work needs it), each with what it means and what brings it about. The stage shows them and who is in which; the resolver says when a deed makes one so or ends it (`becomes`, `no_longer`); what requires one refuses without it.
- **Rules** (`'actions'`): exchanges and odds, which must be the same for everyone and are not a model's to invent: buying, selling and paying, work for a wage, what changes hands by deed, a search that may find something, a repair that may not take. The stage lists them as they stand for a person; the resolver names the rule a deed falls under (the `world` effect), and the engine checks what it needs and applies it (`survival._apply`).

Everything else is the resolver's and the keeper's: moving, taking, making, handing over, wearing, posture, touch, marks, new places, what a made thing does in time. There is one path for every world, with or without money (live cases: logs/review-20261121/ways, logs/review-20261121/rules).
