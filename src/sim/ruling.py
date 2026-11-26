"""What a ruling leaves behind, one way for every point that refuses, strikes or drops something.

A play is made by calls that propose (an actor a line or a deed, the planner a scene, the director
a fact) and calls that rule on what was proposed. Each ruling point dealt with a refusal its own
way, and what a refusal left behind was nowhere the others read: a deed was refused because no
print lay by the cap, the record said only that the deed was not done (in the deed's own words,
print and all), and four scenes later the check of a set-up, finding nothing against a print,
made one; the play then stood on it for a whole part (lost-20261125-154614, and the audit of the
same day in the plan file). The contract every ruling point is held to:

  1. nothing is derived from a proposal, and no shared record sees it, before it is ruled;
  2. what was refused does not go into a shared record in its own words;
  3. the finding is state, kept where every other ruling reads (not_there, below);
  4. the source is told what was refused, why, and what is so instead, and decides once what to
     do about the step it was on (returned, below);
  5. a step given up is kept as unmet (below) and read by the level above at its next decision.
     Nothing is unwound further back, and nothing the audience has seen is taken back.

Kept within one performance and its thread, never from one night to another."""
from sim import ledger
from sim.trace import trace


def _plain(words):
    return ' '.join(str(words or '').split()).strip(' .')


# ------------------------------------------------------------------ what was found not to be there
# The world holds what is, and a thing is modelled when a deed first needs it: so that a thing is
# not listed means only that nothing has needed it yet, and any later call may bring it in. A ruling
# that refuses something because a thing is not there is the one moment when that is decided, and
# the decision is kept: {'what', 'at' (xy), 'since', 'seen_by' [names], 'by', 'ended'}.
def not_there(context, what, at, seen_by=(), by=''):
    """A ruling has found that `what` is not at this spot. Kept for every later ruling at that
    place (absent_note) and for those who saw it. Returns the entry, or None when it is already kept."""
    from sim import survival
    what = _plain(what)
    if not what or at is None:
        return None
    book = ledger.of(context)
    names = [getattr(c, 'name', c) for c in seen_by or ()]
    for a in book.absent:
        if not a['ended'] and a['what'].lower() == what.lower() and survival.at_hand(context, a['at'], at):
            new = [n for n in names if n not in a['seen_by']]
            if new:
                a['seen_by'] += new
                book.wrote('absent')
            return None
    entry = {'what': what, 'at': at, 'since': context.simulation_time, 'seen_by': names, 'by': by, 'ended': None}
    book.absent.append(entry)
    trace.event('not_there', what=what, place=survival._nearest_resource_id(context, at), seen_by=names, by=by)
    return entry


def there_again(context, what, at=None):
    """Something has brought or made what was found not to be there: the finding is over.
    what: the finding's own words. Returns the entry ended, or None."""
    from sim import survival
    low = _plain(what).lower()
    for a in ledger.of(context).absent:
        if low and not a['ended'] and a['what'].lower() == low and (at is None or survival.at_hand(context, a['at'], at)):
            a['ended'] = context.simulation_time
            ledger.of(context).wrote('absent')
            trace.event('there_again', what=a['what'])
            return a
    return None


def absent(context, at=None, reader=None):
    """The findings still in force: all of them; those at or near `at`; for a character, those it saw."""
    from sim import survival
    name = getattr(reader, 'name', reader)
    return [a for a in ledger.of(context).absent if not a['ended']
            and (at is None or survival.at_hand(context, a['at'], at)) and (name is None or name in a['seen_by'])]


def absent_note(context, at=None, reader=None, places=False):
    """What has been found not to be there, in words: '' when nothing has. places: each with its place
    (for whoever plans, who is told of the whole world)."""
    from sim import survival
    found = absent(context, at, reader)[-12:]
    return '; '.join(a['what'] + (f" (at {survival._nearest_resource_id(context, a['at'])})" if places else '') for a in found)


# ------------------------------------------------------------------ a step given up
# A failure used to go upward only as a trace event, which nothing reads: a beat given up after two
# refusals, the tasks a scene closed without playing, what a check struck from a plan. The planner
# of the next scene or act was told none of it, and planned the filling of a water skin at the same
# spring three scenes running. A step given up is kept here, and whoever plans next is shown it.
def unmet(context, level, who, step, why=''):
    """A step was given up. level: 'deed' (an actor's beat, its deed refused twice), 'plan' (something struck
    from a set-up), 'part' (a question nothing could be settled behind)."""
    step = _plain(step)
    if not step:
        return None
    kept = context.__dict__.setdefault('unmet_steps', [])
    if any(u['step'].lower() == step.lower() and u['who'] == who for u in kept[-12:]):
        return None
    scene = getattr(context, 'current_scene', None) or {}
    entry = {'level': level, 'who': who or '', 'step': step[:200], 'why': _plain(why)[:200],
             'scene': scene.get('scene_number') if isinstance(scene, dict) else None, 'at': context.simulation_time}
    kept.append(entry)
    trace.event('unmet', **{k: v for k, v in entry.items() if k != 'at'})
    return entry


def unmet_note(context, most=6):
    """For whoever plans next: what was tried or planned lately and the world would not have, and why.
    '' when nothing was. (The beats a scene ends without playing are not here: it ended because it had got
    where it was going, and they are not wanted.)"""
    kept = (getattr(context, 'unmet_steps', None) or [])[-most:]
    if not kept:
        return ''
    return ("The world would not have these, tried or planned lately. It will not have them now either, unless what stood in the way has changed: "
            "do not set a scene on one as though it could simply be done.\n"
            + '\n'.join(f"- {(u['who'] + ': ') if u['who'] else ''}{u['step']}" + (f" ({u['why']})" if u['why'] else '') for u in kept))


def released(context, promise, why=''):
    """A promise has been taken off the stage (kept, broken or overtaken): whoever made it or was owed it is told,
    once. It was taken from the ledger alone, and each of them went on holding it."""
    import re
    names = {n.strip().lower() for n in re.split(r',|\band\b', str(promise.get('who') or ''))} | {str(n).lower() for n in promise.get('to') or []}
    told = []
    for c in context.everyone():
        if c.name.lower() in names:
            c.add_perceptual_input(f"No longer in force, of what was promised or agreed: {promise.get('who')}: {promise.get('what')}"
                                   + (f" ({_plain(why)})." if _plain(why) else '.'), percept=False, mode='internal')
            told.append(c.name)
    return told


# ------------------------------------------------------------------ back to the source
def returned(what, why='', instead=''):
    """What a source is told of a refusal, the same three things whoever it is: what was refused,
    why, and what is so instead. Sentences, for the source's own prompt or memory."""
    what, why, instead = _plain(what), _plain(why), _plain(instead)
    return ' '.join(filter(None, [f'Not so, and left out: {what}.' if what else '', f'Because: {why}.' if why else '',
                                  f'As it is: {instead}.' if instead else '']))
