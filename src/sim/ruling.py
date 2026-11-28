"""What deeds played have found, and promises taken off the stage.

Nothing in a performance refuses (2026-11-27): a deed is resolved, played as it comes off or as it is tried
and does not, and what the stage cannot take is logged as an error of the run (sim/errors.py), not made a
change to the play. What a deed played finds is state: that a thing it looked for here is not here is kept
(not_there), for every later ruling at the place and for those who saw it, until a deed makes it
(there_again). Only a deed that looks settles that: a check of a text writes nothing to the world.
(A struck 'cans on a low shelf' was kept as cans not there and settled a search nobody had made,
bellona-20261127-142348.)

Kept within one performance and its thread, never from one night to another."""
from sim import ledger
from sim.trace import trace


def _plain(words):
    return ' '.join(str(words or '').split()).strip(' .')


# ------------------------------------------------------------------ what was found not to be there
# The world holds what is, and a thing is modelled when a deed first needs it: so that a thing is
# not listed means only that nothing has needed it yet, and any later call may bring it in. A deed
# that looks or reaches for a thing here and does not find it is the one moment when that is decided,
# and the outcome is kept: {'what', 'at' (xy), 'since', 'seen_by' [names], 'by', 'ended'}.
def not_there(context, what, at, seen_by=(), by=''):
    """A deed played has found that `what` is not at this spot. Kept for every later ruling at that
    place (absent_note) and for those who saw it. Returns the entry, or None when it is already kept."""
    from sim import survival
    what = _plain(what)
    if not what or at is None:
        return None
    # a thing hidden at this spot is there, only unseen: a set-up check struck 'lies a torn notebook page' as hidden
    # until found and gave the page as absent, though told to leave that empty (replay of bellona-20261126-115753)
    low = what.lower()
    if any(p['name'].lower() in low or low in p['name'].lower() for p in survival.lying(context, at, hidden=True) if p.get('hidden')):
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
