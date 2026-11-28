"""What a model answered that the stage could not take, kept as an error of the run, not as a change to the play.

Nothing in a performance refuses: a deed is resolved, a set-up is applied, a line is spoken. Where an answer
would have the stage hold what it does not (a thing in two hands, a person who is not here, a name nothing
lists), the stage stays as it is, that part of the answer is skipped, and it is written here: one JSON line
per error in <run>.errors.jsonl beside the run's trace, and as an 'error' event in the trace itself. Nothing
in the play reads it back.

    from sim import errors
    errors.log('resolve', 'no such thing', name='coat', doer='Kidd')
"""
import json
import time

from sim.trace import trace


def log(where, what, **detail):
    """where: the call or step whose answer it was; what: what the stage could not take, in a few words."""
    detail = {('field_' + k if k in ('kind', 'where', 'what') else k): v for k, v in detail.items()}   # never the trace's own fields
    try:
        trace.event('error', where=where, what=what, **detail)
    except Exception:
        pass   # logging must never affect the play
    if trace.path is None:
        return
    try:
        sim_time, act, scene = trace._position()
        record = {'t': round(time.time(), 3), 'run': trace.run_id, 'sim_time': sim_time, 'act': act, 'scene': scene,
                  'where': where, 'what': what, **detail}
        with open(trace.path.with_suffix('.errors.jsonl'), 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    except Exception:
        pass   # logging must never affect the play
