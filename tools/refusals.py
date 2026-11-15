"""What the stage refused: deeds an actor wrote that are none of the actions a world offers
(sim/record.py mend_deed, trace event deed_refused), tallied across runs. The deeds refused
most are what the plays ask of the world and do not get: the list to grow the worlds from.

    .venv/bin/python tools/refusals.py [RUN ...]      (default: every run of today)

Prints, for each run: Dos played by sort (world action, thing, shared action), deeds sent back,
and deeds not done after being sent back; then every refused deed with the reason given.
"""
import collections, datetime, glob, json, os, sys

TRACES = os.path.expanduser('~/.local/share/alltheworldaplay/logs/traces/')


def runs(args):
    if args:
        return [a if os.path.exists(a) else TRACES + a + ('' if a.endswith('.jsonl') else '.jsonl') for a in args]
    today = datetime.date.today().strftime('%Y%m%d')
    return sorted(p for p in glob.glob(TRACES + f'*-{today}-*.jsonl') if not p.endswith('.prompts.jsonl'))


def main():
    for path in runs(sys.argv[1:]):
        rows = [json.loads(l) for l in open(path)]
        played = collections.Counter()
        for r in rows:
            if r['kind'] == 'affordance' and r.get('applied'):
                played['shared action' if r.get('category') == 'stage' else 'thing' if r.get('action') in
                       ('hand over', 'share', 'hold out', 'read', 'put down', 'pick up', 'take', 'ask for') else 'world action'] += 1
        refused = [r for r in rows if r['kind'] == 'deed_refused']
        first, final = [r for r in refused if not r.get('again')], [r for r in refused if r.get('again')]
        free = sum(1 for r in rows if r['kind'] == 'free_deed')
        print(f"\n{os.path.basename(path)}: played {dict(played)}; sent back {len(first)}; not done {len(final)}; unjudged {free}")
        for r in first:
            print(f"  [{r.get('character')}] {r.get('deed')}\n      why: {'; '.join(r.get('why') or [])}")


if __name__ == '__main__':
    main()
