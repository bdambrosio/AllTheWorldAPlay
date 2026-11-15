"""What the stage refused: deeds an actor wrote that are none of the actions a world offers,
kept as one log across runs. The deeds refused most are what the plays ask of the world and do
not get: the list to grow the worlds and the shared actions from.

    .venv/bin/python tools/refusals.py [RUN ...]      (default: every run not yet in the log)

Adds each run's refusals to logs/refused_deeds.tsv (run, play, character, outcome, label, why,
deed), once, and prints a summary of what it added. Passing acts (small acts that change
nothing, played and then forgotten at the scene's end) go to logs/passing_deeds.tsv the same
way: the ones that keep mattering are the ones to make into actions. A refusal is read from the judgement
itself, in the run's prompt log (every survival.classify_deed call, with the deed and the
answer), so that a deed passed over for the actor's other act is kept with its reason; outcome
says what became of it:
    sent back     written again as one of the acts on offer, and played
    not done      sent back and still refused: not played
    passed over   the actor's alternative act was played instead
"""
import csv, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.expanduser('~/.local/share/alltheworldaplay/logs/traces/')
LOG = os.path.join(ROOT, 'logs', 'refused_deeds.tsv')
PASSING_LOG = os.path.join(ROOT, 'logs', 'passing_deeds.tsv')
FIELDS = ['run', 'play', 'character', 'outcome', 'label', 'why', 'deed']


def text_of(prompt):
    return '\n'.join(m.get('content', '') for m in prompt) if isinstance(prompt, list) else str(prompt)


def refusals(run):
    """The deeds refused in one run, and its passing acts: ([{...FIELDS}], [{...FIELDS}]), from
    its prompt log and its trace."""
    prompts, trace = TRACES + run + '.prompts.jsonl', TRACES + run + '.jsonl'
    if not (os.path.exists(prompts) and os.path.exists(trace)):
        return [], []
    events = [json.loads(l) for l in open(trace)]
    redone = {e.get('before'): e for e in events if e['kind'] == 'deed_check' and e.get('outcome') in ('redone', 'unfixed')}
    played = {' '.join(str(e.get('action') or '').split()) for e in events if e['kind'] == 'act' and e.get('mode') == 'Do'}
    out, passing = [], []
    for line in open(prompts):
        r = json.loads(line)
        if r.get('tag') != 'survival.classify_deed':
            continue
        text = text_of(r['prompt'])
        deed = re.search(r'<deed>\s*(.*?)\s*</deed>', text, re.S)
        claim = re.search(r'Its actor calls it: (.*)', text)
        who = re.search(r'^(.+?) is about to:\s*$', text, re.M)   # the call is the world's, not the character's
        try:
            answer = json.loads(re.search(r'\{.*\}', r.get('response') or '', re.S).group(0))
        except Exception:
            answer = {'action': None, 'why': 'the judgement could not be read'}
        parts = answer.get('actions', answer.get('action'))   # 'action': the first day's form of the answer
        parts = [parts] if isinstance(parts, str) else list(parts or [])
        parts = [str(x).strip() for x in parts if str(x).strip().lower() not in ('null', 'none', '')]
        if not deed:
            continue
        deed = ' '.join(deed.group(1).split())
        row = {'run': run, 'play': run.split('-')[0], 'character': who.group(1).strip() if who else '',
               'label': claim.group(1).strip() if claim else '', 'deed': deed}
        if parts and parts != ['several']:
            if any(x.lower() == 'passing' for x in parts):
                # every Do an actor offers is judged; only the one chosen is played
                passing.append({**row, 'outcome': 'played' if deed in played else 'judged, not chosen', 'why': ' + '.join(parts)})
            continue   # made of the actions on offer: done
        check = redone.get(deed)
        out.append({**row, 'outcome': 'passed over' if check is None else 'sent back' if check['outcome'] == 'redone' else 'not done',
                    'why': 'more than one act' if parts == ['several'] else ' '.join(str(answer.get('why') or '').split())})
    return out, passing


def main():
    logged = set()
    if os.path.exists(LOG):
        logged = {row['run'] for row in csv.DictReader(open(LOG), delimiter='\t')}
    runs = [a[:-6] if a.endswith('.jsonl') else a for a in sys.argv[1:]] or sorted(
        os.path.basename(p)[:-14] for p in glob.glob(TRACES + '*.prompts.jsonl'))
    new = [r for r in runs if os.path.basename(r) not in logged]
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    fresh = not os.path.exists(LOG)
    fresh_passing = not os.path.exists(PASSING_LOG)
    with open(LOG, 'a', newline='') as f, open(PASSING_LOG, 'a', newline='') as g:
        w, wp = csv.DictWriter(f, FIELDS, delimiter='\t'), csv.DictWriter(g, FIELDS, delimiter='\t')
        if fresh:
            w.writeheader()
        if fresh_passing:
            wp.writeheader()
        for run in new:
            rows, passing = refusals(os.path.basename(run))
            wp.writerows(passing)
            if passing:
                print(f"{run}: {len(passing)} passing acts")
            if not rows:
                continue
            w.writerows(rows)
            count = {o: sum(1 for x in rows if x['outcome'] == o) for o in ('sent back', 'not done', 'passed over')}
            print(f"{run}: {len(rows)} refused {count}")
            for x in rows:
                print(f"  [{x['character']}] {x['outcome']}: {x['deed'][:150]}\n      why: {x['why']}")
    print(f'logs: {LOG}, {PASSING_LOG}')


if __name__ == '__main__':
    main()
