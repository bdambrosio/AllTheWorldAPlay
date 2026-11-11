#!/usr/bin/env python3
"""Read a run's prompt log (<run>.prompts.jsonl, written by src/sim/trace.py beside the trace).

    tools/prompts.py                         # the latest run: one line per call
    tools/prompts.py --tag generate_acts     # calls with this tag (substring)
    tools/prompts.py --owner Qiu             # calls made for this character
    tools/prompts.py --grep photograph       # calls whose prompt or reply mentions this
    tools/prompts.py --show 412              # call number 412 in full: prompt, then reply
    tools/prompts.py --first photograph      # the first call whose REPLY mentions it, and whether its prompt did
Add a path to read another run. --where reply|prompt limits --grep to one side.
"""
import argparse
import json
import sys
from pathlib import Path

TRACE_DIR = Path.home() / '.local/share/alltheworldaplay/logs/traces'


def text_of(prompt):
    return '\n\n'.join(f"[{m.get('role')}]\n{m.get('content')}" for m in (prompt or []))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', nargs='?')
    p.add_argument('--tag'); p.add_argument('--owner'); p.add_argument('--grep'); p.add_argument('--first')
    p.add_argument('--where', choices=('reply', 'prompt'))
    p.add_argument('--show', type=int)
    a = p.parse_args()
    path = Path(a.path) if a.path else max(TRACE_DIR.glob('*.prompts.jsonl'), key=lambda f: f.stat().st_mtime, default=None)
    if path is None or not path.exists():
        sys.exit('no prompt log found')
    calls = [json.loads(line) for line in open(path)]
    print(f'{path.name}: {len(calls)} calls')
    if a.show is not None:
        c = calls[a.show]
        print(f"#{a.show} {c['tag']} owner={c['owner']} sim={c['sim_time']} act={c['act']} scene={c['scene']}\n")
        print(text_of(c['prompt'])); print('\n=== REPLY ===\n' + str(c['response']))
        return
    if a.first:
        word = a.first.lower()
        for i, c in enumerate(calls):
            if word in str(c['response']).lower():
                told = word in text_of(c['prompt']).lower()
                print(f"first reply to say '{a.first}': #{i} {c['tag']} owner={c['owner']} sim={c['sim_time']}; "
                      f"its prompt {'already said it' if told else 'did NOT say it: it starts here'}")
                print('  reply:', ' '.join(str(c['response']).split())[:400])
                return
        print(f"no reply says '{a.first}'")
        return
    for i, c in enumerate(calls):
        if a.tag and a.tag.lower() not in str(c['tag']).lower(): continue
        if a.owner and a.owner.lower() != str(c['owner']).lower(): continue
        if a.grep:
            word = a.grep.lower()
            in_prompt, in_reply = word in text_of(c['prompt']).lower(), word in str(c['response']).lower()
            if not ((in_prompt and a.where != 'reply') or (in_reply and a.where != 'prompt')): continue
        print(f"#{i:4d} {str(c['sim_time'])[11:16]} {str(c['tag'])[:34]:34} {str(c['owner'] or '')[:10]:10} "
              f"{' '.join(str(c['response']).split())[:110]}")


if __name__ == '__main__':
    main()
