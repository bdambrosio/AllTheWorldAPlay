"""Pair a prompt's old and new versions on recorded calls: the same call, as it was sent in a play
and as the revised template would send it, asked of the same models several times each.

  extract RUN LINE NAME DIR    write DIR/NAME.old.json (the recorded messages, tag, max_tokens,
                               stops) and DIR/NAME.rec.txt (the reply the play got)
  run DIR [--models fireworks,vllm] [--n 3] [--only NAME,...]
                               for each DIR/NAME.new.json beside its .old.json, ask both of each
                               model n times; write DIR/NAME.pairs.md (old and new replies side by side)

A .new.json is the .old.json with its messages rewritten as the new template renders them
(made by the script that goes with each revision). Nothing here judges a reply: the pairs are
read, or given to a judging model with the facts.

  set -a; . ~/.config/secrets.env; set +a; .venv/bin/python tools/pair_prompts.py run DIR
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

TRACES = os.path.expanduser('~/.local/share/alltheworldaplay/logs/traces/')


def extract(run, line, name, out):
    with open(TRACES + run + '.prompts.jsonl') as f:
        for i, l in enumerate(f, 1):
            if i == line:
                r = json.loads(l)
                break
        else:
            sys.exit(f'{run} has no line {line}')
    os.makedirs(out, exist_ok=True)
    case = {'run': run, 'line': line, 'tag': r['tag'], 'owner': r['owner'],
            'messages': [{'role': m['role'], 'content': m['content']} for m in r['prompt']]}
    json.dump(case, open(os.path.join(out, name + '.old.json'), 'w'), indent=1)
    open(os.path.join(out, name + '.rec.txt'), 'w').write(r['response'] or '')
    print(name, r['tag'], sum(len(m['content']) for m in case['messages']), 'chars')


def ask(llm, case, max_tokens):
    from utils.Messages import UserMessage, SystemMessage, AssistantMessage
    kind = {'system': SystemMessage, 'user': UserMessage, 'assistant': AssistantMessage}
    msgs = [kind[m['role']](content=m['content']) for m in case['messages']]
    return llm.ask({}, msgs, tag=case['tag'], temp=case.get('temp'), max_tokens=max_tokens,
                   stops=case.get('stops', ['</end>']), log=False) or ''


def run(out, models, n, only):
    from utils.llm_api import LLM
    names = sorted(f[:-9] for f in os.listdir(out) if f.endswith('.new.json'))
    if only:
        names = [x for x in names if x in only]
    for name in names:
        old = json.load(open(os.path.join(out, name + '.old.json')))
        new = json.load(open(os.path.join(out, name + '.new.json')))
        size = lambda c: sum(len(m['content']) for m in c['messages'])
        rows = [f"# {name}: {old['run']}:{old['line']} {old['tag']} ({old['owner']})",
                f"old {size(old)} chars, new {size(new)} chars", '',
                '## recorded reply', open(os.path.join(out, name + '.rec.txt')).read().strip(), '']
        for server in models:
            llm = LLM(server)
            for version, case in (('old', old), ('new', new)):
                rows.append(f'## {server} / {version}')
                for k in range(n):
                    reply = ask(llm, case, case.get('max_tokens', 400)).strip()
                    rows.append(f'--- {k + 1}\n{reply}')
                rows.append('')
        open(os.path.join(out, name + '.pairs.md'), 'w').write('\n'.join(rows))
        print('wrote', name + '.pairs.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract'); e.add_argument('run'); e.add_argument('line', type=int); e.add_argument('name'); e.add_argument('dir')
    r = sub.add_parser('run'); r.add_argument('dir'); r.add_argument('--models', default='fireworks,vllm')
    r.add_argument('--n', type=int, default=3); r.add_argument('--only', default='')
    a = p.parse_args()
    if a.cmd == 'extract':
        extract(a.run, a.line, a.name, a.dir)
    else:
        run(a.dir, a.models.split(','), a.n, set(filter(None, a.only.split(','))))
