"""Replay logged speech/narration checks (today's prompt logs) with the current JSON prompts and print where old and new differ.
    set -a; . ~/.config/secrets.env; set +a; .venv/bin/python tools/replay_checks.py speech|check N"""
import json, glob, os, re, sys, random
sys.path.insert(0, 'src')
from utils.llm_api import LLM
from utils.Messages import UserMessage
from utils import json_reply
import sim.record as record

which = sys.argv[1]; n = int(sys.argv[2])
tag, old_marker, new_tmpl, new_marker = {
    'speech': ('record.speech', 'Find only statements of these kinds.', record.SPEECH, 'Find only statements of these kinds.'),
    'check': ('record.check', 'A statement is unsupported when it is one of these:', record.CHECK, 'A statement is unsupported when it is one of these:'),
}[which]
tail = new_tmpl[new_tmpl.index(new_marker):]
cases = []
for f in sorted(glob.glob(os.path.expanduser('~/.local/share/alltheworldaplay/logs/traces/*-20261111-*.prompts.jsonl'))):
    for l in open(f):
        p = json.loads(l)
        if p['tag'] != tag: continue
        t = p['prompt'] if isinstance(p['prompt'], str) else ''.join(m.get('content', '') for m in p['prompt'])
        if old_marker not in t or '"findings"' in t: continue
        cases.append((os.path.basename(f).split('.')[0], t, p['response'] or ''))
random.seed(7)
must = [c for c in cases if any(k in c[1] for k in ('take one. Fair', 'Morning, notebook boy', 'Found two cans', 'pocketing a can'))]
pick = must + random.sample([c for c in cases if c not in must], min(n, len(cases)))
llm = LLM('fireworks')
for run, t, old in pick:
    name = (re.search(r'Speaker: (\S+)', t) or re.search(r'What (\S+) tried', t)).group(1)
    prompt = t[:t.index(old_marker)] + tail.replace('{{$name}}', name).replace('</end>', '').rstrip() + '\n'
    new = llm.ask({}, [UserMessage(content=prompt)], tag='replay', temp=0.1, stops=['</end>'], max_tokens=220) or ''
    found = json_reply.parse(new)
    kinds = record.SPEECH_KINDS if which == 'speech' else record.KINDS
    newf = json_reply.items(found, 'findings', kinds) if isinstance(found, dict) else 'UNREADABLE'
    oldf = re.findall(r'#claim\s+(\w+)\s*:\s*(.+)', old)
    m = re.search(r'<line>\s*(.*?)\s*</line>', t, re.S) or re.search(r'<narration>\s*(.*?)\s*</narration>', t, re.S)
    if not oldf and newf == []: continue
    print(f'[{run}] {name}: {m.group(1)[:230]!r}')
    print('   OLD raw:', oldf)
    print('   NEW    :', newf if newf == 'UNREADABLE' else [(x["kind"], x["words"]) for x in newf],
          ('names=' + str(found.get('names')) if which == 'speech' and isinstance(found, dict) and found.get('names') else ''))
print(len(pick), 'replayed')
