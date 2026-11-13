#!/usr/bin/env python3
"""Draw recorded picture moments four ways, for a blind side-by-side:

    A  the prompt the play sent (the old builder), local FLUX.2 klein
    B  the prompt written from the moment's facts (sim/stage.py picture_prompt), klein, same seed
    C  the facts prompt, OpenAI image model, with each person's reference portrait
    D  the old prompt, OpenAI image model, with the same references (C vs D: the prompt's part)

Moments come from `image` events in run traces that carry `facts` (recorded since the
2026-11-13 image work). A contact sheet (index.html) shows each moment as a row with its four
pictures in a shuffled order labelled 1-4 and the facts beneath; key.json says which is which.

    set -a; . ~/.config/secrets.env; set +a          # OPENAI_API_KEY
    .venv/bin/python tools/image_ab.py ~/.local/share/alltheworldaplay/logs/traces/lost-*.jsonl \\
        --out /tmp/image_ab --stage 15 --portraits 5
    --only AB            draw some of the four (no OpenAI calls for A and B)
    --dry-run            list the moments and prompts, draw nothing

The image server must be up (start-local.sh starts it on :5008).
"""
import argparse
import base64
import html
import json
import os
import random
import sys
import zlib
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sim import stage  # noqa: E402

KLEIN_URL = os.environ.get('IMAGE_URL', 'http://127.0.0.1:5008/generate_image')
OPENAI_URL = 'https://api.openai.com/v1/images'
OPENAI_MODEL = 'gpt-image-2.6-flare'
REFERENCE_NOTE = ('The people in this picture are the ones in the reference portraits: keep each one\'s face, '
                  'hair and clothes, and do not add anyone else. ')


def moments(traces):
    """Every image event with facts, in run order: (run, index, event)."""
    found = []
    for path in traces:
        run = Path(path).name.removesuffix('.jsonl')
        for i, line in enumerate(open(path)):
            event = json.loads(line)
            if event.get('kind') == 'image' and event.get('facts') and event.get('builder') == 'old':
                found.append((run, i, event))
    return found


def choose(found, n_stage, n_portraits, seed=7):
    """A spread of moments: scene openings, deeds with something in hand, someone hurt or a danger
    on stage, one person alone, interiors; then the rest at random. Portraits apart."""
    rng = random.Random(seed)
    stage_moments = [m for m in found if m[2]['of'] in ('world', 'act')]
    portraits = [m for m in found if m[2]['of'] not in ('world', 'act')]
    kinds = [lambda f: bool(f.get('setup')),
             lambda f: bool(f.get('deed')) and any(p['holding'] for p in f['people']),
             lambda f: bool(f.get('threat')) or any(p['condition'] for p in f['people']),
             lambda f: len(f['people']) == 1,
             lambda f: f.get('indoor')]
    picked = []
    for _ in range(n_stage):
        for kind in kinds:
            pool = [m for m in stage_moments if m not in picked and kind(m[2]['facts'])]
            if pool and len(picked) < n_stage:
                picked.append(rng.choice(pool))
        rest = [m for m in stage_moments if m not in picked]
        if len(picked) >= n_stage or not rest:
            break
        picked.append(rng.choice(rest))
    picked = picked[:n_stage]
    picked += rng.sample(portraits, min(n_portraits, len(portraits)))
    return picked


def klein(prompt, path, seed, size):
    r = requests.get(KLEIN_URL, params={'prompt': prompt, 'size': size, 'seed': seed}, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)
    return {'server_tokens': r.headers.get('x-prompt-tokens')}


def openai(prompt, path, refs=(), quality='low'):
    headers = {'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}"}
    if refs:
        files = [('image[]', (p.name, open(p, 'rb'), 'image/png')) for p in refs]
        r = requests.post(f'{OPENAI_URL}/edits', headers=headers, files=files, timeout=300,
                          data={'model': OPENAI_MODEL, 'prompt': REFERENCE_NOTE + prompt, 'size': '1024x1024', 'quality': quality})
    else:
        r = requests.post(f'{OPENAI_URL}/generations', headers=headers, timeout=300,
                          json={'model': OPENAI_MODEL, 'prompt': prompt, 'size': '1024x1024', 'quality': quality, 'n': 1})
    if r.status_code != 200:
        raise RuntimeError(f'{r.status_code} from OpenAI: {r.text[:300]}')
    reply = r.json()
    path.write_bytes(base64.b64decode(reply['data'][0]['b64_json']))
    return {'usage': reply.get('usage')}


def references(found, out, dry):
    """One OpenAI portrait a character, from the first portrait facts recorded for it in its run:
    the face C and D hold to. Keyed by (run, name)."""
    refs = {}
    for run, i, event in found:
        if event['of'] in ('world', 'act') or (run, event['of']) in refs:
            continue
        path = out / f"ref_{run}_{event['of']}.png"
        refs[(run, event['of'])] = path
        if not dry and not path.exists():
            openai(stage.portrait_prompt(event['facts']), path)
    return refs


def draw(picked, refs, out, only, dry):
    rows = []
    for n, (run, i, event) in enumerate(picked, 1):
        facts, portrait = event['facts'], event['of'] not in ('world', 'act')
        new = stage.portrait_prompt(facts) if portrait else stage.picture_prompt(facts)
        names = [event['of']] if portrait else [p['name'] for p in facts['people']]
        people_refs = [refs[(run, name)] for name in names if (run, name) in refs]
        if portrait and people_refs and people_refs[0].stem == f"ref_{run}_{event['of']}":
            people_refs = []   # the reference is this character's own first portrait; draw it fresh
        seed = zlib.crc32(f'{run}:{i}'.encode())
        size = '512x512' if portrait else '768x768'
        ways = {'A': ('old prompt, klein', event['prompt']), 'B': ('facts prompt, klein', new),
                'C': ('facts prompt, OpenAI', new), 'D': ('old prompt, OpenAI', event['prompt'])}
        row = {'n': n, 'run': run, 'line': i, 'of': event['of'], 'facts': facts, 'pictures': {}}
        for way, (label, prompt) in ways.items():
            if way not in only:
                continue
            path = out / f'{n:02d}_{way}.png'
            row['pictures'][way] = {'file': path.name, 'label': label, 'prompt': prompt,
                                    'tokens': stage.prompt_tokens(prompt), 'refs': [p.name for p in people_refs] if way in 'CD' else []}
            if dry or path.exists():
                continue
            print(f'{n:02d} {way} {event["of"]} ...', flush=True)
            try:
                info = klein(prompt, path, seed, size) if way in 'AB' else openai(prompt, path, people_refs)
                row['pictures'][way].update(info)
            except Exception as e:
                row['pictures'][way]['error'] = str(e)
                print(f'   failed: {e}', flush=True)
        rows.append(row)
    return rows


def contact_sheet(rows, out, seed=11):
    """index.html with each row's pictures in a shuffled order labelled 1-4; key.json apart."""
    rng = random.Random(seed)
    key, body = {}, []
    for row in rows:
        ways = list(row['pictures'])
        rng.shuffle(ways)
        key[row['n']] = {str(k): w for k, w in enumerate(ways, 1)}
        cells = ''.join(f"<figure><img src='{row['pictures'][w]['file']}' loading='lazy'><figcaption>{k}</figcaption></figure>"
                        for k, w in enumerate(ways, 1))
        facts = row['facts']
        if row['of'] in ('world', 'act'):
            people = '; '.join(f"<b>{html.escape(p['name'])}</b>: {html.escape(p['condition'] or 'as written')}"
                               + (f", carrying {html.escape(', '.join(p['holding']))}" if p['holding'] else '') for p in facts['people'])
            told = ' · '.join(html.escape(x) for x in (facts.get('deed'), facts.get('threat'), facts.get('setup')) if x)
            what = f"{html.escape(facts['place'])}, {html.escape(facts['light'])}. {people}." + (f"<br>{told}" if told else '')
        else:
            p = facts['person']
            what = (f"Portrait of <b>{html.escape(p['name'])}</b>: {html.escape(p['condition'] or 'as written')}"
                    + (f", carrying {html.escape(', '.join(p['holding']))}" if p['holding'] else '') + f". Mood: {html.escape(facts['mood'])}")
        body.append(f"<section><h2>{row['n']}. {html.escape(row['run'])} · {html.escape(row['of'])}</h2>"
                    f"<p class='facts'>{what}</p><div class='row'>{cells}</div></section>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image A/B</title><style>
:root {{ --bg:#fafaf8; --fg:#1d1d1b; --muted:#666; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ecece8; --muted:#a0a09a; --line:#333; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1500px; padding:16px; }}
section {{ border-top:1px solid var(--line); padding:14px 0; }} h2 {{ font-size:15px; margin:0 0 4px; }}
.facts {{ color:var(--muted); margin:0 0 10px; }} .row {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:10px; }}
figure {{ margin:0; }} img {{ width:100%; display:block; border-radius:4px; }} figcaption {{ text-align:center; font-weight:600; padding:4px; }}
</style></head><body><h1>Pictures of the same moments, four ways</h1>
<p>For each row: which picture would you rather watch the play with? Note the row and the number.</p>
{''.join(body)}</body></html>"""
    (out / 'index.html').write_text(page)
    (out / 'key.json').write_text(json.dumps(key, indent=1))
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--out', default='image_ab')
    ap.add_argument('--stage', type=int, default=15)
    ap.add_argument('--portraits', type=int, default=5)
    ap.add_argument('--only', default='ABCD')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found = moments(args.traces)
    if not found:
        sys.exit('no image events with facts in these traces (recorded since the 2026-11-13 image work)')
    picked = choose(found, args.stage, args.portraits)
    refs = references(found, out, args.dry_run or not set(args.only) & set('CD'))
    rows = draw(picked, refs, out, set(args.only), args.dry_run)
    contact_sheet(rows, out)
    if args.dry_run:
        for row in rows:
            for way, pic in row['pictures'].items():
                print(f"{row['n']:02d} {way} {pic['tokens']} tokens: {pic['prompt'][:160]}")
    print(f'{len(rows)} moments; contact sheet {out / "index.html"}; key {out / "key.json"}')


if __name__ == '__main__':
    main()
