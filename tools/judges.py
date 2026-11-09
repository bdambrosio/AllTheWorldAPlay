#!/usr/bin/env python3
"""LLM-judge tiers of a play scorecard (tiers 4-5), from a run trace (src/sim/trace.py).

    set -a; . ~/.config/secrets.env; set +a          # FIREWORKS_API_KEY for --judge fireworks/both
    .venv/bin/python tools/judges.py <trace.jsonl> [--judge local|fireworks|both] [--samples 3]

Judges: 'local' is the actors' own vLLM model (free; judges its own family's writing),
'fireworks' is GLM-5.3-Flash on Fireworks (utils/llm_api.py route 'fireworks').

  tier 4, per scene (improv craft): yes_and, tension, discovery, pacing, purpose (vs the
          scene's planned outcome)
  tier 5, per character: the June 2025 rubrics (tools/rubrics_2025.py, verbatim)
          per play: arc completion vs the act plans, causality, subtext, overall quality
  attribution: blind speaker attribution of sampled lines (who said this?)

Every rubric is sampled --samples times per judge; the median score and mean validity are
kept. With two judges the report gives their agreement (Spearman, mean |difference|).
Writes logs/scores/<run>.judges.json and .judges.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from utils import hash_utils                      # noqa: E402
from utils.llm_api import LLM                     # noqa: E402
from utils.Messages import SystemMessage, UserMessage  # noqa: E402
from rubrics_2025 import RUBRICS_2025             # noqa: E402
from score_play import load, by_kind              # noqa: E402

PLAY_RUBRIC_DIMS = ['goal_directedness', 'social_awareness', 'cognitive_flexibility', 'coherence',
                    'engagement', 'complexity', 'character_development', 'action_effectiveness']
MAX_SCRIPT_CHARS = 120_000

CRITIC = SystemMessage(content="""You are an expert dramaturg, improv coach and screenplay critic.
You are given an improvised play performed by AI characters, as a series of lines:
<character-name>: '<speech>' - dialogue
<character-name>: ...<thoughts>... - internal monologue
<character-name>: does / moves <action> - physical or world action, possibly followed by its result
Lines without a character name are narration (scene settings, situation updates, scene wrap-ups).""")

SCENE_RUBRIC = [CRITIC, UserMessage(content="""Evaluate this one scene as improvised theatre.

The scene was planned to start from:
{{$pre}}
and to arrive at:
{{$planned_post}}

#SCENE
{{$script}}
##

Rate each from 1.0 to 5.0 (3.0-4.0 is competent work):
- yes_and: characters accept and build on each other's offers (new facts, proposals, emotional bids) rather than ignoring, blocking or repeating them
- tension: stakes are clear and pressure rises or shifts during the scene
- discovery: something is revealed or changes the situation; the scene is not a restatement of known facts
- pacing: the scene moves; no stalling, loops, or repeated beats
- purpose: the scene arrives at (or meaningfully subverts) its planned outcome

Respond with this hash-formatted text only:
#yes_and score
#tension score
#discovery score
#pacing score
#purpose score
#validity 0.0-1.0, how much evidence the scene gives for these scores
#justification 6-12 words
<end/>
""")]

PLAY_RUBRIC = [CRITIC, UserMessage(content="""Evaluate this whole performance as a play.

The setting:
{{$scenario}}

The planned acts (intended starting and ending states):
{{$acts}}

#SCRIPT
{{$script}}
##

Rate each from 1.0 to 5.0 (3.0-4.0 is competent work):
- arc: the play travels the planned arc of its acts (or departs from it in a way that works dramatically)
- causality: events follow from characters' choices and earlier events, rather than happening arbitrarily
- subtext: characters want things they do not say outright; dialogue carries more than its surface meaning
- overall: the quality of the play as a piece of theatre

Respond with this hash-formatted text only:
#arc score
#causality score
#subtext score
#overall score
#validity 0.0-1.0, how much evidence the transcript gives for these scores
#justification 6-12 words
<end/>
""")]

ATTRIBUTION = [CRITIC, UserMessage(content="""The cast of a play:
{{$cast}}

One line of dialogue from the play, with the speaker's name removed:
{{$line}}

Which character said it? Respond with this hash-formatted text only:
#speaker the character's name
<end/>
""")]


def make_judge(name):
    if name == 'local':
        return LLM('vllm'), 0.7
    llm = LLM('fireworks')
    return llm, 1.0


def number(text):
    try:
        return float(str(text).strip().split()[0])
    except (ValueError, IndexError, TypeError):
        return None


def sample(llm, temp, prompt, bindings, fields, samples, tag, effort=None):
    """Median of each numeric field over `samples` asks; mean validity; first justification."""
    if effort and hasattr(llm, 'reasoning_effort'):
        llm.reasoning_effort = effort
    values = collections.defaultdict(list)
    justification = None
    for _ in range(samples):
        reply = llm.ask(bindings, prompt, tag=tag, temp=temp, max_tokens=2000, stops=['<end/>'])
        if not reply:
            continue
        for f in fields + ['validity']:
            v = number(hash_utils.find(f, reply))
            if v is not None:
                values[f].append(v)
        justification = justification or hash_utils.find('justification', reply)
    out = {f: (statistics.median(values[f]) if values[f] else None) for f in fields}
    out['validity'] = round(statistics.fmean(values['validity']), 3) if values['validity'] else None
    out['samples'] = max((len(v) for v in values.values()), default=0)
    out['justification'] = (justification or '').strip()[:200]
    return out


def script_lines(lines):
    out = []
    for l in lines:
        speaker, text = l.get('speaker'), l.get('text', '')
        out.append(f'{speaker}: {text}' if speaker else text)
    return '\n'.join(out)


def run_judge(name, ev, cast, settings, samples, attribution_n):
    llm, temp = make_judge(name)
    lines = ev['transcript_line']
    script = script_lines(lines)[-MAX_SCRIPT_CHARS:]
    result = {'judge': name, 'model': llm.model}

    # tier 4: scenes
    scenes = []
    starts = {(e.get('act'), e.get('scene')): e for e in ev['scene_start']}
    by_scene = collections.defaultdict(list)
    for l in lines:
        if l.get('scene') is not None:
            by_scene[(l.get('act'), l.get('scene'))].append(l)
    for key, start in starts.items():
        scene_lines = by_scene.get(key, [])
        if len([l for l in scene_lines if l.get('speaker')]) < 3:
            continue
        r = sample(llm, temp, SCENE_RUBRIC, {'pre': start.get('pre_narrative') or '',
                                             'planned_post': start.get('planned_post') or '',
                                             'script': script_lines(scene_lines)[-40_000:]},
                   ['yes_and', 'tension', 'discovery', 'pacing', 'purpose'], samples, 'judge_scene', 'low')
        scenes.append({'act': key[0], 'scene': key[1], 'title': start.get('title'), **r})
    result['scenes'] = scenes

    # tier 5: per character (2025 rubrics) and per play
    characters = {}
    for cname, description in cast.items():
        if not any(l.get('speaker') == cname for l in lines):
            continue
        characters[cname] = {}
        for dim in PLAY_RUBRIC_DIMS:
            characters[cname][dim] = sample(llm, temp, RUBRICS_2025[dim],
                                            {'scenario': settings, 'character_description': description,
                                             'name': cname, 'script': script},
                                            ['score'], samples, f'judge_{dim}', 'medium')
    result['characters'] = characters
    acts = '\n'.join(f"Act {e.get('number')}: {e.get('title')}\n  from: {e.get('planned_pre')}\n  to: {e.get('planned_post')}"
                     for e in ev['act_start']) or '(no act plan: an unscripted play)'
    result['play'] = sample(llm, temp, PLAY_RUBRIC, {'scenario': settings, 'acts': acts, 'script': script},
                            ['arc', 'causality', 'subtext', 'overall'], samples, 'judge_play', 'medium')

    # blind attribution
    speech = [l for l in lines if l.get('line_kind') == 'speech' and l.get('speaker') in cast]
    counts = collections.Counter(l['speaker'] for l in speech)
    pool = [l for l in speech if counts[l['speaker']] >= 3 and len(l['text']) > 20]
    picks = random.Random(0).sample(pool, min(attribution_n, len(pool))) if len(counts) >= 2 else []
    cast_text = '\n'.join(f'- {n}: {d[:400]}' for n, d in cast.items() if counts.get(n))
    hits = 0
    for l in picks:
        reply = llm.ask({'cast': cast_text, 'line': l['text']}, ATTRIBUTION, tag='judge_attribution',
                        temp=0.2 if name == 'local' else 1.0, max_tokens=1000, stops=['<end/>'])
        guess = (hash_utils.find('speaker', reply or '') or '').strip().lower()
        hits += bool(guess) and guess.split()[0].strip('.,') == l['speaker'].lower()
    result['attribution'] = {'lines': len(picks), 'accuracy': round(hits / len(picks), 3) if picks else None,
                             'chance': round(1 / len([n for n in counts if counts[n] >= 3]), 3) if picks else None}
    return result


def flat_scores(result):
    """(item, score) pairs for agreement between judges."""
    items = {}
    for s in result['scenes']:
        for f in ('yes_and', 'tension', 'discovery', 'pacing', 'purpose'):
            items[f"scene {s['act']}.{s['scene']} {f}"] = s.get(f)
    for c, dims in result['characters'].items():
        for d, r in dims.items():
            items[f'{c} {d}'] = r.get('score')
    for f in ('arc', 'causality', 'subtext', 'overall'):
        items[f'play {f}'] = result['play'].get(f)
    return {k: v for k, v in items.items() if v is not None}


def agreement(a, b):
    from scipy.stats import spearmanr
    fa, fb = flat_scores(a), flat_scores(b)
    keys = sorted(set(fa) & set(fb))
    if len(keys) < 3:
        return {'items': len(keys)}
    xs, ys = [fa[k] for k in keys], [fb[k] for k in keys]
    rho = spearmanr(xs, ys).statistic
    diffs = sorted(((abs(x - y), k, x, y) for k, x, y in zip(keys, xs, ys)), reverse=True)
    return {'items': len(keys), 'spearman': None if rho != rho else round(float(rho), 3),
            'mean_abs_diff': round(statistics.fmean(d for d, *_ in diffs), 3),
            'mean_local_minus_external': round(statistics.fmean(x - y for x, y in zip(xs, ys)), 3),
            'largest_disagreements': [{'item': k, a['judge']: x, b['judge']: y} for _, k, x, y in diffs[:5]]}


def markdown(report):
    out = [f"# Judges: {report['run']}", ""]
    for r in report['judges']:
        out += [f"## {r['judge']} ({r['model']})", ""]
        if r['scenes']:
            out += ["| scene | yes_and | tension | discovery | pacing | purpose | validity |", "|---|---|---|---|---|---|---|"]
            out += [f"| {s['act']}.{s['scene']} {s.get('title') or ''} | {s['yes_and']} | {s['tension']} | {s['discovery']} "
                    f"| {s['pacing']} | {s['purpose']} | {s['validity']} |" for s in r['scenes']]
        dims = PLAY_RUBRIC_DIMS
        if r['characters']:
            out += ["", "| character | " + " | ".join(dims) + " |", "|---" * (len(dims) + 1) + "|"]
            out += [f"| {c} | " + " | ".join(str(v[d]['score']) for d in dims) + " |" for c, v in r['characters'].items()]
        p = r['play']
        out += ["", f"play: arc {p['arc']}, causality {p['causality']}, subtext {p['subtext']}, overall {p['overall']} "
                    f"(validity {p['validity']}): {p['justification']}",
                f"blind attribution: {r['attribution']}", ""]
    if report.get('agreement'):
        out += ["## Agreement", "", json.dumps(report['agreement'], indent=2)]
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('trace')
    p.add_argument('--judge', choices=['local', 'fireworks', 'both'], default='local')
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--attribution', type=int, default=16, help='lines for blind speaker attribution')
    p.add_argument('--out', default=str(ROOT / 'logs' / 'scores'))
    a = p.parse_args()
    events = load(a.trace)
    ev = collections.defaultdict(list, by_kind(events))
    loaded = (ev['play_loaded'] or [{}])[-1]
    cast = {c['name']: c.get('description') or '' for c in loaded.get('characters', [])}
    settings = loaded.get('setting') or ''
    names = ['local', 'fireworks'] if a.judge == 'both' else [a.judge]
    report = {'run': events[0].get('run'), 'trace': a.trace,
              'judges': [run_judge(n, ev, cast, settings, a.samples, a.attribution) for n in names]}
    if len(report['judges']) == 2:
        report['agreement'] = agreement(*report['judges'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{report['run']}.judges.json").write_text(json.dumps(report, indent=2, default=str))
    (out / f"{report['run']}.judges.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == '__main__':
    main()
