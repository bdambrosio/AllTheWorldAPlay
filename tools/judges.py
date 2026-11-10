#!/usr/bin/env python3
"""LLM-judge tiers of a play scorecard (tiers 4-5), from a run trace (src/sim/trace.py).

    set -a; . ~/.config/secrets.env; set +a          # CLAUDE_API_KEY / FIREWORKS_API_KEY
    .venv/bin/python tools/judges.py <trace.jsonl> [--judge local,claude] [--samples 3] [--scene-samples 5]

Judges (--judge takes a comma list; 'both' = local,fireworks):
  local      the actors' own vLLM model: free, for quick loops; judges its own family's writing
  claude     Claude Sonnet 5.5 (Anthropic API): the judge for headline numbers
  fireworks  GLM-5.3-Flash on Fireworks (utils/llm_api.py route 'fireworks')

  tier 4, per scene (improv craft): yes_and, tension, discovery, pacing, purpose, engagement
          (would an audience want the next scene). One call per dimension, with anchors for
          1/3/5 and the evidence cited before the score; the score is the mean of
          --scene-samples asks. A scene the run's timer cut short (no scene_end) is not judged.
          Each scene also carries measured signals from the trace (repetition, acts, hazards,
          needs) for checking the judges against (tools/compare_runs.py, 'Judge validity').
  tier 5, per character: the June 2025 rubrics (tools/rubrics_2025.py, verbatim, plus a reply
          format note). The 2025 'engagement' rubric measures how actively a character keeps
          the conversation going, so it is reported as conversational_initiative.
          per play: arc, causality, subtext, overall quality of the part performed
  attribution: blind speaker attribution of sampled lines (who said this?)

Replies are parsed leniently ('#score 3.5', 'Score: 3.5', a bare '4'); replies with no usable
score are counted per judge in the report ('parse'). With several judges the report gives each
pair's agreement (Spearman, mean |difference|).
Writes logs/scores/<run>.judges.json and .judges.md.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from utils.llm_api import LLM                     # noqa: E402
from utils.Messages import SystemMessage, UserMessage  # noqa: E402
from rubrics_2025 import RUBRICS_2025             # noqa: E402
from score_play import load, by_kind              # noqa: E402

# report name -> 2025 rubric
CHARACTER_DIMS = {'goal_directedness': 'goal_directedness', 'social_awareness': 'social_awareness',
                  'cognitive_flexibility': 'cognitive_flexibility', 'coherence': 'coherence',
                  'conversational_initiative': 'engagement', 'complexity': 'complexity',
                  'character_development': 'character_development', 'action_effectiveness': 'action_effectiveness'}
PLAY_RUBRIC_DIMS = list(CHARACTER_DIMS)
PLAY_FIELDS = ['arc', 'causality', 'subtext', 'overall']
MAX_SCRIPT_CHARS = 120_000
CLAUDE_MODEL = 'claude-sonnet-5-5'
CLAUDE_PRICE = {'input': 2.0, 'cache_write': 2.5, 'cache_read': 0.10, 'output': 10.0}   # $ per million tokens
# Text before this marker is the same across the calls about one scene (or one pair of scenes):
# the Claude judge caches it; other judges just drop the marker.
CACHE_BREAK = '\n[[cache]]\n'

CRITIC = """You are an expert dramaturg, improv coach and screenplay critic.
You are given an improvised play performed by AI characters, as a series of lines:
<character-name>: '<speech>' - dialogue
<character-name>: ...<thoughts>... - internal monologue
<character-name>: <action> - a physical action in the present tense ('pulls a tablet ...', 'moves to ...'), possibly followed by its result
Lines without a character name are narration (scene settings, situation updates, scene wrap-ups)."""

# name -> (what it asks, anchor for 1, anchor for 3, anchor for 5)
SCENE_DIMS = {
    'yes_and': ("characters accept and build on each other's offers (new facts, proposals, emotional bids)",
                "offers are ignored or blocked; the characters talk past each other",
                "offers are acknowledged, but rarely built into anything new",
                "nearly every offer is taken up and extended into something neither character had alone"),
    'tension': ("something is at stake, and the pressure on the characters rises or turns during the scene",
                "nothing is at stake, or the stakes are named but never felt",
                "the stakes are clear, but the pressure stays level from start to end",
                "the stakes are clear and the pressure rises or turns at moments you can point to"),
    'discovery': ("something is revealed, or the situation changes; the scene is not a restatement of what was known",
                  "the scene ends where it began: known facts restated",
                  "one new fact or a small shift, without consequences inside the scene",
                  "a revelation or reversal changes what the characters want or can do"),
    'pacing': ("the scene keeps moving: each beat adds something, with no stalling, loops or repeated beats",
               "the scene stalls or loops: the same beat (a line, a move, a check) comes back three or more times",
               "the scene moves forward, with some repeated or idle beats",
               "every beat advances the scene; nothing is repeated"),
    'purpose': ("the scene arrives at its planned outcome, or subverts it in a way that works dramatically",
                "the scene ends somewhere unrelated to the planned outcome, with no dramatic gain",
                "the scene gets part of the way to the planned outcome",
                "the scene fully arrives at the planned outcome, or subverts it to better effect"),
    'engagement': ("an audience would stay interested and want to see the next scene",
                   "an audience would stop watching: there is nothing they want to see resolved",
                   "watchable: mild curiosity about what comes next",
                   "gripping: the scene leaves an open question the audience needs answered"),
}
SCENE_FIELDS = list(SCENE_DIMS)

SCENE_PROMPT = """One scene of the play. It was planned to start from:
{{$pre}}
and to arrive at:
{{$planned_post}}

#SCENE
{{$script}}
##
""" + CACHE_BREAK + """Judge one quality of this scene as improvised theatre, {{$dim}}: {{$question}}.

Anchors:
1 = {{$a1}}
3 = {{$a3}}
5 = {{$a5}}
Use 2, 4 or half points for scenes between anchors. Judge this quality only: weaknesses of other kinds must not move this score.

First cite the evidence, then score. Respond with exactly these three lines, each starting with '#':
#evidence up to 30 words: the lines or beats that decide this score
#score a number from 1.0 to 5.0
#validity 0.0-1.0, how much evidence the scene gives for this score
<end/>
"""

PLAY_PROMPT = """Evaluate this performance as a play.

The setting:
{{$scenario}}

The planned acts (intended starting and ending states):
{{$acts}}

The run is stopped by a timer, so the transcript may end before the planned acts are finished. Judge the part that was performed; do not mark it down for stopping.

#SCRIPT
{{$script}}
##

Rate each from 1.0 to 5.0 (3.0-4.0 is competent work):
- arc: the part performed travels the planned arc (or departs from it in a way that works dramatically)
- causality: events follow from characters' choices and earlier events, rather than happening arbitrarily
- subtext: characters want things they do not say outright; dialogue carries more than its surface meaning
- overall: the quality of the part performed as a piece of theatre

Respond with exactly these lines, each starting with '#':
#arc score
#causality score
#subtext score
#overall score
#validity 0.0-1.0, how much evidence the transcript gives for these scores
#justification 6-12 words
<end/>
"""

# Appended to the 2025 rubrics: five of them give the #-lines straight after the script with no
# instruction to answer in that form, and judges answered 'Score: 3.5' or a bare number.
FORMAT_NOTE = """

Respond with exactly these three lines and nothing else, each starting with '#':
#score a number from 1.0 (poor) to 5.0 (exceptional); 3.0-4.0 is competent work
#validity a number from 0.0 to 1.0
#justification 3-8 words
<end/>
"""

ATTRIBUTION_PROMPT = """The cast of a play:
{{$cast}}

One line of dialogue from the play, with the speaker's name removed:
{{$line}}

Which character said it? Respond with this line only:
#speaker the character's name
<end/>
"""


# --------------------------------------------------------------------------- judges
class Judge:
    """One judge model. ask() takes rendered text and returns the reply text or None."""
    def __init__(self, name):
        self.name = name
        self.calls = self.failed = 0      # failed: no reply, or no usable score in it
        self.usage = collections.Counter()
        if name == 'claude':
            import anthropic
            self.model = CLAUDE_MODEL
            # the SDK retries rate limits, 5xx and connection errors itself
            self.client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('CLAUDE_API_KEY'),
                                              max_retries=5)
        elif name in ('local', 'fireworks'):
            self.llm = LLM('vllm' if name == 'local' else 'fireworks')
            self.model = self.llm.model
        else:
            raise ValueError(f'unknown judge {name}')

    def ask(self, system, user, tag, effort='low', temp=None):
        self.calls += 1
        if self.name == 'claude':
            return self._ask_claude(system, user, effort)
        if hasattr(self.llm, 'reasoning_effort'):
            self.llm.reasoning_effort = effort
        if temp is None:
            temp = 0.7 if self.name == 'local' else 1.0
        return self.llm.ask({}, [SystemMessage(content=system), UserMessage(content=user.replace(CACHE_BREAK, '\n'))],
                            tag=tag, temp=temp, max_tokens=2000, stops=['<end/>'])

    def _ask_claude(self, system, user, effort):
        import anthropic
        head, sep, tail = user.partition(CACHE_BREAK)
        content = ([{'type': 'text', 'text': head, 'cache_control': {'type': 'ephemeral'}},
                    {'type': 'text', 'text': tail}] if sep else [{'type': 'text', 'text': user}])
        try:
            # Sampling parameters are fixed on this model; effort is the only depth control.
            # fallbacks='default': a request the safety classifiers decline is re-run on the
            # fallback model inside the same call instead of coming back as a refusal.
            r = self.client.beta.messages.create(
                model=self.model, max_tokens=4000, system=system,
                messages=[{'role': 'user', 'content': content}],
                output_config={'effort': effort},
                betas=['server-side-fallback-2026-07-01'], fallbacks='default')
        except anthropic.RateLimitError as e:
            print(f'claude judge: rate limited after retries: {e.message}', file=sys.stderr)
            return None
        except anthropic.APIStatusError as e:
            print(f'claude judge: API error {e.status_code}: {e.message}', file=sys.stderr)
            return None
        except anthropic.APIConnectionError as e:
            print(f'claude judge: connection error: {e}', file=sys.stderr)
            return None
        u = r.usage
        self.usage.update(input=u.input_tokens or 0, output=u.output_tokens or 0,
                          cache_write=getattr(u, 'cache_creation_input_tokens', 0) or 0,
                          cache_read=getattr(u, 'cache_read_input_tokens', 0) or 0)
        if r.stop_reason == 'refusal':
            return None
        return '\n'.join(b.text for b in r.content if b.type == 'text').split('<end/>')[0]

    def cost(self):
        if self.name != 'claude':
            return None
        return round(sum(self.usage[k] * CLAUDE_PRICE[k] for k in CLAUDE_PRICE) / 1e6, 3)


def make_judge(name):
    return Judge(name)


def judge_names(spec):
    names = ['local', 'fireworks'] if spec == 'both' else [n.strip() for n in spec.split(',') if n.strip()]
    for n in names:
        if n not in ('local', 'fireworks', 'claude'):
            raise SystemExit(f"unknown judge '{n}' (local, claude, fireworks)")
    return names


def render(template, bindings):
    for k, v in bindings.items():
        template = template.replace('{{$' + k + '}}', str(v))
    return template


def rubric_2025(dim, bindings):
    """(system, user) text of a 2025 rubric with the reply format note appended."""
    system, user = RUBRICS_2025[dim]
    return system.content, render(user.content, bindings) + FORMAT_NOTE


# --------------------------------------------------------------------------- parsing
def number_field(reply, name, lo=None, hi=None):
    """A numeric field: '#name 3.5', 'Name: 3.5', '**name** = 3.5/5'. None when absent or out of range."""
    m = re.search(rf'(?im)^\W*{re.escape(name)}\b\W*?(-?\d+(?:\.\d+)?)', reply or '')
    if not m:
        return None
    v = float(m.group(1))
    return v if (lo is None or v >= lo) and (hi is None or v <= hi) else None


def text_field(reply, name):
    m = re.search(rf'(?im)^\W*{re.escape(name)}\b\W*?[:=]?\s*(\S.*)$', reply or '')
    return re.sub(r'(?i)^up to \d+ words:\s*', '', m.group(1).strip()) if m else ''


def scores_in(reply, fields):
    """{field: value} for the 1-5 fields found; a bare number answers a single-field rubric."""
    out = {f: number_field(reply, f, 1.0, 5.0) for f in fields}
    if len(fields) == 1 and out[fields[0]] is None:
        m = re.fullmatch(r'\s*(\d(?:\.\d+)?)\s*(?:/\s*5)?\s*', reply or '')
        if m and 1.0 <= float(m.group(1)) <= 5.0:
            out[fields[0]] = float(m.group(1))
    return {f: v for f, v in out.items() if v is not None}


def sample(judge, system, user, fields, samples, tag, effort='low', stat='median', note='justification'):
    """Ask `samples` times. Each field: median or mean of the parsed values (None if none), with
    the spread; mean validity; the first note. A reply with no usable score counts as failed."""
    values = collections.defaultdict(list)
    validity, first_note = [], ''
    for _ in range(samples):
        reply = judge.ask(system, user, tag, effort)
        found = scores_in(reply, fields)
        if not found:
            judge.failed += 1
            continue
        for f, v in found.items():
            values[f].append(v)
        v = number_field(reply, 'validity', 0.0, 1.0)
        if v is not None:
            validity.append(v)
        first_note = first_note or text_field(reply, note)
    agg = statistics.median if stat == 'median' else statistics.fmean
    out = {f: (round(agg(values[f]), 3) if values[f] else None) for f in fields}
    out['sd'] = {f: round(statistics.pstdev(values[f]), 3) for f in fields if len(values[f]) > 1}
    out['validity'] = round(statistics.fmean(validity), 3) if validity else None
    out['samples'] = max((len(v) for v in values.values()), default=0)
    out[note] = first_note[:240]
    return out


# --------------------------------------------------------------------------- trace views
def script_lines(lines):
    out = []
    for l in lines:
        speaker, text = l.get('speaker'), l.get('text', '')
        out.append(f'{speaker}: {text}' if speaker else text)
    return '\n'.join(out)


def scenes_of(ev):
    """Scenes of a run in order: key (act, scene), start event, transcript lines, complete (the
    scene reached its scene_end; the last scene of a timed run usually has not), judgeable."""
    ended = {(e.get('act'), e.get('scene')) for e in ev['scene_end']}
    by_scene = collections.defaultdict(list)
    for l in ev['transcript_line']:
        if l.get('scene') is not None:
            by_scene[(l.get('act'), l.get('scene'))].append(l)
    out = []
    for start in ev['scene_start']:
        key = (start.get('act'), start.get('scene'))
        lines = by_scene.get(key, [])
        out.append({'key': key, 'start': start, 'lines': lines, 'complete': key in ended,
                    'judgeable': len([l for l in lines if l.get('speaker')]) >= 3})
    return out


def scene_signals(ev, scene):
    """Measured properties of one scene, to check judge scores against."""
    key = scene['key']

    def of(kind):
        return [e for e in ev[kind] if (e.get('act'), e.get('scene')) == key]
    lines = scene['lines']
    grams = collections.Counter()
    spoken = [l for l in lines if l.get('line_kind') == 'speech']
    for l in spoken:
        words = re.findall(r"[a-z']+", l.get('text', '').lower())
        grams.update(zip(words, words[1:], words[2:]))
    acts = of('act')
    physical = [a for a in acts if a.get('mode') in ('Do', 'Move')]
    texts = [re.sub(r'\W+', ' ', a.get('action', '').lower()).strip() for a in acts]
    return {
        'lines': len(lines), 'speech_lines': len(spoken), 'acts': len(acts),
        'physical_share': round(len(physical) / len(acts), 3) if acts else None,
        'repeated_trigrams': round(sum(c - 1 for c in grams.values() if c > 1) / sum(grams.values()), 4) if grams else None,
        'repeated_acts': len(texts) - len(set(texts)),
        'hazard_events': len(of('hazard')),
        'affordances_applied': len([e for e in of('affordance') if e.get('applied')]),
        'max_need': max((e.get('level') or 0 for e in of('body')), default=None),
        'commitments': len(of('commitment_extracted')),
    }


# --------------------------------------------------------------------------- one judge over a run
def judge_scene(judge, scene, samples):
    """All SCENE_DIMS for one scene: {dim: mean, 'sd': {...}, 'validity', 'samples', 'evidence': {...}}."""
    start = scene['start']
    out, sds, evidence, validity, n = {}, {}, {}, [], []
    for dim, (question, a1, a3, a5) in SCENE_DIMS.items():
        user = render(SCENE_PROMPT, {'pre': start.get('pre_narrative') or '', 'planned_post': start.get('planned_post') or '',
                                     'script': script_lines(scene['lines'])[-40_000:], 'dim': dim,
                                     'question': question, 'a1': a1, 'a3': a3, 'a5': a5})
        r = sample(judge, CRITIC, user, ['score'], samples, f'judge_scene_{dim}', 'low', stat='mean', note='evidence')
        out[dim] = r['score']
        if 'score' in r['sd']:
            sds[dim] = r['sd']['score']
        evidence[dim] = r['evidence']
        n.append(r['samples'])
        if r['validity'] is not None:
            validity.append(r['validity'])
    out.update(sd=sds, evidence=evidence, samples=min(n),
               validity=round(statistics.fmean(validity), 3) if validity else None)
    return out


def run_judge(name, ev, cast, settings, samples, scene_samples, attribution_n):
    judge = make_judge(name)
    lines = ev['transcript_line']
    script = script_lines(lines)[-MAX_SCRIPT_CHARS:]
    result = {'judge': name, 'model': judge.model}

    # tier 4: scenes
    scenes = []
    for sc in scenes_of(ev):
        if not sc['judgeable']:
            continue
        entry = {'act': sc['key'][0], 'scene': sc['key'][1], 'title': sc['start'].get('title'),
                 'complete': sc['complete'], 'signals': scene_signals(ev, sc)}
        if sc['complete']:
            entry.update(judge_scene(judge, sc, scene_samples))
        scenes.append(entry)
    result['scenes'] = scenes

    # tier 5: per character (2025 rubrics) and per play
    characters = {}
    for cname, description in cast.items():
        if not any(l.get('speaker') == cname for l in lines):
            continue
        characters[cname] = {}
        for label, dim in CHARACTER_DIMS.items():
            system, user = rubric_2025(dim, {'scenario': settings, 'character_description': description,
                                             'name': cname, 'script': script})
            characters[cname][label] = sample(judge, system, user, ['score'], samples, f'judge_{label}', 'medium')
    result['characters'] = characters
    acts = '\n'.join(f"Act {e.get('number')}: {e.get('title')}\n  from: {e.get('planned_pre')}\n  to: {e.get('planned_post')}"
                     for e in ev['act_start']) or '(no act plan: an unscripted play)'
    result['play'] = sample(judge, CRITIC, render(PLAY_PROMPT, {'scenario': settings, 'acts': acts, 'script': script}),
                            PLAY_FIELDS, samples, 'judge_play', 'medium')

    # blind attribution
    speech = [l for l in lines if l.get('line_kind') == 'speech' and l.get('speaker') in cast]
    counts = collections.Counter(l['speaker'] for l in speech)
    pool = [l for l in speech if counts[l['speaker']] >= 3 and len(l['text']) > 20]
    picks = random.Random(0).sample(pool, min(attribution_n, len(pool))) if len(counts) >= 2 else []
    cast_text = '\n'.join(f'- {n}: {d[:400]}' for n, d in cast.items() if counts.get(n))
    hits = 0
    for l in picks:
        reply = judge.ask(CRITIC, render(ATTRIBUTION_PROMPT, {'cast': cast_text, 'line': l['text']}),
                          'judge_attribution', 'low', temp=0.2 if name == 'local' else None)
        guess = text_field(reply, 'speaker').lower()
        hits += bool(guess) and guess.split()[0].strip('.,') == l['speaker'].lower()
    result['attribution'] = {'lines': len(picks), 'accuracy': round(hits / len(picks), 3) if picks else None,
                             'chance': round(1 / len([n for n in counts if counts[n] >= 3]), 3) if picks else None}
    result['parse'] = {'calls': judge.calls, 'failed': judge.failed}
    if judge.cost() is not None:
        result['cost_usd'] = judge.cost()
        result['tokens'] = dict(judge.usage)
    return result


# --------------------------------------------------------------------------- report
def flat_scores(result):
    """(item, score) pairs for agreement between judges."""
    items = {}
    for s in result['scenes']:
        for f in SCENE_FIELDS:
            items[f"scene {s['act']}.{s['scene']} {f}"] = s.get(f)
    for c, dims in result['characters'].items():
        for d, r in dims.items():
            items[f'{c} {d}'] = r.get('score')
    for f in PLAY_FIELDS:
        items[f'play {f}'] = result['play'].get(f)
    return {k: v for k, v in items.items() if v is not None}


def agreement(a, b):
    from scipy.stats import spearmanr
    fa, fb = flat_scores(a), flat_scores(b)
    keys = sorted(set(fa) & set(fb))
    out = {'judges': [a['judge'], b['judge']], 'items': len(keys)}
    if len(keys) < 3:
        return out
    xs, ys = [fa[k] for k in keys], [fb[k] for k in keys]
    rho = spearmanr(xs, ys).statistic
    diffs = sorted(((abs(x - y), k, x, y) for k, x, y in zip(keys, xs, ys)), reverse=True)
    out.update({'spearman': None if rho != rho else round(float(rho), 3),
                'mean_abs_diff': round(statistics.fmean(d for d, *_ in diffs), 3),
                f"mean_{a['judge']}_minus_{b['judge']}": round(statistics.fmean(x - y for x, y in zip(xs, ys)), 3),
                'largest_disagreements': [{'item': k, a['judge']: x, b['judge']: y} for _, k, x, y in diffs[:5]]})
    return out


def markdown(report):
    out = [f"# Judges: {report['run']}", ""]
    for r in report['judges']:
        out += [f"## {r['judge']} ({r['model']})", ""]
        if r['scenes']:
            out += ["| scene | " + " | ".join(SCENE_FIELDS) + " | validity |", "|---" * (len(SCENE_FIELDS) + 2) + "|"]
            for s in r['scenes']:
                name = f"{s['act']}.{s['scene']} {s.get('title') or ''}"
                if not s.get('complete'):
                    out.append(f"| {name} | " + " | ".join(["cut short, not judged"] + [""] * (len(SCENE_FIELDS) - 1)) + " | |")
                else:
                    out.append(f"| {name} | " + " | ".join(str(s.get(f)) for f in SCENE_FIELDS) + f" | {s.get('validity')} |")
        dims = PLAY_RUBRIC_DIMS
        if r['characters']:
            out += ["", "| character | " + " | ".join(dims) + " |", "|---" * (len(dims) + 1) + "|"]
            out += [f"| {c} | " + " | ".join(str(v[d]['score']) for d in dims) + " |" for c, v in r['characters'].items()]
        p = r['play']
        out += ["", f"play: arc {p['arc']}, causality {p['causality']}, subtext {p['subtext']}, overall {p['overall']} "
                    f"(validity {p['validity']}): {p['justification']}",
                f"blind attribution: {r['attribution']}",
                f"replies without a usable score: {r['parse']['failed']} of {r['parse']['calls']} calls"
                + (f"; cost ${r['cost_usd']}" if 'cost_usd' in r else ''), ""]
    if report.get('agreement'):
        out += ["## Agreement", "", json.dumps(report['agreement'], indent=2)]
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('trace')
    p.add_argument('--judge', default='local', help="comma list of local, claude, fireworks ('both' = local,fireworks)")
    p.add_argument('--samples', type=int, default=3, help='asks per character and play rubric (median)')
    p.add_argument('--scene-samples', type=int, default=5, help='asks per scene dimension (mean)')
    p.add_argument('--attribution', type=int, default=16, help='lines for blind speaker attribution')
    p.add_argument('--out', default=str(ROOT / 'logs' / 'scores'))
    a = p.parse_args()
    events = load(a.trace)
    ev = collections.defaultdict(list, by_kind(events))
    loaded = (ev['play_loaded'] or [{}])[-1]
    cast = {c['name']: c.get('description') or '' for c in loaded.get('characters', [])}
    settings = loaded.get('setting') or ''
    report = {'run': events[0].get('run'), 'trace': a.trace,
              'judges': [run_judge(n, ev, cast, settings, a.samples, a.scene_samples, a.attribution)
                         for n in judge_names(a.judge)]}
    if len(report['judges']) > 1:
        report['agreement'] = [agreement(x, y) for x, y in itertools.combinations(report['judges'], 2)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{report['run']}.judges.json").write_text(json.dumps(report, indent=2, default=str))
    (out / f"{report['run']}.judges.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == '__main__':
    main()
