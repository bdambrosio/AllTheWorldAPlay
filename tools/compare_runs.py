#!/usr/bin/env python3
"""Compare ablation conditions from a batch (tools/batch_run.py).

    .venv/bin/python tools/compare_runs.py logs/batches/<name>.json [--pairwise local|fireworks]

Per metric and condition: n, mean and a 95% bootstrap interval. Against 'base': the
difference, a Mann-Whitney p-value, and Holm-corrected significance across the metrics of
that condition. --pairwise also asks a judge which transcript is better for base-vs-condition
pairs, both orders, and reports the condition's win rate. The frontier (mean tokens vs mean
judged quality per condition) answers "does a feature earn its tokens". Writes
logs/batches/<name>.compare.json, .compare.md and .frontier.png.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def get(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def mean_of(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(values) if values else None


# metric name -> (higher_is_better, extractor(scorecard, judges))
METRICS = {
    "tokens_total": (False, lambda c, j: (get(c, "cost", "tokens", "prompt") or 0) + (get(c, "cost", "tokens", "completion") or 0)),
    "llm_calls": (False, lambda c, j: get(c, "cost", "llm_calls")),
    "gpu_energy_wh": (False, lambda c, j: sum(g["energy_wh"] for g in (get(c, "cost", "hardware", "gpus") or {}).values()) or None),
    "tokens_per_speech_line": (False, lambda c, j: get(c, "cost", "per_unit", "tokens_per_speech_line")),
    "speech_lines": (True, lambda c, j: get(c, "integrity", "speech_lines")),
    "truncated_calls": (False, lambda c, j: get(c, "cost", "truncated_calls")),
    "llm_error_rate": (False, lambda c, j: get(c, "integrity", "llm_error_rate")),
    "duplicate_speech_lines": (False, lambda c, j: get(c, "integrity", "duplicate_speech_lines")),
    "task_satisfied_rate": (True, lambda c, j: get(c, "agency", "task_satisfied_rate")),
    "scene_plan_adherence": (True, lambda c, j: get(c, "agency", "scene_plan_adherence")),
    "scene_post_state_achieved": (True, lambda c, j: get(c, "agency", "scene_post_state_achieved")),
    "commitment_follow_through": (True, lambda c, j: get(c, "agency", "commitments", "follow_through_rate")),
    "stylometric_attribution": (True, lambda c, j: get(c, "character", "voice_distinctiveness", "stylometric_accuracy")),
    "scene_consistency": (True, lambda c, j: mean_of((get(c, "character", "scene_to_scene_consistency") or {}).values())),
    "repeated_trigrams": (False, lambda c, j: mean_of(r.get("repeated_trigram_fraction") for r in (get(c, "character", "repetition") or {}).values())),
    "theory_of_mind": (True, lambda c, j: get(c, "character", "theory_of_mind_similarity")),
}


def judge_metrics(j):
    """Judge scores per judge name: play dims, mean 2025 dims over characters, mean scene dims."""
    out = {}
    for r in (j or {}).get("judges", []):
        name = r["judge"]
        for f in ("overall", "arc", "causality", "subtext"):
            out[f"{name}:play_{f}"] = get(r, "play", f)
        dims = collections.defaultdict(list)
        for c in (r.get("characters") or {}).values():
            for d, v in c.items():
                dims[d].append(v.get("score"))
        for d, vs in dims.items():
            out[f"{name}:{d}"] = mean_of(vs)
        for f in ("yes_and", "tension", "discovery", "pacing", "purpose"):
            out[f"{name}:scene_{f}"] = mean_of(s.get(f) for s in r.get("scenes", []))
        out[f"{name}:attribution"] = get(r, "attribution", "accuracy")
    return out


def bootstrap_ci(xs, n=2000, seed=0):
    if len(xs) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, size=(n, len(xs)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def holm(pvalues):
    """Holm step-down: {name: significant at 0.05}."""
    order = sorted((p, k) for k, p in pvalues.items() if p is not None)
    m, result, still = len(order), {}, True
    for i, (p, k) in enumerate(order):
        still = still and p <= 0.05 / (m - i)
        result[k] = still
    return result


PAIRWISE = """Two improvised performances of the same play follow, A and B.

#A
{{$a}}
##

#B
{{$b}}
##

Which is the better piece of theatre: more engaging, coherent, character-driven and dramatically alive?
Respond with this hash-formatted text only:
#better A or B
#justification 6-12 words
<end/>
"""


def pairwise(manifest, judge, pairs_per_condition=4):
    from judges import make_judge, script_lines, CRITIC
    from score_play import load, by_kind
    from utils import hash_utils
    from utils.Messages import UserMessage
    llm, temp = make_judge(judge)
    scripts = {}
    for r in manifest["runs"]:
        if r.get("trace"):
            ev = collections.defaultdict(list, by_kind(load(r["trace"])))
            scripts[r["label"]] = script_lines(ev["transcript_line"])[-50_000:]
    base = [r["label"] for r in manifest["runs"] if r["condition"] == "base" and r["label"] in scripts]
    out = {}
    for cond in sorted({r["condition"] for r in manifest["runs"]} - {"base"}):
        others = [r["label"] for r in manifest["runs"] if r["condition"] == cond and r["label"] in scripts]
        pairs = [(b, o) for b in base for o in others]
        random.Random(0).shuffle(pairs)
        wins = trials = 0
        for b, o in pairs[:pairs_per_condition]:
            for first, second, cond_is in ((b, o, "B"), (o, b, "A")):   # both orders
                reply = llm.ask({"a": scripts[first], "b": scripts[second]}, [CRITIC, UserMessage(content=PAIRWISE)],
                                tag="judge_pairwise", temp=temp, max_tokens=2000, stops=["<end/>"])
                pick = (hash_utils.find("better", reply or "") or "").strip().upper()[:1]
                if pick in ("A", "B"):
                    trials += 1
                    wins += pick == cond_is
        out[cond] = {"judge": judge, "trials": trials, "condition_win_rate": round(wins / trials, 3) if trials else None}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--pairwise", choices=["local", "fireworks"])
    p.add_argument("--quality", default="local:play_overall", help="judge metric for the frontier")
    a = p.parse_args()
    from scipy.stats import mannwhitneyu
    manifest = json.loads(Path(a.manifest).read_text())

    rows = collections.defaultdict(lambda: collections.defaultdict(list))   # condition -> metric -> values
    higher_better = {k: hb for k, (hb, _) in METRICS.items()}
    for r in manifest["runs"]:
        card = json.loads(Path(r["scorecard"]).read_text()) if r.get("scorecard") and Path(r["scorecard"]).exists() else None
        jd = json.loads(Path(r["judges"]).read_text()) if r.get("judges") and Path(r["judges"]).exists() else None
        if not card:
            continue
        for name, (_, fn) in METRICS.items():
            v = fn(card, jd)
            if isinstance(v, (int, float)):
                rows[r["condition"]][name].append(float(v))
        for name, v in judge_metrics(jd).items():
            higher_better.setdefault(name, True)
            if isinstance(v, (int, float)):
                rows[r["condition"]][name].append(float(v))

    result = {"name": manifest["name"], "conditions": {}}
    base = rows.get("base", {})
    for cond, metrics in rows.items():
        entry, pvals = {}, {}
        for m, xs in sorted(metrics.items()):
            e = {"n": len(xs), "mean": round(statistics.fmean(xs), 4), "ci95": bootstrap_ci(xs)}
            if cond != "base" and len(base.get(m, [])) >= 2 and len(xs) >= 2:
                e["diff_vs_base"] = round(e["mean"] - statistics.fmean(base[m]), 4)
                try:
                    pvals[m] = float(mannwhitneyu(xs, base[m], alternative="two-sided").pvalue)
                except ValueError:
                    pvals[m] = None
                e["p"] = None if pvals[m] is None else round(pvals[m], 4)
                e["direction"] = ("better" if (e["diff_vs_base"] > 0) == higher_better.get(m, True) else "worse") \
                    if e["diff_vs_base"] else "same"
            entry[m] = e
        for m, sig in holm(pvals).items():
            entry[m]["holm_significant"] = sig
        result["conditions"][cond] = entry
    if a.pairwise:
        result["pairwise"] = pairwise(manifest, a.pairwise)

    # frontier: tokens vs judged quality, per condition
    frontier = {}
    for cond, metrics in rows.items():
        frontier[cond] = {"tokens": mean_of(metrics.get("tokens_total", [])), "quality": mean_of(metrics.get(a.quality, []))}
    result["frontier"] = {"quality_metric": a.quality, "points": frontier}

    stem = Path(a.manifest).with_suffix("")
    Path(f"{stem}.compare.json").write_text(json.dumps(result, indent=2))
    lines = [f"# Comparison: {manifest['name']}", ""]
    conds = sorted(result["conditions"], key=lambda c: (c != "base", c))
    metrics = sorted({m for c in conds for m in result["conditions"][c]})
    lines += ["| metric | " + " | ".join(conds) + " |", "|---" * (len(conds) + 1) + "|"]
    for m in metrics:
        cells = []
        for c in conds:
            e = result["conditions"][c].get(m)
            if not e:
                cells.append("")
                continue
            cell = f"{e['mean']:.3g}"
            if "diff_vs_base" in e:
                cell += f" ({e['diff_vs_base']:+.3g}{' *' if e.get('holm_significant') else ''})"
            cells.append(cell)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "\\* Holm-significant at 0.05 vs base. Runs per condition: " +
              ", ".join(f"{c}={max((e['n'] for e in result['conditions'][c].values()), default=0)}" for c in conds)]
    if result.get("pairwise"):
        lines += ["", "## Pairwise preference vs base", "", json.dumps(result["pairwise"], indent=2)]
    lines += ["", f"## Frontier ({a.quality} vs tokens)", ""] + \
             [f"- {c}: {v['tokens']:.0f} tokens, quality {v['quality']}" if v["tokens"] else f"- {c}: n/a" for c, v in frontier.items()]
    Path(f"{stem}.compare.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for c, v in frontier.items():
            if v["tokens"] and v["quality"] is not None:
                ax.scatter(v["tokens"] / 1e6, v["quality"])
                ax.annotate(c, (v["tokens"] / 1e6, v["quality"]), textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("tokens per run (millions)")
        ax.set_ylabel(a.quality)
        ax.set_title(f"{manifest['name']}: quality vs cost")
        fig.tight_layout()
        fig.savefig(f"{stem}.frontier.png", dpi=120)
    except Exception as e:
        print(f"(no frontier plot: {e})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
