#!/usr/bin/env python3
"""Compare ablation conditions from a batch (tools/batch_run.py).

    .venv/bin/python tools/compare_runs.py logs/batches/<name>.json [--pairwise local|claude|fireworks]

Per metric and condition: n, mean and a 95% bootstrap interval. Against 'base': the
difference, a Mann-Whitney p-value, and Holm-corrected significance across the metrics of
that condition. --pairwise also asks a judge to compare base and condition runs scene by scene
(same act and scene number), one question each for tension, pacing and engagement, both orders,
and reports the condition's win rate per dimension. 'Judge validity' correlates each judge's
scene scores with measured signals of the same scenes. The frontier (mean tokens vs mean
judged quality per condition) answers "does a feature earn its tokens". Writes
logs/batches/<name>.compare.json, .compare.md and .frontier.png.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from judges import (CACHE_BREAK, CRITIC, SCENE_FIELDS, make_judge, render, scenes_of,   # noqa: E402
                    script_lines)
from score_play import load, by_kind   # noqa: E402


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
    "physical_share_of_acts": (True, lambda c, j: get(c, "agency", "physical", "physical_share_of_acts")),
    "speech_share_of_acts": (False, lambda c, j: get(c, "agency", "physical", "speech_share_of_acts")),
    "concrete_action_rate": (True, lambda c, j: get(c, "agency", "physical", "concrete_action_rate")),
}


def judge_metrics(j):
    """Judge scores per judge name: play dims, mean 2025 dims over characters, mean scene dims
    (scenes the run's timer cut short are not judged and not counted)."""
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
        for f in SCENE_FIELDS:
            out[f"{name}:scene_{f}"] = mean_of(s.get(f) for s in r.get("scenes", []) if s.get("complete", True))
        out[f"{name}:attribution"] = get(r, "attribution", "accuracy")
    return out


# judge dimension -> measured scene signals it should move with (sign expected)
VALIDITY_PAIRS = [("pacing", "repeated_trigrams", "-"), ("pacing", "repeated_acts", "-"),
                  ("tension", "hazard_events", "+"), ("tension", "max_need", "+"),
                  ("engagement", "repeated_trigrams", "-"), ("discovery", "affordances_applied", "+")]


def judge_validity(manifest):
    """Spearman, over every judged scene of the manifest's runs, between a judge's scene score
    and a measured signal of the same scene (judges.py scene_signals)."""
    from scipy.stats import spearmanr
    rows = collections.defaultdict(list)     # judge -> scenes
    for r in manifest["runs"]:
        jd = json.loads(Path(r["judges"]).read_text()) if r.get("judges") and Path(r["judges"]).exists() else None
        for res in (jd or {}).get("judges", []):
            rows[res["judge"]] += [s for s in res.get("scenes", []) if s.get("complete") and s.get("signals")]
    out = {}
    for judge, scenes in rows.items():
        entry = {"scenes": len(scenes)}
        for dim, signal, sign in VALIDITY_PAIRS:
            pairs = [(s[dim], s["signals"][signal]) for s in scenes
                     if s.get(dim) is not None and s["signals"].get(signal) is not None]
            if len(pairs) >= 6 and len({y for _, y in pairs}) > 1 and len({x for x, _ in pairs}) > 1:
                rho = spearmanr([x for x, _ in pairs], [y for _, y in pairs])
                entry[f"{dim}~{signal}"] = {"expected": sign, "spearman": round(float(rho.statistic), 3),
                                            "p": round(float(rho.pvalue), 4), "n": len(pairs)}
        out[judge] = entry
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


PAIRWISE_DIMS = {
    "tension": "has more at stake, with pressure that rises or turns during the scene",
    "pacing": "keeps moving, with less stalling, looping and repetition of beats",
    "engagement": "would hold an audience better and leave them wanting the next scene",
}

PAIRWISE = """Two performances of the same point in the same play follow, scene A and scene B.

#A
{{$a}}
##

#B
{{$b}}
##
""" + CACHE_BREAK + """Compare them on one quality only, {{$dim}}: which scene {{$question}}?
Other differences must not decide the answer.

First cite the evidence, then answer. Respond with exactly these two lines, each starting with '#':
#evidence up to 30 words: what decides it
#better A or B
<end/>
"""


def picked(reply):
    """'A' or 'B' from a pairwise reply: '#better B', 'Better: A', or a last line that is just '#B'."""
    found = re.findall(r"(?im)^\W*(?:better\b\W*)?(?:scene\s+)?([AB])\W*$", reply or "")
    return found[-1].upper() if found else ""


def pairwise(manifest, judge_name, pairs_per_condition=4):
    """Base run vs condition run, scene by scene (same act and scene number, both played to
    their end), one question per dimension, each pair asked in both orders."""
    from scipy.stats import binomtest
    judge = make_judge(judge_name)
    scenes = {}
    for r in manifest["runs"]:
        if r.get("trace") and Path(r["trace"]).exists():
            ev = collections.defaultdict(list, by_kind(load(r["trace"])))
            scenes[r["label"]] = {sc["key"]: script_lines(sc["lines"])[-20_000:] for sc in scenes_of(ev)
                                  if sc["complete"] and sc["judgeable"]}
    base = [r["label"] for r in manifest["runs"] if r["condition"] == "base" and r["label"] in scenes]
    out = {}
    for cond in sorted({r["condition"] for r in manifest["runs"]} - {"base"}):
        others = [r["label"] for r in manifest["runs"] if r["condition"] == cond and r["label"] in scenes]
        pairs = [(b, o) for b in base for o in others]
        random.Random(0).shuffle(pairs)
        tally = {d: [0, 0] for d in PAIRWISE_DIMS}     # dim -> [condition wins, trials]
        matched = 0
        for b, o in pairs[:pairs_per_condition]:
            for key in sorted(set(scenes[b]) & set(scenes[o])):
                matched += 1
                for first, second, cond_is in ((scenes[b][key], scenes[o][key], "B"), (scenes[o][key], scenes[b][key], "A")):
                    for dim, question in PAIRWISE_DIMS.items():
                        reply = judge.ask(CRITIC, render(PAIRWISE, {"a": first, "b": second, "dim": dim, "question": question}),
                                          "judge_pairwise", "low")
                        pick = picked(reply)
                        if pick in ("A", "B"):
                            tally[dim][1] += 1
                            tally[dim][0] += pick == cond_is
                        else:
                            judge.failed += 1
        entry = {"judge": judge_name, "model": judge.model, "matched_scenes": matched}
        for dim, (wins, trials) in tally.items():
            entry[dim] = {"condition_win_rate": round(wins / trials, 3) if trials else None, "trials": trials,
                          "p_vs_even": round(float(binomtest(wins, trials).pvalue), 4) if trials else None}
        out[cond] = entry
    out["_judge"] = {"calls": judge.calls, "failed": judge.failed, "cost_usd": judge.cost()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--pairwise", choices=["local", "claude", "fireworks"])
    p.add_argument("--quality", default="local:play_overall", help="judge metric for the frontier")
    p.add_argument("--max-llm-errors", type=float, default=0.05,
                   help="exclude runs whose LLM error rate exceeds this (server outages)")
    a = p.parse_args()
    from scipy.stats import mannwhitneyu
    manifest = json.loads(Path(a.manifest).read_text())

    rows = collections.defaultdict(lambda: collections.defaultdict(list))   # condition -> metric -> values
    excluded = []   # runs whose LLM server failed part-way (vLLM crashes)
    higher_better = {k: hb for k, (hb, _) in METRICS.items()}
    for r in manifest["runs"]:
        card = json.loads(Path(r["scorecard"]).read_text()) if r.get("scorecard") and Path(r["scorecard"]).exists() else None
        jd = json.loads(Path(r["judges"]).read_text()) if r.get("judges") and Path(r["judges"]).exists() else None
        if not card:
            continue
        err = get(card, "integrity", "llm_error_rate") or 0
        if err > a.max_llm_errors:
            excluded.append({"label": r["label"], "condition": r["condition"], "llm_error_rate": err})
            continue
        for name, (_, fn) in METRICS.items():
            v = fn(card, jd)
            if isinstance(v, (int, float)):
                rows[r["condition"]][name].append(float(v))
        for name, v in judge_metrics(jd).items():
            higher_better.setdefault(name, True)
            if isinstance(v, (int, float)):
                rows[r["condition"]][name].append(float(v))

    result = {"name": manifest["name"], "conditions": {}, "excluded_runs": excluded}
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
    result["judge_validity"] = judge_validity(manifest)

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
    if excluded:
        lines += ["", f"Excluded (LLM error rate > {a.max_llm_errors}): " +
                  ", ".join(f"{e['label']} ({e['llm_error_rate']:.0%})" for e in excluded)]
    lines += ["", "\\* Holm-significant at 0.05 vs base. Runs per condition: " +
              ", ".join(f"{c}={max((e['n'] for e in result['conditions'][c].values()), default=0)}" for c in conds)]
    if result.get("pairwise"):
        lines += ["", "## Pairwise preference vs base", "", json.dumps(result["pairwise"], indent=2)]
    if any(len(v) > 1 for v in result["judge_validity"].values()):
        lines += ["", "## Judge validity (scene score vs measured signal, Spearman over scenes)", "",
                  "| judge | pair | expected | rho | p | scenes |", "|---|---|---|---|---|---|"]
        for judge, entry in result["judge_validity"].items():
            lines += [f"| {judge} | {k} | {v['expected']} | {v['spearman']} | {v['p']} | {v['n']} |"
                      for k, v in entry.items() if isinstance(v, dict)]
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
