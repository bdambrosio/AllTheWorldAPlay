#!/usr/bin/env python3
"""Check the scene judges (tools/judges.py) against a person's blind scores.

    .venv/bin/python tools/calibrate_judges.py sample [--n 20] [--traces <dir>]
        picks N finished scenes from stored traces and writes, in logs/calibration/:
          scenes.md    the scenes, numbered S01.., with no run names
          scores.csv   one row per scene to fill in: tension, pacing, engagement (1-5),
                       using the anchors printed at the top of scenes.md
          key.json     which trace and scene each number is

    set -a; . ~/.config/secrets.env; set +a
    .venv/bin/python tools/calibrate_judges.py compare [--judge local,claude] [--scene-samples 5]
        judges the same scenes (cached in logs/calibration/judge-<name>.json) and prints, per
        judge and dimension, Spearman and mean |difference| against scores.csv.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from judges import SCENE_DIMS, judge_names, judge_scene, make_judge, scenes_of, script_lines   # noqa: E402
from score_play import load, by_kind   # noqa: E402

DIMS = ["tension", "pacing", "engagement"]
OUT = ROOT / "logs" / "calibration"
TRACES = Path.home() / ".local/share/alltheworldaplay/logs/traces"


def scene_of(entry):
    ev = collections.defaultdict(list, by_kind(load(entry["trace"])))
    return next(sc for sc in scenes_of(ev) if list(sc["key"]) == entry["key"])


def sample(a):
    pool = []
    for trace in sorted(Path(a.traces).glob("*.jsonl")):
        ev = collections.defaultdict(list, by_kind(load(str(trace))))
        pool += [(str(trace), sc) for sc in scenes_of(ev)
                 if sc["complete"] and sc["judgeable"] and len(script_lines(sc["lines"])) < 12_000]
    picks = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Scenes to score", "", "Score each scene 1-5 (halves allowed) in scores.csv.", ""]
    for d in DIMS:
        q, a1, a3, a5 = SCENE_DIMS[d]
        md += [f"**{d}**: {q}.", f"- 1 = {a1}", f"- 3 = {a3}", f"- 5 = {a5}", ""]
    key = {}
    for i, (trace, sc) in enumerate(picks, 1):
        sid = f"S{i:02d}"
        key[sid] = {"trace": trace, "key": list(sc["key"])}
        md += [f"## {sid}", "", f"Planned to start from: {sc['start'].get('pre_narrative') or ''}",
               f"Planned to arrive at: {sc['start'].get('planned_post') or ''}", "", "```",
               script_lines(sc["lines"]), "```", ""]
    (OUT / "scenes.md").write_text("\n".join(md))
    (OUT / "key.json").write_text(json.dumps(key, indent=2))
    with open(OUT / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene"] + DIMS)
        w.writerows([sid, "", "", ""] for sid in key)
    print(f"wrote {len(key)} scenes to {OUT}/scenes.md; fill in {OUT}/scores.csv")


def compare(a):
    from scipy.stats import spearmanr
    key = json.loads((OUT / "key.json").read_text())
    human = {}
    with open(OUT / "scores.csv") as f:
        for row in csv.DictReader(f):
            vals = {d: float(row[d]) for d in DIMS if (row.get(d) or "").strip()}
            if vals:
                human[row["scene"]] = vals
    if len(human) < 5:
        raise SystemExit(f"only {len(human)} scored scenes in {OUT}/scores.csv; score more first")
    for name in judge_names(a.judge):
        cache_path = OUT / f"judge-{name}.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        judge = None
        for sid in human:
            if sid not in cache:
                judge = judge or make_judge(name)
                cache[sid] = judge_scene(judge, scene_of(key[sid]), a.scene_samples)
                cache_path.write_text(json.dumps(cache, indent=2))
        print(f"\n{name}" + (f" (cost ${judge.cost()})" if judge and judge.cost() is not None else ""))
        for d in DIMS:
            pairs = [(human[s][d], cache[s][d]) for s in human if d in human[s] and cache[s].get(d) is not None]
            if len(pairs) < 5:
                continue
            xs, ys = zip(*pairs)
            rho = spearmanr(xs, ys).statistic
            print(f"  {d:11} n={len(pairs):2}  spearman {rho:+.2f}  mean |diff| {statistics.fmean(abs(x - y) for x, y in pairs):.2f}"
                  f"  judge minus you {statistics.fmean(y - x for x, y in pairs):+.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--traces", default=str(TRACES))
    c = sub.add_parser("compare")
    c.add_argument("--judge", default="local,claude")
    c.add_argument("--scene-samples", type=int, default=5)
    a = p.parse_args()
    sample(a) if a.cmd == "sample" else compare(a)


if __name__ == "__main__":
    main()
