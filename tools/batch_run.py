#!/usr/bin/env python3
"""Run a play several times under each ablation condition, then score and judge each run.

    set -a; . ~/.config/secrets.env; set +a     # only needed with --judge fireworks/both
    .venv/bin/python tools/batch_run.py --name stylizer-test --play lost.py \\
        --conditions base,stylizer --repeats 3 --run-secs 600 [--judge local]

A condition is 'base' (everything on) or a '+'-joined list of features to disable
(src/sim/ablation.py: stylizer, commitments, stance, drive_checks), e.g. 'stylizer+stance'.
Each run: tools/smoke_run.py (with --hw), tools/score_play.py, then tools/judges.py.
Writes logs/batches/<name>.json, the input to tools/compare_runs.py. Runs are sequential:
the engine serves one play at a time. Needs the stack up (./start-local.sh).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
GPU_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"}  # keep off vLLM's GPU


def run(cmd, log):
    import os
    env = {**os.environ, **GPU_ENV}
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--play", default="lost.py")
    p.add_argument("--conditions", default="base")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--run-secs", type=float, default=600)
    p.add_argument("--judge", choices=["none", "local", "fireworks", "both"], default="local")
    p.add_argument("--samples", type=int, default=3)
    a = p.parse_args()

    out_dir = ROOT / "logs" / "batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{a.name}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else \
        {"name": a.name, "play": a.play, "runs": []}
    log = out_dir / f"{a.name}.log"
    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    for i in range(a.repeats):           # interleave conditions so drift over time hits all alike
        for cond in conditions:
            label = f"{a.name}-{cond}-{int(time.time())}"
            cmd = [PY, "tools/smoke_run.py", "--play", a.play, "--steps", "1", "--run-secs", str(a.run_secs),
                   "--hw", "--label", label]
            if cond != "base":
                cmd += ["--ablate", cond.replace("+", ",")]
            print(f"[{i + 1}/{a.repeats}] {cond}: {label}", flush=True)
            code = run(cmd, log)
            smoke_path = ROOT / "logs" / f"smoke-{label}.json"
            entry = {"condition": cond, "repeat": i, "label": label, "smoke": str(smoke_path), "smoke_exit": code}
            if smoke_path.exists():
                trace = json.loads(smoke_path.read_text()).get("trace")
                entry["trace"] = trace
                if trace:
                    run([PY, "tools/score_play.py", "--smoke", str(smoke_path)], log)
                    run_id = Path(trace).stem
                    entry["scorecard"] = str(ROOT / "logs" / "scores" / f"{run_id}.json")
                    if a.judge != "none":
                        run([PY, "tools/judges.py", trace, "--judge", a.judge, "--samples", str(a.samples)], log)
                        entry["judges"] = str(ROOT / "logs" / "scores" / f"{run_id}.judges.json")
            manifest["runs"].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"    -> {entry.get('scorecard', 'no scorecard')} (smoke exit {code})", flush=True)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
