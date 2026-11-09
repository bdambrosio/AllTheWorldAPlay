#!/usr/bin/env python3
"""Headless smoke run of a play against a running ATWAP stack (./start-local.sh).

Drives the backend over the same websocket protocol the React UI uses:
initialize -> toggle speech off -> load_play -> step x N, waiting for the engine
to go quiet (no websocket traffic and no backend.log growth) after each command.
Then summarises the backend.log slice written during the run: tracebacks,
error-ish lines, embedding-model loads, and per-tag LLM call counts/timings.

    .venv/bin/python tools/smoke_run.py --play lost.py --steps 3 --label baseline

Writes logs/smoke-<label>.json. Exit code 1 if the run saw tracebacks or play errors.
Note: the engine is a single process shared by all sessions, so this replaces
whatever play is loaded in an open browser tab.
"""
import argparse
import asyncio
import collections
import json
import re
import sys
import time
from pathlib import Path

import requests
import websockets

ROOT = Path(__file__).resolve().parent.parent
TAG_LINE = re.compile(r"^([A-Za-z_][\w.]*)\.\.\.([0-9]+\.[0-9]+)\s*$")
ERRORISH = re.compile(r"error|exception|failed|not found|invalid|unbound|cannot|could not", re.I)
NOISE = re.compile(r"elevenlabs|text-to-speech|^\s+(File|raise|~|\^)|Loading weights", re.I)


class Run:
    def __init__(self, args):
        self.args = args
        self.log = ROOT / args.log
        self.msgs = []                 # (t, type, short)
        self.last_msg = time.time()
        self.problems = []             # play_error / error messages from the engine

    def log_size(self):
        try:
            return self.log.stat().st_size
        except FileNotFoundError:
            return 0

    async def reader(self, ws):
        async for raw in ws:
            try:
                m = json.loads(raw)
            except ValueError:
                continue
            t = m.get("type", "?")
            self.last_msg = time.time()
            self.msgs.append((round(self.last_msg - self.t0, 1), t, m.get("command") or m.get("error") or ""))
            if t in ("play_error", "error"):
                self.problems.append(m)
                print(f"  !! {t}: {str(m.get('error'))[:200]}")

    async def quiesce(self, label, min_secs=3):
        """Wait until neither the websocket nor backend.log has changed for quiet_secs."""
        start = time.time()
        size, size_t = self.log_size(), time.time()
        await asyncio.sleep(min_secs)
        while True:
            now = time.time()
            s = self.log_size()
            if s != size:
                size, size_t = s, now
            quiet_for = now - max(size_t, self.last_msg)
            if quiet_for >= self.args.quiet_secs:
                break
            if now - start > self.args.max_step_secs:
                print(f"  {label}: still busy after {self.args.max_step_secs}s, moving on")
                self.problems.append({"type": "timeout", "error": f"{label} did not go quiet"})
                break
            await asyncio.sleep(1)
        took = time.time() - start - self.args.quiet_secs
        print(f"  {label}: done in ~{max(took, 0):.0f}s")
        return took

    async def main(self):
        a = self.args
        self.log_offset = self.log_size()
        self.t0 = time.time()
        sid = requests.get(f"{a.backend}/api/session/new", timeout=10).json()["session_id"]
        ws_url = a.backend.replace("http", "ws", 1) + f"/ws/{sid}"
        timings = {}
        async with websockets.connect(ws_url, max_size=None) as ws:
            reader = asyncio.create_task(self.reader(ws))

            async def cmd(action, **payload):
                await ws.send(json.dumps({"type": "command", "action": action, **payload}))

            print(f"session {sid}: initialize")
            await cmd("initialize")
            await asyncio.sleep(2)
            if not a.speech:
                await cmd("toggle_speech")   # engine default is speech on; toggle flips it
            print(f"load_play {a.play}")
            await cmd("load_play", play=a.play)
            timings["load"] = await self.quiesce("load_play", min_secs=10)
            for i in range(1, a.steps + 1):
                await cmd("step")
                timings[f"step{i}"] = await self.quiesce(f"step {i}")
            reader.cancel()
        self.timings = timings

    def summarise(self):
        with open(self.log, "rb") as f:
            f.seek(self.log_offset)
            text = f.read().decode("utf-8", "replace").replace("\x00", "")
        lines = text.splitlines()
        tags = collections.defaultdict(lambda: [0, 0.0])
        for ln in lines:
            m = TAG_LINE.match(ln.strip())
            if m:
                tags[m.group(1)][0] += 1
                tags[m.group(1)][1] += float(m.group(2))
        tracebacks = []
        for i, ln in enumerate(lines):
            if ln.startswith("Traceback"):
                # last line of the traceback block names the exception
                j = i + 1
                while j < len(lines) and (lines[j].startswith(" ") or not lines[j].strip()):
                    j += 1
                tracebacks.append(lines[j][:200] if j < len(lines) else "?")
        errorish = collections.Counter(
            re.sub(r"\d+(\.\d+)?", "N", ln.strip())[:140]
            for ln in lines if ERRORISH.search(ln) and not NOISE.search(ln) and not ln.startswith("Traceback"))
        summary = {
            "label": self.args.label, "play": self.args.play, "steps": self.args.steps,
            "wall_secs": round(time.time() - self.t0),
            "phase_secs": {k: round(v) for k, v in self.timings.items()},
            "llm_calls": sum(n for n, _ in tags.values()),
            "llm_secs": round(sum(s for _, s in tags.values()), 1),
            "by_tag": {k: {"calls": n, "secs": round(s, 1)}
                       for k, (n, s) in sorted(tags.items(), key=lambda kv: -kv[1][1])},
            "embedding_model_loads": sum(1 for ln in text.split("\r") if "Loading weights" in ln and "100%" in ln),
            "tracebacks": collections.Counter(tracebacks).most_common(),
            "errorish_lines": errorish.most_common(40),
            "engine_problems": self.problems,
            "ws_message_types": collections.Counter(t for _, t, _ in self.msgs).most_common(),
        }
        out = ROOT / "logs" / f"smoke-{self.args.label}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\n== {self.args.label}: {summary['wall_secs']}s wall, {summary['llm_calls']} LLM calls "
              f"({summary['llm_secs']}s), {len(tracebacks)} tracebacks, {len(self.problems)} engine problems")
        for exc, n in summary["tracebacks"]:
            print(f"   {n}x {exc}")
        print(f"   wrote {out.relative_to(ROOT)}")
        return 1 if (tracebacks or self.problems) else 0


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--play", default="lost.py")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--label", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--backend", default="http://localhost:8010")
    p.add_argument("--log", default="logs/backend.log", help="backend stdout log, relative to repo root")
    p.add_argument("--quiet-secs", type=float, default=15, help="idle time that counts as 'done'")
    p.add_argument("--max-step-secs", type=float, default=900)
    p.add_argument("--speech", action="store_true", help="leave TTS on (default: toggle it off)")
    return p.parse_args()


if __name__ == "__main__":
    r = Run(parse())
    asyncio.run(r.main())
    sys.exit(r.summarise())
