#!/usr/bin/env python3
"""Headless smoke run of a play against a running ATWAP stack (./start-local.sh).

Drives the backend over the same websocket protocol the React UI uses:
initialize -> toggle speech off -> load_play -> step x N, waiting for the engine
to go quiet (no websocket traffic and no backend.log growth) after each command.
Then summarises the backend.log slice written during the run: tracebacks,
error-ish lines, embedding-model loads, and per-tag LLM call counts/timings.

    .venv/bin/python tools/smoke_run.py --play lost.py --steps 3 --label baseline

Writes logs/smoke-<label>.json. Exit code 1 if the run saw tracebacks or play errors.
--hw also samples nvidia-smi every 5 s and takes vLLM /metrics deltas over the run; the
summary names the run's trace file (sim/trace.py) for tools/score_play.py.
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
TRACE_DIR = Path.home() / ".local/share/alltheworldaplay/logs/traces"
VLLM_COUNTERS = ("prompt_tokens_total", "generation_tokens_total", "prefix_cache_queries_total",
                 "prefix_cache_hits_total", "spec_decode_num_draft_tokens_total",
                 "spec_decode_num_accepted_tokens_total", "request_success_total")


def vllm_counters(url):
    """vLLM /metrics counters summed over labels, keyed by name (request_success by finish_reason)."""
    out = collections.Counter()
    try:
        text = requests.get(f"{url}/metrics", timeout=5).text
    except requests.RequestException:
        return {}
    for ln in text.splitlines():
        if not ln.startswith("vllm:"):
            continue
        m = re.match(r"vllm:(\w+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", ln)
        if not m or m.group(1) not in VLLM_COUNTERS:
            continue
        key = m.group(1)
        if key == "request_success_total":
            reason = re.search(r'finished_reason="(\w+)"', m.group(2) or "")
            key = f"finished_{reason.group(1) if reason else 'unknown'}"
        out[key] += float(m.group(3))
    return dict(out)


async def gpu_sampler(samples, interval=5.0):
    """Append one nvidia-smi reading per GPU every interval seconds until cancelled."""
    query = "index,name,utilization.gpu,memory.used,power.draw"
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out, _ = await proc.communicate()
            t = time.time()
            for row in out.decode().strip().splitlines():
                idx, name, util, mem, power = [c.strip() for c in row.split(",")]
                samples.append({"t": round(t, 1), "gpu": int(idx), "name": name, "util": float(util),
                                "mem_mib": float(mem), "power_w": float(power)})
        except (OSError, ValueError):
            pass
        await asyncio.sleep(interval)


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
        self.gpu_samples, sampler = [], None
        if a.hw:
            self.vllm_start = vllm_counters(a.vllm)
            sampler = asyncio.create_task(gpu_sampler(self.gpu_samples))
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
            await cmd("load_play", play=a.play, **({"ablate": a.ablate} if a.ablate else {}),
                      **({"reasoning": a.reasoning} if a.reasoning else {}))
            timings["load"] = await self.quiesce("load_play", min_secs=10)
            for i in range(1, a.steps + 1):
                await cmd("step")
                timings[f"step{i}"] = await self.quiesce(f"step {i}")
            if a.run_secs:
                acks_before = sum(1 for _, t, c in self.msgs if t == "command_ack" and c == "step")
                print(f"run for {a.run_secs:.0f}s")
                await cmd("run")
                await asyncio.sleep(a.run_secs)
                await cmd("pause")
                timings["run"] = await self.quiesce("pause")
                self.step_acks_during_run = sum(
                    1 for _, t, c in self.msgs if t == "command_ack" and c == "step") - acks_before
                print(f"  step acks while running: {self.step_acks_during_run}")
            if a.reload_play:
                print(f"load_play {a.reload_play} (second play in same engine)")
                await cmd("load_play", play=a.reload_play)
                timings["reload"] = await self.quiesce("reload", min_secs=10)
                await cmd("step")
                timings["reload_step"] = await self.quiesce("reload step")
            reader.cancel()
        if sampler:
            sampler.cancel()
            self.vllm_end = vllm_counters(a.vllm)
        self.timings = timings

    def trace_path(self):
        """The trace file this run wrote: the newest one started after the run began."""
        traces = [p for p in TRACE_DIR.glob("*.jsonl") if p.stat().st_mtime >= self.t0] if TRACE_DIR.exists() else []
        return str(max(traces, key=lambda p: p.stat().st_mtime)) if traces else None

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
            "step_acks_during_run": getattr(self, "step_acks_during_run", None),
            "ws_message_types": collections.Counter(t for _, t, _ in self.msgs).most_common(),
            "trace": self.trace_path(),
        }
        if self.args.hw:
            start, end = getattr(self, "vllm_start", {}), getattr(self, "vllm_end", {})
            summary["hw"] = {"gpu_samples": self.gpu_samples,
                             "vllm_delta": {k: end[k] - start.get(k, 0) for k in end}}
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
    p.add_argument("--run-secs", type=float, default=0, help="after stepping, 'run' for this long then 'pause'")
    p.add_argument("--reload-play", help="then load this second play in the same engine and step once")
    p.add_argument("--hw", action="store_true", help="sample GPUs and vLLM /metrics during the run")
    p.add_argument("--ablate", help="comma-separated features to disable (src/sim/ablation.py)")
    p.add_argument("--reasoning", help="calls that think first, e.g. 'page=medium' (src/utils/llm_api.py set_reasoning)")
    p.add_argument("--vllm", default="http://localhost:5000", help="vLLM server for --hw")
    return p.parse_args()


if __name__ == "__main__":
    r = Run(parse())
    asyncio.run(r.main())
    sys.exit(r.summarise())
