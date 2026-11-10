#!/usr/bin/env python3
"""Score a play performance from its run trace (src/sim/trace.py).

    .venv/bin/python tools/score_play.py <trace.jsonl> [--smoke logs/smoke-<label>.json]
    .venv/bin/python tools/score_play.py --smoke logs/smoke-<label>.json      # trace named in it

Writes logs/scores/<run>.json (the scorecard) and logs/scores/<run>.md (a readable report).
No LLM calls; the embedding tier uses the engine's sentence-transformer (utils/embedding_model).

Tiers:
  0 cost        wall time, LLM time and tokens (in/out/cached), truncations, GPU energy, vLLM
                prefix-cache and speculative-decode rates, $ at Fireworks GLM prices; by tag,
                character and scene; per line / act / scene
  1 integrity   LLM errors, engine errors, empty / duplicate / placeholder-leaking lines,
                unscripted speakers, staleness score
  2 agency      task outcomes, goal checks, scene-plan adherence, scene post-states achieved,
                commitment funnel (extracted -> placed -> performed -> satisfied), joint
                commitments kept by both, promise keeping per pair
  3 character   persona fidelity, voice distinctiveness, scene-to-scene consistency, long-term
                drift, repetition, stance volatility, theory-of-mind accuracy
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Fireworks serverless prices for GLM-5.3-Flash, $ per million tokens (Cognitive_workbench
# measure/models/fw_glm53flash.yaml, 2026-09-02): what the run would cost on a paid endpoint.
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.15, 0.03, 0.50
PLACEHOLDER = re.compile(r"\{\{\$?\w+\}?\}?|\{\$\w+\}")


def load(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    return events


def by_kind(events):
    groups = collections.defaultdict(list)
    for e in events:
        groups[e["kind"]].append(e)
    return groups


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def ratio(a, b):
    return round(a / b, 4) if b else None


# --------------------------------------------------------------------------- tier 0
def tier_cost(ev, events, smoke):
    calls = ev["llm_call"]
    t0, t1 = events[0]["t"], events[-1]["t"]
    wall = t1 - t0
    tok = lambda c, k: c.get(k) or 0
    prompt = sum(tok(c, "prompt_tokens") for c in calls)
    completion = sum(tok(c, "completion_tokens") for c in calls)
    cached = sum(tok(c, "cached_tokens") for c in calls)

    def breakdown(key):
        out = collections.defaultdict(lambda: {"calls": 0, "secs": 0.0, "prompt": 0, "completion": 0})
        for c in calls:
            k = c.get(key) if key != "scene" else f"{c.get('act')}.{c.get('scene')}"
            d = out[str(k)]
            d["calls"] += 1
            d["secs"] += c.get("secs") or 0
            d["prompt"] += tok(c, "prompt_tokens")
            d["completion"] += tok(c, "completion_tokens")
        return {k: {**v, "secs": round(v["secs"], 1)}
                for k, v in sorted(out.items(), key=lambda kv: -kv[1]["prompt"] - kv[1]["completion"])}

    lines = ev["transcript_line"]
    speech = [l for l in lines if l.get("line_kind") == "speech"]
    scenes = len(ev["scene_start"]) or None
    acts = len(ev["act"]) or None
    cost = (prompt - cached) * PRICE_IN / 1e6 + cached * PRICE_CACHED / 1e6 + completion * PRICE_OUT / 1e6
    out = {
        "wall_secs": round(wall, 1),
        "llm_calls": len(calls),
        "llm_secs": round(sum(c.get("secs") or 0 for c in calls), 1),
        "llm_busy_fraction": ratio(sum(c.get("secs") or 0 for c in calls), wall),
        "calls_without_usage": sum(1 for c in calls if c.get("prompt_tokens") is None),
        "tokens": {"prompt": prompt, "completion": completion, "cached": cached,
                   "cached_fraction": ratio(cached, prompt),
                   "mean_prompt_per_call": ratio(prompt, len(calls)),
                   "mean_completion_per_call": ratio(completion, len(calls))},
        "truncated_calls": sum(1 for c in calls if c.get("finish_reason") == "length"),
        "fireworks_equivalent_usd": round(cost, 4),
        "per_unit": {
            "tokens_per_speech_line": ratio(prompt + completion, len(speech)),
            "tokens_per_act": ratio(prompt + completion, acts),
            "tokens_per_scene": ratio(prompt + completion, scenes),
            "secs_per_act": ratio(wall, acts),
            "llm_calls_per_act": ratio(len(calls), acts),
        },
        "by_tag": breakdown("tag"),
        "by_character": breakdown("owner"),
        "by_scene": breakdown("scene"),
    }
    hw = (smoke or {}).get("hw")
    if hw:
        gpus = collections.defaultdict(list)
        for s in hw.get("gpu_samples", []):
            gpus[(s["gpu"], s["name"])].append(s)
        gpu_out = {}
        for (idx, name), samples in sorted(gpus.items()):
            samples.sort(key=lambda s: s["t"])
            wh = sum((b["t"] - a["t"]) * (a["power_w"] + b["power_w"]) / 2
                     for a, b in zip(samples, samples[1:])) / 3600
            gpu_out[f"{idx}:{name}"] = {"util_mean": mean([s["util"] for s in samples]),
                                        "mem_mib_max": max(s["mem_mib"] for s in samples),
                                        "power_w_mean": mean([s["power_w"] for s in samples]),
                                        "energy_wh": round(wh, 2)}
        d = hw.get("vllm_delta", {})
        out["hardware"] = {
            "gpus": gpu_out,
            "vllm": {"prompt_tokens": d.get("prompt_tokens_total"), "generation_tokens": d.get("generation_tokens_total"),
                     "prefix_cache_hit_rate": ratio(d.get("prefix_cache_hits_total", 0), d.get("prefix_cache_queries_total", 0)),
                     "spec_decode_acceptance": ratio(d.get("spec_decode_num_accepted_tokens_total", 0),
                                                     d.get("spec_decode_num_draft_tokens_total", 0)),
                     "finished_length": d.get("finished_length"), "finished_error": d.get("finished_error")},
            "trace_vs_vllm_prompt_tokens": ratio(prompt, d.get("prompt_tokens_total")),
        }
        total_wh = sum(g["energy_wh"] for g in gpu_out.values())
        out["per_unit"]["wh_per_scene"] = ratio(total_wh, scenes)
    return out


# --------------------------------------------------------------------------- tier 1
def tier_integrity(ev, smoke, cast):
    calls = ev["llm_call"]
    lines = ev["transcript_line"]
    speech = [l for l in lines if l.get("line_kind") == "speech"]
    texts = [l["text"].strip() for l in speech]
    dup = sum(n - 1 for n in collections.Counter(texts).values() if n > 1)
    empty = sum(1 for t in texts if t.strip("' .") == "")
    leaks = [l["text"][:120] for l in lines if PLACEHOLDER.search(l.get("text", ""))]
    speakers = {l["speaker"] for l in speech}
    errors = collections.Counter(e.get("message", "")[:100] for e in ev["error"])
    stale = [((e.get("staleness") or {}).get("staleness_score")) for e in ev["scene_end"]]
    out = {
        "llm_error_rate": ratio(sum(1 for c in calls if not c.get("ok")), len(calls)),
        "llm_truncation_rate": ratio(sum(1 for c in calls if c.get("finish_reason") == "length"), len(calls)),
        "engine_errors": sum(errors.values()),
        "top_engine_errors": errors.most_common(8),
        "speech_lines": len(speech),
        "duplicate_speech_lines": dup,
        "empty_speech_lines": empty,
        "placeholder_leaks": leaks[:10],
        "unscripted_speakers": sorted(speakers - set(cast)),
        "staleness_score_mean": mean(stale),
    }
    if smoke:
        out["tracebacks"] = sum(n for _, n in smoke.get("tracebacks", []))
    return out


# --------------------------------------------------------------------------- tier 2
def tier_agency(ev):
    task_ends = ev["task_end"]
    outcomes = collections.Counter(e["outcome"] for e in task_ends)
    # a task checked but neither satisfied nor timed out is left unfinished when the scene moves on
    ended = {e["task_id"] for e in task_ends}
    checked = {e["task_id"] for e in ev["task_check"]}
    outcomes["unfinished"] = len(checked - ended)
    satisfied_ids = {e["task_id"] for e in task_ends if e["outcome"] == "satisfied"}
    # run_scene's refresh_task may replace a planned task with a revised one (new id): follow it
    replaced_by = collections.defaultdict(set)
    for e in ev["scene_task"]:
        if e.get("planned_task_id") and e.get("task_id"):
            replaced_by[e["planned_task_id"]].add(e["task_id"])
    satisfied_ids |= {planned for planned, actual in replaced_by.items() if actual & satisfied_ids}
    performed_ids = {e.get("task_id") for e in ev["act"]} | {e.get("task_id") for e in ev["scene_task"]} \
        | {e.get("planned_task_id") for e in ev["scene_task"]}

    # scene plan adherence: planned entries of each scene that were run
    planned, run = 0, 0
    run_by_scene = collections.defaultdict(set)
    for e in ev["scene_task"]:
        run_by_scene[(e.get("act"), e.get("scene"))].add(e.get("planned_task_id"))
    for e in ev["scene_plan"]:
        ids = [x["task_id"] for x in e.get("entries", [])]
        planned += len(ids)
        run += sum(1 for i in ids if i in run_by_scene[(e.get("act"), e.get("scene"))])

    # scene post-states: check_post_state_ambiguity outcomes, each {'outcome', 'test'}
    post_tests = []
    for e in ev["scene_end"]:
        for o in e.get("outcomes") or []:
            if isinstance(o, dict) and "test" in o:
                post_tests.append(bool(o["test"]))

    # commitments
    extracted = ev["commitment_extracted"]
    placements = {e["task_id"]: e for e in ev["commitment"]}
    funnel = {"extracted": len(extracted), "in_scene": sum(1 for e in extracted if e.get("in_scene")),
              "joint": sum(1 for e in extracted if e.get("joint")),
              "significance": dict(collections.Counter(e.get("significance") for e in extracted)),
              "placed_tasks": sum(1 for p in placements.values() if p.get("placed")),
              "optional_placed": sum(1 for p in placements.values() if p.get("placed") and p.get("optional")),
              "optional_skipped": len(ev["optional_skipped"]),
              "drop_reasons": dict(collections.Counter(p.get("reason", "")[:40] for p in placements.values()
                                                        if not p.get("placed")))}
    placed_ids = [tid for tid, p in placements.items() if p.get("placed")]
    funnel["performed_tasks"] = sum(1 for t in placed_ids if t in performed_ids)
    funnel["satisfied_tasks"] = sum(1 for t in placed_ids if t in satisfied_ids)
    funnel["follow_through_rate"] = ratio(funnel["performed_tasks"], len(placed_ids))
    funnel["fulfilment_rate"] = ratio(funnel["satisfied_tasks"], len(placed_ids))
    joint = [e for e in extracted if e.get("joint")]
    joint_kept = [all(tid in performed_ids for _, tid in e.get("owner_tasks", [])) for e in joint
                  if any(tid in placements and placements[tid].get("placed") for _, tid in e.get("owner_tasks", []))]
    funnel["joint_kept_by_all_parties"] = ratio(sum(joint_kept), len(joint_kept))
    pairs = collections.defaultdict(lambda: [0, 0])
    for e in extracted:
        pair = " & ".join(sorted([e.get("speaker", "?"), e.get("listener", "?")]))
        for owner, tid in e.get("owner_tasks", []):
            if tid in placements and placements[tid].get("placed"):
                pairs[pair][0] += 1
                pairs[pair][1] += tid in performed_ids
    funnel["promise_keeping_by_pair"] = {p: {"placed": a, "performed": b, "rate": ratio(b, a)}
                                         for p, (a, b) in pairs.items()}

    goal_checks = ev["goal_check"]
    return {
        "tasks_ended": len(task_ends),
        "task_outcomes": dict(outcomes),
        "task_satisfied_rate": ratio(outcomes.get("satisfied", 0), sum(outcomes.values())),
        "acts_per_task": mean([e.get("acts") for e in task_ends]),
        "goals_created": len(ev["goal_created"]),
        "goal_checks": len(goal_checks),
        "goals_satisfied": len({e["goal_id"] for e in goal_checks if e.get("satisfied")}),
        "scene_plan_adherence": ratio(run, planned),
        "scenes": len(ev["scene_start"]),
        "scene_post_state_achieved": ratio(sum(post_tests), len(post_tests)),
        "commitments": funnel,
        "physical": physical(ev),
    }


def physical(ev):
    """Concrete physical action (sim/survival.py and the act mix): how much of the play is doing."""
    acts = ev["act"]
    modes = collections.Counter(a.get("mode") for a in acts)
    aff = ev["affordance"]
    applied = [a for a in aff if a.get("applied")]
    hazards = collections.Counter(h.get("phase") for h in ev["hazard"])
    return {
        "act_modes": dict(modes),
        "speech_share_of_acts": ratio(modes.get("Say", 0), len(acts)),
        "physical_share_of_acts": ratio(modes.get("Do", 0) + modes.get("Move", 0), len(acts)),
        "affordances_applied": len(applied),
        "affordance_refusals": len(aff) - len(applied),
        "affordances_by_action": dict(collections.Counter(a.get("action") for a in applied)),
        "affordances_via": dict(collections.Counter(a.get("via", "text") for a in applied)),
        "affordances_after_approach": sum(1 for a in applied if a.get("approached")),
        "do_acts_labelled": ratio(sum(1 for a in acts if a.get("mode") == "Do" and a.get("affordance")), modes.get("Do", 0)),
        "concrete_action_rate": ratio(len(applied), len(acts)),
        "need_warnings": dict(collections.Counter(f"{b.get('need')}>={0.8 if b.get('level', 0) >= 0.8 else 0.5}" for b in ev["body"])),
        "hazards": dict(hazards),
        "travel_minutes": round(sum(t.get("minutes") or 0 for t in ev["travel"]), 1),
        "world_events": len(ev["world_event"]),
    }


# --------------------------------------------------------------------------- tier 3
def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


FUNCTION_WORDS = ("i me my we our you your he she it they them the a an and but or so if then not no "
                  "yes is are was were be been am do does did have has had will would can could should "
                  "just very really maybe perhaps well oh what why how who where when that this there "
                  "here of to in on at for with about from by as like").split()


def style_vector(text):
    words = re.findall(r"[a-z']+", text.lower())
    n = max(1, len(words))
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    counts = collections.Counter(words)
    feats = [counts[w] / n for w in FUNCTION_WORDS]
    feats += [len(words) / sentences / 30, sum(len(w) for w in words) / n / 10,
              text.count(',') / n, text.count('?') / sentences, text.count('!') / sentences,
              (text.count('—') + text.count('--') + text.count('...')) / sentences]
    return np.array(feats)


def stylometric_attribution(eligible):
    """Leave-one-out nearest-centroid speaker attribution on standardised style features."""
    if len(eligible) < 2:
        return None
    rows = [(n, style_vector(l["text"])) for n, items in eligible.items() for l, _ in items]
    X = np.array([v for _, v in rows])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    labels = [n for n, _ in rows]
    correct = 0
    for i, (n, _) in enumerate(rows):
        best, best_d = None, None
        for m in eligible:
            idx = [j for j, lab in enumerate(labels) if lab == m and j != i]
            d = float(np.linalg.norm(X[i] - X[idx].mean(axis=0)))
            if best_d is None or d < best_d:
                best, best_d = m, d
        correct += best == n
    return ratio(correct, len(rows))


def tier_character(ev, cast):
    from utils.embedding_model import get_embedding_model
    model = get_embedding_model()
    speech = [l for l in ev["transcript_line"] if l.get("line_kind") == "speech" and l["speaker"] in cast]
    if not speech or not cast:
        return {"note": "no speech lines from cast"}
    texts = [l["text"].strip("' ") for l in speech]
    embs = model.encode(texts, normalize_embeddings=True)
    names = sorted(cast)
    desc = {n: model.encode(cast[n], normalize_embeddings=True) for n in names}
    by_char = collections.defaultdict(list)
    for l, e in zip(speech, embs):
        by_char[l["speaker"]].append((l, e))

    # persona fidelity: own description vs the others'
    fidelity = {}
    for n, items in by_char.items():
        own = [cos(e, desc[n]) for _, e in items]
        others = [cos(e, desc[m]) for _, e in items for m in names if m != n]
        fidelity[n] = {"own": mean(own), "others": mean(others),
                       "margin": round(mean(own) - mean(others), 4) if others else None, "lines": len(items)}

    # voice distinctiveness: leave-one-out nearest-centroid attribution
    eligible = {n: items for n, items in by_char.items() if len(items) >= 3}
    correct = total = 0
    if len(eligible) >= 2:
        sums = {n: np.sum([e for _, e in items], axis=0) for n, items in eligible.items()}
        for n, items in eligible.items():
            for _, e in items:
                best = max(eligible, key=lambda m: cos(e, (sums[m] - e) / (len(eligible[m]) - 1)
                                                      if m == n else sums[m] / len(eligible[m])))
                correct += best == n
                total += 1
    distinct = {"embedding_accuracy": ratio(correct, total), "chance": ratio(1, len(eligible)) if eligible else None,
                "lines": total, "stylometric_accuracy": stylometric_attribution(eligible),
                "note": "embedding attribution measures topic more than voice; stylometric uses "
                        "function words, sentence length and punctuation"}

    # consistency across scenes; drift first act -> last act
    consistency, drift = {}, {}
    for n, items in by_char.items():
        scenes = collections.defaultdict(list)
        acts = collections.defaultdict(list)
        for l, e in items:
            scenes[(l.get("act"), l.get("scene"))].append(e)
            acts[l.get("act")].append(e)
        keys = [k for k in sorted(scenes, key=lambda k: (k[0] or 0, k[1] or 0))]
        cents = [np.mean(scenes[k], axis=0) for k in keys]
        consistency[n] = mean([cos(a, b) for a, b in zip(cents, cents[1:])])
        act_keys = sorted(k for k in acts if k is not None)
        if len(act_keys) >= 2:
            first, last = np.mean(acts[act_keys[0]], axis=0), np.mean(acts[act_keys[-1]], axis=0)
            drift[n] = {"first_to_description": round(cos(first, desc[n]), 4),
                        "last_to_description": round(cos(last, desc[n]), 4),
                        "first_to_last": round(cos(first, last), 4)}

    # repetition: near-duplicate lines and repeated trigrams, per character
    repetition = {}
    for n, items in by_char.items():
        es = [e for _, e in items]
        near = sum(1 for a, b in combinations(range(len(es)), 2) if cos(es[a], es[b]) > 0.92)
        grams = collections.Counter()
        for l, _ in items:
            words = re.findall(r"[a-z']+", l["text"].lower())
            grams.update(zip(words, words[1:], words[2:]))
        repeated = sum(c - 1 for c in grams.values() if c > 1)
        repetition[n] = {"near_duplicate_pairs": near, "lines": len(items),
                         "repeated_trigram_fraction": ratio(repeated, sum(grams.values()))}

    # stance volatility: stance changes per scene
    stances = collections.defaultdict(list)
    for e in ev["stance"]:
        stances[e["character"]].append((e.get("arousal"), e.get("tone"), e.get("orientation")))
    n_scenes = max(1, len(ev["scene_start"]))
    volatility = {n: round(sum(1 for a, b in zip(s, s[1:]) if a != b) / n_scenes, 3) for n, s in stances.items()}

    # theory of mind: inferred vs actual goal of the observed character
    tom = [(e["inferred"], e["actual"]) for e in ev["inferred_goal"] if e.get("inferred") and e.get("actual")]
    tom_score = None
    if tom:
        a = model.encode([x for x, _ in tom], normalize_embeddings=True)
        b = model.encode([y for _, y in tom], normalize_embeddings=True)
        tom_score = mean([cos(x, y) for x, y in zip(a, b)])

    return {
        "persona_fidelity": fidelity,
        "persona_fidelity_margin_mean": mean([v["margin"] for v in fidelity.values()]),
        "voice_distinctiveness": distinct,
        "scene_to_scene_consistency": consistency,
        "drift": drift,
        "repetition": repetition,
        "stance_changes_per_scene": volatility,
        "theory_of_mind_similarity": tom_score,
        "theory_of_mind_samples": len(tom),
    }


# --------------------------------------------------------------------------- report
def headline(card):
    c, i, a, ch = card["cost"], card["integrity"], card["agency"], card["character"]
    cm = a["commitments"]
    return [
        ("wall time (s)", c["wall_secs"]), ("LLM calls", c["llm_calls"]),
        ("tokens in / out", f'{c["tokens"]["prompt"]:,} / {c["tokens"]["completion"]:,}'),
        ("cached prompt fraction", c["tokens"]["cached_fraction"]),
        ("truncated calls", c["truncated_calls"]), ("$ at Fireworks prices", c["fireworks_equivalent_usd"]),
        ("tokens per speech line", c["per_unit"]["tokens_per_speech_line"]),
        ("GPU energy (Wh)", sum(g["energy_wh"] for g in c.get("hardware", {}).get("gpus", {}).values()) or None),
        ("LLM error rate", i["llm_error_rate"]), ("engine errors", i["engine_errors"]),
        ("duplicate speech lines", i["duplicate_speech_lines"]),
        ("task satisfied rate", a["task_satisfied_rate"]), ("scene-plan adherence", a["scene_plan_adherence"]),
        ("scene post-states achieved", a["scene_post_state_achieved"]),
        ("commitments placed / performed / satisfied",
         f'{cm["placed_tasks"]} / {cm["performed_tasks"]} / {cm["satisfied_tasks"]}'),
        ("persona fidelity margin (weak: descriptions are mostly appearance)", ch.get("persona_fidelity_margin_mean")),
        ("speaker attribution: stylometric / embedding (chance)",
         f'{ch.get("voice_distinctiveness", {}).get("stylometric_accuracy")} / '
         f'{ch.get("voice_distinctiveness", {}).get("embedding_accuracy")} ({ch.get("voice_distinctiveness", {}).get("chance")})'),
        ("theory-of-mind similarity", ch.get("theory_of_mind_similarity")),
        ("speech / physical share of acts", f'{a["physical"]["speech_share_of_acts"]} / {a["physical"]["physical_share_of_acts"]}'),
        ("survival actions applied (refused)", f'{a["physical"]["affordances_applied"]} ({a["physical"]["affordance_refusals"]})'),
        ("hazards", a["physical"]["hazards"]),
    ]


def markdown(card):
    out = [f"# Scorecard: {card['run']}", "", f"play `{card['play']}`, trace `{card['trace']}`", "",
           "| metric | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in headline(card)]
    out += ["", "## Cost by tag (top 12)", "", "| tag | calls | secs | prompt | completion |", "|---|---|---|---|---|"]
    for tag, d in list(card["cost"]["by_tag"].items())[:12]:
        out.append(f"| {tag} | {d['calls']} | {d['secs']} | {d['prompt']:,} | {d['completion']:,} |")
    out += ["", "## Character", ""]
    for n, f in card["character"].get("persona_fidelity", {}).items():
        rep = card["character"]["repetition"].get(n, {})
        out.append(f"- **{n}**: fidelity margin {f['margin']} ({f['lines']} lines), "
                   f"scene-to-scene consistency {card['character']['scene_to_scene_consistency'].get(n)}, "
                   f"repeated trigrams {rep.get('repeated_trigram_fraction')}, "
                   f"stance changes/scene {card['character']['stance_changes_per_scene'].get(n)}")
    out += ["", "Full detail in the JSON scorecard."]
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", nargs="?")
    p.add_argument("--smoke", help="logs/smoke-<label>.json from tools/smoke_run.py (hardware, tracebacks)")
    p.add_argument("--out", default=str(ROOT / "logs" / "scores"))
    p.add_argument("--no-embeddings", action="store_true", help="skip tier 3")
    a = p.parse_args()
    smoke = json.loads(Path(a.smoke).read_text()) if a.smoke else None
    trace_path = a.trace or (smoke or {}).get("trace")
    if not trace_path:
        p.error("no trace: pass one, or a --smoke summary that names it")
    events = load(trace_path)
    if not events:
        sys.exit(f"empty trace {trace_path}")
    ev = collections.defaultdict(list, by_kind(events))
    loaded = (ev["play_loaded"] or [{}])[-1]
    cast = {c["name"]: c.get("description") or "" for c in loaded.get("characters", [])}
    card = {
        "run": events[0].get("run"), "play": (ev["run_start"] or [{}])[0].get("play"), "trace": str(trace_path),
        "cast": sorted(cast),
        "cost": tier_cost(ev, events, smoke),
        "integrity": tier_integrity(ev, smoke, cast),
        "agency": tier_agency(ev),
        "character": {} if a.no_embeddings else tier_character(ev, cast),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{card['run']}.json").write_text(json.dumps(card, indent=2, default=str))
    (out / f"{card['run']}.md").write_text(markdown(card))
    for k, v in headline(card):
        print(f"{k:45s} {v}")
    print(f"\nwrote {out / (card['run'] + '.json')} and .md")


if __name__ == "__main__":
    main()
