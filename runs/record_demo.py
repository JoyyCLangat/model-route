"""Produce the committed runs/demo-run.jsonl used for offline replay.

Why this file exists, stated plainly. The demo needs a recorded run that
replays with no API key, so someone in the audience with no credit can still
see exactly what the router does. This script writes that recording.

What is real and what is not:

  Real. Every routing decision here (the chosen model, the score breakdown for
  each candidate, and the list of models filtered out with the reason) is
  produced by core/router.py, the same code a live run uses, scoring a
  snapshot of the catalog. Nothing about the routing is hand written.

  Representative. The token counts, latencies and costs attached to each call
  are chosen to look like a plausible run. They are not measurements, because
  this recording is made in an environment with no provider access. Run
  main.py with a real key to record a true run into runs/latest.jsonl.

The snapshot below is a small, plausible slice of the catalog. The prices are
in dollars per million tokens, the same units core/catalog.py produces. The
slugs here are data, not logic. No model slug is hardcoded in the router.
"""

import json
import os
import sys
import time

# Allow running this file directly from the repo root or from runs/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import Model
from core import router, classify
from agents import base

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo-run.jsonl")


# A plausible snapshot of the catalog. caps come from what core/registry.py
# would attach for each family, so they match capabilities.yaml.
def _snapshot():
    opus_caps = {"code": 0.95, "reasoning": 0.95, "long_context": 0.8,
                 "extraction": 0.8, "vision": 0.8, "speed": 0.4}
    gem_caps = {"code": 0.8, "reasoning": 0.85, "long_context": 0.95,
                "extraction": 0.85, "vision": 0.9, "speed": 0.9}
    kimi_caps = {"code": 0.85, "reasoning": 0.75, "long_context": 0.8,
                 "extraction": 0.9, "vision": 0.7, "speed": 0.8}
    default_caps = {"code": 0.6, "reasoning": 0.6, "long_context": 0.6,
                    "extraction": 0.6, "vision": 0.3, "speed": 0.6}
    return [
        Model("anthropic/claude-opus-4.1", 200000, 15.0, 75.0, opus_caps,
              ["text", "image"], False),
        Model("google/gemini-2.5-flash", 1000000, 0.30, 2.5, gem_caps,
              ["text", "image"], False),
        Model("moonshotai/kimi-k2", 128000, 0.14, 0.28, kimi_caps,
              ["text"], False),
        Model("openai/gpt-4o-mini", 128000, 0.15, 0.60, default_caps,
              ["text", "image"], False),
        Model("meta-llama/llama-3.3-70b-instruct:free", 65536, 0.0, 0.0,
              default_caps, ["text"], True),
    ]


class SnapshotRegistry:
    """A registry backed by the snapshot, no catalog and no network."""

    def __init__(self, models):
        self._models = models
        self.free_only = False

    def candidates(self):
        return list(self._models)

    def is_healthy(self, model_id):
        return True


def _spec(task, agent_caps, stakes, context_chars, exclude):
    # Mirror what agents/base.py does: classify with the keyword rules, then
    # merge the role's declared needs, then set stakes and context size.
    spec = classify._rules(task)
    merged = list(spec["required_capabilities"])
    for cap in agent_caps:
        if cap not in merged:
            merged.append(cap)
    spec["required_capabilities"] = merged
    spec["stakes"] = stakes
    spec["est_context_tokens"] = context_chars // classify.CHARS_PER_TOKEN
    spec["exclude"] = list(exclude)
    return spec


def _result(ok, model, in_tokens, out_tokens, in_price, out_price, latency,
            error=None):
    cost = 0.0 if not ok else (in_tokens * in_price + out_tokens * out_price) / 1_000_000
    return {"ok": ok, "model": model if ok else model, "text": "",
            "in_tokens": in_tokens if ok else 0,
            "out_tokens": out_tokens if ok else 0,
            "cost": round(cost, 6), "latency": latency, "error": error}


def build_events():
    reg = SnapshotRegistry(_snapshot())
    goal = "Audit this repo and propose a fix"
    events = []
    clock = [1_726_000_000.0]  # a fixed start time for a stable recording

    def add(agent, task, agent_caps, stakes, ctx_chars, exclude, result,
            retry_of=None, think=0.6):
        base._step = len(events)  # keep step numbers in order
        spec = _spec(task, agent_caps, stakes, ctx_chars, exclude)
        decision = router.select(spec, reg)
        step = len(events) + 1
        clock[0] += result["latency"] + think
        ev = base._event(step, agent, task, "rules", spec, decision, result,
                        retry_of)
        ev["ts"] = round(clock[0], 3)
        events.append(ev)
        return ev

    P = {"opus": (15.0, 75.0), "gem": (0.30, 2.5), "kimi": (0.14, 0.28)}

    # The goal is passed as context, not baked into the task string, so the
    # keyword classifier labels each step by the role's own verb.
    goal_note = f"Goal: {goal}"

    # 1. Plan. High stakes, reasoning. The all rounder wins on value.
    add("planner", "Plan the steps needed to reach the goal",
        ["reasoning"], "high", len(goal_note), [],
        _result(True, "google/gemini-2.5-flash", 130, 190, *P["gem"], 2.1))

    # 2. Research the target. Medium stakes, large context, so the big window
    #    model is favoured.
    add("researcher", "Research the target files for the facts and risks",
        ["long_context", "reasoning"], "medium", 5200, [],
        _result(True, "google/gemini-2.5-flash", 1300, 420, *P["gem"], 4.6))

    # 3. Summarize. Low stakes and bulk, so the cheap specialist wins.
    add("summarizer", "Summarize these findings into a short brief",
        ["extraction"], "low", 2100, [],
        _result(True, "moonshotai/kimi-k2", 520, 160, *P["kimi"], 1.8))

    # 4. Coder attempt one. The chosen model has a bad moment and the call
    #    fails. This is the health path: a failed model forces a change.
    add("coder", "Write the smallest patch that fixes the bugs",
        ["code", "reasoning"], "high", 900, [],
        _result(False, "google/gemini-2.5-flash", 0, 0, *P["gem"], 0.7,
                error="http 503: model temporarily overloaded, try again"))

    # 5. Critic reviews the empty result and fails it.
    add("critic", f"Goal: {goal}\n\nReview this work and decide if it passes.",
        ["reasoning", "code"], "high", 40, [],
        _result(True, "google/gemini-2.5-flash", 300, 24, *P["gem"], 1.1))

    # 6. Coder attempt two, with the first model excluded. The router picks a
    #    different, cheaper capable model. This is the exclusion retry.
    add("coder", "Write the smallest patch that fixes the bugs",
        ["code", "reasoning"], "high", 900, ["google/gemini-2.5-flash"],
        _result(True, "moonshotai/kimi-k2", 610, 520, *P["kimi"], 3.2),
        retry_of=4)

    # 7. Critic reviews the patch and still fails it, on quality this time.
    add("critic", f"Goal: {goal}\n\nReview this work and decide if it passes.",
        ["reasoning", "code"], "high", 2200, [],
        _result(True, "google/gemini-2.5-flash", 780, 30, *P["gem"], 1.3))

    # 8. Coder attempt three, with two models excluded. Only the most capable
    #    survivor is left, and the router pays for it because stakes are high.
    add("coder", "Write the smallest patch that fixes the bugs",
        ["code", "reasoning"], "high", 900,
        ["google/gemini-2.5-flash", "moonshotai/kimi-k2"],
        _result(True, "anthropic/claude-opus-4.1", 620, 700, *P["opus"], 9.4),
        retry_of=4)

    # 9. Critic passes the final patch.
    add("critic", f"Goal: {goal}\n\nReview this work and decide if it passes.",
        ["reasoning", "code"], "high", 2600, [],
        _result(True, "google/gemini-2.5-flash", 900, 28, *P["gem"], 1.3))

    return events


def main():
    events = build_events()
    with open(OUT_PATH, "w") as handle:
        for ev in events:
            handle.write(json.dumps(ev) + "\n")
    chosen = [e["decision"]["chosen"] for e in events]
    print(f"wrote {len(events)} events to {OUT_PATH}")
    print("chosen per step:", chosen)


if __name__ == "__main__":
    main()
