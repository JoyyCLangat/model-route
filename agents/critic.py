"""Critic. Reviews another agent's output and either passes it or sends it back.

This is where autonomy actually shows. Choosing once is easy. Recovering when
the first choice produced bad work is the harder and more honest thing. When
the critic fails an output, main.py retries the original step with the failed
model added to the exclude list, which forces the router to pick a different
model.

The critic returns a plain verdict, {"pass": bool, "reason": str}. It reads
the work. It does not name a replacement model. Python does the re selection.
"""

import json

from agents.base import Agent


class Critic(Agent):
    name = "critic"
    required_capabilities = ["reasoning", "code"]
    stakes = "high"
    system = (
        "You are a strict reviewer. Judge whether the work meets the goal. "
        "Reply with one JSON object only and nothing else: "
        '{"pass": true or false, "reason": "one short sentence"}. '
        "No prose, no code fences."
    )

    def review(self, goal: str, work: str, exclude=()) -> dict:
        # Reuse the normal agent pipeline, so the review call is routed and
        # traced exactly like every other step.
        task = f"Goal: {goal}\n\nReview this work and decide if it passes."
        outcome = self.run(task, context=work, exclude=exclude)
        outcome["verdict"] = _verdict(outcome["result"])
        return outcome


def _verdict(result: dict) -> dict:
    # Fail closed. If anything is off, treat it as a rejection with a clear
    # reason, so a broken critic call never silently passes bad work.
    if not result.get("ok"):
        return {"pass": False,
                "reason": f"critic call failed: {result.get('error')}"}
    text = (result.get("text") or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        return {"pass": bool(data.get("pass")),
                "reason": str(data.get("reason", ""))}
    except (ValueError, AttributeError):
        return {"pass": False,
                "reason": "critic did not return a clear verdict"}
