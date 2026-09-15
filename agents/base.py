"""The Agent role.

An agent is a role with a job to do and an opinion about what that job needs.
It does not know which model will run it. It never has. Grep this whole
directory for a provider name and you will find nothing, which is the point:
the model is chosen by the router at call time, from whatever the key can
reach right now.

Every agent's run() does the same six steps: classify the task, add its own
declared needs, ask the router to pick a model, call that model through the
one gateway, record health, emit a trace event. The concrete agents in the
other files only set four attributes and a system prompt.

There is deliberately no model id and no provider name anywhere in this file
or in any agent. The router owns that choice.
"""

import time

from core import classify as classify_mod
from core import llm, router, trace

# A module level step counter, so every event across a run has an order.
_step = 0


def _next_step() -> int:
    global _step
    _step += 1
    return _step


def reset_steps() -> None:
    # Called by the runner at the start of a run so numbering starts at 1.
    global _step
    _step = 0


class Agent:
    name = "agent"
    system = ""
    required_capabilities: list = ["reasoning"]
    stakes = "medium"

    def __init__(self, registry):
        # The registry is shared, so every agent sees the same live catalog and
        # the same health state. It is a resource the agent uses, not a model
        # the agent is tied to.
        self.registry = registry

    def run(self, task: str, context: str = "", exclude=(), retry_of=None) -> dict:
        step = _next_step()

        # 1. Classify the task into a spec. This may use a cheap model or fall
        #    back to keyword rules. Either way it only describes the task.
        classified = classify_mod.classify(
            task, context_chars=len(context), registry=self.registry)
        spec = dict(classified["task_spec"])

        # 2. Merge in what this role knows it needs. The classifier's guess and
        #    the role's declared needs are combined.
        spec["required_capabilities"] = _merge(
            spec.get("required_capabilities", []), self.required_capabilities)
        # The role's stakes win. A critic is high stakes even on a tiny input.
        spec["stakes"] = self.stakes
        spec["exclude"] = list(exclude)

        # 3. Ask the router to pick a model. Pure Python, no model involved.
        decision = router.select(spec, self.registry)

        # 4. Call the chosen model through the one gateway, if there is one.
        if not decision.chosen:
            result = {"ok": False, "model": "", "cost": 0.0, "latency": 0.0,
                      "in_tokens": 0, "out_tokens": 0, "text": "",
                      "error": "no model passed the filters for this task"}
        else:
            result = llm.call(decision.chosen, _prompt(task, context),
                              system=self.system)
            # 5. Record health, so a failing model steps aside for the next
            #    call and a working one keeps its place.
            if result["ok"]:
                self.registry.mark_success(decision.chosen)
            else:
                self.registry.mark_failure(decision.chosen)

        # 6. Emit one trace event. The trace is the demo.
        event = _event(step, self.name, task, classified["classifier"],
                       spec, decision, result, retry_of)
        trace.emit(event)
        return {"result": result, "decision": decision, "event": event,
                "spec": spec}


def _merge(found: list, declared: list) -> list:
    out = list(found)
    for cap in declared:
        if cap not in out:
            out.append(cap)
    return out


def _prompt(task: str, context: str) -> str:
    if context:
        return f"{task}\n\nMaterial:\n{context}"
    return task


def _event(step, agent, task, classifier, spec, decision, result, retry_of):
    return {
        "ts": time.time(),
        "step": step,
        "agent": agent,
        "task": task,
        "classifier": classifier,
        "task_spec": spec,
        "decision": {
            "chosen": decision.chosen,
            "score": decision.score,
            "reason": decision.reason,
            "breakdown": decision.breakdown,
            "filtered_out": decision.filtered_out,
        },
        "result": {
            "ok": result["ok"],
            "model": result.get("model", ""),
            "in_tokens": result.get("in_tokens", 0),
            "out_tokens": result.get("out_tokens", 0),
            "cost": result.get("cost", 0.0),
            "latency": result.get("latency", 0.0),
            "error": result.get("error"),
        },
        "retry_of": retry_of,
    }
