"""The router. This is the file to read first.

An agent hands the router a task spec, which says what the task needs. The
router looks at every model the key can reach, throws out the ones that cannot
do the job, scores the rest on capability, price, speed and context fit, and
returns the winner together with its full reasoning. It does all of this in
well under two hundred lines of plain Python, and no model is ever asked to
choose.

The one idea worth taking away: the weights change with the stakes. On a cheap
bulk job, price dominates, so a small model wins. On a high stakes job,
capability dominates, so a strong model wins. Same code, different weights.
"""

from dataclasses import dataclass
from statistics import mean


@dataclass
class Decision:
    chosen: str
    score: float
    reason: str            # one plain sentence, written by a template
    breakdown: list        # every surviving candidate, with its sub scores
    filtered_out: list     # every removed model, with the reason it went


# How much each factor matters, by stakes. These four numbers per row are the
# whole argument of the talk. Read across a row: low stakes leans on cost,
# high stakes leans on capability.
WEIGHTS = {
    "low":    {"cap": 0.30, "cost": 0.45, "speed": 0.15, "ctx": 0.10},
    "medium": {"cap": 0.50, "cost": 0.30, "speed": 0.10, "ctx": 0.10},
    "high":   {"cap": 0.70, "cost": 0.10, "speed": 0.05, "ctx": 0.15},
}

# Headroom on the context window. If a task needs N tokens of input we want
# room for the answer too, so we demand the window hold N times this.
CTX_HEADROOM = 1.3


def select(task_spec: dict, registry, budget: dict = None) -> Decision:
    """Pick one model for one task. Never calls a model. Pure Python."""
    required = task_spec.get("required_capabilities") or ["reasoning"]
    est_tokens = int(task_spec.get("est_context_tokens", 0) or 0)
    needs_vision = bool(task_spec.get("needs_vision", False))
    stakes = task_spec.get("stakes", "medium")
    exclude = set(task_spec.get("exclude", []))
    weights = WEIGHTS.get(stakes, WEIGHTS["medium"])
    # An optional hard ceiling on the blended price, for a caller on a budget.
    max_price = (budget or {}).get("max_blended_price")

    survivors = []
    filtered_out = []

    # Step 1, the hard filters. A model is either fit for this task or it is
    # not. We record why each reject was dropped, because showing the excluded
    # set is more convincing on stage than showing only the winner.
    for model in registry.candidates():
        why = _reject_reason(model, est_tokens, needs_vision, exclude,
                             registry, max_price)
        if why:
            filtered_out.append({"id": model.id, "reason": why})
        else:
            survivors.append(model)

    if not survivors:
        return Decision(
            chosen="", score=0.0,
            reason="No model passed the hard filters for this task.",
            breakdown=[], filtered_out=filtered_out,
        )

    # Step 2, score every survivor from 0 to 1. Cost is normalised across the
    # survivors, not against a fixed constant, so adding a model reshapes the
    # field on its own.
    prices = [m.blended_price for m in survivors]
    breakdown = []
    for model in survivors:
        capability = mean(model.caps.get(c, 0.6) for c in required)
        cost_score = 1 - _normalise(model.blended_price, prices)
        speed_score = model.caps.get("speed", 0.6)
        ctx_score = 1.0 if est_tokens < 0.5 * model.ctx else 0.6

        score = (weights["cap"] * capability
                 + weights["cost"] * cost_score
                 + weights["speed"] * speed_score
                 + weights["ctx"] * ctx_score)

        breakdown.append({
            "id": model.id,
            "score": round(score, 4),
            "capability": round(capability, 4),
            "cost_score": round(cost_score, 4),
            "speed_score": round(speed_score, 4),
            "ctx_score": round(ctx_score, 4),
            "blended_price": round(model.blended_price, 4),
        })

    breakdown.sort(key=lambda b: b["score"], reverse=True)
    winner = breakdown[0]

    # Step 3, return the decision with its full reasoning attached.
    return Decision(
        chosen=winner["id"],
        score=winner["score"],
        reason=_reason(winner, task_spec, required, stakes),
        breakdown=breakdown,
        filtered_out=filtered_out,
    )


def _reject_reason(model, est_tokens, needs_vision, exclude, registry,
                   max_price):
    # Returns a short reason string when the model is unfit, else None.
    if model.id in exclude:
        return "excluded by caller (already tried or rejected)"
    if est_tokens * CTX_HEADROOM > model.ctx:
        need = int(est_tokens * CTX_HEADROOM)
        return f"context too small ({model.ctx} window, task needs {need})"
    if needs_vision and "image" not in model.modalities:
        return "no image input, task needs vision"
    if not registry.is_healthy(model.id):
        return "unhealthy, in cooldown after recent failures"
    if registry.free_only and not model.is_free:
        return "not free, and free mode is on"
    if max_price is not None and model.blended_price > max_price:
        return f"over budget (blended {model.blended_price:.2f} > {max_price})"
    return None


def _normalise(value: float, values: list) -> float:
    # Min max across the current survivors. When every survivor has the same
    # price this returns 0, which makes cost neutral rather than undefined.
    low, high = min(values), max(values)
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def _reason(winner: dict, task_spec: dict, required: list, stakes: str) -> str:
    # A template writes this sentence, not a model. The router must be able to
    # explain itself without asking anything to generate prose.
    caps = ", ".join(required)
    tokens = int(task_spec.get("est_context_tokens", 0) or 0)
    task_type = task_spec.get("task_type", "task")
    lean = {
        "low": "price was weighted heavily",
        "medium": "capability and price were balanced",
        "high": "capability outweighed price",
    }.get(stakes, "capability and price were balanced")
    return (f"Picked {winner['id']} for {task_type}. Needed {caps}, "
            f"about {tokens} tokens of input, {stakes} stakes, so {lean}.")
