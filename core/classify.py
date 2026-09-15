"""Turn a free text task into a small structured spec.

The router needs to know what a task requires before it can score models
against it. This file produces that spec. It asks the cheapest healthy model
to read the task and label it as JSON. That is the only place a model touches
the routing pipeline, and even here the model only describes the task. It
never names a model. Python does the choosing.

A live demo cannot depend on that one call. If the model is down, returns
prose instead of JSON, or returns a value we do not allow, we fall back to a
plain keyword matcher that returns the exact same shape. The trace records
which path ran, so you can pull the network cable and watch the rules path
take over with the run still completing.
"""

import hashlib
import json

from core import llm
from core.registry import Registry

DIMENSIONS = ["code", "reasoning", "long_context", "extraction", "vision", "speed"]
TASK_TYPES = ["code_edit", "research", "extraction", "summarize", "review",
              "plan", "other"]
STAKES = ["low", "medium", "high"]

# Rough tokens per character. Good enough to size a context window, and we
# would rather use a measured character count than trust a model to count.
CHARS_PER_TOKEN = 4

_SYSTEM = (
    "You label a task for a router. Reply with one JSON object and nothing "
    "else. No prose, no code fences. Keys: task_type (one of code_edit, "
    "research, extraction, summarize, review, plan, other), "
    "required_capabilities (a list drawn from code, reasoning, long_context, "
    "extraction, vision, speed), est_context_tokens (integer), stakes (low, "
    "medium or high), needs_vision (true or false), max_latency_s (a number). "
    "Do not name any model or provider."
)

# Cache by a hash of the task, so repeated demo runs are fast and cheap.
_cache: dict = {}


def classify(task: str, context_chars: int = 0, registry: Registry = None) -> dict:
    """Return {"task_spec": {...}, "classifier": "llm" | "rules"}.

    context_chars is the size of any material handed to the task, used to size
    the context need. The registry is optional. Passing the caller's registry
    lets the classifier share health state and pick a truly cheap live model.
    """
    key = _key(task, context_chars)
    if key in _cache:
        return _cache[key]

    spec, source = _via_llm(task, registry)
    if spec is None:
        spec = _rules(task)
        source = "rules"

    # A measured input size is more trustworthy than a model's guess, so it
    # wins whenever we have one.
    if context_chars > 0:
        spec["est_context_tokens"] = context_chars // CHARS_PER_TOKEN

    result = {"task_spec": spec, "classifier": source}
    _cache[key] = result
    return result


def _via_llm(task: str, registry: Registry):
    # Returns (spec, "llm") on success, or (None, None) on any problem so the
    # caller falls back to the rules. classify must never take a run down.
    try:
        reg = registry or Registry()
        model = _cheapest_healthy(reg)
        if model is None:
            return None, None
        prompt = f"Task:\n{task}\n\nReturn the JSON now."
        result = llm.call(model.id, prompt, system=_SYSTEM, max_tokens=256)
        if not result["ok"]:
            return None, None
        spec = _parse(result["text"])
        if spec is None or not _valid(spec):
            return None, None
        return spec, "llm"
    except Exception:
        return None, None


def _cheapest_healthy(registry: Registry):
    healthy = [m for m in registry.candidates() if registry.is_healthy(m.id)]
    if registry.free_only:
        healthy = [m for m in healthy if m.is_free]
    if not healthy:
        return None
    return min(healthy, key=lambda m: m.blended_price)


def _parse(text: str):
    # Models sometimes wrap JSON in code fences even when told not to. Strip a
    # leading fence and a trailing fence before parsing.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def _valid(spec: dict) -> bool:
    # Every field is checked against the allowed values. One bad field sends
    # the whole spec to the rules path, which is the safe default.
    try:
        if spec.get("task_type") not in TASK_TYPES:
            return False
        caps = spec.get("required_capabilities")
        if not isinstance(caps, list) or not caps:
            return False
        if any(c not in DIMENSIONS for c in caps):
            return False
        if spec.get("stakes") not in STAKES:
            return False
        if not isinstance(spec.get("needs_vision"), bool):
            return False
        int(spec.get("est_context_tokens"))
        float(spec.get("max_latency_s"))
    except (TypeError, ValueError):
        return False
    return True


def _rules(task: str) -> dict:
    # A plain keyword matcher. No model, no network. It returns the same shape
    # as the model path, so the rest of the pipeline cannot tell them apart.
    low = task.lower()

    def has(*words):
        return any(w in low for w in words)

    needs_vision = has("image", "screenshot", "photo", "diagram", "chart",
                       "picture")

    if has("review", "audit", "critique", "check for bugs"):
        spec = _spec("review", ["reasoning", "code"], "high")
    elif has("fix", "patch", "implement", "refactor", "bug", "write code",
             "function"):
        spec = _spec("code_edit", ["code", "reasoning"], "high")
    elif has("summarize", "summary", "condense", "tl;dr", "shorten"):
        spec = _spec("summarize", ["extraction"], "low")
    elif has("extract", "pull out", "list all", "find all", "collect"):
        spec = _spec("extraction", ["extraction"], "low")
    elif has("plan", "break down", "outline", "steps", "roadmap"):
        spec = _spec("plan", ["reasoning"], "high")
    elif has("research", "investigate", "read the", "across", "analyse",
             "analyze"):
        spec = _spec("research", ["long_context", "reasoning"], "medium")
    else:
        spec = _spec("other", ["reasoning"], "medium")

    spec["needs_vision"] = needs_vision
    if needs_vision and "vision" not in spec["required_capabilities"]:
        spec["required_capabilities"].append("vision")
    return spec


def _spec(task_type: str, caps: list, stakes: str) -> dict:
    return {
        "task_type": task_type,
        "required_capabilities": list(caps),
        "est_context_tokens": 2000,
        "stakes": stakes,
        "needs_vision": False,
        "max_latency_s": 30,
    }


def _key(task: str, context_chars: int) -> str:
    raw = f"{context_chars}:{task}".encode()
    return hashlib.sha1(raw).hexdigest()
