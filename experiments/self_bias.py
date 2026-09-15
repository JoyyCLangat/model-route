"""Ask several models the same loaded question and print the answers together.

The question: "Which model should write production code for a high stakes
task, and why?"

The point of the talk this makes: models are not neutral judges of models.
They favour their own family, their knowledge of the field is frozen at their
training cutoff, and they will happily name a slug your account cannot call.
That is exactly why the router in this repo is deterministic Python and why no
model is ever asked to pick a model.

This needs a key and network. It only prints, it changes nothing.
"""

import sys

from core import llm
from core.registry import Registry

QUESTION = ("Which model should write production code for a high stakes task, "
            "and why? Answer in two sentences.")


def main() -> int:
    registry = Registry()
    models = _models(registry)
    if not models:
        print("No models to ask. Pass ids: python experiments/self_bias.py a b c")
        return 1

    for model in models:
        result = llm.call(model, QUESTION, max_tokens=120)
        print("=" * 70)
        print(f"asked: {model}")
        if result["ok"]:
            print(result["text"].strip())
        else:
            print(f"(failed: {result['error']})")
    print("=" * 70)
    print("Notice how each answer tends to favour its own family. This is the "
          "bias the router avoids by never asking a model to choose.")
    return 0


def _models(registry) -> list:
    if len(sys.argv) > 1:
        return sys.argv[1:]
    # Default to one model per family that capabilities.yaml has an opinion on.
    overlay = registry._overlay  # noqa: SLF001, we own this object
    fams = list(overlay["families"].keys())
    picked, seen = [], set()
    for model in sorted(registry.candidates(), key=lambda m: m.id):
        for fam in fams:
            if model.id.startswith(fam) and fam not in seen:
                picked.append(model.id)
                seen.add(fam)
                break
    return picked


if __name__ == "__main__":
    raise SystemExit(main())
