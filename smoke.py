"""Step 1 of the talk: prove the gateway works, and show the problem with it.

This script calls three models directly and prints their cost and latency. It
proves the gateway returns real numbers. But look closely at the flaw. The
three model names are written into this file by hand. Someone guessed them at
the time of writing. They go out of date, they may not be callable by your
key, and nothing here reacts to the task. Removing that guess is the whole
point of the rest of the repo. See core/router.py for the version that picks a
model from the live catalog instead.

If the default slugs have gone stale, pass your own:

    python smoke.py anthropic/claude-3.5-haiku openai/gpt-4o-mini ...
"""

import sys

from rich.console import Console
from rich.table import Table

from core import llm

# Hardcoded slugs. This is the anti pattern the repo exists to remove.
DEFAULT_MODELS = [
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
]

PROMPT = "In one short sentence, say why routing beats hardcoding a model."


def main() -> int:
    models = sys.argv[1:] or DEFAULT_MODELS
    console = Console()

    table = Table(title="smoke test: one prompt, three hardcoded models")
    table.add_column("model")
    table.add_column("ok")
    table.add_column("latency s", justify="right")
    table.add_column("cost $", justify="right")
    table.add_column("text or error")

    for model in models:
        result = llm.call(model, PROMPT, max_tokens=64)
        ok = "yes" if result["ok"] else "no"
        shown = result["text"] if result["ok"] else result["error"]
        table.add_row(result.get("model", model), ok,
                      f"{result['latency']:.2f}",
                      f"{result['cost']:.6f}",
                      _clip(shown))

    console.print(table)
    return 0


def _clip(text: str, width: int = 60) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
