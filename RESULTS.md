# Benchmark results

Status: not run yet.

This file is meant to hold measured evidence: how each model did on the tasks
in `bench/tasks.yaml`, with real cost and latency, and pairwise judgements
from a judge that is never from the same family as the answers it compares.

It has not been run in this environment, because a real benchmark needs an
`OPENROUTER_API_KEY` and live network access to the providers, and this repo
was prepared without either. So there are no numbers here yet, on purpose. The
capability scores the router uses still come from the human guess in
`core/capabilities.yaml`, which is exactly the thing this benchmark exists to
replace.

## How to produce real numbers

```bash
# set OPENROUTER_API_KEY in your .env first
python bench/run.py            # runs one model per known family, cheap
python bench/run.py --all      # runs every model your key can reach
python bench/run.py --models anthropic/claude-opus-4.1,google/gemini-2.5-flash
```

When it finishes it overwrites this file with the run date, the commit SHA,
the exact model slugs, the cost and latency per task, and an aggregate table.
Then update `core/capabilities.yaml` so the scores match what you measured, and
commit both files together.

## An honest note

Any fixed ranking of models goes stale within weeks. New models ship, prices
change, and a model that led last month trails this month. That is the whole
reason the scores live in one small YAML file and the evidence comes from a
script you can re run, rather than a paragraph of claims written into this
README once and left to rot.
