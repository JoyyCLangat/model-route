# model-route

A small demo where agents declare what a task needs and a router picks the
model at run time. Nothing is hardcoded. No model ever picks a model.

![The live trace panel, showing each step, the chosen model, the sub scores for every candidate, and the two retry cards](docs/panel.png)

> The panel above is the real view from replaying `runs/demo-run.jsonl`. Run
> `python main.py` and open `http://127.0.0.1:8000` to watch it live, or replay
> the recorded run with no key (see Quickstart).

## The idea

Most multi agent code hardwires a model to an agent, for example
`researcher = some_model`. That is a guess made at the time you write the code,
about a task you have not seen yet. It is wrong the moment a cheaper model gets
good enough, or a provider has a bad minute, or the task turns out to be bigger
than you expected.

Here, agents declare what a task needs, things like code, long_context or
speed, plus how much is at stake. The router scores every model the key can
reach against that need, plus price, context fit and speed, and picks at call
time. The same coder role can land on a cheap model for a small job and a
strong one for a risky job, with no change to the code.

No model ever picks a model. A cheap model may read the task and label it, but
that is all. Only Python chooses. This matters because models are biased about
models. They favour their own family, their knowledge of the field is frozen
at their training cutoff, and they will happily name a slug your account cannot
call. `experiments/self_bias.py` shows this in one screen.

When a model fails, or its work is rejected by a reviewer, the router picks
again with that model excluded. Autonomy is not choosing once. It is choosing
again after a failure. That retry is the closing moment of the demo.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Replay the recorded run. No API key needed, nothing is called.
python main.py --replay runs/demo-run.jsonl

# 2. A live run against the bundled buggy project. Needs a key in .env.
cp .env.example .env      # then add your OPENROUTER_API_KEY
python main.py --goal "Audit this repo and propose a fix" --path ./sample_target

# 3. A live run restricted to free models, so it costs nothing.
FREE_MODE=true python main.py
```

`FREE_MODE=true` restricts the router to free models, so you can run the whole
thing with no credit. `--replay` needs no key at all, so anyone can see the
trace. Add `--no-panel` to print only to the terminal.

## What happens when you run it

Four roles, one goal, and the router sending each step to a different model for
a different reason. This is a trimmed excerpt of the recorded run in
`runs/demo-run.jsonl`, showing the one line reason the router writes for each
pick:

```
step 1 planner    -> google/gemini-2.5-flash
  Needed reasoning, high stakes, so capability outweighed price.
step 2 researcher -> google/gemini-2.5-flash
  Needed long_context, about 1300 tokens of input, medium stakes.
step 3 summarizer -> moonshotai/kimi-k2
  Needed extraction, low stakes, so price was weighted heavily.
step 4 coder      -> google/gemini-2.5-flash   FAILED (http 503, overloaded)
step 6 coder      -> moonshotai/kimi-k2        (retry of step 4, gemini excluded)
step 8 coder      -> anthropic/claude-opus-4.1 (retry of step 4, two models excluded)
  Needed code and reasoning, high stakes, so capability outweighed price.
```

Read the low stakes summary step and the high stakes coder step side by side.
The cheap model wins the bulk job on price. On the coder job the first pick has
a bad moment and fails, so the router excludes it and tries again, then the
reviewer rejects that patch too, so the router excludes that model and lands on
the most capable one it has left. Same router, different weights, and a clean
recovery when a choice does not work out.

One more thing worth saying out loud. In that run the single call to the most
capable model cost about twenty times as much as every other call combined.
The router only reaches for it when the stakes justify it. That is the router
protecting your bill, and you can argue with it by editing one table of
weights.

Note on this recording: the routing in `runs/demo-run.jsonl` is real, produced
by `core/router.py` scoring a snapshot of the catalog. The token counts and
latencies are representative, since the recording was prepared without live
provider access. Run a live goal to record your own into `runs/latest.jsonl`.

## The steps

The repo is built in five stages, each a git tag, so you can walk the build
from a single call function up to the full demo. Check out any tag to see the
repo at that level:

```bash
git checkout step-3-router
```

| tag | what it adds | what it still cannot do |
| --- | --- | --- |
| `step-1-gateway` | one call function, three models, real cost and latency | everything is hardcoded, there is no choice |
| `step-2-registry` | live catalog, capability overlay, health tracking | knows the options, still cannot pick |
| `step-3-router` | hard filters and weighted scoring | picks per task, but no agents yet |
| `step-4-agents` | roles that declare needs and never name a model | no recovery when output is bad |
| `step-5-fallback` | critic loop, exclusion, retry on a different model | this is the finished demo |

### step 1, the gateway

Every provider call in the whole repo goes through one function,
`core/llm.py::call`. It returns a plain dict and it never raises. On any
problem it returns `{"ok": False, "error": ...}`, because a failed call is a
normal event the router recovers from, not a crash. `smoke.py` calls three
models and prints their cost and latency. It also shows the problem the rest of
the repo fixes: the three model names are typed into the file by hand.

```python
result = llm.call("some/model", "hello")
# {"ok": True, "text": "...", "cost": 0.0004, "latency": 1.2, ...}
```

### step 2, the registry

`core/catalog.py` fetches the live model list from OpenRouter and caches it for
a day, so prices and context limits are never a number typed into the code.
`core/capabilities.yaml` holds the one thing a catalog cannot tell you, a human
opinion about what each family is good at, keyed by family prefix so it
survives a version bump. `core/registry.py` joins the two and tracks health
with a small circuit breaker, so a model that fails twice steps aside for a
minute.

### step 3, the router

This is the file to read first, `core/router.py`. It filters, scores and picks,
in well under two hundred lines. First it drops any model that cannot do the
job. Then it scores the survivors from 0 to 1 on four things: capability for
the needed skills, cost, speed, and context fit.

```
capability = mean(model.caps[c] for c in required_capabilities)
cost_score = 1 - normalised(blended_price)     # cheapest survivor gets 1
speed_score = model.caps["speed"]
ctx_score   = 1 if est_tokens < 0.5 * ctx else 0.6
blended_price = in_price + 3 * out_price        # output dominates a real bill
score = w_cap*capability + w_cost*cost_score + w_speed*speed_score + w_ctx*ctx_score
```

The one idea worth taking away is that the weights change with the stakes:

| stakes | w_cap | w_cost | w_speed | w_ctx |
| --- | --- | --- | --- | --- |
| low | 0.30 | 0.45 | 0.15 | 0.10 |
| medium | 0.50 | 0.30 | 0.10 | 0.10 |
| high | 0.70 | 0.10 | 0.05 | 0.15 |

On a low stakes bulk job, cost carries almost half the weight, so the cheap
model wins. On a high stakes job, capability carries most of the weight, so the
capable model wins even though it costs more. It is the same code and the same
models both times. Only the weights move. The cost score is normalised across
the surviving candidates, not against a fixed constant, so adding a new model
reshapes the field on its own.

The router also returns why. Every decision carries the sub scores for each
survivor and the list of models it filtered out with the reason. On stage,
showing why a model was excluded is more convincing than showing the winner.

### step 4, the agents

An agent is a role with a job and an opinion about what that job needs. It
never names a model. You can grep the whole `agents/` directory for a provider
name and find nothing. A concrete agent is tiny:

```python
class Coder(Agent):
    name = "coder"
    required_capabilities = ["code", "reasoning"]
    stakes = "high"
    system = "You are a careful engineer..."
```

`agents/base.py` does the same six steps for every role: classify the task,
merge in the role's declared needs, ask the router to pick, call the chosen
model, record health, emit a trace event.

### step 5, the fallback

`agents/critic.py` reviews another agent's output and returns a plain verdict.
When it fails an output, `main.py` retries the step with the failed model added
to an exclude list, which forces the router to pick a different model. The
trace marks the retry with `retry_of`, so you can see the recovery. This is the
whole point: choosing again after a failure.

## Which model for what

This repo does not hand you an opinion. The scores in
`core/capabilities.yaml` are a starting guess by a human, clearly labelled as
such. The way to replace them with evidence is `bench/run.py`, which runs each
task on each model, records real cost and latency, and judges the answers
pairwise with a judge that is never from the same family as the answers it is
comparing.

See [RESULTS.md](RESULTS.md) for the measured table. To produce it against your
own tasks:

```bash
python bench/run.py
```

Any fixed ranking of models written into a README is stale within weeks. New
models ship, prices move, and last month's leader trails this month. That is
the whole reason the scores live in one small YAML file and the evidence comes
from a script you re run, not a paragraph of claims frozen in this file.

## Adding your own model or agent

A new model needs nothing in the code. If it is in the OpenRouter catalog, the
registry already sees it, with its live price and context window. To give it a
capability opinion, add one row to `core/capabilities.yaml` under its family
prefix. A model with no row gets neutral default scores and is still picked
when it fits, so it is never excluded just for being new.

A new agent needs a name, a system prompt, a capability list and a stakes
level. That is the whole file:

```python
from agents.base import Agent

class Translator(Agent):
    name = "translator"
    required_capabilities = ["reasoning"]
    stakes = "low"
    system = "You translate text faithfully and keep the formatting."
```

## Costs

A full live run of the five step pipeline is cheap, because most steps land on
small models and only the high stakes steps reach for a strong one. In the
recorded run the whole pipeline came to a few cents, and a single call to the
most capable model was most of that. Your real cost depends on live prices and
on how big your target files are. If you want zero cost, run with
`FREE_MODE=true`, which restricts the router to free models, or use `--replay`,
which calls nothing at all.

## Limitations

Honest list, because the talk is about being honest with the numbers.

- The capability scores in `capabilities.yaml` are hand set until you run the
  benchmark. They are a guess, not a measurement.
- The scores are keyed by family, so the router cannot tell a fast small
  variant from a large one in the same family. It gives them the same skill
  scores and lets price and context break the tie.
- The classifier adds one cheap call of latency before the real work. If it
  fails or returns junk, the keyword rules take over and the run still finishes.
- Pairwise judging in the benchmark is still one model judging another. Using a
  judge from a different family reduces the bias, it does not remove it.
- The circuit breaker is naive. Two failures and a fixed minute of cooldown. It
  is enough for a live demo, not a production breaker.
- Routing is per task, not per token. A long task is sent to one model, not
  split across several.

## Talk

Slides: (link to come)

Video: (link to come)

Speaker: Langat.

License: [MIT](LICENSE).
