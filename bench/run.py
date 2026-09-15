"""Benchmark the models against the tasks in tasks.yaml and write RESULTS.md.

Why this exists: the capability scores in core/capabilities.yaml are a human
guess. This script replaces the guess with evidence. It runs each task on each
model, records real cost and latency, then judges the answers.

Two decisions that matter for honest results:

  1. The judge model is never from the same family as either answer it is
     comparing. Models favour their own family, so a same family judge would
     quietly inflate that family. See experiments/self_bias.py for why.

  2. Judging is pairwise, A against B, not an absolute score out of ten.
     Absolute ratings from a model are noisy and drift. Asking which of two
     answers is better for a concrete rubric is a steadier signal.

This needs a key and network, so it does not run in CI. By default it only
runs the families that capabilities.yaml has an opinion about, to keep a run
cheap. Pass --all to run every model the key can reach, or --models a,b,c to
name your own.
"""

import argparse
import datetime
import itertools
import os
import subprocess

import yaml

from core import llm
from core.registry import Registry

TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks.yaml")
RESULTS_PATH = "RESULTS.md"


def main() -> int:
    args = _parse_args()
    tasks = _load_tasks()
    registry = Registry()
    families = _family_keys(registry)
    models = _pick_models(registry, args, families)
    if len(models) < 2:
        print("Need at least two models to judge pairwise. "
              "Try --all or --models a,b.")
        return 1

    print(f"Running {len(tasks)} tasks on {len(models)} models.")
    # answers[task_id][model_id] = result dict from llm.call
    answers: dict = {}
    for task in tasks:
        answers[task["id"]] = {}
        for model in models:
            result = llm.call(model, task["task"], max_tokens=700)
            answers[task["id"]][model] = result
            state = "ok" if result["ok"] else f"failed ({result['error'][:40]})"
            print(f"  {task['id']:26} {model:34} {state}")

    wins = _judge_all(tasks, answers, models, families)
    _write_results(tasks, answers, models, wins)
    print(f"\nWrote {RESULTS_PATH}")
    return 0


def _judge_all(tasks, answers, models, families):
    # wins[model_id] = number of pairwise comparisons this model won.
    wins = {m: 0 for m in models}
    for task in tasks:
        good = [m for m in models if answers[task["id"]][m]["ok"]]
        for a, b in itertools.combinations(good, 2):
            judge = _pick_judge(models, families, exclude={a, b})
            if judge is None:
                continue
            winner = _judge_pair(
                judge, task, answers[task["id"]][a]["text"],
                answers[task["id"]][b]["text"])
            if winner == "A":
                wins[a] += 1
            elif winner == "B":
                wins[b] += 1
            # A tie adds nothing to either side.
    return wins


def _judge_pair(judge, task, answer_a, answer_b) -> str:
    system = (
        "You are judging two answers to the same task against a rubric. Reply "
        "with exactly one character: A if the first answer is better, B if the "
        "second is better, or T for a tie. No other text."
    )
    prompt = (
        f"Task:\n{task['task']}\n\nRubric:\n{task['rubric']}\n\n"
        f"Answer A:\n{answer_a}\n\nAnswer B:\n{answer_b}\n\nYour verdict:"
    )
    result = llm.call(judge, prompt, system=system, max_tokens=4)
    if not result["ok"]:
        return "T"
    letter = (result["text"] or "").strip().upper()[:1]
    return letter if letter in ("A", "B") else "T"


def _pick_judge(models, families, exclude):
    # A judge from a family that is not in the excluded set of families.
    bad_families = {families.get(m) for m in exclude}
    for m in models:
        if m in exclude:
            continue
        if families.get(m) not in bad_families:
            return m
    return None


def _family_keys(registry) -> dict:
    # Map each candidate model id to its family prefix, or "unknown".
    overlay = registry._overlay  # noqa: SLF001, we own this object
    fams = list(overlay["families"].keys())
    mapping = {}
    for model in registry.candidates():
        best = None
        for fam in fams:
            if model.id.startswith(fam) and (best is None or len(fam) > len(best)):
                best = fam
        mapping[model.id] = best or "unknown"
    return mapping


def _pick_models(registry, args, families) -> list:
    ids = [m.id for m in registry.candidates()]
    if args.models:
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        return [i for i in ids if i in wanted]
    if args.all:
        return ids
    # Default: only the families capabilities.yaml has an opinion about, and
    # only one model per family prefix to keep a run cheap. First match wins.
    picked, seen = [], set()
    for i in sorted(ids):
        fam = families.get(i)
        if fam and fam != "unknown" and fam not in seen:
            picked.append(i)
            seen.add(fam)
    return picked


def _load_tasks() -> list:
    with open(TASKS_PATH) as handle:
        return yaml.safe_load(handle)["tasks"]


def _commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_results(tasks, answers, models, wins) -> None:
    today = datetime.date.today().isoformat()
    lines = [
        "# Benchmark results",
        "",
        f"Generated: {today}",
        f"Commit: {_commit_sha()}",
        f"Models: {', '.join(models)}",
        "",
        "These numbers replace the guessed scores in core/capabilities.yaml.",
        "The judge for each comparison is from a different family than both",
        "answers, and judging is pairwise. See bench/run.py for the method.",
        "",
        "## Per task",
    ]
    for task in tasks:
        lines += ["", f"### {task['id']} ({task['category']})", "",
                  f"rubric: {task['rubric']}", "",
                  "| model | ok | latency s | cost $ |",
                  "| --- | --- | --- | --- |"]
        for model in models:
            r = answers[task["id"]][model]
            ok = "yes" if r["ok"] else "no"
            lines.append(f"| {model} | {ok} | {r['latency']:.2f} | "
                         f"{r['cost']:.5f} |")

    lines += ["", "## Aggregate", "",
              "| model | tasks ok | total cost $ | avg latency s | pairwise wins |",
              "| --- | --- | --- | --- | --- |"]
    for model in models:
        oks = sum(1 for t in tasks if answers[t["id"]][model]["ok"])
        cost = sum(answers[t["id"]][model]["cost"] for t in tasks)
        lat = [answers[t["id"]][model]["latency"] for t in tasks]
        avg = sum(lat) / len(lat) if lat else 0.0
        lines.append(f"| {model} | {oks}/{len(tasks)} | {cost:.5f} | "
                     f"{avg:.2f} | {wins.get(model, 0)} |")

    with open(RESULTS_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark models and write RESULTS.md")
    parser.add_argument("--all", action="store_true",
                        help="run every model the key can reach")
    parser.add_argument("--models", default=None,
                        help="comma separated model ids to run")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
