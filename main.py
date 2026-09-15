"""The demo runner.

Five roles, one goal. A planner breaks the goal down, a researcher reads the
target, a summarizer condenses, a coder writes a patch, and a critic reviews
it. Each step is routed to a model at call time by core/router.py. If the
critic rejects the patch, the coder runs again with the failed model excluded,
and the router is forced to choose differently. That retry is the closing
moment of the demo.

Three ways to run it:

    python main.py --goal "..." --path ./sample_target   live, needs a key
    python main.py --free                                live, free models only
    python main.py --replay runs/demo-run.jsonl          recorded, no key needed

Add --no-panel to skip the web panel and print only to the terminal.
"""

import argparse
import os
import threading
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core import trace

console = Console()

DEFAULT_GOAL = "Audit this repo and propose a fix"
DEFAULT_PATH = "sample_target"
MAX_ATTEMPTS = 3


def main() -> int:
    args = _parse_args()
    if args.free:
        os.environ["FREE_MODE"] = "true"

    _reset_latest()

    server = None
    if not args.no_panel:
        server = _start_panel()

    if args.replay:
        code = run_replay(args.replay, args.speed)
    else:
        code = run_live(args.goal, args.path)

    if server is not None:
        console.print("\nPanel is still live. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return code


def run_replay(path: str, speed: float = 1.0) -> int:
    if not os.path.exists(path):
        console.print(f"[red]No recorded run at {path}[/red]")
        return 1
    console.rule("replay")
    console.print(f"Replaying [bold]{path}[/bold]. No API key needed.\n")
    total = 0.0
    count = 0
    for event in trace.replay(path, speed=speed):
        _print_event(event)
        total += (event.get("result") or {}).get("cost", 0.0) or 0.0
        count += 1
    console.print(f"\nReplayed {count} events. Recorded cost ${total:.4f}.")
    return 0


def run_live(goal: str, path: str) -> int:
    # Imported here so replay never needs the live dependencies or the network.
    from agents import base
    from agents.planner import Planner
    from agents.researcher import Researcher
    from agents.summarizer import Summarizer
    from agents.coder import Coder
    from agents.critic import Critic
    from core.registry import Registry

    console.rule("live run")
    console.print(f"Goal: [bold]{goal}[/bold]\n")
    target = _read_target(path)

    try:
        base.reset_steps()
        registry = Registry()
    except Exception as exc:
        console.print(f"[red]Could not build the registry: {exc}[/red]")
        console.print("Check your network, or use --replay to run offline.")
        return 1

    planner = Planner(registry)
    researcher = Researcher(registry)
    summarizer = Summarizer(registry)
    coder = Coder(registry)
    critic = Critic(registry)

    start = time.time()
    used = []       # list of (role, model) in order
    total_cost = 0.0
    retries = 0

    def record(role, outcome):
        nonlocal total_cost
        _print_event(outcome["event"])
        used.append((role, outcome["event"]["result"]["model"]))
        total_cost += outcome["event"]["result"]["cost"] or 0.0

    # The goal is passed as context, not baked into the task string, so the
    # classifier labels each step by the role's own verb (plan, research and
    # so on) rather than by a stray word in the goal.
    goal_note = f"Goal: {goal}"

    # 1. Plan the work.
    plan = planner.run("Plan the steps needed to reach the goal",
                       context=goal_note)
    record("planner", plan)

    # 2. Research the target. Large input, so the router should favour a big
    #    context model.
    research = researcher.run("Research the target files for the facts and risks",
                              context=f"{goal_note}\n\n{target}")
    record("researcher", research)
    findings = _text(research, fallback=target)

    # 3. Summarize the findings. Low stakes and bulk, so price should win.
    summary = summarizer.run("Summarize these findings into a short brief",
                             context=findings)
    record("summarizer", summary)
    brief = _text(summary, fallback=findings)

    # 4. Write the patch, then 5. review it, and retry on a rejection.
    exclude = []
    first_code_step = None
    for _ in range(MAX_ATTEMPTS):
        patch = coder.run("Write the smallest patch that fixes the bugs",
                          context=f"{goal_note}\n\nFindings:\n{brief}",
                          exclude=exclude, retry_of=first_code_step)
        record("coder", patch)
        if first_code_step is None:
            first_code_step = patch["event"]["step"]

        reviewed = critic.review(goal, _text(patch), exclude=exclude)
        record("critic", reviewed)
        verdict = reviewed["verdict"]
        mark = "PASS" if verdict["pass"] else "FAIL"
        console.print(f"  critic verdict: [bold]{mark}[/bold] {verdict['reason']}\n")
        if verdict["pass"]:
            break

        # Rejected. Exclude the model that wrote the failed patch and try
        # again, which forces the router to a different model.
        failed = patch["event"]["result"]["model"]
        if failed and failed not in exclude:
            exclude.append(failed)
        retries += 1

    _summary(used, time.time() - start, total_cost, retries)
    return 0


def _text(outcome: dict, fallback: str = "") -> str:
    result = outcome["result"]
    return result.get("text", "") if result.get("ok") else fallback


def _print_event(event: dict) -> None:
    decision = event.get("decision") or {}
    result = event.get("result") or {}
    chosen = decision.get("chosen") or "(none)"
    ok = result.get("ok")
    tag = "[green]ok[/green]" if ok else "[red]failed[/red]"

    lines = []
    header = f"step {event.get('step')}  [bold]{event.get('agent')}[/bold]"
    if event.get("retry_of"):
        header += f"  [yellow](retry of step {event['retry_of']})[/yellow]"
    lines.append(header)
    lines.append(f"task: {event.get('task', '')[:80]}")
    lines.append(f"classifier: {event.get('classifier')}   chosen: [bold]{chosen}[/bold]  {tag}")
    if ok:
        lines.append(f"cost ${result.get('cost', 0.0):.5f}   latency {result.get('latency', 0.0):.2f}s"
                     f"   tokens {result.get('in_tokens', 0)} in / {result.get('out_tokens', 0)} out")
    elif result.get("error"):
        lines.append(f"[red]{result['error'][:90]}[/red]")
    lines.append(f"why: {decision.get('reason', '')}")

    # Show the top few survivors and a couple of the excluded, since the
    # excluded set is the convincing part on stage.
    for row in (decision.get("breakdown") or [])[:3]:
        lines.append(f"  keep {row['id']}  score {row['score']}  "
                     f"(cap {row['capability']} cost {row['cost_score']} "
                     f"spd {row['speed_score']} ctx {row['ctx_score']})")
    for row in (decision.get("filtered_out") or [])[:2]:
        lines.append(f"  [dim]drop {row['id']}: {row['reason']}[/dim]")

    colour = "cyan" if ok else "red"
    if event.get("retry_of"):
        colour = "yellow"
    console.print(Panel("\n".join(lines), border_style=colour, expand=True))


def _summary(used, elapsed, total_cost, retries) -> None:
    table = Table(title="run summary")
    table.add_column("order", justify="right")
    table.add_column("role")
    table.add_column("model that ran it")
    for i, (role, model) in enumerate(used, start=1):
        table.add_row(str(i), role, model or "(none)")
    console.print(table)
    console.print(f"total cost ${total_cost:.5f}   total time {elapsed:.1f}s   "
                  f"retries {retries}")
    distinct = sorted({m for _, m in used if m})
    console.print(f"distinct models used: {len(distinct)}  ->  {', '.join(distinct) or '(none)'}")


def _read_target(path: str) -> str:
    if not path or not os.path.isdir(path):
        return "def add(a, b):\n    return a - b\n"
    chunks = []
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if name.endswith((".py", ".md", ".txt")):
                full = os.path.join(root, name)
                try:
                    with open(full) as handle:
                        rel = os.path.relpath(full, path)
                        chunks.append(f"# file: {rel}\n{handle.read()}")
                except OSError:
                    pass
    return "\n\n".join(chunks) if chunks else "(no readable files found)"


def _reset_latest() -> None:
    # Start each run with a clean live log, so the panel does not show the
    # previous run mixed in.
    os.makedirs("runs", exist_ok=True)
    open(os.path.join("runs", "latest.jsonl"), "w").close()


def _start_panel():
    import uvicorn
    from panel.server import app, port_from_env
    port = port_from_env()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    # Give the socket a moment to come up before the run starts emitting.
    time.sleep(0.5)
    console.print(f"panel live at http://127.0.0.1:{port}\n")
    return server


def _parse_args():
    parser = argparse.ArgumentParser(description="Autonomous model router demo.")
    parser.add_argument("--goal", default=DEFAULT_GOAL,
                        help="the goal for the agent team")
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help="folder of files to work on")
    parser.add_argument("--replay", default=None,
                        help="replay a recorded JSONL trace, no key needed")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="replay speed multiplier, higher is faster")
    parser.add_argument("--free", action="store_true",
                        help="restrict routing to free models, so it costs nothing")
    parser.add_argument("--no-panel", action="store_true",
                        help="do not start the web trace panel")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
