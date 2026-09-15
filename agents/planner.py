"""Planner. Breaks a goal into a short list of concrete subtasks.

High stakes and a single call, because a bad plan wastes every step after it.
Notice there is no model here. The planner says it needs strong reasoning and
lets the router decide what actually runs.
"""

from agents.base import Agent


class Planner(Agent):
    name = "planner"
    required_capabilities = ["reasoning"]
    stakes = "high"
    system = (
        "You are a planner. Break the goal into three to six concrete "
        "subtasks. Return one subtask per line. No numbering, no prose."
    )
