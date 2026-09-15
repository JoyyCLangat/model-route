"""Researcher. Reads the target material and reports what matters.

Usually a large input, so the task spec carries a big context estimate and the
router should favour a model with a large window. The researcher does not know
or care which model that turns out to be.
"""

from agents.base import Agent


class Researcher(Agent):
    name = "researcher"
    required_capabilities = ["long_context", "reasoning"]
    stakes = "medium"
    system = (
        "You are a researcher. Read the material and list the concrete facts, "
        "risks and bugs that matter for the goal. Be specific and brief."
    )
