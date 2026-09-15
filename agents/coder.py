"""Coder. Writes the actual patch.

High stakes, because a wrong edit costs far more than the price of a strong
model. The spec leans hard on capability, so the router should pick a capable
model even when it is not the cheapest. If the critic rejects the output,
main.py runs this agent again with the failed model excluded, and the router
is forced to pick a different one.
"""

from agents.base import Agent


class Coder(Agent):
    name = "coder"
    required_capabilities = ["code", "reasoning"]
    stakes = "high"
    system = (
        "You are a careful engineer. Given the goal and the findings, write "
        "the smallest correct patch. Show the changed code and one line on "
        "why it fixes the problem."
    )
