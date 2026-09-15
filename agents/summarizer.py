"""Summarizer. Condenses findings into a short brief.

Low stakes and high volume, the kind of job you run a lot of. The spec leans
on price, so the router should pick a cheap model here. Same router as the
coder, different weights, different winner.
"""

from agents.base import Agent


class Summarizer(Agent):
    name = "summarizer"
    required_capabilities = ["extraction"]
    stakes = "low"
    system = (
        "You are a summarizer. Turn the input into at most five short bullet "
        "points. Keep only what a busy reader needs."
    )
