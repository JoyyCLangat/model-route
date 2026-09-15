"""Joins the live catalog with the capability overlay and tracks health.

The catalog gives ids, context sizes, prices and modalities. The overlay in
capabilities.yaml gives the human opinion about what each family is good at,
matched by longest family prefix. Health is tracked here at run time, so a
model that starts failing is taken out of the running for a while and then
given another chance.
"""

import os
import time
from dataclasses import dataclass

import yaml

from core import catalog

CAPS_PATH = os.path.join(os.path.dirname(__file__), "capabilities.yaml")

# Circuit breaker settings. Two strikes takes a model out for a minute. These
# are deliberately blunt. A live demo needs a model that has a bad thirty
# seconds to step aside and come back, not a clever failure model.
FAILURE_THRESHOLD = 2
COOLDOWN_SECONDS = 60


@dataclass
class Model:
    id: str
    ctx: int
    in_price: float
    out_price: float
    caps: dict
    modalities: list
    is_free: bool

    @property
    def blended_price(self) -> float:
        # Output tokens dominate a real bill, so weight them more heavily. The
        # router scores cost with the same blend.
        return self.in_price + 3 * self.out_price


@dataclass
class _Health:
    fails: int = 0
    down_until: float = 0.0


class Registry:
    def __init__(self, free_only: bool = None):
        # free_only defaults to the FREE_MODE env var. Passing it explicitly is
        # handy in tests.
        if free_only is None:
            free_only = os.environ.get("FREE_MODE", "false").lower() == "true"
        self.free_only = free_only
        self._overlay = _load_overlay()
        self._models = [self._build(entry) for entry in catalog.fetch()]
        self._health: dict = {}

    def candidates(self) -> list:
        # Return the whole field. The router applies the hard filters and
        # records why each model was kept or dropped, so we hand it everything
        # rather than pre filtering here. The router reads self.free_only to
        # apply free mode, which keeps the excluded set visible on stage.
        return list(self._models)

    def _build(self, entry: dict) -> Model:
        return Model(
            id=entry["id"],
            ctx=entry["ctx"],
            in_price=entry["in_price"],
            out_price=entry["out_price"],
            modalities=entry["modalities"],
            is_free=entry["is_free"],
            caps=self._caps_for(entry["id"]),
        )

    def _caps_for(self, model_id: str) -> dict:
        # Match by longest family prefix, so a version bump keeps its scores.
        # An unknown model gets the neutral default rather than being dropped.
        best_key = None
        for family in self._overlay["families"]:
            if model_id.startswith(family):
                if best_key is None or len(family) > len(best_key):
                    best_key = family
        if best_key is not None:
            return dict(self._overlay["families"][best_key])
        return dict(self._overlay["default"])

    # Health, a small circuit breaker.

    def mark_failure(self, model_id: str) -> None:
        h = self._health.setdefault(model_id, _Health())
        h.fails += 1
        if h.fails >= FAILURE_THRESHOLD:
            h.down_until = time.time() + COOLDOWN_SECONDS

    def mark_success(self, model_id: str) -> None:
        # A success wipes the slate clean.
        self._health[model_id] = _Health()

    def is_healthy(self, model_id: str) -> bool:
        h = self._health.get(model_id)
        if h is None:
            return True
        if h.down_until and time.time() < h.down_until:
            return False
        if h.down_until and time.time() >= h.down_until:
            # Cooldown is over. Give it a clean slate and another chance.
            self._health[model_id] = _Health()
        return True


def _load_overlay() -> dict:
    with open(CAPS_PATH) as handle:
        return yaml.safe_load(handle)
