"""Unit tests for the router.

These are the proof that the idea works, so they are written to be read. They
use a fake registry with a couple of made up models, so they need no network
and no key. Each test states a claim from the talk and checks it:

  low stakes picks the cheap model,
  high stakes picks the capable model,
  an oversized context filters a small window model out,
  free mode drops the paid models,
  an unhealthy model is skipped.

The model ids here are deliberately plain labels, not real slugs. The router
does not care what a model is called. It cares about the numbers.
"""

import unittest

from core.registry import Model
from core import router


def cheap_model():
    # Low price, modest ability, small context, fast.
    return Model(
        id="vendor/cheap-small", ctx=8000, in_price=0.10, out_price=0.30,
        caps={"code": 0.6, "reasoning": 0.6, "long_context": 0.5,
              "extraction": 0.7, "vision": 0.3, "speed": 0.9},
        modalities=["text"], is_free=False)


def capable_model():
    # High price, high ability, large context, slower.
    return Model(
        id="vendor/capable-big", ctx=200000, in_price=3.0, out_price=15.0,
        caps={"code": 0.95, "reasoning": 0.95, "long_context": 0.9,
              "extraction": 0.8, "vision": 0.8, "speed": 0.4},
        modalities=["text", "image"], is_free=False)


def free_model():
    return Model(
        id="vendor/free-tiny", ctx=32000, in_price=0.0, out_price=0.0,
        caps={"code": 0.6, "reasoning": 0.6, "long_context": 0.6,
              "extraction": 0.6, "vision": 0.3, "speed": 0.7},
        modalities=["text"], is_free=True)


class FakeRegistry:
    """A stand in for the real registry, with no catalog and no network."""

    def __init__(self, models, free_only=False, unhealthy=()):
        self._models = list(models)
        self.free_only = free_only
        self._unhealthy = set(unhealthy)

    def candidates(self):
        return list(self._models)

    def is_healthy(self, model_id):
        return model_id not in self._unhealthy


class RouterTests(unittest.TestCase):

    def test_low_stakes_picks_the_cheap_model(self):
        registry = FakeRegistry([cheap_model(), capable_model()])
        spec = {"task_type": "summarize",
                "required_capabilities": ["reasoning"],
                "est_context_tokens": 1000, "stakes": "low"}
        decision = router.select(spec, registry)
        self.assertEqual(decision.chosen, "vendor/cheap-small")

    def test_high_stakes_picks_the_capable_model(self):
        registry = FakeRegistry([cheap_model(), capable_model()])
        spec = {"task_type": "code_edit",
                "required_capabilities": ["code", "reasoning"],
                "est_context_tokens": 1000, "stakes": "high"}
        decision = router.select(spec, registry)
        self.assertEqual(decision.chosen, "vendor/capable-big")

    def test_oversized_context_filters_the_small_model_out(self):
        registry = FakeRegistry([cheap_model(), capable_model()])
        spec = {"task_type": "research",
                "required_capabilities": ["long_context"],
                "est_context_tokens": 100000, "stakes": "medium"}
        decision = router.select(spec, registry)
        # Only the large window model can hold this input, so it must win.
        self.assertEqual(decision.chosen, "vendor/capable-big")
        dropped = [f["id"] for f in decision.filtered_out]
        self.assertIn("vendor/cheap-small", dropped)

    def test_free_mode_drops_the_paid_models(self):
        registry = FakeRegistry([cheap_model(), capable_model(), free_model()],
                                free_only=True)
        spec = {"task_type": "summarize",
                "required_capabilities": ["reasoning"],
                "est_context_tokens": 1000, "stakes": "low"}
        decision = router.select(spec, registry)
        self.assertEqual(decision.chosen, "vendor/free-tiny")
        dropped = [f["id"] for f in decision.filtered_out]
        self.assertIn("vendor/cheap-small", dropped)
        self.assertIn("vendor/capable-big", dropped)

    def test_unhealthy_model_is_skipped(self):
        # The capable model would win a high stakes job, but it is unhealthy,
        # so the router must fall back to the cheap one.
        registry = FakeRegistry([cheap_model(), capable_model()],
                                unhealthy=["vendor/capable-big"])
        spec = {"task_type": "code_edit",
                "required_capabilities": ["code", "reasoning"],
                "est_context_tokens": 1000, "stakes": "high"}
        decision = router.select(spec, registry)
        self.assertEqual(decision.chosen, "vendor/cheap-small")

    def test_exclude_forces_a_different_pick(self):
        # This is what a retry after a rejection does: exclude the model that
        # just failed and pick again.
        registry = FakeRegistry([cheap_model(), capable_model()])
        spec = {"task_type": "code_edit",
                "required_capabilities": ["code", "reasoning"],
                "est_context_tokens": 1000, "stakes": "high",
                "exclude": ["vendor/capable-big"]}
        decision = router.select(spec, registry)
        self.assertEqual(decision.chosen, "vendor/cheap-small")

    def test_no_survivors_returns_an_empty_choice(self):
        registry = FakeRegistry([cheap_model()],
                                unhealthy=["vendor/cheap-small"])
        spec = {"task_type": "other", "required_capabilities": ["reasoning"],
                "est_context_tokens": 1000, "stakes": "medium"}
        decision = router.select(spec, registry)
        self.assertEqual(decision.chosen, "")
        self.assertEqual(decision.breakdown, [])


if __name__ == "__main__":
    unittest.main()
