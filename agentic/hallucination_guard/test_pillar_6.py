"""
Unit tests for Pillar 6 anti-anchor module.

Run on the sandbox VM AFTER copying the package in place:
    cd /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31
    python -m pytest agentic/hallucination_guard/test_pillar_6.py -v

These tests are SELF-CONTAINED — no network, no model calls, no DB.
"""

import os
import unittest
from unittest import mock

from . import flags
from . import pillar_6_anti_anchor as p6


class TestFlags(unittest.TestCase):

    def setUp(self):
        # Snapshot env so each test starts clean
        self._env_snapshot = {
            k: os.environ.get(k) for k in [
                flags.MASTER_FLAG, "HG_PILLAR_1", "HG_PILLAR_4",
                "HG_PILLAR_6", "HG_PILLAR_7", "HG_PILLAR_11",
            ]
        }
        for k in self._env_snapshot:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env_snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_master_off_by_default(self):
        self.assertFalse(flags.is_enabled())

    def test_pillar_off_when_master_off(self):
        os.environ["HG_PILLAR_6"] = "true"
        # Master flag is OFF — pillar must report disabled
        self.assertFalse(flags.is_pillar_enabled(6))

    def test_pillar_on_when_both_on(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_6"] = "true"
        self.assertTrue(flags.is_pillar_enabled(6))

    def test_truthy_values(self):
        os.environ[flags.MASTER_FLAG] = "true"
        for val in ("true", "TRUE", "1", "yes", "on", "enabled"):
            os.environ["HG_PILLAR_6"] = val
            self.assertTrue(flags.is_pillar_enabled(6), f"Failed for value: {val}")

    def test_falsy_values(self):
        os.environ[flags.MASTER_FLAG] = "true"
        for val in ("false", "0", "no", "off", "", "anything-else"):
            os.environ["HG_PILLAR_6"] = val
            self.assertFalse(flags.is_pillar_enabled(6), f"Should be false for: '{val}'")

    def test_get_active_pillars(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_6"] = "true"
        os.environ["HG_PILLAR_7"] = "true"
        active = flags.get_active_pillars()
        self.assertEqual(active, {6, 7})

    def test_get_active_pillars_empty_when_master_off(self):
        os.environ["HG_PILLAR_6"] = "true"
        os.environ["HG_PILLAR_7"] = "true"
        # Master is OFF
        active = flags.get_active_pillars()
        self.assertEqual(active, set())

    def test_describe_state(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_6"] = "true"
        state = flags.describe_state()
        self.assertTrue(state["master"])
        self.assertTrue(state["pillars"][6])
        self.assertFalse(state["pillars"][1])
        self.assertIn(6, state["active_set"])


class TestPillar6Compose(unittest.TestCase):

    def setUp(self):
        self._env_snapshot = {
            k: os.environ.get(k) for k in [flags.MASTER_FLAG, "HG_PILLAR_6"]
        }
        for k in self._env_snapshot:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env_snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_disabled_returns_base_unchanged(self):
        base = "You are a helpful assistant for construction docs."
        result = p6.compose_system_prompt(base)
        self.assertEqual(result, base)

    def test_enabled_prepends_rules(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_6"] = "true"
        base = "You are a helpful assistant for construction docs."
        result = p6.compose_system_prompt(base)
        self.assertGreater(len(result), len(base))
        self.assertIn("NON-NEGOTIABLE", result)
        self.assertIn("Hold the line.", result)
        self.assertIn(base, result)
        # Marker should be present
        self.assertIn("Project-specific instructions below", result)

    def test_force_enable_overrides_flag(self):
        # Even though flags are unset (OFF), force_enable=True should prepend
        base = "x"
        result = p6.compose_system_prompt(base, force_enable=True)
        self.assertIn("NON-NEGOTIABLE", result)

    def test_force_disable_overrides_flag(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_6"] = "true"
        base = "x"
        result = p6.compose_system_prompt(base, force_enable=False)
        self.assertEqual(result, base)

    def test_all_six_rules_present(self):
        rules_text = p6.get_anti_anchor_prompt()
        for i in range(1, 7):
            self.assertIn(f"{i}.", rules_text)

    def test_applied_metadata(self):
        meta = p6.applied_metadata()
        self.assertEqual(meta["pillar"], 6)
        self.assertEqual(meta["name"], "anti_anchor")
        self.assertEqual(meta["rules_count"], 6)
        self.assertIn("version", meta)

    def test_drift_placeholder_returns_false(self):
        # Placeholder for full UMAP detector; safe default = False
        self.assertFalse(p6.is_drift_detected())


class TestSafetyInvariants(unittest.TestCase):
    """Invariants from §10.3 HARD RULES."""

    def test_master_flag_acts_as_kill_switch(self):
        os.environ.pop(flags.MASTER_FLAG, None)
        os.environ["HG_PILLAR_6"] = "true"
        # Master OFF → pillar must be OFF
        self.assertFalse(flags.is_pillar_enabled(6))
        # And compose_system_prompt is a no-op
        base = "base"
        self.assertEqual(p6.compose_system_prompt(base), base)

    def test_no_side_effects_on_disabled_call(self):
        # Disabled call must not log, must not raise, must not modify input.
        base = "an immutable system prompt"
        result = p6.compose_system_prompt(base)
        self.assertEqual(result, base)
        self.assertEqual(base, "an immutable system prompt")  # unchanged


if __name__ == "__main__":
    unittest.main()
