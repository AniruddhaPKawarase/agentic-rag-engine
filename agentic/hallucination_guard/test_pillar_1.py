"""Unit tests for Pillar 1 anaphora resolver."""

import os
import unittest
from unittest.mock import MagicMock

from . import flags
from . import pillar_1_anaphora as p1


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


def _fake_client(canned_text: str):
    client = MagicMock()
    client.messages.create.return_value = _FakeMessage(canned_text)
    return client


class TestAnaphoraSignal(unittest.TestCase):

    def test_pronoun_its(self):
        self.assertTrue(p1._has_anaphora_signal("What's its CFM rating?"))

    def test_pronoun_that(self):
        self.assertTrue(p1._has_anaphora_signal("Is that correct?"))

    def test_implicit_next(self):
        self.assertTrue(p1._has_anaphora_signal("And on the next sheet?"))

    def test_same(self):
        self.assertTrue(p1._has_anaphora_signal("Same for L3?"))

    def test_short_question_treated_as_followup(self):
        # short question ending in ? is likely a followup
        self.assertTrue(p1._has_anaphora_signal("How big?"))

    def test_self_contained_query_no_signal(self):
        # A self-contained question typically won't trigger
        self.assertFalse(p1._has_anaphora_signal("How many FCUs are on Level 3 of project 7224 in the mechanical drawings"))


class TestHistoryFormatting(unittest.TestCase):

    def test_empty_history(self):
        self.assertIn("no prior", p1._format_history_block([]))

    def test_single_turn(self):
        hist = [{"role": "user", "content": "Hello"}]
        block = p1._format_history_block(hist)
        self.assertIn("user: Hello", block)

    def test_truncation(self):
        long_content = "a" * 500
        hist = [{"role": "user", "content": long_content}]
        block = p1._format_history_block(hist)
        self.assertLess(len(block), 300)
        self.assertIn("...", block)

    def test_max_turns(self):
        hist = [{"role": "user", "content": f"turn{i}"} for i in range(20)]
        block = p1._format_history_block(hist, max_turns=3)
        # Only the most-recent 3 turns appear
        self.assertIn("turn19", block)
        self.assertIn("turn18", block)
        self.assertIn("turn17", block)
        self.assertNotIn("turn16", block)


class TestResolveAnaphoraFlagGate(unittest.TestCase):

    def setUp(self):
        self._env = {k: os.environ.get(k) for k in [flags.MASTER_FLAG, "HG_PILLAR_1"]}
        for k in self._env:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_disabled_returns_original(self):
        # Flags off -> resolver is a no-op even with anaphora present
        client = _fake_client("REWRITTEN VERSION")
        out = p1.resolve_anaphora(
            "What's its CFM?",
            history=[{"role": "user", "content": "Show me FCU-101"}],
            anthropic_client=client,
        )
        self.assertEqual(out, "What's its CFM?")
        client.messages.create.assert_not_called()

    def test_enabled_returns_rewrite(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_1"] = "true"
        client = _fake_client("What is the CFM rating of FCU-101?")
        out = p1.resolve_anaphora(
            "What's its CFM?",
            history=[{"role": "user", "content": "Show me FCU-101"}],
            anthropic_client=client,
        )
        self.assertEqual(out, "What is the CFM rating of FCU-101?")
        client.messages.create.assert_called_once()


class TestResolveAnaphoraNoOpPaths(unittest.TestCase):

    def setUp(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_1"] = "true"

    def tearDown(self):
        os.environ.pop(flags.MASTER_FLAG, None)
        os.environ.pop("HG_PILLAR_1", None)

    def test_empty_query(self):
        client = _fake_client("X")
        self.assertEqual(p1.resolve_anaphora("", history=[{"role": "user", "content": "x"}], anthropic_client=client), "")
        client.messages.create.assert_not_called()

    def test_no_history(self):
        client = _fake_client("X")
        self.assertEqual(p1.resolve_anaphora("What's its CFM?", history=[], anthropic_client=client), "What's its CFM?")
        client.messages.create.assert_not_called()

    def test_no_anaphora_signal(self):
        # Self-contained long query -> short-circuits before LLM call
        client = _fake_client("X")
        long_q = "How many fan coil units are on Level 3 of project 7224 according to the mechanical schedule on sheet M-601"
        out = p1.resolve_anaphora(
            long_q,
            history=[{"role": "user", "content": "Earlier turn"}],
            anthropic_client=client,
        )
        self.assertEqual(out, long_q)
        client.messages.create.assert_not_called()

    def test_force_enable_overrides_flag(self):
        os.environ.pop(flags.MASTER_FLAG, None)
        os.environ.pop("HG_PILLAR_1", None)
        client = _fake_client("REWRITTEN")
        out = p1.resolve_anaphora(
            "What's its CFM?",
            history=[{"role": "user", "content": "FCU-101"}],
            anthropic_client=client,
            force_enable=True,
        )
        self.assertEqual(out, "REWRITTEN")


class TestResolveAnaphoraDefensive(unittest.TestCase):

    def setUp(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_1"] = "true"

    def tearDown(self):
        os.environ.pop(flags.MASTER_FLAG, None)
        os.environ.pop("HG_PILLAR_1", None)

    def test_empty_rewrite_falls_back(self):
        client = _fake_client("")
        out = p1.resolve_anaphora(
            "What's its CFM?",
            history=[{"role": "user", "content": "x"}],
            anthropic_client=client,
        )
        self.assertEqual(out, "What's its CFM?")

    def test_too_long_rewrite_falls_back(self):
        client = _fake_client("x" * 800)
        out = p1.resolve_anaphora(
            "What's its CFM?",
            history=[{"role": "user", "content": "x"}],
            anthropic_client=client,
        )
        self.assertEqual(out, "What's its CFM?")

    def test_anthropic_exception_falls_back(self):
        client = MagicMock()
        client.messages.create.side_effect = Exception("API down")
        out = p1.resolve_anaphora(
            "What's its CFM?",
            history=[{"role": "user", "content": "x"}],
            anthropic_client=client,
        )
        self.assertEqual(out, "What's its CFM?")

    def test_preamble_stripped(self):
        client = _fake_client("Rewritten query: What is the CFM rating of FCU-101?")
        out = p1.resolve_anaphora(
            "What's its CFM?",
            history=[{"role": "user", "content": "x"}],
            anthropic_client=client,
        )
        self.assertEqual(out, "What is the CFM rating of FCU-101?")

    def test_identical_rewrite_falls_back(self):
        client = _fake_client("What's its CFM?")
        out = p1.resolve_anaphora(
            "What's its CFM?",
            history=[{"role": "user", "content": "x"}],
            anthropic_client=client,
        )
        self.assertEqual(out, "What's its CFM?")


class TestAppliedMetadata(unittest.TestCase):

    def test_metadata_shape(self):
        meta = p1.applied_metadata(was_rewritten=True, original="What's its CFM?", rewritten="FCU-101 CFM?")
        self.assertEqual(meta["pillar"], 1)
        self.assertEqual(meta["name"], "anaphora_resolver")
        self.assertTrue(meta["rewritten"])
        self.assertIn("CFM", meta["original_preview"])

    def test_metadata_no_rewrite(self):
        meta = p1.applied_metadata(was_rewritten=False)
        self.assertFalse(meta["rewritten"])
        self.assertIsNone(meta["original_preview"])


if __name__ == "__main__":
    unittest.main()
