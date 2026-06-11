"""Unit tests for Pillar 7 citation validator."""

import os
import unittest

from . import flags
from . import pillar_7_citation as p7


class TestCitationDetection(unittest.TestCase):

    def test_ref_pattern(self):
        self.assertTrue(p7._has_citation_marker("7 FCUs on Level 3 (Ref: M-231)."))
        self.assertTrue(p7._has_citation_marker("(Ref: M-231) (Ref: M-232)"))

    def test_drawing_id(self):
        self.assertTrue(p7._has_citation_marker("Per drawing A-211, the corridor is 6 ft."))
        self.assertTrue(p7._has_citation_marker("M-303 shows 10 fan coil units."))

    def test_spec_section(self):
        self.assertTrue(p7._has_citation_marker("Per specification 23 81 26 section 3.4."))
        self.assertTrue(p7._has_citation_marker("Per section 07 92 00."))

    def test_sheet_reference(self):
        self.assertTrue(p7._has_citation_marker("see sheet M-201"))
        self.assertTrue(p7._has_citation_marker("[Page 289 p1]"))

    def test_section_symbol(self):
        self.assertTrue(p7._has_citation_marker("§3.4 of the spec."))

    def test_no_citation(self):
        self.assertFalse(p7._has_citation_marker("There are 7 fan coil units."))
        self.assertFalse(p7._has_citation_marker("The standard width is 36 inches."))


class TestFactualDetection(unittest.TestCase):

    def test_numeric_claim_is_factual(self):
        self.assertTrue(p7._is_factual("There are 7 FCUs on Level 3 of project 7224."))

    def test_drawing_reference_is_factual(self):
        self.assertTrue(p7._is_factual("Drawing M-303 specifies 10 fan coil units."))

    def test_short_sentence_not_factual(self):
        self.assertFalse(p7._is_factual("Yes."))
        self.assertFalse(p7._is_factual("No data."))

    def test_method_preamble_not_factual(self):
        self.assertFalse(p7._is_factual("Method: equipment-tag enumeration."))

    def test_conversational_not_factual(self):
        self.assertFalse(p7._is_factual("However, this is unclear."))
        self.assertFalse(p7._is_factual("If you have more details, please share."))


class TestValidate(unittest.TestCase):

    def test_empty_answer(self):
        r = p7.validate_citations("")
        self.assertEqual(r["verdict"], "no_factual_content")
        self.assertEqual(r["factual_count"], 0)

    def test_well_cited_answer(self):
        ans = ("7 FCUs on Level 3 (Ref: M-231). The mechanical schedule "
               "on drawing M-303 specifies these units (Ref: M-303).")
        r = p7.validate_citations(ans)
        self.assertGreaterEqual(r["density"], 0.5)
        self.assertEqual(r["verdict"], "pass")

    def test_uncited_answer(self):
        ans = ("There are 7 FCUs on Level 3 of project 7224. "
               "The mechanical schedule specifies these units. "
               "The system has 3500 CFM capacity. "
               "All units are rated for 800 watts.")
        r = p7.validate_citations(ans)
        self.assertLess(r["density"], 0.5)
        self.assertEqual(r["verdict"], "tag_unverified")

    def test_partial_citation(self):
        ans = ("7 FCUs on Level 3 (Ref: M-231). "
               "The system has 3500 CFM capacity. "
               "All units are rated for 800 watts.")
        # 1 of 3 cited = 33% < 50% threshold
        r = p7.validate_citations(ans)
        self.assertLess(r["density"], 0.5)
        self.assertEqual(r["verdict"], "tag_unverified")


class TestPostValidate(unittest.TestCase):

    def setUp(self):
        self._env = {k: os.environ.get(k) for k in [flags.MASTER_FLAG, "HG_PILLAR_7"]}
        for k in self._env:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_disabled_no_op(self):
        ans = "There are 7 FCUs on Level 3 of project 7224."
        out, meta = p7.post_validate(ans)
        self.assertEqual(out, ans)
        self.assertFalse(meta["applied"])

    def test_enabled_well_cited_no_tag(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_7"] = "true"
        ans = "7 FCUs on Level 3 (Ref: M-231). Mechanical schedule (Ref: M-303) specifies these."
        out, meta = p7.post_validate(ans)
        self.assertEqual(out, ans)
        self.assertEqual(meta["verdict"], "pass")

    def test_enabled_uncited_gets_tagged(self):
        os.environ[flags.MASTER_FLAG] = "true"
        os.environ["HG_PILLAR_7"] = "true"
        ans = ("There are 7 FCUs on Level 3. "
               "The system has 3500 CFM capacity. "
               "All units are rated 800 watts.")
        out, meta = p7.post_validate(ans)
        self.assertIn("Pillar 7 note", out)
        self.assertEqual(meta["verdict"], "tag_unverified")

    def test_force_enable(self):
        ans = "There are 7 FCUs."
        out, meta = p7.post_validate(ans, force_enable=True)
        self.assertTrue(meta["applied"])

    def test_master_flag_kill_switch(self):
        os.environ.pop(flags.MASTER_FLAG, None)
        os.environ["HG_PILLAR_7"] = "true"
        ans = "There are 7 FCUs."
        out, meta = p7.post_validate(ans)
        self.assertEqual(out, ans)
        self.assertFalse(meta["applied"])


class TestAppliedMetadata(unittest.TestCase):

    def test_metadata_shape(self):
        result = {"density": 0.4, "factual_count": 5, "cited_count": 2, "verdict": "tag_unverified"}
        meta = p7.applied_metadata(result)
        self.assertEqual(meta["pillar"], 7)
        self.assertEqual(meta["density"], 0.4)
        self.assertEqual(meta["verdict"], "tag_unverified")


if __name__ == "__main__":
    unittest.main()
