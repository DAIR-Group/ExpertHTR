"""Numerical regressions for the original notebook's region-stripped CER."""

import re
import unittest

from experthtr.evaluation.metrics import register_metrics
from experthtr.runtime.context import RuntimeState


class PageCERTests(unittest.TestCase):
    def setUp(self):
        self.state = register_metrics(RuntimeState("test", {"re": re}))
        self.normalize = self.state["page_htr_text_for_cer"]

    def test_region_indices_are_not_transcription_characters(self):
        ref = self.normalize("[R1]: Red\n[R2]: River")
        pred = self.normalize("[R9]: Red\n[R7]: River")
        self.assertEqual(ref, "Red\nRiver")
        self.assertEqual(self.state["character_error_rate"](ref, pred), 0.0)

    def test_notebook_whitespace_and_legacy_header_policy(self):
        self.assertEqual(self.normalize(" \r\n[Para1]: A  B\r\n[R2]: C!\n "), "A  B\nC!")
        self.assertEqual(self.normalize("[R1]:\nA\n\n[R2]:\nB"), "A\n\n\nB")
        self.assertEqual(self.normalize("[Para 1]: A"), "[Para 1]: A")
        self.assertEqual(self.normalize("literal [R1]: R"), "literal [R1]: R")

    def test_markers_each_count_as_one_distinct_unit(self):
        ref = self.normalize("[R1]: A<del><gap>")
        pred = self.normalize("[R2]: A<gap><gap>")
        self.assertEqual(len(ref), 3)
        self.assertAlmostEqual(self.state["character_error_rate"](ref, pred), 1 / 3)

    def test_micro_cer_uses_total_reference_characters(self):
        tracker = self.state["MetricTracker"]()
        tracker.update(["b", "123456789"], ["a", "123456789"])
        self.assertAlmostEqual(tracker.compute()["cer"], 0.1)

    def test_internal_newlines_punctuation_and_insertions_count(self):
        cer = self.state["character_error_rate"]
        self.assertAlmostEqual(cer(self.normalize("[R1]: A\nB"), "AB"), 1 / 3)
        self.assertAlmostEqual(cer("A!", "A"), 0.5)
        self.assertEqual(cer("A", "AAAA"), 3.0)


if __name__ == "__main__":
    unittest.main()
