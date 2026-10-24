"""Selection of picture-refinement rounds; runs without Blender or a pose detector."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measure"))
from picture import assess_round  # noqa: E402


ROWS = [("start", "torso_lean", 0.0), ("end", "torso_lean", 22.0), ("end", "hip_included", 70.0)]
START, LEAN, HIP = [(p, m) for p, m, _ in ROWS]


class PictureRoundTests(unittest.TestCase):
    def test_losing_the_bad_reading_cannot_beat_a_complete_round(self):
        complete = assess_round(ROWS, {START: 0.0, LEAN: 9.0, HIP: 0.0}, 5)
        disappeared = assess_round(ROWS, {START: 0.0, HIP: 0.0}, 5)
        self.assertLess(complete["score"], disappeared["score"])
        self.assertEqual(disappeared["missing"], [LEAN])
        self.assertEqual(disappeared["status"], "incomplete")

    def test_no_detected_body_never_counts_as_a_match(self):
        absent = assess_round(ROWS, {}, 5)
        self.assertFalse(absent["complete"])
        self.assertEqual(absent["status"], "incomplete")
        self.assertEqual(set(absent["missing"]), {START, LEAN, HIP})
        partial = assess_round(ROWS, {START: 90.0}, 5)
        self.assertLess(partial["score"], absent["score"])

    def test_nonfinite_or_boolean_errors_are_missing(self):
        invalid = assess_round(ROWS, {START: float("nan"), LEAN: float("inf"), HIP: False}, 5)
        self.assertEqual(invalid, assess_round(ROWS, {}, 5))

    def test_complete_rounds_keep_failure_then_lean_then_total_order(self):
        candidates = [
            {START: 6, LEAN: 7, HIP: 0},  # two failures
            {START: 0, LEAN: 0, HIP: 20},  # one failure, even with a larger total error
            {START: 4, LEAN: 4, HIP: 4},  # no failures
            {START: 1, LEAN: 1, HIP: 5},  # smaller worst lean
            {START: 1, LEAN: 1, HIP: 0},  # smaller total error
        ]
        scores = [assess_round(ROWS, c, 5)["score"] for c in candidates]
        for worse, better in zip(scores, scores[1:]):
            self.assertLess(better, worse)
        self.assertEqual(assess_round(ROWS, candidates[0], 5)["status"], "limited")
        self.assertEqual(assess_round(ROWS, candidates[-1], 5)["status"], "matched")

    def test_best_round_remains_an_earlier_snapshot_after_detection_loss(self):
        candidates = [{START: 0, LEAN: 10, HIP: 8}, {START: 0, LEAN: 2, HIP: 2}, {START: 0}, {}]
        chosen = min(range(len(candidates)), key=lambda n: assess_round(ROWS, candidates[n], 5)["score"])
        self.assertEqual(chosen, 1)

    def test_unexpected_rows_and_empty_targets_do_not_produce_a_false_match(self):
        self.assertEqual(assess_round(ROWS, {("middle", "hip_included"): 0}, 5),
                         assess_round(ROWS, {}, 5))
        self.assertNotEqual(assess_round([], {}, 5)["status"], "matched")


if __name__ == "__main__":
    unittest.main()
