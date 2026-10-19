from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forced_choice  # noqa: E402
from proofwriter_test import build_or_load_eval_set  # noqa: E402


class ForcedChoiceTests(unittest.TestCase):
    def test_record_margin_and_prediction(self):
        example = dict(id="x", theory="T", question="Q", answer="False", QDep=3)
        row = forced_choice.record_for(0, example, {"True": -2.0, "False": -0.5, "Unknown": -1.5}, True, 10, 0.1)
        self.assertEqual(row["prediction"], "False")
        self.assertTrue(row["correct"])
        self.assertAlmostEqual(row["margin"], 1.0)
        row = forced_choice.record_for(0, example, {"True": -0.2, "False": -0.5, "Unknown": -3.0}, True, 10, 0.1)
        self.assertFalse(row["correct"])
        self.assertAlmostEqual(row["margin"], -0.3)

    def test_resume(self):
        examples = build_or_load_eval_set()
        header = forced_choice.header_for("fake", examples, {"True": 1, "False": 2, "Unknown": 3})
        score = lambda _: ({"True": -1.0, "False": -2.0, "Unknown": -3.0}, True, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fc.jsonl"
            forced_choice.run(examples, score, path, header, limit=5)
            rows = forced_choice.run(examples, score, path, header, limit=12)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(path.read_text().splitlines()), 13)


if __name__ == "__main__":
    unittest.main()
