import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).resolve().parents[1] / "src/proofwriter_test.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_normalization(self):
        for raw, expected in [("True", "True"), ("True.", "True"), (" False ", "False"),
                              ("Unknown\n\nExplanation: ...", "Unknown"), ("False: True", "False"),
                              ("", "INVALID"), ("I think True", "INVALID"), ("Trueish", "INVALID"),
                              ("True_false", "INVALID")]:
            with self.subTest(raw=raw):
                self.assertEqual(evaluation.normalize_prediction(raw), expected)

    def test_sampling_and_cache(self):
        rows = [dict(id=f"{d}-{label}-{i}", theory="Facts", question=f"Q{i}", answer=label, QDep=d)
                for d in evaluation.DEPTHS for label in evaluation.LABELS for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            first = evaluation.build_or_load_eval_set(Path(directory) / "a.jsonl", rows)
            second = evaluation.build_or_load_eval_set(Path(directory) / "b.jsonl", list(reversed(rows)) + rows)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 540)
            self.assertEqual(first, evaluation.build_or_load_eval_set(Path(directory) / "a.jsonl", []))
            with self.assertRaises(ValueError):
                evaluation.build_or_load_eval_set(Path(directory) / "short.jsonl", rows[:1])
            (Path(directory) / "a.jsonl").write_text('{}\n')
            with self.assertRaises(ValueError):
                evaluation.build_or_load_eval_set(Path(directory) / "a.jsonl", rows)

    def test_scoring_keeps_invalid_in_denominator(self):
        rows = [dict(id=i, theory="Facts", question="Q", answer=label, QDep=0)
                for i, label in enumerate(evaluation.LABELS)]
        outputs = iter(["True.", "", "False"])
        metrics, records = evaluation.evaluate_depth(rows, lambda _: next(outputs))
        self.assertEqual(metrics, dict(correct=1, total=3, invalid=1, accuracy=1 / 3))
        self.assertEqual(records[1]["gold"], "False")


if __name__ == "__main__":
    unittest.main()
