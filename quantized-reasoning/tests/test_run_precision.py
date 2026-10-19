from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_precision  # noqa: E402
from proofwriter_test import build_or_load_eval_set  # noqa: E402


class RunPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.examples = build_or_load_eval_set()
        self.header = run_precision.run_header("fake-model", self.examples)

    def test_interleaved_prefix_is_balanced(self):
        order = run_precision.interleaved_order(self.examples)
        self.assertEqual(sorted(order), list(range(540)))
        prefix = Counter((self.examples[i]["QDep"], self.examples[i]["answer"]) for i in order[:36])
        self.assertEqual(set(prefix.values()), {2})

    def test_resume_skips_finished_examples(self):
        calls = []
        def predict(prompt):
            calls.append(prompt)
            if len(calls) == 4:
                raise KeyboardInterrupt  # simulate a crash mid-run
            return "True"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            with self.assertRaises(KeyboardInterrupt):
                run_precision.run(self.examples, predict, path, self.header, limit=10)
            records = run_precision.run(self.examples, lambda _: "True", path, self.header, limit=10)
            self.assertEqual(len(records), 10)
            self.assertEqual(len(path.read_text().splitlines()), 11)  # header + 10, no duplicates
            with self.assertRaises(ValueError):
                run_precision.run(self.examples, predict, path, dict(self.header, model="other"), limit=10)


if __name__ == "__main__":
    unittest.main()
