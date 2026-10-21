from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import thinking  # noqa: E402
from proofwriter_test import build_or_load_eval_set  # noqa: E402


class ThinkingTests(unittest.TestCase):
    def test_extract_answer(self):
        self.assertEqual(thinking.extract_answer("<think>steps</think>\n\nTrue")[0], "True")
        self.assertEqual(thinking.extract_answer("steps... </think>Unknown.")[0], "Unknown")
        self.assertEqual(thinking.extract_answer("<think>never finishes")[0], thinking.TRUNCATED)
        self.assertEqual(thinking.extract_answer("<think>x</think>The answer is True")[0], "INVALID")

    def test_resume_with_balanced_prefix(self):
        examples = build_or_load_eval_set()
        header = thinking.header_for("fake", examples, 64)
        gen = lambda _: ("<think>a b c</think>\n\nFalse", 9, "stop", 50.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.jsonl"
            thinking.run(examples, gen, path, header, n=4)
            rows = thinking.run(examples, gen, path, header, n=18)
            self.assertEqual(len(rows), 18)
            self.assertEqual(len({(r["depth"], r["gold"]) for r in rows}), 18)  # one per stratum
            self.assertEqual(rows[0]["thinking_words"], 3)


if __name__ == "__main__":
    unittest.main()
