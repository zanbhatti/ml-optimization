from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import thinking  # noqa: E402
from proofwriter_test import build_or_load_eval_set  # noqa: E402


class ThinkingTests(unittest.TestCase):
    def test_lenient_parser(self):
        import analyze_thinking
        L = analyze_thinking.lenient_prediction
        self.assertEqual(L("<think>x</think>\n\nAnswer: Unknown"), "Unknown")
        self.assertEqual(L("<think>x</think>**True**"), "True")
        self.assertEqual(L("<think>never"), "TRUNCATED")
        self.assertEqual(L("<think>x</think> no idea"), "INVALID")

    def test_extract_answer(self):
        self.assertEqual(thinking.extract_answer("<think>steps</think>\n\nTrue")[0], "True")
        self.assertEqual(thinking.extract_answer("steps... </think>Unknown.")[0], "Unknown")
        self.assertEqual(thinking.extract_answer("<think>never finishes")[0], thinking.TRUNCATED)
        self.assertEqual(thinking.extract_answer("<think>x</think>The answer is True")[0], "INVALID")

    def test_resume_with_balanced_prefix(self):
        examples = build_or_load_eval_set()
        header = thinking.header_for("fake", examples, 64)
        gen = lambda prompt, index: ("<think>a b c</think>\n\nFalse", 9, "stop", 50.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.jsonl"
            thinking.run(examples, gen, path, header, n=4)
            rows = thinking.run(examples, gen, path, header, n=18)
            self.assertEqual(len(rows), 18)
            self.assertEqual(len({(r["depth"], r["gold"]) for r in rows}), 18)  # one per stratum
            self.assertEqual(rows[0]["thinking_words"], 3)

    def test_subset_rerun_and_tags(self):
        examples = build_or_load_eval_set()
        sampling = {"temp": 0.6, "top_p": 0.95, "top_k": 20, "seed": 1}
        self.assertEqual(thinking.run_tag(None, 4096), "")
        self.assertEqual(thinking.run_tag(sampling, 8192), "_t0.6_p0.95_k20_s1_max8192")
        greedy = thinking.header_for("fake", examples, 64)
        self.assertNotIn("sampling", greedy)  # old greedy files keep resuming
        header = thinking.header_for("fake", examples, 64, sampling)
        self.assertEqual(header["decoding"], "sample")
        seen = []
        gen = lambda prompt, index: (seen.append(index), ("<think>x</think>True", 3, "stop", 1.0))[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.jsonl"
            first = thinking.run(examples, gen, path, header, n=18)
            pick = {first[2]["index"], first[7]["index"]}
            rows = thinking.run(examples, gen, Path(directory) / "s.jsonl", header, n=18, indices=pick)
            self.assertEqual({r["index"] for r in rows}, pick)
            self.assertEqual(seen[18:], [r["index"] for r in rows])  # generator gets the item index


if __name__ == "__main__":
    unittest.main()
