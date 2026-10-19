from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyze  # noqa: E402


def fake_runs(acc_by_precision):
    """acc_by_precision: {p: [items correct out of 10, per depth]}."""
    runs = {}
    for p, per_depth in acc_by_precision.items():
        runs[p] = {}
        for d, k in enumerate(per_depth):
            for j in range(10):
                i = d * 10 + j
                runs[p][i] = dict(index=i, key=str(i), depth=d, gold="True",
                                  prediction="True" if j < k else "False", correct=j < k)
    return runs


class AnalyzeTests(unittest.TestCase):
    def test_wilson_known_value(self):
        lo, hi = analyze.wilson(45, 90)
        self.assertAlmostEqual(lo, 0.3990, places=3)
        self.assertAlmostEqual(hi, 0.6010, places=3)

    def test_penalty_sign_and_trend(self):
        out = analyze.analyze(fake_runs({"bf16": [10] * 6, "4bit": [10, 9, 8, 7, 6, 5]}))
        deltas = [r["delta"] for r in out["penalty"]]
        self.assertEqual([round(x, 2) for x in deltas], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        trend = out["trend"][0]
        self.assertAlmostEqual(trend["slope_per_depth"], 0.1)
        self.assertAlmostEqual(trend["delta_d5_minus_d0"], 0.5)
        self.assertGreater(trend["slope_ci"][0], 0)


if __name__ == "__main__":
    unittest.main()
