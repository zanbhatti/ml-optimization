from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loop_classifier import repeat_fraction  # noqa: E402
from recheck_rate import recheck_rate  # noqa: E402


class RepeatFractionTests(unittest.TestCase):
    def test_pure_repeat(self):
        self.assertEqual(repeat_fraction("a b a b a b", 2), 1.0)

    def test_no_repeats(self):
        
        self.assertEqual(repeat_fraction('a b c d e', 2), 0.0)

    def test_partial(self):
        
        self.assertEqual(repeat_fraction('a b a b c', 2), 0.5)

    def test_too_short(self):
        
        self.assertEqual(repeat_fraction('a b', 5), 0.0)


class RecheckRateTests(unittest.TestCase):
    def test_rate(self):
        # TODO: a text with 1,000 words containing "but wait" twice should give 2.0
        text = "but wait " * 2 + "word " * 996
        self.assertEqual(recheck_rate(text), 2.0)


if __name__ == "__main__":
    unittest.main()
