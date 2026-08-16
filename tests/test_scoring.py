import math
import unittest

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestScoringBasics(unittest.TestCase):
    def test_two_way_prob(self):
        from persona_eval.scoring import two_way_prob
        self.assertAlmostEqual(two_way_prob(-1.0, -1.0), 0.5)
        self.assertAlmostEqual(two_way_prob(0.0, math.log(3)), 0.25)
        self.assertAlmostEqual(two_way_prob(0.0, -1000.0), 1.0, places=6)
        self.assertAlmostEqual(two_way_prob(-1000.0, 0.0), 0.0, places=6)

    def test_encode_pair_fake_tokenizer_lengths(self):
        from persona_eval.fake import FakeTokenizer
        from persona_eval.scoring import encode_pair
        tok = FakeTokenizer()
        _, yes_ids = encode_pair(tok, "Q\nAnswer:", " Yes")
        _, no_ids = encode_pair(tok, "Q\nAnswer:", " No")
        # The documented byte-level artifact: different lengths (4 vs 3 bytes).
        self.assertEqual(len(yes_ids), 4)
        self.assertEqual(len(no_ids), 3)

    def test_tokenization_report_fake(self):
        from persona_eval.fake import FakeTokenizer
        from persona_eval.scoring import tokenization_report
        info = tokenization_report(FakeTokenizer(), "Q\nAnswer:", [" Yes", " No"])
        self.assertFalse(info["answer_lengths_equal"])
        self.assertFalse(info["answers_single_token"])
        self.assertTrue(info["joint_equals_separate"])  # byte-level concat is exact


if __name__ == "__main__":
    unittest.main()
