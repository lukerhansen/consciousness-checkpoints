import re
import unittest

from select_checkpoints import pick_spaced


def steps(names):
    return [int(re.findall(r"\d+", n)[-1]) for n in names]


class TestPickSpaced(unittest.TestCase):
    def setUp(self):
        # Linear 1000-step saves like OLMo stage1 branches, step0 included.
        self.olmo = [f"stage1-step{s}" for s in range(0, 144000, 1000)]
        self.other = ["main", "stage2-step5000", "stage1-final", "refs-pr-1"]

    def test_no_match_raises(self):
        with self.assertRaises(ValueError):
            pick_spaced(["main", "foo"], r"^stage1-step", 6)

    def test_pattern_filters(self):
        picked = pick_spaced(self.olmo + self.other, r"^stage1-step\d", 4)
        self.assertTrue(all(p.startswith("stage1-step") for p in picked))
        self.assertNotIn("stage2-step5000", picked)

    def test_names_without_integers_are_dropped(self):
        # "step-final" matches the pattern but has no integer anywhere -> dropped.
        picked = pick_spaced(["step-final", "step10", "step20"], r"^step", 5)
        self.assertEqual(picked, ["step10", "step20"])

    def test_n1_returns_last(self):
        self.assertEqual(pick_spaced(self.olmo, r"^stage1-step", 1), ["stage1-step143000"])

    def test_fewer_matches_than_n_returns_all_sorted(self):
        few = ["stage2-step30000", "stage2-step1000", "stage2-step9000"]
        self.assertEqual(
            pick_spaced(few, r"^stage2", 10),
            ["stage2-step1000", "stage2-step9000", "stage2-step30000"])

    def test_includes_first_last_and_is_sorted(self):
        picked = pick_spaced(self.olmo, r"^stage1-step", 6)
        vals = steps(picked)
        self.assertEqual(len(picked), 6)
        self.assertEqual(len(set(picked)), 6)
        self.assertEqual(vals[0], 0)
        self.assertEqual(vals[-1], 143000)
        self.assertEqual(vals, sorted(vals))

    def test_log_spacing_front_loads_early_checkpoints(self):
        vals = steps(pick_spaced(self.olmo, r"^stage1-step", 6))
        # Linear spacing would put the 2nd pick near 28600; log spacing keeps it early.
        self.assertLessEqual(vals[1], 5000)
        self.assertEqual(vals[1:], sorted(set(vals[1:])))


if __name__ == "__main__":
    unittest.main()
