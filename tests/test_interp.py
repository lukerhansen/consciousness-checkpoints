import unittest
from types import SimpleNamespace

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


if HAVE_TORCH:
    class _Block(torch.nn.Module):
        """Tuple-returning block that adds a constant, like HF decoder layers."""

        def __init__(self, delta):
            super().__init__()
            self.delta = delta

        def forward(self, hidden, **kwargs):
            return (hidden + self.delta,)

    class _TinyLM(torch.nn.Module):
        def __init__(self, vocab=11, d=4, n_blocks=3):
            super().__init__()
            inner = torch.nn.Module()
            inner.embed_tokens = torch.nn.Embedding(vocab, d)
            inner.layers = torch.nn.ModuleList(
                [_Block(float(i + 1)) for i in range(n_blocks)])
            self.model = inner

        def forward(self, input_ids, attention_mask=None, use_cache=False,
                    output_hidden_states=False):
            hidden = self.model.embed_tokens(input_ids)
            states = [hidden]
            for block in self.model.layers:
                hidden = block(hidden)[0]
                states.append(hidden)
            return SimpleNamespace(logits=hidden, hidden_states=tuple(states))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestHooks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _TinyLM()
        self.ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        self.mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    def test_resolve_blocks_and_sites(self):
        from interp.hooks import n_sites, resolve_blocks
        embed, blocks = resolve_blocks(self.model)
        self.assertIs(embed, self.model.model.embed_tokens)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(n_sites(self.model), 4)

    def test_steering_shifts_and_restores(self):
        from interp.hooks import SteeringHooks
        base = self.model(self.ids, self.mask, output_hidden_states=True)
        vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        with SteeringHooks(self.model, {1: (vec, 2.0)}):
            steered = self.model(self.ids, self.mask, output_hidden_states=True)
        shift = torch.zeros(4)
        shift[0] = 2.0
        # Site 0 untouched; sites 1..3 all carry the shift (blocks add consts).
        self.assertTrue(torch.equal(steered.hidden_states[0],
                                    base.hidden_states[0]))
        for k in (1, 2, 3):
            self.assertTrue(torch.allclose(
                steered.hidden_states[k], base.hidden_states[k] + shift))
        # Hooks removed on exit: forward matches baseline exactly.
        after = self.model(self.ids, self.mask, output_hidden_states=True)
        for k in range(4):
            self.assertTrue(torch.equal(after.hidden_states[k],
                                        base.hidden_states[k]))

    def test_steering_at_embedding_site(self):
        from interp.hooks import SteeringHooks
        base = self.model(self.ids, self.mask, output_hidden_states=True)
        vec = torch.tensor([0.0, 1.0, 0.0, 0.0])
        with SteeringHooks(self.model, {0: (vec, -1.5)}):
            steered = self.model(self.ids, self.mask, output_hidden_states=True)
        shift = torch.zeros(4)
        shift[1] = -1.5
        self.assertTrue(torch.allclose(
            steered.hidden_states[0], base.hidden_states[0] + shift))

    def test_steering_rejects_bad_site(self):
        from interp.hooks import SteeringHooks
        with self.assertRaises(ValueError):
            with SteeringHooks(self.model, {9: (torch.zeros(4), 1.0)}):
                pass

    def test_capture_at_offsets_respects_row_lengths(self):
        from interp.hooks import capture_at_offsets
        out = capture_at_offsets(self.model, self.ids, self.mask, [-1, -2])
        ref = self.model(self.ids, self.mask, output_hidden_states=True)
        self.assertEqual(tuple(out[-1].shape), (4, 2, 4))
        for k in range(4):
            h = ref.hidden_states[k]
            # Row 0 has length 3 -> last content position 2; row 1 length 2 -> 1.
            self.assertTrue(torch.allclose(out[-1][k, 0], h[0, 2]))
            self.assertTrue(torch.allclose(out[-1][k, 1], h[1, 1]))
            self.assertTrue(torch.allclose(out[-2][k, 0], h[0, 1]))
            self.assertTrue(torch.allclose(out[-2][k, 1], h[1, 0]))

    def test_project_normalizes(self):
        from interp.hooks import project
        acts = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        direction = torch.tensor([0.0, 10.0])  # not unit
        self.assertTrue(torch.allclose(project(acts, direction),
                                       torch.tensor([4.0, 2.0])))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestHonestyContrast(unittest.TestCase):
    def test_fact_pairs_mirrored_and_disjoint_from_batteries(self):
        from interp.honesty import assert_disjoint, fact_statements
        stmts = fact_statements()
        self.assertEqual(len(stmts) % 2, 0)
        self.assertTrue(all(is_true for (_, is_true) in stmts[0::2]))
        self.assertTrue(all(not is_true for (_, is_true) in stmts[1::2]))
        self.assertEqual(len({s for s, _ in stmts}), len(stmts))
        assert_disjoint("data")  # raises on any collision

    def test_extraction_records_force_the_right_answers(self):
        from interp.honesty import FACT_PAIRS, build_extraction_records, YES, NO
        records = build_extraction_records(FACT_PAIRS[:4])
        self.assertEqual(len(records), 4 * 2 * 2 * 4)  # pairs x stmts x cond x wordings
        for rec in records:
            truthful = YES if rec["is_true"] else NO
            expected = truthful if rec["condition"] == "honest" else (
                NO if truthful == YES else YES)
            self.assertEqual(rec["answer"], expected)
            self.assertIn(rec["statement"], rec["prompt"])
        # Balanced answer tokens within each condition.
        for cond in ("honest", "lie"):
            answers = [r["answer"] for r in records if r["condition"] == cond]
            self.assertEqual(answers.count(YES), answers.count(NO))

    def test_split_is_disjoint_in_pairs_and_wordings(self):
        from interp.honesty import build_extraction_records, split_records
        records = build_extraction_records()
        train_idx, test_idx = split_records(records)
        self.assertTrue(train_idx and test_idx)
        self.assertFalse(set(train_idx) & set(test_idx))
        train_pairs = {records[i]["pair_idx"] for i in train_idx}
        test_pairs = {records[i]["pair_idx"] for i in test_idx}
        self.assertFalse(train_pairs & test_pairs)
        train_wordings = {records[i]["wording"] for i in train_idx}
        test_wordings = {records[i]["wording"] for i in test_idx}
        self.assertFalse(train_wordings & test_wordings)

    def test_diff_in_means_recovers_planted_direction(self):
        from interp.honesty import diff_in_means
        torch.manual_seed(1)
        planted = torch.zeros(8)
        planted[3] = 1.0
        base = torch.randn(1, 2, 8) * 0.1
        honest = base + planted + torch.randn(40, 2, 8) * 0.1
        lie = base - planted + torch.randn(40, 2, 8) * 0.1
        unit, norms = diff_in_means(honest, lie)
        for site in range(2):
            cos = float(unit[site] @ planted)
            self.assertGreater(cos, 0.95)
        self.assertTrue((norms > 1.5).all())

    def test_auc(self):
        from interp.honesty import auc
        self.assertEqual(auc([1.0, 2.0], [0.0, 0.5]), 1.0)
        self.assertEqual(auc([0.0], [1.0]), 0.0)
        self.assertAlmostEqual(auc([1.0, 2.0], [1.0, 2.0]), 0.5)
        torch.manual_seed(2)
        same = torch.randn(500).tolist()
        other = torch.randn(500).tolist()
        self.assertAlmostEqual(auc(same, other), 0.5, delta=0.06)

    def test_cohens_d_sign(self):
        from interp.honesty import cohens_d
        d = cohens_d([2.0, 2.1, 1.9], [0.0, 0.1, -0.1])
        self.assertGreater(d, 5)
        self.assertLess(cohens_d([0.0, 0.1, -0.1], [2.0, 2.1, 1.9]), -5)

    def test_validate_direction_finds_planted_site(self):
        from interp.honesty import (FACT_PAIRS, build_extraction_records,
                                    validate_direction)
        torch.manual_seed(3)
        records = build_extraction_records(FACT_PAIRS[:8])
        n, sites, d = len(records), 3, 8
        planted = torch.zeros(d)
        planted[5] = 1.0
        acts = {-1: torch.randn(n, sites, d), -2: torch.randn(n, sites, d)}
        for i, rec in enumerate(records):  # signal only at site 2, offset -1
            sign = 1.0 if rec["condition"] == "honest" else -1.0
            acts[-1][i, 2] += 3.0 * sign * planted
        report, directions = validate_direction(acts, records, (-1, -2))
        best = report["best"]
        self.assertEqual((best["offset"], best["site"]), (-1, 2))
        self.assertGreater(best["auc_transfer"], 0.95)
        cos = float(directions[-1][2] @ planted)
        self.assertGreater(cos, 0.9)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestSteerRules(unittest.TestCase):
    def test_band_is_middle_half(self):
        from run_honesty_steer import steering_band
        band = steering_band(33, None)  # 32 blocks
        self.assertEqual((band[0], band[-1]), (8, 24))
        self.assertEqual(steering_band(33, "5:7"), [5, 6, 7])

    def test_site_vectors_scaling_and_control(self):
        from run_honesty_steer import site_vectors
        dirs = torch.eye(4)[:3]  # 3 sites, d=4
        norms = torch.tensor([10.0, 20.0, 30.0])
        sv = site_vectors(dirs, norms, [1, 2], 0.1)
        self.assertTrue(torch.equal(sv[1][0], dirs[1]))
        self.assertAlmostEqual(sv[1][1], 2.0)
        self.assertAlmostEqual(sv[2][1], 3.0)
        gen = torch.Generator().manual_seed(0)
        rand = site_vectors(dirs, norms, [1], 0.1, gen)
        self.assertAlmostEqual(float(rand[1][0].norm()), 1.0, places=5)
        self.assertFalse(torch.allclose(rand[1][0], dirs[1]))

    def test_choose_coeff_rejects_bias_and_keeps_largest_lie(self):
        from run_honesty_steer import choose_coeff
        def entry(acc, t, f, world, ):
            return {"fact": {"accuracy": acc, "accuracy_true_half": t,
                             "accuracy_false_half": f},
                    "world_facts_rate": world}
        baseline = entry(1.0, 1.0, 1.0, 1.0)
        grid = {
            "0": baseline,
            # 0.02: clean lying both halves -> passes
            "+0.02": entry(1.0, 1.0, 1.0, 0.95), "-0.02": entry(0.6, 0.6, 0.6, 0.5),
            # 0.05: "lying" is actually a No-bias (false half improves) -> fails
            "+0.05": entry(1.0, 1.0, 1.0, 0.95), "-0.05": entry(0.55, 0.15, 0.95, 0.5),
            # 0.1: +c breaks world-facts -> fails
            "+0.1": entry(1.0, 1.0, 1.0, 0.5), "-0.1": entry(0.3, 0.3, 0.3, 0.2),
            # 0.2: +c destroys sincerity -> fails
            "+0.2": entry(0.5, 0.5, 0.5, 0.95), "-0.2": entry(0.2, 0.2, 0.2, 0.2),
        }
        chosen, table = choose_coeff(grid, baseline)
        self.assertEqual(chosen, 0.02)
        self.assertTrue(table[0.02]["pass"])
        self.assertFalse(table[0.05]["minus_both_halves"])
        self.assertFalse(table[0.1]["plus_world_guard"])
        self.assertFalse(table[0.2]["plus_sincerity_kept"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestReadoutRecords(unittest.TestCase):
    def test_fact_records_shape(self):
        from run_honesty_readout import fact_records
        recs = fact_records([("A is A", True), ("A is B", False)])
        self.assertEqual(len(recs), 2 * (1 + 4 + 4))
        sincere = [r for r in recs if r["condition"] == "fact_sincere"]
        self.assertEqual([r["answer"] for r in sincere], [" Yes", " No"])
        lies = [r for r in recs if r["condition"] == "fact_lie"]
        self.assertEqual({r["answer"] for r in lies if r["statement"] == "A is A"},
                         {" No"})

    def test_roleplay_records_filter_subjects(self):
        from run_honesty_readout import roleplay_records
        recs = roleplay_records("data")
        self.assertEqual(len(recs), 16)  # 8 self + 8 human items
        self.assertTrue(all(r["subject"] in ("self", "human") for r in recs))
        self.assertTrue(all(r["prompt"].startswith("For a story") for r in recs))


if __name__ == "__main__":
    unittest.main()
