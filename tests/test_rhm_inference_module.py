"""Fast self-contained interface smoke tests (not an exhaustive validation)."""
import unittest

import torch

from rhm_inference import (GrammarConfig, build_grammar,
                           predictor_tensors, estimate_bayes_loss)


class TestInferenceModule(unittest.TestCase):
    def test_shapes_probabilities_and_unknown_layer(self):
        cfg = GrammarConfig(v=4, L=2, m2=1, m3=4, eta=4, seed_rules=123)
        _, binary, ternary = build_grammar(cfg)
        self.assertEqual(binary.shape, (2, 4, 4, 4))
        self.assertEqual(ternary.shape, (2, 4, 4, 4, 4))
        self.assertTrue(torch.allclose(binary.sum((2, 3)) + ternary.sum((2, 3, 4)),
                                       torch.ones((2, 4))))
        b2, t2 = predictor_tensors(cfg, binary, ternary, "bottom-intermediate")
        self.assertTrue(torch.allclose(b2[0], torch.full_like(b2[0], cfg.p2 * cfg.m2 / cfg.v**2)))
        self.assertTrue(torch.equal(binary[1], b2[1]))

    def test_estimate_small(self):
        cfg = GrammarConfig(v=4, L=2, m2=1, m3=4, eta=4, seed_rules=123)
        summary, rows = estimate_bayes_loss(cfg, n_samples=3, sample_seed=42)
        self.assertEqual(summary["n_valid"], 3)
        self.assertEqual(summary["zero_partition_count"], 0)
        self.assertTrue(0 <= summary["mean_entropy"] <= 1)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
