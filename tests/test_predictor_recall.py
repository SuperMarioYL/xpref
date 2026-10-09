"""Predictor recall tests.

Verifies the n-gram + top-k predictor beats blind readahead on the shipped
Kimi K3 Q4 sample trace (recall@16 > 0.6) and on a tiny synthetic trace.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from xpref.predictor import Predictor, evaluate
from xpref.trace import read_trace

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "k3-q4-128tok.bin"


def _tiny_trace(tokens: int = 40, layers: int = 2, experts: int = 32,
                active: int = 4, seed: int = 7):
    """A tiny trace with the same drift structure as the K3 sample."""
    rng = np.random.default_rng(seed)
    logits = np.full((tokens, layers, experts), -2.0, dtype=np.float32)
    fired = np.zeros((tokens, layers, active), dtype=np.uint16)
    start = rng.integers(0, experts, size=layers)
    stride = rng.integers(4, 7, size=layers)
    for layer in range(layers):
        c = start[layer]
        for t in range(tokens):
            c = (start[layer] + (t / stride[layer])) % experts
            block = np.arange(int(c), int(c) + active) % experts
            logits[t, layer, block] = 8.0 + rng.normal(0, 0.5, active)
            fired[t, layer] = block.astype(np.uint16)
    return logits, fired


class TestPredictorRecall(unittest.TestCase):
    def test_recall_on_sample_trace(self):
        tr = read_trace(SAMPLE)
        res = evaluate(tr.logits, tr.fired)
        self.assertIn("recall_at_16", res)
        self.assertGreater(res["recall_at_16"], 0.6,
                           "predictor must beat blind readahead (>0.6 recall@16)")
        self.assertEqual(res["num_experts"], 896)
        self.assertEqual(res["num_active"], 16)
        self.assertEqual(res["predictions_made"], (tr.num_tokens - 1) * tr.num_layers)

    def test_recall_on_tiny_trace(self):
        logits, fired = _tiny_trace()
        res = evaluate(logits, fired)
        self.assertGreater(res["recall_at_k"], 0.5)

    def test_overlap_with(self):
        p = Predictor(num_experts=32, num_active=4)
        obs_logits = np.full((2, 32), -2.0, dtype=np.float32)
        obs_logits[0, :4] = 8.0
        obs_fired = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.uint16)
        p.observe(obs_logits, obs_fired)
        preds = p.predict_next()
        self.assertEqual(len(preds), 2)
        self.assertEqual(len(preds[0].ids), 4)
        actual = np.array([0, 1, 2, 3], dtype=np.uint16)
        self.assertGreaterEqual(preds[0].overlap_with(actual), 0.5)

    def test_predict_before_observe_raises(self):
        p = Predictor(num_experts=16, num_active=2)
        with self.assertRaises(RuntimeError):
            p.predict_next()


class TestNgramPrior(unittest.TestCase):
    """The n-gram prior must actually contribute (v0.1 exact-set matching never fired)."""

    def test_ngram_improves_recall_over_topk_only(self):
        tr = read_trace(SAMPLE)
        default = evaluate(tr.logits, tr.fired)
        topk_only = evaluate(tr.logits, tr.fired, topk_weight=1.0, ngram_weight=0.0)
        self.assertGreater(default["recall_at_k"], topk_only["recall_at_k"],
                           "n-gram prior must add signal beyond the top-k prior")

    def test_ngram_weight_zero_equals_topk_only(self):
        tr = read_trace(SAMPLE)
        zeroed = evaluate(tr.logits, tr.fired, topk_weight=0.6, ngram_weight=0.0)
        topk_only = evaluate(tr.logits, tr.fired, topk_weight=1.0, ngram_weight=0.0)
        self.assertEqual(zeroed["recall_at_k"], topk_only["recall_at_k"],
                         "ngram_weight=0 must reduce exactly to the top-k prior")

    def test_ngram_table_predicts_deterministic_followers(self):
        """A strict alternation must be predicted from the transition table alone."""
        num_experts, num_active = 8, 2
        p = Predictor(num_experts=num_experts, num_active=num_active,
                      topk_weight=0.0, ngram_weight=1.0)
        logits = np.full((1, num_experts), -2.0, dtype=np.float32)  # flat: no top-k signal
        seq = [np.array([[0, 1]], dtype=np.uint16),
               np.array([[2, 3]], dtype=np.uint16)] * 3  # (0,1) <-> (2,3)
        for fired in seq[:-1]:
            p.observe(logits, fired)
        preds = p.predict_next()
        predicted = {int(x) for x in preds[0].ids}
        self.assertEqual(predicted, {2, 3},
                         "n-gram table must predict the followers of (0, 1)")


class TestLazyDecay(unittest.TestCase):
    """v0.3.0: transition decay is folded in lazily on touch/read.

    The eager sweep used to iterate the whole table on every observed token —
    per-token cost grew with every distinct transition ever seen (measured
    0.46 -> 3.09 ms/token across the 127-token sample at 8x896).
    """

    def test_sample_recall_unchanged(self):
        """The rewrite must not move the published sample-trace number."""
        tr = read_trace(SAMPLE)
        res = evaluate(tr.logits, tr.fired)
        self.assertAlmostEqual(res["recall_at_k"], 0.8154, places=4)

    def test_idle_entry_keeps_raw_count_and_decays_by_the_formula(self):
        """An entry left idle for k tokens must keep its raw count/last-touch
        untouched (observe() must not sweep it) and read as count * 0.98**k —
        exactly the value the eager per-observe decay produced."""
        p = Predictor(num_experts=8, num_active=2, topk_weight=0.0, ngram_weight=1.0)
        logits = np.full((1, 8), -2.0, dtype=np.float32)  # flat: no top-k signal
        p.observe(logits, np.array([[0, 1]], dtype=np.uint16))
        p.observe(logits, np.array([[2, 3]], dtype=np.uint16))  # records 0->2, touch t=2
        entry = p._per_layer_trans[0][0][2]
        self.assertEqual(entry[0], 1.0)
        self.assertEqual(entry[1], 2)
        for _ in range(5):  # idle observes never touch source 0 or follower 2
            p.observe(logits, np.array([[4, 5]], dtype=np.uint16))
        self.assertEqual(entry[0], 1.0, "lazy decay must not rewrite idle entries")
        self.assertEqual(entry[1], 2)
        eff = entry[0] * 0.98 ** (p._t - entry[1])
        self.assertAlmostEqual(eff, 0.98 ** 5, places=12)

    def test_observe_touches_only_the_new_transitions(self):
        """Per observe, exactly len(prev)*len(cur) entries get last_touch==now —
        no whole-table sweep (this is what made observe() O(table) per token)."""
        p = Predictor(num_experts=32, num_active=4)
        logits = np.full((1, 32), -2.0, dtype=np.float32)
        fired = np.arange(4, dtype=np.uint16).reshape(1, 4)
        p.observe(logits, fired)  # cold: no transitions yet
        self.assertEqual(len(p._per_layer_trans[0]), 0)
        p.observe(logits, (fired + 1) % 32)
        touched = sum(1 for src in p._per_layer_trans[0].values()
                      for entry in src.values() if entry[1] == p._t)
        self.assertEqual(touched, 4 * 4)  # len(prev) * len(cur) transitions only
        p.observe(logits, fired)
        touched = sum(1 for src in p._per_layer_trans[0].values()
                      for entry in src.values() if entry[1] == p._t)
        self.assertEqual(touched, 4 * 4)


if __name__ == "__main__":
    unittest.main()
