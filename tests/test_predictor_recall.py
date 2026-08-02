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


if __name__ == "__main__":
    unittest.main()
