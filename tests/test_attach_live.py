"""Attach-path tests: live ring recall + single-pass prefetch accounting.

Regression tests for the v0.2.0 attach fixes:

* ``live_attach`` must actually measure next-token recall (it used to leave
  its ``recalls`` list empty and always report recall=0.000 / 4 t/s).
* replay/live must run exactly one predict + prefetch pass per token (they
  used to predict and ``prefetch_many`` twice per token).
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from xpref.attach import REACTIVE_TPS, live_attach, replay_attach
from xpref.predictor import evaluate
from xpref.prefetch import Prefetcher
from xpref.ringbuf import RingBuffer
from xpref.trace import read_trace

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "k3-q4-128tok.bin"
CKPT_BYTES = 256 * 1024  # 256 KiB synthetic checkpoint


def _fake_checkpoint(td: str) -> Path:
    ckpt = Path(td) / "ckpt.gguf"
    ckpt.write_bytes(b"\0" * CKPT_BYTES)
    return ckpt


class TestLiveAttach(unittest.TestCase):
    def test_live_attach_measures_recall(self):
        """A ring fed the sample-trace prefix must report real next-token recall."""
        tr = read_trace(SAMPLE)
        n_live = 16  # tokens written into the ring (>= max_tokens so we terminate)
        with tempfile.TemporaryDirectory() as td:
            ring_path = str(Path(td) / "ring.xring")
            ckpt = _fake_checkpoint(td)
            with RingBuffer(ring_path, create=True, num_experts=tr.num_experts,
                            num_active=tr.num_active, num_layers=tr.num_layers,
                            capacity=256) as rb:
                for t in range(n_live):
                    for lay in range(tr.num_layers):
                        rb.write_record(t, lay, tr.logits[t, lay], tr.fired[t, lay])
            with contextlib.redirect_stdout(io.StringIO()):
                stats = live_attach(ring_path, str(ckpt), max_tokens=8)
        self.assertEqual(stats.tokens_seen, 8)
        # the predictor has strong signal on this trace — recall must be real,
        # not the pre-fix constant 0.0
        self.assertGreater(stats.recall, 0.5)
        self.assertGreater(stats.projected_tps, REACTIVE_TPS)

    def test_live_attach_recall_matches_offline_scoring(self):
        """Live recall equals evaluate() over the same observed prefix."""
        tr = read_trace(SAMPLE)
        n_live = 24
        with tempfile.TemporaryDirectory() as td:
            ring_path = str(Path(td) / "ring.xring")
            ckpt = _fake_checkpoint(td)
            with RingBuffer(ring_path, create=True, num_experts=tr.num_experts,
                            num_active=tr.num_active, num_layers=tr.num_layers,
                            capacity=256) as rb:
                for t in range(n_live):
                    for lay in range(tr.num_layers):
                        rb.write_record(t, lay, tr.logits[t, lay], tr.fired[t, lay])
            with contextlib.redirect_stdout(io.StringIO()):
                stats = live_attach(ring_path, str(ckpt), max_tokens=n_live)
        # live scored predictions made after tokens 0..n_live-2 against
        # tokens 1..n_live-1 — the same pairs evaluate() scores over the
        # full n_live-token prefix
        offline = evaluate(tr.logits[:n_live], tr.fired[:n_live])
        self.assertAlmostEqual(stats.recall, offline["recall_at_k"], places=6)


class TestSinglePassPrefetch(unittest.TestCase):
    def test_replay_prefetches_each_prediction_once(self):
        """prefetch_many fires exactly once per layer-prediction (no double pass)."""
        tr = read_trace(SAMPLE)
        calls = {"n": 0}
        real = Prefetcher.prefetch_many

        def counting(self, pairs):
            calls["n"] += 1
            return real(self, pairs)

        with tempfile.TemporaryDirectory() as td:
            ckpt = _fake_checkpoint(td)
            with unittest.mock.patch.object(Prefetcher, "prefetch_many", counting):
                with contextlib.redirect_stdout(io.StringIO()):
                    replay_attach(SAMPLE, str(ckpt), verbose=False)
        # one call per (layer, predicted token): 8 layers x 127 predictions
        self.assertEqual(calls["n"], tr.num_layers * (tr.num_tokens - 1))

    def test_replay_recall_matches_evaluate(self):
        tr = read_trace(SAMPLE)
        with tempfile.TemporaryDirectory() as td:
            ckpt = _fake_checkpoint(td)
            with contextlib.redirect_stdout(io.StringIO()):
                stats = replay_attach(SAMPLE, str(ckpt), verbose=False)
        offline = evaluate(tr.logits, tr.fired)
        # tokens_seen counts scored tokens; predictions_made counts
        # (token, layer) pairs — one prediction per layer per scored token
        self.assertEqual(stats.tokens_seen, tr.num_tokens - 1)
        self.assertEqual(offline["predictions_made"], stats.tokens_seen * tr.num_layers)
        self.assertAlmostEqual(stats.recall, offline["recall_at_k"], places=6)


if __name__ == "__main__":
    unittest.main()
