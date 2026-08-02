"""Ring buffer + trace + prefetch roundtrip tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from xpref.prefetch import ExpertLayout, Prefetcher
from xpref.ringbuf import RingBuffer, default_ring_path
from xpref.trace import Trace, read_trace, write_trace


def _mini_trace(tokens=4, layers=2, experts=16, active=4, seed=3):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (tokens, layers, experts)).astype(np.float32)
    fired = rng.integers(0, experts, (tokens, layers, active)).astype(np.uint16)
    return logits, fired


class TestTrace(unittest.TestCase):
    def test_roundtrip(self):
        logits, fired = _mini_trace()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bin"
            write_trace(p, logits, fired)
            tr: Trace = read_trace(p)
            np.testing.assert_allclose(tr.logits, logits)
            np.testing.assert_array_equal(tr.fired, fired)
            self.assertEqual(tr.num_experts, 16)
            self.assertEqual(tr.num_active, 4)
            self.assertEqual(tr.num_tokens, 4)
            self.assertEqual(tr.num_layers, 2)

    def test_bad_magic_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.bin"
            p.write_bytes(b"GARBAGEX" + b"\0" * 40)
            with self.assertRaises(ValueError):
                read_trace(p)


class TestRingBuffer(unittest.TestCase):
    def test_write_read_roundtrip(self):
        logits, fired = _mini_trace(tokens=3, layers=1, experts=8, active=2)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ring"
            with RingBuffer(p, create=True, num_experts=8, num_active=2,
                            num_layers=1, capacity=16) as w:
                for t in range(3):
                    w.write_record(t, 0, logits[t, 0], fired[t, 0])
            with RingBuffer(p, create=False) as r:
                recs = r.read_new()
            self.assertEqual(len(recs), 3)
            self.assertEqual(recs[0]["token_idx"], 0)
            np.testing.assert_allclose(recs[0]["logits"], logits[0, 0])
            np.testing.assert_array_equal(recs[0]["fired"], fired[0, 0])

    def test_default_ring_path(self):
        p = default_ring_path("xpref_router")
        self.assertTrue(str(p).endswith("xpref_router.xring"))

    def test_wraparound(self):
        logits, fired = _mini_trace(tokens=6, layers=1, experts=8, active=2)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ring"
            cap = 4
            with RingBuffer(p, create=True, num_experts=8, num_active=2,
                            num_layers=1, capacity=cap) as w:
                for t in range(6):
                    w.write_record(t, 0, logits[t, 0], fired[t, 0])
            with RingBuffer(p, create=False) as r:
                recs = r.read_new()
            # capacity-sized ring keeps the last `cap` records
            self.assertEqual(len(recs), cap)
            self.assertEqual(recs[0]["token_idx"], 2)


class TestPrefetcher(unittest.TestCase):
    def test_prefetch_no_crash(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh:
            fh.write(b"\0" * (128 * 1024))
            ckpt = fh.name
        try:
            layout = ExpertLayout.uniform(num_layers=4, num_experts=32,
                                          file_size=128 * 1024)
            with Prefetcher(ckpt, layout) as pf:
                # madvise may be a no-op if unavailable; must not raise.
                pf.prefetch(layer=1, expert_id=5)
                pf.prefetch(layer=2, expert_id=17)
                n = pf.prefetch_many([(0, 0), (3, 31)])
                self.assertGreaterEqual(n, 0)
        finally:
            Path(ckpt).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
