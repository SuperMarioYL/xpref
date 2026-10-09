"""Ring buffer + trace + prefetch roundtrip tests."""

from __future__ import annotations

import mmap
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xpref.prefetch import ExpertLayout, Prefetcher
from xpref.ringbuf import (
    HEADER_SIZE,
    RingBuffer,
    _record_dtype,
    default_ring_path,
)
from xpref.trace import Trace, read_trace, write_trace

ROOT = Path(__file__).resolve().parents[1]


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


class TestRingLayoutContract(unittest.TestCase):
    """Producer (engine patch) and consumer must agree on the ring layout.

    v0.3.0 regression: the bundled C patch documented a 40-byte header and
    mapped records from offset 40 while the consumer reads from offset 44
    (the header fields — 8s magic + 5x u32 + 2x u64 — sum to 44), so every
    live attach against a real patched engine crashed with
    ValueError("buffer is smaller than requested size").
    """

    def test_header_size_is_the_documented_44(self):
        self.assertEqual(HEADER_SIZE, 44)
        self.assertEqual(HEADER_SIZE, 8 + 5 * 4 + 2 * 8)

    def test_engine_patch_constants_match_consumer(self):
        patch = (ROOT / "src" / "xpref" / "engine_patch" /
                 "llama_cpp_router_logit.patch").read_text()
        self.assertIn("header (44 bytes", patch)
        # record base / total sizing must use the 44-byte header everywhere
        self.assertNotIn("+ 40", patch, "stale 40-byte record arithmetic in patch")
        self.assertIn("xpref_ring + 44", patch)
        self.assertIn("total = 44 + (size_t)capacity * rec", patch)

    def test_producer_file_sizing_matches_consumer_view(self):
        # a create=True ring (the fixed patch's layout: HEADER_SIZE + cap*rec)
        # must be exactly what the consumer's numpy view expects to map.
        E, A, L, CAP = 896, 16, 8, 256
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ring"
            with RingBuffer(p, create=True, num_experts=E, num_active=A,
                            num_layers=L, capacity=CAP) as w:
                w.write_record(0, 0, np.zeros(E, np.float32), np.zeros(A, np.uint16))
            expected = HEADER_SIZE + CAP * _record_dtype(E, A).itemsize
            self.assertEqual(p.stat().st_size, expected)


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

    def test_prefetch_unaligned_layout_hints(self):
        """v0.3.0 regression: madvise(2) needs a page-aligned address (EINVAL
        otherwise on Linux), so unaligned expert ranges used to be dropped and
        bytes_prefetched stayed 0. The hint must now succeed and report the
        page-rounded range actually advised."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            fh.write(b"\0" * (16 * mmap.PAGESIZE))
            ckpt = fh.name
        try:
            # 16384/3-byte experts: offsets land mid-page
            layout = ExpertLayout.uniform(num_layers=1, num_experts=3,
                                          file_size=16 * mmap.PAGESIZE)
            with Prefetcher(ckpt, layout) as pf:
                if not pf.available:
                    self.skipTest("madvise unavailable")
                off, n = pf.prefetch(0, 1)
                self.assertGreater(n, 0, "unaligned expert range was dropped")
                self.assertEqual(off % mmap.PAGESIZE, 0)
                self.assertLessEqual(off + n, 16 * mmap.PAGESIZE)
        finally:
            Path(ckpt).unlink(missing_ok=True)

    def test_prefetch_page_aligned_layout_is_exact(self):
        page = mmap.PAGESIZE  # 4K on x86 Linux, 16K on arm64 macOS — both valid
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            fh.write(b"\0" * (4 * page))
            ckpt = fh.name
        try:
            layout = ExpertLayout.uniform(num_layers=1, num_experts=4,
                                          file_size=4 * page)  # one page per expert
            with Prefetcher(ckpt, layout) as pf:
                if not pf.available:
                    self.skipTest("madvise unavailable")
                self.assertEqual(pf.prefetch(0, 1), (page, page))
        finally:
            Path(ckpt).unlink(missing_ok=True)

    def test_prefetch_range_clamped_to_eof(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            fh.write(b"\0" * (16384))
            ckpt = fh.name
        try:
            layout = ExpertLayout(num_layers=1, num_experts=2, file_size=16384,
                                  table={(0, 0): (16000, 4000)})
            with Prefetcher(ckpt, layout) as pf:
                if not pf.available:
                    self.skipTest("madvise unavailable")
                off, n = pf.prefetch(0, 0)
                self.assertGreater(n, 0)
                self.assertLessEqual(off + n, 16384)
        finally:
            Path(ckpt).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
