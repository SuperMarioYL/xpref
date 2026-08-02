"""Shared ring buffer carrying router logits from the patched engine to xpref.

Two processes share one ring:

* the **producer** — a patched llama.cpp kimi-k3 fork — writes one record per
  ``(token, layer)`` after computing the router gating logits (and, once the
  routing decision is made, the fired expert ids);
* the **consumer** — ``xpref attach`` — reads new records, feeds them to the
  predictor, and issues prefetch hints.

The ring is a single-producer / single-consumer lock-free structure backed by
a memory-mapped file. On Linux the file lives under ``/dev/shm`` (tmpfs); on
macOS under ``/tmp``. Both are shared-memory-equivalent on a single host and
side-step the ``shm_open`` slash-naming divergence between the two platforms
(macOS forbids a leading ``/``, Linux requires it). An ``shm_open``-backed
backend is the documented upgrade path — see :func:`shm_open`.

Header (44 bytes, little-endian, ``struct`` format ``<8sIIIIIQQ``):

    magic      : 8 bytes  = b"XPRFRING"
    version    : uint32
    num_experts: uint32
    num_active : uint32
    num_layers : uint32
    capacity   : uint32
    head       : uint64   # monotonic write counter (producer)
    tail       : uint64   # monotonic read counter (consumer)

Record (per token, per layer):

    token_idx  : uint32
    layer      : uint32
    logits     : num_experts * float32   # router gating logits
    fired      : num_active  * uint16     # ids that actually fired
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RING_MAGIC = b"XPRFRING"
RING_VERSION = 1
HEADER_FMT = "<8sIIIIIQQ"  # magic, version, E, A, L, cap, head, tail
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 44
HEAD_OFF = 28  # offset of head u64 within the header
TAIL_OFF = 36  # offset of tail u64 within the header


def default_ring_path(name: str) -> Path:
    """Platform default for the shared ring file."""
    base = "/dev/shm" if sys.platform.startswith("linux") else "/tmp"
    return Path(base) / f"{name}.xring"


def _record_dtype(num_experts: int, num_active: int) -> np.dtype:
    return np.dtype(
        [
            ("token_idx", "<u4"),
            ("layer", "<u4"),
            ("logits", "<f4", (num_experts,)),
            ("fired", "<u2", (num_active,)),
        ]
    )


@dataclass
class RingMeta:
    num_experts: int
    num_active: int
    num_layers: int
    capacity: int


class RingBuffer:
    """A lock-free SPSC shared ring backed by a memory-mapped file."""

    def __init__(self, path: str | Path, *, create: bool = False,
                 num_experts: int = 0, num_active: int = 0,
                 num_layers: int = 0, capacity: int = 256):
        self.path = Path(path)
        self._create = create
        self._mmap: mmap.mmap | None = None
        self._fh = None
        self.meta: RingMeta | None = None
        self._records: np.ndarray | None = None
        if create:
            if not (num_experts and num_active and num_layers):
                raise ValueError("create needs num_experts/num_active/num_layers")
            self._rec_dt = _record_dtype(num_experts, num_active)
            self._init_create(num_experts, num_active, num_layers, capacity)
        else:
            self._attach()

    # ------------------------------------------------------------ creation --
    def _init_create(self, E: int, A: int, L: int, cap: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec_bytes = self._rec_dt.itemsize
        total = HEADER_SIZE + cap * rec_bytes
        self._fh = self.path.open("w+b")
        self._fh.truncate(total)
        self._mmap = mmap.mmap(self._fh.fileno(), total, access=mmap.ACCESS_WRITE)
        struct.pack_into(HEADER_FMT, self._mmap, 0, RING_MAGIC, RING_VERSION,
                         E, A, L, cap, 0, 0)
        self._records = np.frombuffer(self._mmap, dtype=self._rec_dt,
                                      count=cap, offset=HEADER_SIZE)
        self.meta = RingMeta(E, A, L, cap)

    def _attach(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"ring not found at {self.path} (is the engine running?)")
        self._fh = self.path.open("r+b")
        self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_WRITE)
        magic, _ver, E, A, L, cap, _head, _tail = struct.unpack_from(
            HEADER_FMT, self._mmap, 0)
        if magic != RING_MAGIC:
            raise ValueError(f"bad ring magic {magic!r} at {self.path}")
        self._rec_dt = _record_dtype(E, A)
        self.meta = RingMeta(E, A, L, cap)
        self._records = np.frombuffer(self._mmap, dtype=self._rec_dt,
                                      count=cap, offset=HEADER_SIZE)

    # --------------------------------------------------------------- I/O --
    def write_record(self, token_idx: int, layer: int,
                     logits: np.ndarray, fired: np.ndarray) -> None:
        """Producer: append one (token, layer) record (call after computing logits)."""
        if self.meta is None or self._records is None or self._mmap is None:
            raise RuntimeError("ring not initialised")
        head = struct.unpack_from("<Q", self._mmap, HEAD_OFF)[0]
        slot = head % self.meta.capacity
        rec = self._records[slot]
        rec["token_idx"] = token_idx
        rec["layer"] = layer
        rec["logits"] = np.ascontiguousarray(logits, dtype="<f4")
        rec["fired"] = np.ascontiguousarray(fired, dtype="<u2")
        # bump head last (acts as a release barrier for the consumer)
        struct.pack_into("<Q", self._mmap, HEAD_OFF, head + 1)

    def read_new(self) -> list[dict]:
        """Consumer: drain all records written since the last call.

        If the consumer fell behind by more than ``capacity`` records, only
        the newest ``capacity`` survive (older slots were overwritten); the
        consumer's read pointer advances past the dropped records.
        """
        if self.meta is None or self._records is None or self._mmap is None:
            return []
        head = struct.unpack_from("<Q", self._mmap, HEAD_OFF)[0]
        tail = struct.unpack_from("<Q", self._mmap, TAIL_OFF)[0]
        cap = self.meta.capacity
        n = min(head - tail, cap)
        start = (head - n) % cap
        out: list[dict] = []
        for i in range(n):
            slot = (start + i) % cap
            rec = self._records[slot]
            out.append(
                {
                    "token_idx": int(rec["token_idx"]),
                    "layer": int(rec["layer"]),
                    "logits": np.array(rec["logits"], dtype=np.float32, copy=True),
                    "fired": np.array(rec["fired"], dtype=np.uint16, copy=True),
                }
            )
        # consumer caught up (drops anything lost to overrun)
        struct.pack_into("<Q", self._mmap, TAIL_OFF, head)
        return out

    def close(self) -> None:
        # Release the numpy view first — it holds an exported buffer pointer
        # over the mmap, which would otherwise block mmap.close(). The ring
        # file itself persists so a consumer can attach after the producer
        # closes (or between producer writes); callers manage its lifetime.
        self._records = None
        self.meta = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> RingBuffer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------- shm --
def _libc():
    name = ctypes.util.find_library("c")
    return ctypes.CDLL(name, use_errno=True) if name else None


def shm_open(name: str, *, create: bool = False, size: int = 0) -> int:
    """Thin ctypes wrapper over ``shm_open`` for the shm-backed variant.

    ``name`` must be slash-prefixed on Linux and slash-free on macOS. Returns
    an fd the caller owns (and must ``close``). Raises if unavailable.
    """
    libc = _libc()
    if libc is None:
        raise RuntimeError("libc not found; shm_open unavailable")
    flags = os.O_RDWR
    if create:
        flags |= os.O_CREAT
    if sys.platform == "darwin" and name.startswith("/"):
        name = name.lstrip("/")
    elif sys.platform not in ("darwin",) and not name.startswith("/"):
        name = "/" + name
    libc.shm_open.restype = ctypes.c_int
    fd = libc.shm_open(name.encode(), flags, 0o600)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(f"shm_open({name}) failed: errno {err}")
    if create and size:
        if os.ftruncate(fd, size) != 0:
            raise OSError("ftruncate failed")
    return fd
