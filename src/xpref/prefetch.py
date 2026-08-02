"""Expert-weight prefetch via ``madvise(MADV_WILLNEED)``.

The checkpoint (a Q4/Q1 GGUF) is ``mmap``'d read-only. When the predictor
says expert ``(layer, expert_id)`` will fire next, xpref asks the kernel to
stage that expert's weight pages from NVMe into the page cache *before* the
router fires — turning a synchronous NVMe read at decode time into a
background readahead. VRAM / CUDA host-pinned staging is explicitly out of
scope for v0.1 (a v0.2 concern); v0.1 pages to DDR, which is enough to clear
the reactive-paging decode valley.

The expert→byte-range mapping defaults to a *uniform* layout (each expert is
an equal slice of its layer's block) which is correct for the synthetic
demo checkpoint and a reasonable first approximation for real GGUFs; supply
``--layout`` (a JSON ``{layer: {expert_id: [offset, length]}}``) to override
once the real GGUF tensor layout is known.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import mmap
from dataclasses import dataclass
from pathlib import Path

MADV_WILLNEED = 3  # portable across Linux + macOS


@dataclass
class ExpertLayout:
    """Maps ``(layer, expert_id)`` → ``(offset, length)`` in the checkpoint."""

    num_layers: int
    num_experts: int
    file_size: int
    table: dict  # {(layer, expert_id): (offset, length)}

    def range_of(self, layer: int, expert_id: int) -> tuple[int, int]:
        return self.table.get((layer, expert_id), (0, 0))

    @classmethod
    def uniform(cls, num_layers: int, num_experts: int, file_size: int) -> ExpertLayout:
        """Equal slices: layer L's experts occupy [L*layer_size, (L+1)*layer_size)."""
        if num_layers <= 0 or num_experts <= 0:
            raise ValueError("need positive num_layers/num_experts")
        layer_size = file_size // num_layers
        expert_size = layer_size // num_experts
        table: dict[tuple[int, int], tuple[int, int]] = {}
        for layer in range(num_layers):
            base = layer * layer_size
            for eid in range(num_experts):
                table[(layer, eid)] = (base + eid * expert_size, expert_size)
        return cls(num_layers, num_experts, file_size, table)

    @classmethod
    def from_json(cls, path: str | Path, num_layers: int, num_experts: int,
                  file_size: int) -> ExpertLayout:
        raw = json.loads(Path(path).read_text())
        table: dict[tuple[int, int], tuple[int, int]] = {}
        for lstr, inner in raw.items():
            layer = int(lstr)
            for estr, rng in inner.items():
                table[(layer, int(estr))] = (int(rng[0]), int(rng[1]))
        return cls(num_layers, num_experts, file_size, table)


class Prefetcher:
    """``mmap`` the checkpoint and ``madvise(WILLNEED)`` predicted expert pages."""

    def __init__(self, checkpoint: str | Path, layout: ExpertLayout):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        self.layout = layout
        self._fh = self.checkpoint.open("rb")
        # ACCESS_COPY maps a private copy-on-write view: read-only backing,
        # but a *writable* buffer so ctypes.from_buffer can hand its address to
        # madvise. madvise(MADV_WILLNEED) on a COW mapping still hints the
        # kernel to readahead the backing file pages.
        try:
            self._mm: mmap.mmap | None = mmap.mmap(
                self._fh.fileno(), 0, access=mmap.ACCESS_COPY
            )
        except (ValueError, OSError):
            # zero-length or unreadable checkpoint — degrade to no-op prefetch
            self._mm = None
        self._libc = None
        lib = ctypes.util.find_library("c")
        if lib is not None and self._mm is not None:
            try:
                self._libc = ctypes.CDLL(lib, use_errno=True)
                self._libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                self._libc.madvise.restype = ctypes.c_int
            except OSError:
                self._libc = None

    @property
    def available(self) -> bool:
        return self._libc is not None and self._mm is not None

    def _base_addr(self) -> int:
        # address of the mmap's first byte via a ctypes view over the buffer.
        return ctypes.addressof(ctypes.c_char.from_buffer(self._mm))  # type: ignore[arg-type]

    def prefetch(self, layer: int, expert_id: int) -> tuple[int, int]:
        """Hint the kernel to readahead expert ``(layer, expert_id)``.

        Returns ``(offset, length)`` actually hinted (``(0,0)`` if no-op).
        Prefetch is advisory and must never raise — a failure here degrades to
        a no-op so the decode loop is unaffected.
        """
        try:
            offset, length = self.layout.range_of(layer, expert_id)
            if length == 0 or self._mm is None or self._libc is None:
                return (0, 0)
            end = min(offset + length, len(self._mm))
            length = max(0, end - offset)
            if length == 0:
                return (0, 0)
            addr = ctypes.c_void_p(self._base_addr() + offset)
            rc = self._libc.madvise(addr, ctypes.c_size_t(length),
                                    ctypes.c_int(MADV_WILLNEED))
            return (offset, length) if rc == 0 else (0, 0)
        except (OSError, ValueError, TypeError):
            return (0, 0)

    def prefetch_many(self, pairs: list[tuple[int, int]]) -> int:
        """Prefetch a set of ``(layer, expert_id)``; return total bytes hinted."""
        total = 0
        for layer, eid in pairs:
            _, n = self.prefetch(layer, eid)
            total += n
        return total

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Prefetcher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
