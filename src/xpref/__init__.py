"""xpref — predictive MoE expert prefetch.

Reads router gating logits and predicts which experts the router will fire
next, then pre-stages their weights from NVMe into the kernel page cache
(``madvise(MADV_WILLNEED)``) before the fire — collapsing the cold-start
decode valley of ultra-sparse MoE (e.g. Kimi K3's 896/16) from ~4 t/s
reactive paging toward the bandwidth ceiling.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .predictor import Predictor
from .trace import Trace, read_trace, write_trace

try:  # pragma: no cover - resolved at runtime
    __version__ = version("xpref")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["Predictor", "Trace", "read_trace", "write_trace", "__version__"]
