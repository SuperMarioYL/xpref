"""Binary router-logit trace format.

A trace carries, for every ``(token, layer)`` pair:

* the router **gating logits** over all experts (the predictor's input), and
* the **fired** expert ids (ground truth, for offline eval / live recall).

Layout (all integers little-endian, on-disk):

    magic    : 8 bytes  = b"XPREFTR1"
    num_tok  : uint32
    num_lay  : uint32
    num_exp  : uint32       # experts per layer (e.g. 896 for Kimi K3)
    num_act  : uint32       # experts fired per (token, layer) (e.g. 16)
    for t in num_tok:
        for l in num_lay:
            logits : num_exp  * float32        # gating logits
            fired  : num_act  * uint16         # ids of experts that fired

The format is intentionally self-describing and trivial to emit from C: a
patched engine just appends ``num_exp`` floats + ``num_act`` uint16s per
(token, layer). See ``src/xpref/engine_patch/llama_cpp_router_logit.patch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"XPREFTR1"
_HEADER = np.dtype(
    [
        ("magic", "S8"),
        ("num_tok", "<u4"),
        ("num_lay", "<u4"),
        ("num_exp", "<u4"),
        ("num_act", "<u4"),
    ]
)


@dataclass
class Trace:
    """An in-memory router-logit + fired-expert trace."""

    logits: np.ndarray  # (T, L, E) float32
    fired: np.ndarray  # (T, L, A) uint16
    num_experts: int
    num_active: int

    @property
    def num_tokens(self) -> int:
        return int(self.logits.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.logits.shape[1])

    def __len__(self) -> int:  # number of tokens
        return self.num_tokens

    def summary(self) -> dict:
        return {
            "num_tokens": self.num_tokens,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "num_active": self.num_active,
            "bytes": self.logits.nbytes + self.fired.nbytes + _HEADER.itemsize,
        }


def write_trace(path: str | Path, logits: np.ndarray, fired: np.ndarray) -> None:
    """Write a trace. ``logits``: (T, L, E) float32, ``fired``: (T, L, A) uint16."""
    logits = np.ascontiguousarray(logits, dtype="<f4")
    fired = np.ascontiguousarray(fired, dtype="<u2")
    if logits.ndim != 3 or fired.ndim != 3:
        raise ValueError("logits/fired must be (T, L, E)/(T, L, A)")
    if logits.shape[:2] != fired.shape[:2]:
        raise ValueError("logits and fired must share (T, L)")
    num_tok, num_lay, num_exp = logits.shape
    _num_tok2, _num_lay2, num_act = fired.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros((), dtype=_HEADER)
    header["magic"] = MAGIC
    header["num_tok"] = num_tok
    header["num_lay"] = num_lay
    header["num_exp"] = num_exp
    header["num_act"] = num_act
    with path.open("wb") as fh:
        fh.write(header.tobytes())
        # interleave per (token, layer): logits then fired — matches the C emit order.
        for t in range(num_tok):
            for lay in range(num_lay):
                fh.write(logits[t, lay].tobytes())
                fh.write(fired[t, lay].tobytes())


def read_trace(path: str | Path) -> Trace:
    """Read a trace written by :func:`write_trace` (or the patched engine)."""
    path = Path(path)
    raw = np.fromfile(path, dtype=_HEADER, count=1)
    if raw.size == 0:
        raise ValueError(f"empty trace: {path}")
    hdr = raw[0]
    if bytes(hdr["magic"]) != MAGIC:
        raise ValueError(f"bad magic {bytes(hdr['magic'])!r}, expected {MAGIC!r}")
    num_tok = int(hdr["num_tok"])
    num_lay = int(hdr["num_lay"])
    num_exp = int(hdr["num_exp"])
    num_act = int(hdr["num_act"])

    rec_logits = np.dtype((np.float32, num_exp))
    rec_fired = np.dtype((np.uint16, num_act))
    rec = np.dtype([("logits", rec_logits), ("fired", rec_fired)])
    data = np.fromfile(path, dtype=rec, offset=_HEADER.itemsize, count=num_tok * num_lay)
    if data.size != num_tok * num_lay:
        raise ValueError(
            f"truncated trace: expected {num_tok * num_lay} records, got {data.size}"
        )
    logits = data["logits"].reshape(num_tok, num_lay, num_exp).astype("<f4")
    fired = data["fired"].reshape(num_tok, num_lay, num_act).astype("<u2")
    return Trace(logits=logits, fired=fired, num_experts=num_exp, num_active=num_act)
