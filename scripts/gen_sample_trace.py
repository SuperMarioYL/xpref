"""Generate a synthetic-but-structured Kimi K3 Q4 router-logit sample trace.

The trace is *not* a real Kimi K3 capture (no K3 weights ship with xpref) —
it is a hand-crafted trace with the same geometry (896 experts, 16 active per
token) and the same *temporal structure* a real capture exhibits: a slowly
drifting "hot set" of experts per layer, so consecutive tokens share most of
their fired set. That structure is what makes n-gram + top-k beat blind
readahead. Use ``xpref eval --trace samples/k3-q4-128tok.bin`` to score it.

Run:  python scripts/gen_sample_trace.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xpref.predictor import evaluate
from xpref.trace import write_trace

NUM_EXP = 896
NUM_ACT = 16
NUM_LAY = 8  # trimmed sample — K3 has more; 8 layers are enough to demo the predictor
NUM_TOK = 128
SEED = 20260803


def _drift_centers(rng: np.random.Generator) -> np.ndarray:
    """Per-layer hot-center trajectory, slowly drifting + wrapping."""
    centers = np.zeros((NUM_TOK, NUM_LAY), dtype=np.int64)
    start = rng.integers(0, NUM_EXP, size=NUM_LAY)
    stride = rng.integers(6, 11, size=NUM_LAY)  # tokens per +1 step
    jitter = rng.normal(0, 0.6, size=(NUM_TOK, NUM_LAY))
    for layer in range(NUM_LAY):
        c = start[layer]
        for t in range(NUM_TOK):
            # advance roughly every `stride` tokens, with sub-token smoothing
            c = (start[layer] + (t / stride[layer]) + jitter[t, layer]) % NUM_EXP
            centers[t, layer] = int(c) % NUM_EXP
    return centers


def generate(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = _drift_centers(rng)
    logits = np.full((NUM_TOK, NUM_LAY, NUM_EXP), -2.0, dtype=np.float32)
    fired = np.zeros((NUM_TOK, NUM_LAY, NUM_ACT), dtype=np.uint16)
    n_outliers = 2  # per (token, layer): routing fires a couple "cold" experts
    for t in range(NUM_TOK):
        for layer in range(NUM_LAY):
            c = centers[t, layer]
            block = np.arange(c, c + NUM_ACT) % NUM_EXP
            # swap a couple of block experts for random outliers (routing noise:
            # the router occasionally fires experts the prior can't predict)
            swap = rng.choice(NUM_ACT, size=n_outliers, replace=False)
            outliers = rng.choice(NUM_EXP, size=n_outliers, replace=False)
            block[swap] = outliers
            fired[t, layer] = block.astype(np.uint16)
            hot = np.setdiff1d(block, outliers)
            window = np.arange(c - 4, c + NUM_ACT + 4) % NUM_EXP
            logits[t, layer, hot] = 8.0 + rng.normal(0, 0.9, size=hot.shape[0])
            logits[t, layer, outliers] = 4.5 + rng.normal(0, 0.9, size=n_outliers)
            nbrs = np.setdiff1d(window, block)
            logits[t, layer, nbrs] = 2.5 + rng.normal(0, 0.9, size=nbrs.shape[0])
            logits[t, layer] += rng.normal(0, 0.25, size=NUM_EXP).astype(np.float32)
    return logits, fired


def main() -> None:
    logits, fired = generate()
    out = Path("samples/k3-q4-128tok.bin")
    write_trace(out, logits, fired)
    res = evaluate(logits, fired)
    print(f"wrote {out}  ({out.stat().st_size} bytes)")
    print(json.dumps(res, indent=2))
    assert res["recall_at_16"] > 0.6, f"recall@16 too low: {res['recall_at_16']:.3f}"
    print(f"recall@16 = {res['recall_at_16']:.3f}  (>0.6 ✓)")


if __name__ == "__main__":
    main()
