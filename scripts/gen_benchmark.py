"""Regenerate benchmarks/k3-q4-baseline-vs-xpref.json from the shipped sample.

Runs ``xpref``'s predictor on ``samples/k3-q4-128tok.bin`` and writes the
recall + projected t/s curve to the committed benchmark JSON so the numbers
in the README stay tied to the code.

Run:  python scripts/gen_benchmark.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from xpref.attach import REACTIVE_TPS, projected_tps
from xpref.predictor import evaluate
from xpref.trace import read_trace

TRACE = Path("samples/k3-q4-128tok.bin")
OUT = Path("benchmarks/k3-q4-baseline-vs-xpref.json")


def main() -> None:
    tr = read_trace(TRACE)
    res = evaluate(tr.logits, tr.fired)
    tps = projected_tps(res["recall_at_k"], res["num_active"])
    doc = {
        "model": "Kimi K3 (Q4 sample trace, 8 of N layers, 128 tokens)",
        "geometry": {
            "num_experts": res["num_experts"],
            "num_active": res["num_active"],
            "num_layers": res["num_layers"],
            "num_tokens": res["num_tokens"],
        },
        "baseline": {
            "mode": "reactive on-demand expert paging (llama.cpp kimi-k3 fork)",
            "decode_tps": REACTIVE_TPS,
        },
        "xpref": {
            "predictor": "n-gram + top-k heuristic (v0.1, non-learned)",
            "recall_at_16": round(res["recall_at_k"], 4),
            "projected_decode_tps": round(tps, 2),
            "speedup": round(tps / REACTIVE_TPS, 2),
            "per_layer_recall": [round(x, 4) for x in res["per_layer_recall"]],
        },
        "note": (
            "Reactive t/s is the observed ~4 t/s on 768 GB DDR5 + NVMe paging; "
            "xpref projected t/s models decode cost as seek-dominated with misses "
            "falling from num_active to (1-recall)*num_active, capped at the "
            "bandwidth ceiling. Real measured t/s is read off llama-cli's output "
            "after applying the bundled patch + rebuild."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(doc["xpref"], indent=2))


if __name__ == "__main__":
    main()
