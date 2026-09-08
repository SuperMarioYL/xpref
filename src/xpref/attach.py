"""Live attach: read the router-logit ring, predict next, prefetch, measure.

Two entry points:

* ``live_attach`` — runs against a real ``RingBuffer`` the patched engine is
  writing. Reads new ``(token, layer)`` records, feeds them to the predictor,
  prefetches the predicted experts, and prints live predicted-vs-actual
  recall plus a projected decode t/s. The real 4→12 t/s is read off the
  engine's own output (llama-cli prints t/s); xpref prints the *recall* and
  the projected gain so you can correlate the two.

* ``replay_attach`` — replays a shipped trace (``--replay``) through the same
  predictor + prefetcher to demo the 4→12 effect without a llama.cpp build.
  It simulates the decode loop: every token, observe the router logits +
  fired set, predict next, prefetch, then score recall against the *next*
  token's actual firing. Projected t/s is modelled from the recall and the
  measured ~4 t/s reactive baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .predictor import Prediction, Predictor
from .prefetch import ExpertLayout, Prefetcher
from .ringbuf import RingBuffer
from .trace import Trace, read_trace

# Calibrated to the observed Kimi K3 reactive-paging baseline (~4 t/s on
# 768 GB DDR5 + NVMe) and the bandwidth ceiling (~12 t/s once the hot expert
# set is resident). The model is seek-dominated: decode cost scales with the
# number of *missed* experts (those not in the page cache when the router
# fires). Prediction turns hits into background readahead.
REACTIVE_TPS = 4.0
BANDWIDTH_CEILING_TPS = 12.0


def projected_tps(recall: float, num_active: int) -> float:
    """Project decode t/s from recall: misses fall from num_active to (1-r)*num_active."""
    misses_reactive = float(num_active)
    misses_predictive = max(1.0, (1.0 - recall) * num_active)
    ratio = misses_reactive / misses_predictive
    return min(BANDWIDTH_CEILING_TPS, REACTIVE_TPS * ratio)


@dataclass
class AttachStats:
    tokens_seen: int
    recall: float
    projected_tps: float
    bytes_prefetched: int


def _group_by_token(records: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in records:
        out.setdefault(r["token_idx"], []).append(r)
    return out


def _run_token(predictor: Predictor, logits_tok: np.ndarray, fired_tok: np.ndarray,
               prefetcher: Prefetcher | None) -> tuple[list[Prediction], int]:
    """Observe a token, predict next, prefetch. Returns (predictions, bytes hinted)."""
    predictor.observe(logits_tok, fired_tok)
    preds = predictor.predict_next()
    bytes_hinted = 0
    if prefetcher is not None:
        for p in preds:
            pairs = [(p.layer, int(eid)) for eid in p.ids]
            bytes_hinted += prefetcher.prefetch_many(pairs)
    return preds, bytes_hinted


def replay_attach(trace_path: str | Path, checkpoint_path: str | Path,
                  *, ngram_n: int = 3, topk_weight: float = 0.6,
                  ngram_weight: float = 0.4, layout: str | None = None,
                  verbose: bool = True) -> AttachStats:
    """Replay a trace through the predictor + prefetcher (no live engine).

    Simulates the decode loop token-by-token and prints a live-style log of
    recall and projected t/s climbing 4 → 12.
    """
    tr: Trace = read_trace(trace_path)
    E, A, L, T = tr.num_experts, tr.num_active, tr.num_layers, tr.num_tokens

    ckpt = Path(checkpoint_path)
    layout_obj: ExpertLayout
    if layout is not None:
        layout_obj = ExpertLayout.from_json(layout, L, E, ckpt.stat().st_size)
    else:
        layout_obj = ExpertLayout.uniform(L, E, ckpt.stat().st_size)
    prefetcher = Prefetcher(ckpt, layout_obj)

    predictor = Predictor(E, A, ngram_n=ngram_n, topk_weight=topk_weight,
                          ngram_weight=ngram_weight)

    recalls: list[float] = []
    bytes_total = 0
    if verbose:
        mode = "live madvise" if prefetcher.available else "no-op (madvise unavailable)"
        print(f"xpref attach — replay {Path(trace_path).name}  [{mode}]")
        print(f"  checkpoint={ckpt.name} ({ckpt.stat().st_size} bytes)  "
              f"experts={E} active={A} layers={L} tokens={T}")
        print(f"  {'tok':>4} {'recall':>8} {'pred/act':>9} {'proj t/s':>9} "
              f"{'prefetch':>10}")
    for t in range(T - 1):
        logits_tok = tr.logits[t]  # (L, E)
        fired_tok = tr.fired[t]  # (L, A)
        preds, layer_bytes = _run_token(predictor, logits_tok, fired_tok, prefetcher)
        # score the prediction made for token t+1 against what actually fired
        actual = tr.fired[t + 1]  # (L, A)
        rec = float(np.mean([p.overlap_with(actual[p.layer]) for p in preds]))
        recalls.append(rec)
        bytes_total += layer_bytes
        tps = projected_tps(rec, A)
        if verbose and (t % max(1, T // 16) == 0 or t == T - 2):
            print(f"  {t:>4} {rec:>8.3f} "
                  f"{int(round(rec * A)):>4}/{A:<4} {tps:>9.2f} "
                  f"{layer_bytes / 1024:>8.0f} KB")
    prefetcher.close()
    if not recalls:
        return AttachStats(0, 0.0, REACTIVE_TPS, 0)
    mean_recall = float(np.mean(recalls))
    return AttachStats(
        tokens_seen=len(recalls),
        recall=mean_recall,
        projected_tps=projected_tps(mean_recall, A),
        bytes_prefetched=bytes_total,
    )


def live_attach(ring_path: str | Path, checkpoint_path: str | Path,
                *, ngram_n: int = 3, topk_weight: float = 0.6,
                ngram_weight: float = 0.4, layout: str | None = None,
                max_tokens: int | None = None,
                poll_interval: float = 0.05) -> AttachStats:
    """Attach to a live ring the patched engine is writing. Blocks until interrupted
    or ``max_tokens`` tokens complete."""
    ring = RingBuffer(ring_path, create=False)
    meta = ring.meta
    assert meta is not None
    ckpt = Path(checkpoint_path)
    layout_obj = (ExpertLayout.from_json(layout, meta.num_layers, meta.num_experts,
                                         ckpt.stat().st_size) if layout is not None
                  else ExpertLayout.uniform(meta.num_layers, meta.num_experts, ckpt.stat().st_size))
    prefetcher = Prefetcher(ckpt, layout_obj)
    predictor = Predictor(meta.num_experts, meta.num_active, ngram_n=ngram_n,
                          topk_weight=topk_weight, ngram_weight=ngram_weight)

    pending: dict[int, list[dict]] = {}
    recalls: list[float] = []
    bytes_total = 0
    last_token = -1
    # predictions made after the previous token, scored when the next token's
    # ground truth arrives (same next-token semantics as evaluate()/replay)
    prev_preds: list[Prediction] | None = None
    print(f"xpref attach — ring={ring_path} checkpoint={checkpoint_path} "
          f"(experts={meta.num_experts} active={meta.num_active} layers={meta.num_layers})")
    try:
        while True:
            recs = ring.read_new()
            for r in recs:
                pending.setdefault(r["token_idx"], []).append(r)
            # process complete tokens in order
            while True:
                tok = last_token + 1
                if tok not in pending or len(pending[tok]) < meta.num_layers:
                    break
                rows = sorted(pending.pop(tok), key=lambda x: x["layer"])
                logits_tok = np.stack([r["logits"] for r in rows])  # (L, E)
                fired_tok = np.stack([r["fired"] for r in rows])  # (L, A)
                if prev_preds is not None:
                    hits = [p.overlap_with(fired_tok[p.layer]) for p in prev_preds]
                    recalls.append(float(np.mean(hits)))
                preds, hinted = _run_token(predictor, logits_tok, fired_tok, prefetcher)
                prev_preds = preds
                bytes_total += hinted
                last_token = tok
                if max_tokens and last_token + 1 >= max_tokens:
                    raise StopIteration
            time.sleep(poll_interval)
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        ring.close()
        prefetcher.close()
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    return AttachStats(
        tokens_seen=last_token + 1,
        recall=mean_recall,
        projected_tps=projected_tps(mean_recall, meta.num_active) if recalls else REACTIVE_TPS,
        bytes_prefetched=bytes_total,
    )
