"""Heuristic expert-firing predictor (n-gram + top-k of router gating).

This is the v0.1 primitive: a *non-learned* union of

* **top-k prior** — the router already "almost" fires the top-k of its current
  gating distribution; gating is temporally smooth, so the current top-k is a
  strong prior on the *next* token's firings, and
* **n-gram prior** — over the recent fired-expert sequences, which experts
  co-occur with / follow the most recent firings.

It deliberately beats blind readahead (the OS page cache cannot see router
logits) without the cold-start latency budget a learned predictor would burn.
A learned predictor (MLP/transformer) is explicitly out of scope for v0.1.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Prediction:
    """A single per-layer prediction."""

    layer: int
    ids: np.ndarray  # (K,) uint16, ranked best-first
    scores: np.ndarray  # (K,) float32, in the same order
    confidence: float  # mean of top-K scores, a rough "how sure"

    def overlap_with(self, actual: np.ndarray) -> float:
        """Fraction of ``actual`` recovered by the predicted set (recall)."""
        k = min(len(self.ids), len(actual))
        if k == 0:
            return 0.0
        topk = set(int(x) for x in self.ids[:k])
        return len(topk & {int(x) for x in actual}) / len(actual)


@dataclass
class Predictor:
    """n-gram + top-k expert-firing predictor."""

    num_experts: int
    num_active: int
    ngram_n: int = 3
    topk_weight: float = 0.6
    ngram_weight: float = 0.4
    _history_logits: list = field(default_factory=list, repr=False)
    _history_fired: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # per-layer transition table: {source expert: {follower expert: decayed count}}
        self._per_layer_trans: list[dict[int, dict[int, float]]] = []

    # ------------------------------------------------------------------ API --
    def observe(self, logits: np.ndarray, fired: np.ndarray) -> None:
        """Feed one token's ``logits`` (L, E) and ``fired`` (L, A) into history.

        Also updates the per-layer transition table: every expert that fired
        at token t-1 is a "source" whose followers we expect to fire at t.
        Transitions are recorded per source expert (not per fired *set*) so a
        slowly drifting hot set still produces matches.
        """
        L = logits.shape[0]
        if not self._per_layer_trans:
            self._per_layer_trans = [
                defaultdict(lambda: defaultdict(float)) for _ in range(L)
            ]
        self._history_logits.append(logits)
        self._history_fired.append(fired)
        if len(self._history_fired) >= 2:
            prev = self._history_fired[-2]
            cur = self._history_fired[-1]
            for layer in range(min(len(prev), len(cur), L)):
                tab = self._per_layer_trans[layer]
                # gentle decay so the table tracks expert-set drift
                for followers in tab.values():
                    for f in followers:
                        followers[f] *= 0.98
                for e in prev[layer]:
                    followers = tab[int(e)]
                    for f in cur[layer]:
                        followers[int(f)] += 1.0
        # bound history
        self._history_logits = self._history_logits[-self.ngram_n - 1 :]
        self._history_fired = self._history_fired[-self.ngram_n - 1 :]

    def predict_next(self) -> list[Prediction]:
        """Predict the next token's per-layer fired experts (top-K each)."""
        if not self._history_logits:
            raise RuntimeError("predict_next called before any observation")
        K = self.num_active
        preds: list[Prediction] = []
        L = self._history_logits[-1].shape[0]
        for layer in range(L):
            scores = np.zeros(self.num_experts, dtype=np.float32)
            # --- top-k prior: softmax over the latest gating logits, take soft mass
            latest = self._history_logits[-1][layer].astype(np.float64)
            latest = latest - latest.max()
            w = np.exp(latest)
            w = w / w.sum()
            scores += self.topk_weight * w.astype(np.float32)
            # --- n-gram prior: followers of the most recent fired experts.
            # Each recently-fired expert votes a normalized follower
            # distribution; votes are averaged over the sources so the total
            # n-gram mass stays <= ngram_weight and the two priors remain
            # score-commensurate (the weights act as a real blend).
            if self._history_fired and self.ngram_weight > 0:
                tab = (
                    self._per_layer_trans[layer]
                    if layer < len(self._per_layer_trans)
                    else {}
                )
                last = [int(x) for x in self._history_fired[-1][layer]]
                votes: dict[int, float] = defaultdict(float)
                for e in last:
                    followers = tab.get(e)
                    if not followers:
                        continue
                    tot = sum(followers.values())
                    if tot <= 0:
                        continue
                    for f, cnt in followers.items():
                        votes[f] += cnt / tot
                if votes and last:
                    scale = self.ngram_weight / len(last)
                    for f, v in votes.items():
                        scores[f] += scale * v
            order = np.argsort(scores)[::-1][:K]
            top_scores = scores[order]
            preds.append(
                Prediction(
                    layer=layer,
                    ids=order.astype(np.uint16),
                    scores=top_scores,
                    confidence=float(top_scores.mean()) if len(top_scores) else 0.0,
                )
            )
        return preds


def evaluate(
    logits: np.ndarray,
    fired: np.ndarray,
    *,
    ngram_n: int = 3,
    topk_weight: float = 0.6,
    ngram_weight: float = 0.4,
) -> dict:
    """Walk a trace, predicting token t+1 from tokens [0..t].

    Returns recall@K (K = num_active), recall@all-active and friends.
    ``logits``: (T, L, E), ``fired``: (T, L, A).
    """
    T, L, E = logits.shape
    _T2, _L2, A = fired.shape
    K = A
    p = Predictor(
        num_experts=E,
        num_active=A,
        ngram_n=ngram_n,
        topk_weight=topk_weight,
        ngram_weight=ngram_weight,
    )
    recalls = []
    per_layer_recall = [0.0] * L
    per_layer_n = [0] * L
    # cold start: the first prediction is purely top-k (no history) — still fair.
    for t in range(T - 1):
        p.observe(logits[t], fired[t])
        preds = p.predict_next()
        actual = fired[t + 1]  # (L, A)
        for layer in range(L):
            r = preds[layer].overlap_with(actual[layer])
            recalls.append(r)
            per_layer_recall[layer] += r
            per_layer_n[layer] += 1
    recall_at_k = float(np.mean(recalls)) if recalls else 0.0
    per_layer = [
        (per_layer_recall[i] / per_layer_n[i] if per_layer_n[i] else 0.0)
        for i in range(L)
    ]
    return {
        "num_tokens": T,
        "num_layers": L,
        "num_experts": E,
        "num_active": A,
        "recall_at_k": recall_at_k,  # K == num_active (e.g. 16)
        "recall_at_16": recall_at_k if K == 16 else None,
        "per_layer_recall": per_layer,
        "predictions_made": len(recalls),
    }
