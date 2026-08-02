"""``xpref`` command-line interface.

Subcommands:

* ``eval``      — run the predictor on a shipped router-logit trace, print
                  recall@16 and a projected t/s curve.
* ``attach``    — live or replay attach: read the router-logit ring, predict
                  next, prefetch via madvise, print live recall + projected t/s.
* ``patch-path``— print the path to the bundled llama.cpp router-logit patch
                  (for ``git -C llama.cpp apply "$(xpref patch-path)"``).
* ``trace-info``— print a trace's metadata.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import click

from . import __version__
from .attach import REACTIVE_TPS, live_attach, projected_tps, replay_attach
from .predictor import evaluate
from .trace import read_trace


def _patch_path() -> Path:
    """Resolve the bundled patch file shipped inside the package."""
    pkg = resources.files("xpref").joinpath("engine_patch", "llama_cpp_router_logit.patch")
    # importlib.resources returns a Traversable; materialise to a real path.
    with resources.as_file(pkg) as p:  # type: ignore[arg-type]
        return Path(p)


def _bundled_sample() -> Path:
    """Resolve the shipped Kimi K3 Q4 sample trace.

    Looks first inside the installed package (the wheel force-includes the
    sample), then falls back to the repo-root ``samples/`` copy so a fresh
    clone works before any install.
    """
    try:
        pkg = resources.files("xpref").joinpath("samples", "k3-q4-128tok.bin")
        with resources.as_file(pkg) as p:  # type: ignore[arg-type]
            cand = Path(p)
            if cand.exists():
                return cand
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        pass
    for cand in (Path.cwd() / "samples" / "k3-q4-128tok.bin",
                 Path(__file__).resolve().parents[2] / "samples" / "k3-q4-128tok.bin"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "sample trace not found — install xpref or pass --trace PATH")


@click.group()
@click.version_option(__version__, prog_name="xpref")
def main() -> None:
    """xpref — predictive MoE expert prefetch."""


@main.command()
@click.option("--trace", "trace", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Router-logit trace (default: the bundled Kimi K3 Q4 sample).")
@click.option("--ngram-n", default=3, show_default=True, type=int)
@click.option("--topk-weight", default=0.6, show_default=True, type=float)
@click.option("--ngram-weight", default=0.4, show_default=True, type=float)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def eval(trace: str, ngram_n: int, topk_weight: float, ngram_weight: float,
         as_json: bool) -> None:
    """Score the predictor on a trace; print recall@16 + projected t/s."""
    if trace is None:
        trace = str(_bundled_sample())
    tr = read_trace(trace)
    res = evaluate(tr.logits, tr.fired, ngram_n=ngram_n, topk_weight=topk_weight,
                   ngram_weight=ngram_weight)
    tps = projected_tps(res["recall_at_k"], res["num_active"])
    out = {
        **res,
        "reactive_tps": REACTIVE_TPS,
        "projected_predictive_tps": round(tps, 2),
        "speedup": round(tps / REACTIVE_TPS, 2),
    }
    if as_json:
        click.echo(json.dumps(out, indent=2))
        return
    click.echo(f"xpref eval — {Path(trace).name}")
    click.echo(f"  tokens={res['num_tokens']} layers={res['num_layers']} "
               f"experts={res['num_experts']} active={res['num_active']}")
    click.echo(f"  recall@{res['num_active']} = {res['recall_at_k']:.4f}")
    click.echo(f"  reactive t/s  = {REACTIVE_TPS:.1f}")
    click.echo(f"  xpref  t/s   = {tps:.2f}  ({tps / REACTIVE_TPS:.2f}x)")
    click.echo("  per-layer recall:")
    for i, r in enumerate(res["per_layer_recall"]):
        click.echo(f"    layer {i:>2}: {r:.4f}")


@main.command()
@click.option("--ring", default=None, type=str,
              help="Live ring path (or a bare name resolved to /dev/shm on Linux).")
@click.option("--replay", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Replay a trace through the predictor (default: bundled sample).")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to the mmap'd GGUF checkpoint (real or synthetic).")
@click.option("--layout", default=None, type=click.Path(exists=True, dir_okay=False),
              help="JSON expert→range layout (default: uniform).")
@click.option("--ngram-n", default=3, show_default=True, type=int)
@click.option("--topk-weight", default=0.6, show_default=True, type=float)
@click.option("--ngram-weight", default=0.4, show_default=True, type=float)
@click.option("--max-tokens", default=None, type=int, help="Stop after N tokens (live).")
def attach(ring, replay, checkpoint, layout, ngram_n, topk_weight, ngram_weight,
           max_tokens) -> None:
    """Read router logits, predict next experts, prefetch via madvise."""
    if replay is None and ring is None:
        replay = str(_bundled_sample())  # default: replay the shipped sample
    if replay is not None:
        stats = replay_attach(replay, checkpoint, ngram_n=ngram_n,
                              topk_weight=topk_weight, ngram_weight=ngram_weight,
                              layout=layout, verbose=True)
        click.echo(f"\nreplay done: tokens={stats.tokens_seen} recall={stats.recall:.3f} "
                   f"proj t/s={stats.projected_tps:.2f} "
                   f"prefetched={stats.bytes_prefetched} bytes")
        return
    if ring is None:
        raise click.UsageError("attach needs --ring (live) or --replay (offline).")
    from .ringbuf import default_ring_path
    ring_path = ring if "/" in ring or "\\" in ring else str(default_ring_path(ring))
    stats = live_attach(ring_path, checkpoint, ngram_n=ngram_n,
                        topk_weight=topk_weight, ngram_weight=ngram_weight,
                        layout=layout, max_tokens=max_tokens)
    click.echo(f"\nlive attach: tokens={stats.tokens_seen} recall={stats.recall:.3f} "
               f"proj t/s={stats.projected_tps:.2f}")


@main.command(name="patch-path")
def patch_path() -> None:
    """Print the path to the bundled llama.cpp router-logit patch."""
    click.echo(str(_patch_path()))


@main.command(name="trace-info")
@click.option("--trace", required=True, type=click.Path(exists=True, dir_okay=False))
def trace_info(trace: str) -> None:
    """Print a trace's metadata."""
    tr = read_trace(trace)
    click.echo(json.dumps(tr.summary(), indent=2))


if __name__ == "__main__":
    main()
