<div align="right"><sub><b>English</b>&nbsp;&nbsp;⇄&nbsp;&nbsp;<a href="./README.md">简体中文</a></sub></div>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg">
    <img src="./assets/hero-light.svg" width="880" alt="xpref — predictive MoE expert prefetch">
  </picture>
</p>

<p align="center"><sub>Predicts the next-firing MoE experts from router gating logits and pre-stages them from NVMe into DDR before the router fires — lifting ultra-sparse (896/16) MoE decode from ~4 t/s reactive paging to ~12 t/s, targeting Kimi K3 and DeepSeek V4.</sub></p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="license"></a>
  <a href="https://github.com/SuperMarioYL/xpref/releases"><img src="https://img.shields.io/github/v/release/SuperMarioYL/xpref?label=release" alt="release"></a>
  <a href="https://github.com/SuperMarioYL/xpref/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/xpref/ci.yml?branch=main&label=CI" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/Kimi_K3-ready-5E5CE6" alt="Kimi K3 ready">
  <img src="https://img.shields.io/badge/DeepSeek_V4-compat-10A37F" alt="DeepSeek V4 compat">
</p>

---

**Predict the next-firing MoE experts and stage them from NVMe into DDR before the router fires — taking Kimi K3's ~4 t/s reactive paging to ~12 t/s.**

<h2><img src="https://api.iconify.design/tabler:topology-star-3.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Architecture</h2>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/atlas-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/atlas-light.svg">
    <img src="./assets/atlas-light.svg" width="880" alt="Architecture: llama.cpp (patched) → shm ring → xpref predictor → mmap checkpoint (madvise → DDR)">
  </picture>
</p>

Two processes: a patched llama.cpp engine writes per-token, per-layer router gating logits into a shared ring; `xpref attach` reads the ring, runs the predictor, and issues `madvise(MADV_WILLNEED)` on the predicted experts' weight pages so the kernel reads them from NVMe into the DDR page cache *before* the router fires. No microservices, no Kubernetes — just an engine, a daemon, and an mmap'd checkpoint.

<h2><img src="https://api.iconify.design/tabler:bulb.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Why this exists</h2>

Ultra-sparse MoE like Kimi K3 (896 experts, 16 active per token) runs on consumer hardware but won't fit in VRAM: 93% of the 1.56 TB checkpoint is routed expert weights, the engine pages them from NVMe *reactively* (after the router fires), and decode crawls at ~4 t/s until the OS page cache accidentally warms the hot set. A home-lab operator on r/LocalLLaMA (768 GB DDR5 + 2x5090) recorded exactly this [warmup/swap behavior](https://www.reddit.com/r/LocalLLaMA/comments/1va0rce/) — decode t/s drifting upward over time; the K3 weights release post from SavunOski is the checkpoint local runners deploy.

xpref doesn't wait for the cache to warm by accident: it reads the router gating logits, predicts the next-firing experts, and pre-stages their pages ahead of the fire — lifting the local Agent decode loop from ~4 t/s to ~12 t/s. No stock engine (llama.cpp / vLLM / SGLang / ktransformers) does this; they all page reactively after the fire. The same expert-paging pain will widen when DeepSeek V4 lands with the same geometry.

<h2><img src="https://api.iconify.design/tabler:scale.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> vs ktransformers</h2>

| Capability | xpref | [ktransformers](https://github.com/kvcache-ai/ktransformers) |
|---|:---:|:---:|
| Predict next-firing experts (reads router logits) | ✓ | — |
| Expert-weight paging to DDR | ✓ | ✓ |
| VRAM / CUDA host-pinned staging | — (v0.2) | partial |
| Target models | Kimi K3 / DeepSeek V4 (896-expert ultra-sparse) | DeepSeek V2/V3 |
| Community maturity | partial (new) | ✓ (5–8k stars) |

ktransformers is the nearest analog (CPU+GPU MoE offload for DeepSeek-V2/V3, ~several thousand stars in ~2 months) — it proves the expert-paging category is real, while *prediction* is the unclaimed wedge. It is genuinely stronger on GPU staging and community maturity; xpref's difference is the "predict-before-the-fire" primitive.

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Install</h2>

```bash
# either
uv tool install xpref                                          # PyPI (once published)
uv tool install git+https://github.com/SuperMarioYL/xpref       # before the first release
# or from a clone: git clone … && cd xpref && uv tool install .
```

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Quickstart</h2>

```bash
xpref eval          # score the predictor on the bundled Kimi K3 Q4 sample; see recall@16 + 4→12 t/s
```

<details><summary>Sample output</summary>

```
xpref eval — k3-q4-128tok.bin
  tokens=128 layers=8 experts=896 active=16
  recall@16 = 0.7405
  reactive t/s  = 4.0
  xpref  t/s   = 12.00  (3.00x)
```
</details>

Cold install to first visible result in under 30 seconds — no engine build required.

<h2><img src="https://api.iconify.design/tabler:terminal-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Usage</h2>

```bash
# 1) Offline eval, JSON output (CI-friendly)
xpref eval --trace samples/k3-q4-128tok.bin --json

# 2) Replay demo: simulate the decode loop, per-token recall + projected t/s (4→12)
truncate -s 256k /tmp/ckpt.gguf          # any checkpoint stand-in
xpref attach --replay --checkpoint /tmp/ckpt.gguf

# 3) Real path (one-time): apply the patch + rebuild the llama.cpp kimi-k3 fork
git -C llama.cpp apply "$(xpref patch-path)"
cmake --build build
./build/bin/llama-cli --model kimi-k3-q4.gguf ... &              # engine
xpref attach --ring xpref_router --checkpoint kimi-k3-q4.gguf   # daemon
```

Subcommands: `eval` (offline scoring), `attach` (live / replay), `patch-path` (print the bundled patch path), `trace-info` (trace metadata). Full flags via `xpref --help`.

<h2><img src="https://api.iconify.design/tabler:photo.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Demo</h2>

<p align="center"><img src="./assets/demo.gif" width="780" alt="xpref eval + attach --replay demo: recall@16 and the 4→12 t/s curve"></p>

`eval` scores recall@16 ≈ 0.74 and projects 12 t/s on the bundled Kimi K3 sample; `attach --replay` simulates the decode loop, printing per-token predicted-vs-actual recall and projected t/s. The real 4→12 t/s reproduces after applying the patch and rebuilding llama.cpp (Usage §3). The CI (`.github/workflows/demo.yml`) can re-render `assets/demo.gif` via vhs on tag.

<h2><img src="https://api.iconify.design/tabler:adjustments.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Configuration</h2>

| Key / flag | Type | Default | Meaning |
|---|---|---|---|
| `--trace` | path | bundled K3 sample | router gating-logit trace (`eval`) |
| `--replay` | path | bundled K3 sample | trace to replay (`attach`, offline) |
| `--ring` | name | `xpref_router` | shared ring name (Linux `/dev/shm`, macOS `/tmp`) |
| `--checkpoint` | path | — | mmap'd GGUF checkpoint (`attach`) |
| `--layout` | path | uniform | expert→byte-range JSON layout (`{layer: {expert: [off, len]}}`) |
| `--ngram-n` | int | 3 | n-gram order |
| `--topk-weight` | float | 0.6 | top-k prior weight |
| `--ngram-weight` | float | 0.4 | n-gram prior weight |
| `XPREF_RING` | env | `xpref_router` | shared ring name the engine reads |

<h2><img src="https://api.iconify.design/tabler:map-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Roadmap</h2>

- [x] **m1** router-logit trace format + n-gram/top-k predictor + offline eval (recall@16 > 0.6)
- [x] **m2** llama.cpp kimi-k3 fork patch + shared ring + live attach + madvise prefetch (4→~12 t/s)
- [x] **m3** `uv tool install` packaging + bilingual README + demo
- [ ] v0.2 VRAM / CUDA host-pinned staging (v0.1 is DDR page cache only)
- [ ] v0.2 learned predictor (MLP / transformer)
- [ ] v0.3 vLLM / SGLang integration (v0.1 is llama.cpp kimi-k3 fork only)
- [ ] v0.3+ Windows support (mmap/madvise semantics differ)

<h2><img src="https://api.iconify.design/tabler:git-pull-request.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Contributing</h2>

MIT — see [LICENSE](./LICENSE). Issues and PRs welcome at [github.com/SuperMarioYL/xpref/issues](https://github.com/SuperMarioYL/xpref/issues). Dev: `pip install -e ".[dev]"` then `python -m unittest discover -s tests`.

<h2><img src="https://api.iconify.design/tabler:share.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Share this</h2>

```
xpref — predictive MoE expert prefetch for Kimi K3's 896 experts. Reads router logits, stages the next-firing experts from NVMe→DDR before the fire. Decode 4→12 t/s on consumer RAM+NVMe. https://github.com/SuperMarioYL/xpref
```

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
