# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-09

### Fixed
- `xpref attach --ring` (live mode) never measured recall: the scoring loop was
  missing, so it always reported recall=0.000 and projected t/s=4.00 regardless
  of predictor quality. Live attach now scores the previous token's predictions
  against the next token's actual fired set — the same next-token semantics as
  `eval` and `attach --replay`.
- Replay and live attach loops ran the predictor and issued every
  `madvise(MADV_WILLNEED)` hint twice per token (once inside the observe step,
  once inside the scoring step). Each prediction is now hinted exactly once.
- The three-line quickstart (examples/quickstart.sh) documented
  `xpref attach --replay --checkpoint ...`, an invalid form: `--replay` takes a
  path, so the command exited 2 before doing anything. The quickstart now uses
  the working no-flag form (`xpref attach --checkpoint ...` replays the bundled
  sample; pass `--replay PATH` to replay a custom trace).
- The n-gram prior of the predictor was inert: transitions were keyed by the
  exact sorted fired *set*, which never repeats on drifting hot-set traces, so
  the n-gram contributed zero on the shipped sample and `--ngram-weight` had no
  effect at any setting. Transitions are now recorded per source expert (order-1
  over individual experts) with normalized vote fusion, making the two priors
  score-commensurate. Sample-trace recall@16 improves 0.7405 → 0.8154 with
  default weights (topk-only stays 0.7405; n-gram-only 0.0026 → 0.7712).

### Changed
- Version bumped to 0.2.0 across all surfaces (VERSION, pyproject.toml,
  `__version__` fallback, web/site.json `meta.content_version`); README example
  output, the benchmark JSON, and docs/demo-results.json now quote the
  corrected recall numbers.

### Notes
- v0.2 remains DDR/page-cache prefetch via `madvise`; VRAM / CUDA host-pinned
  staging and a learned predictor are still planned (they require GPU-rig
  validation this release did not have).

## [0.1.0] - 2026-08-03

### Added
- Predictive MoE expert prefetch primitive: reads router gating logits and
  predicts the next-firing experts before the router fires, then pre-stages
  their weights from NVMe into the DDR page cache via `madvise(MADV_WILLNEED)`.
- `xpref` CLI (click): `eval` (offline recall + projected t/s), `attach`
  (live ring reader + predictor + prefetcher, with `--replay` offline mode),
  `patch-path` (bundled llama.cpp router-logit patch), `trace-info`.
- n-gram + top-k heuristic predictor (v0.1 is non-learned by design).
- Binary, self-describing router-logit trace format (`XPREFTR1`) with a
  shipped synthetic Kimi K3 Q4 sample trace (8 experts of N, 128 tokens).
- Lock-free single-producer/single-consumer shared ring (`XPRFRING`) backed by
  a memory-mapped file (`/dev/shm` on Linux, `/tmp` on macOS).
- Reference unified diff against the llama.cpp kimi-k3 fork exporting router
  logits + fired experts to the shared ring.
- Bilingual README (Chinese primary + English sibling), animated dark/light
  hero + architecture SVGs (SMIL, no script), rendered demo gif, CI
  (lint + test), release (wheel + sdist), and demo (vhs) workflows.

### Notes
- v0.1 prefetches to DDR via `madvise`; VRAM / CUDA host-pinned staging and a
  learned predictor are v0.2. Only the llama.cpp kimi-k3 fork is supported;
  vLLM/SGLang integration is v0.3. Linux + macOS only (Windows is v0.3+).

[Unreleased]: https://github.com/SuperMarioYL/xpref/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/SuperMarioYL/xpref/releases/tag/v0.2.0
[0.1.0]: https://github.com/SuperMarioYL/xpref/releases/tag/v0.1.0
