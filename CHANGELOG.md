# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/SuperMarioYL/xpref/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SuperMarioYL/xpref/releases/tag/v0.1.0
