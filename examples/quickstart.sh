#!/usr/bin/env bash
# xpref in three lines — install, score the predictor, replay the decode loop.
set -euo pipefail

uv tool install git+https://github.com/SuperMarioYL/xpref      # 1) install
xpref eval                                                     # 2) recall@16 + 4→12 t/s on the bundled K3 sample
truncate -s 256k /tmp/ckpt.gguf && xpref attach --replay --checkpoint /tmp/ckpt.gguf   # 3) simulate the 4→12 decode loop
