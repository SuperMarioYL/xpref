"""Pytest bootstrap: ensure the in-tree ``src/`` package is importable even
before ``pip install -e .`` (the canonical CI path installs first). unittest
users should ``pip install -e .`` or set ``PYTHONPATH=src``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
