"""Make the repo root importable so tests can reach benchmarks/ and scripts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
