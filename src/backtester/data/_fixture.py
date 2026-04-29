"""Helper for the ``BACKTESTER_FIXTURE_MODE`` env-var path.

When set to ``1``, every live data loader returns a bundled fixture
from ``samples/`` instead of hitting the network. Used by CI so the
notebook-execution job is hermetic and fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "samples" / "fixtures"


def fixture_mode_active() -> bool:
    return os.environ.get("BACKTESTER_FIXTURE_MODE", "").strip() in ("1", "true", "TRUE")


def load_fixture(filename: str) -> pd.DataFrame | None:
    path = FIXTURES_DIR / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    return df
