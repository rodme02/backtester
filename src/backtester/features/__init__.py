"""Leakage-free feature builders.

Every public function in this package satisfies the **causality
invariant**: the value at time ``t`` depends only on input data with
index ``<= t``. This is enforced by property tests in
``tests/test_features_leakage.py``.

Modules:

- ``technical``       — single-asset price/return features.
- ``macro``           — FRED-driven macro features, lagged.
- ``cross_sectional`` — universe-wide ranks (momentum, vol).
"""

from .cross_sectional import momentum_rank, vol_rank
from .macro import macro_features
from .technical import (
    atr,
    log_returns,
    macd,
    rolling_volatility,
    rsi,
)

__all__ = [
    "atr",
    "log_returns",
    "macd",
    "macro_features",
    "momentum_rank",
    "rolling_volatility",
    "rsi",
    "vol_rank",
]
