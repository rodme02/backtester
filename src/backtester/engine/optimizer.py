"""Parameter-grid optimisation (sequential, deterministic)."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import Any

import pandas as pd

from .runner import run_backtest


def optimize(
    strategy_cls,
    data: pd.DataFrame,
    grid: dict[str, Iterable[Any]],
    *,
    cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
) -> pd.DataFrame:
    """Sweep ``grid`` cartesian product, return one row per combination."""
    keys = list(grid.keys())
    rows: list[dict[str, Any]] = []
    for combo in product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo, strict=True))
        result = run_backtest(
            strategy_cls, data, params=params,
            cash=cash, commission=commission, slippage=slippage,
        )
        rows.append({**params, **result.metrics, "final_value": result.final_value})
    return pd.DataFrame(rows)
