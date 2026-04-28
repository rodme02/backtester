"""Convert backtrader analyzer output into plain dicts and pandas Series."""

from __future__ import annotations

from typing import Any

import pandas as pd


def equity_curve_from_strategy(strat: Any, *, starting_cash: float) -> pd.Series:
    """Reconstruct equity curve from the TimeReturn analyzer."""
    timereturn = strat.analyzers.timereturn.get_analysis()
    if not timereturn:
        return pd.Series(dtype=float, name="equity")
    series = pd.Series(timereturn).sort_index()
    equity = (1.0 + series).cumprod() * starting_cash
    equity.name = "equity"
    return equity


def summarize(strat: Any, *, starting_cash: float, final_value: float) -> dict[str, Any]:
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    dd = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    total_closed = getattr(trades.total, "closed", 0) if hasattr(trades, "total") else 0
    won = getattr(trades.won, "total", 0) if hasattr(trades, "won") else 0
    win_rate = (100.0 * won / total_closed) if total_closed else 0.0
    total_return_pct = (final_value / starting_cash - 1.0) * 100.0

    return {
        "sharpe": sharpe,
        "max_drawdown_pct": dd.get("max", {}).get("drawdown"),
        "total_trades": total_closed,
        "win_rate_pct": win_rate,
        "total_return_pct": total_return_pct,
    }
