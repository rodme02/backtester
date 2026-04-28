"""Cerebro wiring + single-run backtest entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import backtrader as bt
import pandas as pd

from ..reporting.metrics import equity_curve_from_strategy, summarize


@dataclass
class BacktestResult:
    final_value: float
    metrics: dict[str, Any]
    equity_curve: pd.Series
    trades: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


def _add_analyzers(cerebro: bt.Cerebro) -> None:
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn",
                        timeframe=bt.TimeFrame.Days)


def run_backtest(
    strategy_cls: type[bt.Strategy],
    data: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
) -> BacktestResult:
    """Run a single backtest and return aggregated results."""
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=data))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage, slip_open=True, slip_limit=True,
                                     slip_match=True, slip_out=True)
    cerebro.addstrategy(strategy_cls, **(params or {}))
    _add_analyzers(cerebro)

    strat = cerebro.run()[0]
    final_value = cerebro.broker.getvalue()
    curve = equity_curve_from_strategy(strat, starting_cash=cash)
    summary = summarize(strat, starting_cash=cash, final_value=final_value)
    trades_info = strat.analyzers.trades.get_analysis()
    return BacktestResult(
        final_value=final_value,
        metrics=summary,
        equity_curve=curve,
        trades=_flatten_trades(trades_info),
        params=dict(strat.params._getkwargs()),
    )


def _flatten_trades(info: Any) -> list[dict[str, Any]]:
    total = getattr(info.total, "closed", 0) if hasattr(info, "total") else 0
    won = getattr(info.won, "total", 0) if hasattr(info, "won") else 0
    lost = getattr(info.lost, "total", 0) if hasattr(info, "lost") else 0
    return [{"closed": total, "won": won, "lost": lost}]
