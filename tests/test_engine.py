from backtester.data import load_samples
from backtester.engine import run_backtest
from backtester.strategies import MaCrossover


def test_run_backtest_produces_metrics():
    data = load_samples("AAPL").iloc[:500]
    result = run_backtest(MaCrossover, data, cash=100_000.0)
    assert result.final_value > 0
    assert "sharpe" in result.metrics
    assert "max_drawdown_pct" in result.metrics
    assert "total_trades" in result.metrics
    assert result.metrics["total_return_pct"] is not None
