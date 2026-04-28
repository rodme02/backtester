from types import SimpleNamespace

from backtester.reporting.metrics import summarize


class _Analyzer:
    def __init__(self, data):
        self._data = data

    def get_analysis(self):
        return self._data


def test_summarize_handles_no_trades():
    fake = SimpleNamespace(
        analyzers=SimpleNamespace(
            sharpe=_Analyzer({"sharperatio": None}),
            drawdown=_Analyzer({"max": {"drawdown": 0.0}}),
            trades=_Analyzer(SimpleNamespace(
                total=SimpleNamespace(closed=0),
                won=SimpleNamespace(total=0),
                lost=SimpleNamespace(total=0),
            )),
        )
    )
    summary = summarize(fake, starting_cash=100_000, final_value=100_000)
    assert summary["total_trades"] == 0
    assert summary["win_rate_pct"] == 0.0
    assert summary["total_return_pct"] == 0.0
