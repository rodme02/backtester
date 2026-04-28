from backtester.data import load_samples


def test_load_samples_returns_ohlcv():
    df = load_samples("AAPL")
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert len(df) > 100
    assert df.index.is_monotonic_increasing
