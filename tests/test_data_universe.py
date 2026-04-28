import pandas as pd

from backtester.data.universe import eligible_tickers, load_universe


def test_load_universe_has_expected_columns():
    df = load_universe()
    assert {"ticker", "first_eligible"}.issubset(df.columns)
    assert df["first_eligible"].notna().all()
    assert df["ticker"].is_unique


def test_eligible_tickers_filters_by_date():
    df = load_universe()
    early = eligible_tickers(df, as_of="2010-01-01")
    later = eligible_tickers(df, as_of="2020-01-01")
    assert set(early).issubset(set(later))
    # Tesla IPO is 2010-06-29: shouldn't be in early.
    assert "TSLA" not in early
    assert "TSLA" in later


def test_eligible_tickers_respects_last_eligible():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "first_eligible": pd.to_datetime(["2010-01-01", "2010-01-01"]),
            "last_eligible": pd.to_datetime(["2015-12-31", pd.NaT]),
        }
    )
    assert "AAA" in eligible_tickers(df, "2014-01-01")
    assert "AAA" not in eligible_tickers(df, "2016-01-01")
    assert "BBB" in eligible_tickers(df, "2030-01-01")
