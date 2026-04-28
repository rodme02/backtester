"""Macro features built from FRED series.

FRED daily series are *as-of-close* values. To avoid same-day leakage
into a model that predicts next-day returns, every macro feature is
**lagged by one trading day** before being aligned to the equity
calendar. ``align_macro`` does the lag + reindex + forward-fill.
"""

from __future__ import annotations

import pandas as pd


def align_macro(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Lag a daily macro series by one period, then align to ``calendar``.

    The one-period lag prevents using same-day macro prints to predict
    same-day returns. Forward-filling inside the calendar covers
    weekends and FRED publication gaps.
    """
    return series.shift(1).reindex(calendar, method="ffill")


def macro_features(
    fred_series: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    *,
    change_window: int = 5,
) -> pd.DataFrame:
    """Build a macro-feature DataFrame indexed by ``calendar``.

    For each FRED series, two columns are produced:
    ``{name}_level`` (lagged-and-aligned) and
    ``{name}_change`` (level minus its ``change_window``-day lag).
    """
    out = {}
    for name, series in fred_series.items():
        aligned = align_macro(series, calendar)
        out[f"{name}_level"] = aligned
        out[f"{name}_change"] = aligned - aligned.shift(change_window)
    return pd.DataFrame(out, index=calendar)
