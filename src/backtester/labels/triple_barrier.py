"""Triple-barrier labelling and sample uniqueness weighting.

Triple-barrier labels (AFML §3.2-3.4)
-------------------------------------
For each event time ``t0``, define three exit conditions:

1. **Profit-take** at ``+pt_mult × σ_t`` cumulative log-return.
2. **Stop-loss**   at ``−sl_mult × σ_t`` cumulative log-return.
3. **Vertical**    at ``t1 = t0 + max_holding`` (calendar or trading bars).

The label is the sign of the realised log-return at the *first* barrier
touched. This is closer to a real trader's exit pattern than the binary
"sign of next-bar return" used in naive ML setups.

Sample uniqueness weights (AFML §4.4-4.5)
-----------------------------------------
With horizon > 1 bar, neighbouring labels overlap (share information),
which double-counts. The fix: weight each label inversely proportional
to the average concurrency over its own ``[t0, t1]`` window. We then
normalise so weights sum to N (preserving sklearn's interpretation of
``sample_weight``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_events(
    close: pd.Series,
    *,
    t0_index: pd.DatetimeIndex,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 5,
    vol: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute triple-barrier exit times and realised returns.

    Parameters
    ----------
    close
        Price series indexed by trading-day timestamps. Log-returns
        are computed against ``close.loc[t0]``.
    t0_index
        Timestamps at which a position is hypothetically opened. Must
        be a subset of ``close.index``.
    pt_mult, sl_mult
        Profit-take / stop-loss multiples on ``vol``.
    max_holding
        Vertical-barrier length in bars (rows of ``close``).
    vol
        Per-bar volatility used to scale the barriers. If ``None``,
        a 20-bar EWMA std of log-returns is used.
    side
        Optional per-event ``+1`` (long) / ``-1`` (short) bias. When
        provided, profit-take and stop-loss are flipped for shorts.
        Defaults to long-only (``+1`` everywhere).

    Returns
    -------
    DataFrame indexed by ``t0_index`` with columns:

    - ``t1``        : actual exit time (first barrier touch).
    - ``ret``       : realised log-return at ``t1``.
    - ``bin``       : ``+1``, ``-1``, or ``0`` (vertical without
      crossing a horizontal barrier and no clear sign).
    - ``barrier``   : which barrier was hit ('pt', 'sl', or 'vert').
    """
    if vol is None:
        log_ret = np.log(close).diff()
        vol = log_ret.ewm(span=20, min_periods=20).std()

    side_series = pd.Series(1.0, index=t0_index) if side is None else side.reindex(t0_index).fillna(1.0)

    out = pd.DataFrame(index=t0_index, columns=["t1", "ret", "bin", "barrier"], dtype=object)
    close_log = np.log(close)

    for t0 in t0_index:
        if t0 not in close.index:
            continue
        pos = close.index.get_loc(t0)
        end_pos = min(pos + max_holding, len(close) - 1)
        if end_pos <= pos:
            continue
        path = close.index[pos : end_pos + 1]
        ret_path = (close_log.loc[path] - close_log.loc[t0]) * side_series.loc[t0]

        sigma = vol.loc[t0] if t0 in vol.index else np.nan
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        upper = pt_mult * sigma
        lower = -sl_mult * sigma

        t_pt = ret_path[ret_path >= upper].index.min()
        t_sl = ret_path[ret_path <= lower].index.min()
        t_vert = path[-1]

        candidates = [
            (t_pt, "pt"),
            (t_sl, "sl"),
            (t_vert, "vert"),
        ]
        candidates = [(t, name) for t, name in candidates if pd.notna(t)]
        candidates.sort(key=lambda pair: pair[0])
        t_first, barrier = candidates[0]

        realised = ret_path.loc[t_first]
        if barrier == "pt":
            label = 1
        elif barrier == "sl":
            label = -1
        else:
            label = int(np.sign(realised))  # 0 if exactly zero

        out.loc[t0] = {
            "t1": t_first,
            "ret": float(realised),
            "bin": int(label),
            "barrier": barrier,
        }

    out = out.dropna(subset=["t1"])
    out["t1"] = pd.to_datetime(out["t1"])
    out["ret"] = out["ret"].astype(float)
    out["bin"] = out["bin"].astype(int)
    return out


def triple_barrier_labels(events: pd.DataFrame) -> pd.Series:
    """Convenience: return the ``bin`` column as a labelled Series."""
    return events["bin"].rename("y")


def avg_uniqueness_weights(events: pd.DataFrame) -> pd.Series:
    """Average-uniqueness sample weights (AFML §4.4-4.5).

    For each event ``i`` with span ``[t0_i, t1_i]``, compute the
    concurrency ``c_t = #{j : t ∈ [t0_j, t1_j]}`` and the weight
    ``w_i = mean(1 / c_t over t ∈ [t0_i, t1_i])``. Weights are
    normalised so their sum equals ``len(events)`` (preserving the
    interpretation of an "effective sample size" equal to N when
    weights are uniform).
    """
    if events.empty:
        return pd.Series(dtype=float)

    t0 = events.index
    t1 = pd.DatetimeIndex(events["t1"].values)

    # Build the union of all event timestamps + their t1's so we can
    # tally concurrency on a common index.
    all_ts = pd.DatetimeIndex(sorted(set(t0).union(t1)))
    concurrency = pd.Series(0, index=all_ts, dtype=int)
    for s, e in zip(t0, t1, strict=True):
        # Increment all bars in [s, e].
        concurrency.loc[s:e] += 1

    weights = pd.Series(index=t0, dtype=float)
    for i, (s, e) in enumerate(zip(t0, t1, strict=True)):
        bars = concurrency.loc[s:e]
        weights.iloc[i] = (1.0 / bars).mean() if not bars.empty else 0.0

    n = len(weights)
    total = weights.sum()
    if total > 0:
        weights = weights * (n / total)
    return weights
