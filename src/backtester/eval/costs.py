"""Trading-cost models applied to a position/return series.

A ``CostModel`` charges three components per turnover-equivalent unit:

- ``commission_bps``  — exchange fees + broker commission, basis points
  on notional traded.
- ``half_spread_bps`` — paying half the bid-ask spread on each fill.
- ``impact_coef``     — square-root market-impact term;
  ``impact_coef * sqrt(participation)`` bps. Set to 0 to disable.

``apply_costs(returns, positions, model, ...)`` returns the cost-
adjusted return series. Costs are charged on the **change** in
position from one period to the next (turnover), not on the position
itself.

Default profiles roughly match real-world friction:

- ``EQUITIES_LIQUID``  — US blue-chip retail-broker reality (Alpaca-ish).
- ``CRYPTO_PERP``      — Binance USDT perpetual taker fees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    commission_bps: float
    half_spread_bps: float
    impact_coef: float = 0.0

    @property
    def per_turnover_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps


EQUITIES_LIQUID = CostModel(commission_bps=0.5, half_spread_bps=1.0, impact_coef=0.0)
CRYPTO_PERP = CostModel(commission_bps=4.0, half_spread_bps=2.0, impact_coef=0.0)


def apply_costs(
    returns: pd.Series | np.ndarray,
    positions: pd.Series | np.ndarray,
    model: CostModel,
    *,
    participation: float | pd.Series | None = None,
) -> pd.Series:
    """Apply costs to a strategy-return series.

    Costs are charged each period proportional to ``|Δ position|``.

    Parameters
    ----------
    returns
        Per-period strategy return *before* costs (e.g. position * asset_return).
    positions
        Position weight at each period. Same length as ``returns``.
    model
        Cost model with bps charges per unit turnover.
    participation
        Optional participation rate (fraction of period volume the
        trade represents). Drives the sqrt market-impact term. Scalar
        or per-period series. Ignored when ``model.impact_coef`` is 0.
    """
    r = pd.Series(np.asarray(returns, dtype=float))
    pos = pd.Series(np.asarray(positions, dtype=float))
    if r.size != pos.size:
        raise ValueError("returns and positions must have the same length")

    turnover = pos.diff().abs().fillna(pos.iloc[0])
    cost_bps = pd.Series(model.per_turnover_bps, index=turnover.index)

    if model.impact_coef:
        if participation is None:
            raise ValueError("impact_coef > 0 requires participation")
        part = pd.Series(participation, index=turnover.index, dtype=float)
        cost_bps = cost_bps + model.impact_coef * np.sqrt(part.clip(lower=0.0))

    cost = turnover * cost_bps * BPS
    return r - cost
