"""Trading-cost models applied to a position/return series.

A ``CostModel`` charges per-turnover (commission + half-spread +
sqrt-impact) and optionally a per-day *holding* cost. Holding costs
matter for two cases:

- **Equity short borrow.** Brokers charge an annualised rate (5–25 bps
  for liquid names) on the absolute short notional, accruing daily.
- **Crypto perp funding.** Longs pay positive funding to shorts (and
  vice versa) every 8h. We model this as a holding cost where the
  per-period rate is the funding-rate series itself, applied to the
  signed position (so positive-funding longs pay, shorts receive).

``apply_costs(returns, positions, model, funding_rate=None,
borrow_rate_bps_year=None, ...)`` returns the cost-adjusted return
series.

Default profiles roughly match real-world friction:

- ``EQUITIES_LIQUID``  — US blue-chip retail-broker; trade only.
- ``EQUITIES_LIQUID_WITH_BORROW`` — same + 5 bps/yr short borrow.
- ``CRYPTO_PERP``      — Binance USDT perpetual taker fees; trade only.
- ``CRYPTO_PERP_WITH_FUNDING`` — same; ``apply_costs`` will pull in a
  funding-rate series at call time.
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
    borrow_bps_year: float = 0.0
    """Annualised short-borrow rate in bps. Charged daily on |position|
    when position < 0. 0 → disabled."""
    fund_with_funding_rate: bool = False
    """When True, ``apply_costs`` expects a ``funding_rate`` series and
    charges/credits each period: long pays positive funding, short
    receives. Independent of ``commission_bps`` / ``half_spread_bps``
    which still apply to turnover."""

    @property
    def per_turnover_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps


EQUITIES_LIQUID = CostModel(commission_bps=0.5, half_spread_bps=1.0)
EQUITIES_LIQUID_WITH_BORROW = CostModel(
    commission_bps=0.5, half_spread_bps=1.0, borrow_bps_year=5.0
)
CRYPTO_PERP = CostModel(commission_bps=4.0, half_spread_bps=2.0)
CRYPTO_PERP_WITH_FUNDING = CostModel(
    commission_bps=4.0, half_spread_bps=2.0, fund_with_funding_rate=True
)


def apply_costs(
    returns: pd.Series | np.ndarray,
    positions: pd.Series | np.ndarray,
    model: CostModel,
    *,
    participation: float | pd.Series | None = None,
    funding_rate: pd.Series | None = None,
    periods_per_year: int = 252,
) -> pd.Series:
    """Apply commission + spread + impact + borrow + funding costs.

    Parameters
    ----------
    returns
        Per-period strategy return *before* costs (e.g.
        ``position * asset_return``).
    positions
        Signed position weight at each period. Same length as
        ``returns``.
    model
        Cost model.
    participation
        Optional participation rate (fraction of period volume the
        trade represents). Drives the sqrt market-impact term.
    funding_rate
        Per-period funding rate to charge (long pays positive funding,
        short receives) when ``model.fund_with_funding_rate`` is True.
    periods_per_year
        Used to convert ``borrow_bps_year`` to a per-period rate.
    """
    r = pd.Series(np.asarray(returns, dtype=float)).reset_index(drop=True)
    pos = pd.Series(np.asarray(positions, dtype=float)).reset_index(drop=True)
    if r.size != pos.size:
        raise ValueError("returns and positions must have the same length")

    turnover = pos.diff().abs().fillna(pos.iloc[0].__abs__())
    cost_bps = pd.Series(model.per_turnover_bps, index=turnover.index)

    if model.impact_coef:
        if participation is None:
            raise ValueError("impact_coef > 0 requires participation")
        part = pd.Series(participation).reset_index(drop=True).astype(float)
        cost_bps = cost_bps + model.impact_coef * np.sqrt(part.clip(lower=0.0))

    cost = turnover * cost_bps * BPS

    if model.borrow_bps_year > 0:
        per_period = (model.borrow_bps_year / periods_per_year) * BPS
        short_notional = (-pos).clip(lower=0.0)
        cost = cost + short_notional * per_period

    if model.fund_with_funding_rate:
        if funding_rate is None:
            raise ValueError(
                "fund_with_funding_rate=True requires a funding_rate series"
            )
        fr = pd.Series(np.asarray(funding_rate, dtype=float)).reset_index(drop=True)
        if fr.size != r.size:
            raise ValueError(
                "funding_rate must have same length as returns/positions"
            )
        # Long pays positive funding (cost = +pos * funding); short
        # receives (cost = -|pos| * funding when pos < 0 and funding > 0
        # → negative cost = revenue). Both cases captured by pos * fr.
        cost = cost + pos * fr

    return r - cost
