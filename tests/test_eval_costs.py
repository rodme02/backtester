import numpy as np
import pandas as pd
import pytest

from backtester.eval.costs import (
    BPS,
    CRYPTO_PERP,
    CRYPTO_PERP_WITH_FUNDING,
    EQUITIES_LIQUID,
    EQUITIES_LIQUID_WITH_BORROW,
    CostModel,
    apply_costs,
)


def test_no_turnover_no_cost():
    pos = pd.Series([0.0] * 10)
    ret = pd.Series([0.001] * 10)
    out = apply_costs(ret, pos, EQUITIES_LIQUID)
    pd.testing.assert_series_equal(out, ret)


def test_constant_position_only_initial_cost():
    pos = pd.Series([1.0] * 10)
    ret = pd.Series([0.0] * 10)
    out = apply_costs(ret, pos, EQUITIES_LIQUID)
    expected_first = -EQUITIES_LIQUID.per_turnover_bps * BPS
    assert out.iloc[0] == pytest.approx(expected_first)
    assert (out.iloc[1:] == 0.0).all()


def test_flip_charges_double_turnover():
    pos = pd.Series([1.0, -1.0, 1.0])
    ret = pd.Series([0.0, 0.0, 0.0])
    out = apply_costs(ret, pos, EQUITIES_LIQUID)
    # Initial entry: 1 unit. Flip: 2 units. Flip back: 2 units.
    expected = -np.array([1.0, 2.0, 2.0]) * EQUITIES_LIQUID.per_turnover_bps * BPS
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_crypto_costs_higher_than_equities():
    assert CRYPTO_PERP.per_turnover_bps > EQUITIES_LIQUID.per_turnover_bps


def test_impact_requires_participation():
    model = CostModel(commission_bps=1.0, half_spread_bps=1.0, impact_coef=10.0)
    with pytest.raises(ValueError):
        apply_costs(pd.Series([0.0]), pd.Series([1.0]), model)


def test_impact_increases_cost():
    base = CostModel(commission_bps=1.0, half_spread_bps=1.0, impact_coef=0.0)
    impact = CostModel(commission_bps=1.0, half_spread_bps=1.0, impact_coef=20.0)
    pos = pd.Series([1.0, 1.0, 1.0])
    ret = pd.Series([0.0, 0.0, 0.0])
    base_out = apply_costs(ret, pos, base).sum()
    impact_out = apply_costs(ret, pos, impact, participation=0.05).sum()
    assert impact_out < base_out


# --- borrow ---


def test_borrow_charges_only_shorts():
    """Long-only book should be borrow-free; short positions accrue
    daily borrow on |pos|."""
    long_pos = pd.Series([1.0] * 10)
    short_pos = pd.Series([-1.0] * 10)
    ret = pd.Series([0.0] * 10)

    long_out = apply_costs(ret, long_pos, EQUITIES_LIQUID_WITH_BORROW)
    short_out = apply_costs(ret, short_pos, EQUITIES_LIQUID_WITH_BORROW)

    # Long: only initial turnover.
    expected_long_first = -EQUITIES_LIQUID_WITH_BORROW.per_turnover_bps * BPS
    assert long_out.iloc[0] == pytest.approx(expected_long_first)
    assert (long_out.iloc[1:] == 0.0).all()

    # Short: turnover on day 0, then 9 days of daily borrow.
    daily_borrow = (5.0 / 252) * BPS  # bps_year / periods_per_year * BPS
    np.testing.assert_allclose(
        short_out.iloc[1:].to_numpy(), -daily_borrow, atol=1e-12
    )


def test_borrow_zero_when_disabled():
    """Default EQUITIES_LIQUID has borrow=0, so shorts cost the same
    as longs (only commission + spread)."""
    short_pos = pd.Series([-1.0] * 5)
    ret = pd.Series([0.0] * 5)
    out = apply_costs(ret, short_pos, EQUITIES_LIQUID)
    assert (out.iloc[1:] == 0.0).all()


# --- funding ---


def test_funding_long_pays_short_receives():
    """When funding is positive, long pays (negative cost-adjusted
    return), short receives (positive cost-adjusted return). Symmetry:
    long_cost == -short_cost on the same funding."""
    funding = pd.Series([0.0001] * 5)  # 1 bp/period
    long_pos = pd.Series([1.0] * 5)
    short_pos = pd.Series([-1.0] * 5)
    ret = pd.Series([0.0] * 5)

    long_out = apply_costs(
        ret, long_pos, CRYPTO_PERP_WITH_FUNDING, funding_rate=funding
    )
    short_out = apply_costs(
        ret, short_pos, CRYPTO_PERP_WITH_FUNDING, funding_rate=funding
    )
    # Subtract the symmetric turnover cost (initial entry only) and
    # check the funding-only difference is symmetric.
    long_funding = long_out.iloc[1:] - 0.0
    short_funding = short_out.iloc[1:] - 0.0
    np.testing.assert_allclose(
        long_funding.to_numpy(), -short_funding.to_numpy(), atol=1e-12
    )


def test_funding_zero_equivalent_to_base():
    """Zero-funding series should give the same result as the
    funding-free profile (within floating tolerance)."""
    funding = pd.Series([0.0] * 5)
    pos = pd.Series([1.0] * 5)
    ret = pd.Series([0.0] * 5)

    base = apply_costs(ret, pos, CRYPTO_PERP)
    with_funding = apply_costs(
        ret, pos, CRYPTO_PERP_WITH_FUNDING, funding_rate=funding
    )
    np.testing.assert_allclose(
        base.to_numpy(), with_funding.to_numpy(), atol=1e-12
    )


def test_funding_required_when_enabled():
    pos = pd.Series([1.0] * 3)
    ret = pd.Series([0.0] * 3)
    with pytest.raises(ValueError):
        apply_costs(ret, pos, CRYPTO_PERP_WITH_FUNDING)
