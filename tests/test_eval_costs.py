import numpy as np
import pandas as pd
import pytest

from backtester.eval.costs import (
    BPS,
    CRYPTO_PERP,
    EQUITIES_LIQUID,
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
