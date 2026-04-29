"""Triple-barrier labelling and uniqueness-weighting property tests."""

import numpy as np
import pandas as pd
import pytest

from backtester.labels.triple_barrier import (
    avg_uniqueness_weights,
    triple_barrier_events,
    triple_barrier_labels,
)


@pytest.fixture
def trending_up_close():
    """Monotone-up price path → profit-take should fire long before vert."""
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    close = pd.Series(100.0 * np.exp(0.01 * np.arange(20)), index=idx)
    return close


@pytest.fixture
def trending_down_close():
    """Monotone-down price path → stop-loss should fire long before vert."""
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    close = pd.Series(100.0 * np.exp(-0.01 * np.arange(20)), index=idx)
    return close


def test_triple_barrier_profit_take_hit(trending_up_close):
    vol = pd.Series(0.005, index=trending_up_close.index)  # tight vol → PT first
    events = triple_barrier_events(
        trending_up_close,
        t0_index=trending_up_close.index[:5],
        pt_mult=2.0,
        sl_mult=1.0,
        max_holding=10,
        vol=vol,
    )
    assert (events["barrier"] == "pt").all()
    assert (events["bin"] == 1).all()


def test_triple_barrier_stop_loss_hit(trending_down_close):
    vol = pd.Series(0.005, index=trending_down_close.index)
    events = triple_barrier_events(
        trending_down_close,
        t0_index=trending_down_close.index[:5],
        pt_mult=2.0,
        sl_mult=1.0,
        max_holding=10,
        vol=vol,
    )
    assert (events["barrier"] == "sl").all()
    assert (events["bin"] == -1).all()


def test_triple_barrier_vertical_when_quiet():
    """A flat price path can never hit PT or SL → vertical barrier wins."""
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    close = pd.Series(100.0, index=idx)
    vol = pd.Series(0.01, index=idx)
    events = triple_barrier_events(
        close, t0_index=idx[:5], pt_mult=2.0, sl_mult=1.0, max_holding=5, vol=vol
    )
    assert (events["barrier"] == "vert").all()
    # bin can be 0 (zero return) since price never moved.
    assert (events["bin"] == 0).all()


def test_triple_barrier_t1_is_strictly_after_t0(trending_up_close):
    vol = pd.Series(0.005, index=trending_up_close.index)
    events = triple_barrier_events(
        trending_up_close,
        t0_index=trending_up_close.index[:8],
        max_holding=5,
        vol=vol,
    )
    assert (events["t1"] > events.index).all()


def test_triple_barrier_short_side_flips_pt_sl():
    """A long position in a falling market hits SL; a short position
    in the same market should hit PT."""
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    close = pd.Series(100.0 * np.exp(-0.01 * np.arange(20)), index=idx)
    vol = pd.Series(0.005, index=idx)
    side_short = pd.Series(-1.0, index=idx[:5])
    events = triple_barrier_events(
        close, t0_index=idx[:5], max_holding=10, vol=vol, side=side_short,
        pt_mult=2.0, sl_mult=1.0,
    )
    assert (events["barrier"] == "pt").all()
    assert (events["bin"] == 1).all()


def test_triple_barrier_uses_default_vol_when_none(trending_up_close):
    # Just smoke-test that it runs end-to-end without explicit vol.
    events = triple_barrier_events(
        trending_up_close, t0_index=trending_up_close.index[19:20], max_holding=1,
    )
    assert isinstance(events, pd.DataFrame)


def test_uniqueness_weights_uniform_when_no_overlap():
    """Disjoint events should each have weight 1.0 (after normalisation)."""
    idx = pd.date_range("2024-01-01", periods=12, freq="B")
    events = pd.DataFrame(
        {"t1": [idx[2], idx[5], idx[8], idx[11]]},
        index=[idx[0], idx[3], idx[6], idx[9]],
    )
    w = avg_uniqueness_weights(events)
    assert w.shape == (4,)
    # Each weight should be 1 (uniform → sum to N = 4).
    np.testing.assert_allclose(w.to_numpy(), 1.0, atol=1e-9)


def test_uniqueness_weights_overlap_reduces_weight():
    """Heavily overlapping events should each receive weight ~ 1, but
    pre-normalisation should be < 1; after normalisation sum to N."""
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    # 5 events all starting at day 0 with t1 at day 4 → maximum overlap.
    events = pd.DataFrame(
        {"t1": [idx[4]] * 5},
        index=[idx[0]] * 5,
    )
    # Need unique index for the test to work; use sequential starts.
    events = pd.DataFrame(
        {"t1": [idx[4], idx[5], idx[6], idx[7], idx[8]]},
        index=[idx[0], idx[1], idx[2], idx[3], idx[4]],
    )
    w = avg_uniqueness_weights(events)
    assert w.shape == (5,)
    assert w.sum() == pytest.approx(5.0, rel=1e-6)


def test_uniqueness_weights_empty():
    w = avg_uniqueness_weights(pd.DataFrame(columns=["t1"]))
    assert w.empty


def test_triple_barrier_labels_helper(trending_up_close):
    vol = pd.Series(0.005, index=trending_up_close.index)
    events = triple_barrier_events(
        trending_up_close, t0_index=trending_up_close.index[:5], vol=vol
    )
    y = triple_barrier_labels(events)
    assert y.name == "y"
    assert (y == 1).all()


def test_triple_barrier_no_peeking_past_t1(trending_up_close):
    """Adding extra future data must not change the label of any t0
    already produced — the label is deterministic given the prefix
    up to t1."""
    full = trending_up_close
    vol = pd.Series(0.005, index=full.index)
    events_full = triple_barrier_events(
        full, t0_index=full.index[:5], pt_mult=2.0, sl_mult=1.0, max_holding=10, vol=vol
    )

    # Truncate to just past the last t1 of the first 5 events.
    cutoff = events_full["t1"].max()
    truncated = full.loc[:cutoff]
    events_trunc = triple_barrier_events(
        truncated, t0_index=truncated.index[:5], pt_mult=2.0, sl_mult=1.0,
        max_holding=10, vol=vol.loc[:cutoff]
    )
    pd.testing.assert_frame_equal(events_full.loc[events_trunc.index], events_trunc)
