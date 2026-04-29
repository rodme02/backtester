import numpy as np
import pytest

from backtester.eval.statistics import (
    annualised_sharpe,
    bootstrap_ci,
    deflated_sharpe_ratio,
    holm_correct,
    probabilistic_sharpe_ratio,
)


def _normal_returns(n: int, mu: float, sigma: float, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=n)


def test_annualised_sharpe_matches_definition():
    r = _normal_returns(2_000, mu=0.001, sigma=0.01)
    sr = annualised_sharpe(r)
    expected = (r.mean() / r.std(ddof=1)) * np.sqrt(252)
    assert sr == pytest.approx(expected, rel=1e-9)


def test_annualised_sharpe_zero_volatility_returns_zero():
    assert annualised_sharpe(np.zeros(50)) == 0.0


def test_psr_high_for_clearly_positive_strategy():
    r = _normal_returns(1_000, mu=0.001, sigma=0.005, seed=1)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert 0.99 < psr <= 1.0


def test_psr_low_for_zero_mean_strategy():
    r = _normal_returns(1_000, mu=0.0, sigma=0.01, seed=2)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.5)
    assert psr < 0.2


def test_dsr_more_conservative_than_psr_against_zero():
    r = _normal_returns(1_000, mu=0.0008, sigma=0.01, seed=3)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    dsr = deflated_sharpe_ratio(r, n_trials=100, trials_sr_var=0.5)
    assert dsr <= psr


def test_dsr_increases_with_more_data():
    # A clearly winning strategy: mu/sigma annualises to SR ~ 3.17,
    # well above any reasonable deflated benchmark. With more samples the
    # standard error shrinks, so DSR should rise.
    rng = np.random.default_rng(4)
    short_r = rng.normal(0.002, 0.01, size=300)
    long_r = rng.normal(0.002, 0.01, size=3000)
    short = deflated_sharpe_ratio(short_r, n_trials=20, trials_sr_var=0.5)
    long = deflated_sharpe_ratio(long_r, n_trials=20, trials_sr_var=0.5)
    assert long > short


def test_bootstrap_ci_brackets_truth_for_mean():
    r = _normal_returns(500, mu=0.0005, sigma=0.01, seed=5)
    res = bootstrap_ci(r, statistic=lambda x: float(x.mean()), n_resamples=500)
    assert res.lower < res.point < res.upper
    assert res.lower < 0.0005 < res.upper


def test_bootstrap_ci_block_resample_runs():
    r = _normal_returns(400, mu=0.0, sigma=0.01, seed=6)
    res = bootstrap_ci(r, statistic=annualised_sharpe, n_resamples=200, block_size=20)
    assert res.samples.shape == (200,)
    assert res.lower <= res.point <= res.upper


def test_bootstrap_ci_method_choices_run():
    r = _normal_returns(400, mu=0.0, sigma=0.01, seed=7)
    for method in ("stationary", "fixed", "iid"):
        res = bootstrap_ci(
            r, statistic=annualised_sharpe, n_resamples=100,
            block_size=20, method=method,
        )
        assert res.samples.shape == (100,)
        assert res.lower <= res.point <= res.upper


def test_bootstrap_ci_invalid_method_raises():
    import pytest
    r = _normal_returns(50, mu=0.0, sigma=0.01)
    with pytest.raises(ValueError):
        bootstrap_ci(r, statistic=annualised_sharpe, method="bogus")


def test_dsr_sensitivity_decreases_with_var():
    """At fixed n_trials, DSR should be monotone non-increasing in
    trials_sr_var (more variance → harder benchmark)."""
    from backtester.eval.statistics import dsr_sensitivity

    rng = np.random.default_rng(11)
    r = rng.normal(0.001, 0.01, size=2000)
    out = dsr_sensitivity(r, n_trials=20, var_grid=(0.1, 0.25, 0.5, 1.0))
    values = [out[v] for v in (0.1, 0.25, 0.5, 1.0)]
    # Non-increasing within a small tolerance for sampling noise.
    assert all(values[i] >= values[i + 1] - 1e-6 for i in range(len(values) - 1))


def test_holm_rejects_only_smallest_when_appropriate():
    # 4 p-values: only the smallest should clearly survive Holm at alpha=0.05.
    decisions = holm_correct([0.001, 0.04, 0.5, 0.9], alpha=0.05)
    assert decisions == [True, False, False, False]


def test_holm_handles_empty():
    assert holm_correct([]) == []
