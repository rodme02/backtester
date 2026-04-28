# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project actually is

Not a generic backtesting framework. This is a **research artifact**: an honest evaluation of ML-driven trading signals across US equities and crypto. The pitch is "most retail backtests lie; here's a worked example with the rigour they skip." The backtrader engine and Streamlit dashboard exist but are deliberately demoted in framing — the headline is the eval harness, the case-study notebooks (week 2+), and the long-form writeup at `docs/writeup.md` (week 4).

When in doubt about whether to add a feature: **does it make the evaluation more honest, or just more impressive-looking?** Prefer the former.

## Commands

```bash
pip install -e ".[dev]"
pytest                                       # all
pytest tests/test_eval_walkforward.py        # one file
pytest -k deflated                           # by keyword

ruff check .
ruff check . --fix

backtester run --strategy ma_crossover --symbol AAPL  # legacy backtrader baseline
streamlit run dashboard/app.py                         # legacy dashboard
```

Env vars (loaded automatically from `.env` via `python-dotenv`):
- `ALPHA_VANTAGE_API_KEY` — only for `--source alphavantage` runs.
- `FRED_API_KEY` — for macro features (VIX, yield-curve slope, credit spreads).
- Yahoo Finance and Binance public endpoints need no key.

## Architecture

Two parallel layers, intentionally not yet unified:

### 1. The honest-eval harness (the project's centre of gravity)

`src/backtester/eval/` — operates on plain `pandas.Series` of returns / positions.

- `walkforward.walk_forward_splits(n, n_splits, label_horizon, embargo, min_train_size)` — yields `(train_idx, test_idx)` pairs with **purge** (rows whose label window leaks into the test fold are removed) and **embargo** (buffer rows after each test fold). Implements Lopez de Prado AFML §7.4. The label-leak invariant is asserted in `tests/test_eval_walkforward.py::test_purge_removes_label_horizon_rows` — keep that green.
- `statistics.{annualised_sharpe, probabilistic_sharpe_ratio, deflated_sharpe_ratio, bootstrap_ci, holm_correct}` — pure functions on a numpy array of returns. DSR uses the Gumbel approximation for the expected max SR across `n_trials`; default `trials_sr_var=1.0` is conservative — pass the empirical cross-trial Sharpe variance when comparing many models. PSR/DSR can be NaN for n<4 or degenerate variance; callers should handle.
- `costs.{CostModel, apply_costs, EQUITIES_LIQUID, CRYPTO_PERP}` — costs charged per period proportional to `|Δ position|` (turnover). Initial entry counts as turnover too. Profile constants approximate retail-broker reality, not institutional fees.
- `regimes.{trend_regimes, vol_regimes, per_regime_metrics}` — both regime taggers use only past data (200d SMA / expanding-quantile of rolling vol) so they are leakage-free.

**Invariant:** every public eval function takes a numpy/pandas series of returns and is asset-class agnostic. Don't bake market-specific assumptions into eval/.

### 2. The legacy backtrader simulator (baseline-strategy playground)

`src/backtester/{strategies,engine,reporting}/` — the original backtrader wrapper. Use `engine.run_backtest(strategy_cls, df, ...)` to get a `BacktestResult` containing `equity_curve` (TimeReturn-based pandas Series) and `metrics` (sharpe / max_drawdown_pct / etc.). The Streamlit dashboard at `dashboard/app.py` consumes this. Adding a new bt.Strategy means: subclass, import in `strategies/__init__.py`, add to `REGISTRY`.

When the case-study notebooks land they will produce per-period strategy-return Series and feed them into the eval/ harness directly — bypassing the backtrader engine — because backtrader's bar-by-bar simulation is overkill for daily ML signals. The two layers are kept separate intentionally.

### Data layer (`src/backtester/data/`)

All loaders return either:
- A DataFrame indexed by `datetime` with lower-cased `open/high/low/close/volume` columns, OR
- A pandas Series indexed by `datetime` (FRED scalars).

Every loader has its own date-keyed disk cache under `data_cache/<source>/`. The cache directory is gitignored; samples in `samples/` are committed.

`data/universe.py` + `samples/universe_us_liquid.csv` give point-in-time-aware ticker eligibility — use it whenever building cross-sectional features for equities to avoid survivorship bias.

## Conventions

- Public functions/classes get type hints; tests do not.
- `ruff` config (in `pyproject.toml`) enforces import sorting + select lints. CI fails on violations.
- New deps go in `pyproject.toml` with version pins/ranges, never bare `pip install`.
- New eval functionality: write the test first, hit a known closed-form or degenerate case for the assertion. The eval layer's credibility is the project's credibility.
