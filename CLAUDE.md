# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Editable install with dev tools
pip install -e ".[dev]"

# Lint
ruff check .
ruff check . --fix

# Tests
pytest                                    # all
pytest tests/test_engine.py               # one file
pytest tests/test_engine.py::test_run_backtest_produces_metrics  # one test
pytest -k metrics                         # by keyword

# CLI
backtester run --strategy ma_crossover --symbol AAPL
backtester run --strategy advanced_trend --symbol SPY --source samples

# Dashboard
streamlit run dashboard/app.py
```

`ALPHA_VANTAGE_API_KEY` must be set in the environment (or `.env`) only when `--source alphavantage` is used. Bundled samples cover AAPL, AMZN, MSFT, SPY, TSLA and require no network.

## Architecture

The codebase is a thin, opinionated wrapper around **backtrader**. Code under `src/backtester/` is organised by responsibility, not by strategy — adding a new strategy should not require touching the engine.

- `data/` — `csv_loader.py` (bundled samples + arbitrary CSVs) and `alpha_vantage.py` (live fetcher with on-disk cache under `data_cache/alpha_vantage/`, keyed by symbol + date so repeated runs don't burn API quota). All loaders return a `pandas.DataFrame` indexed by `datetime` with `open/high/low/close/volume` columns — that's the contract every other layer assumes.
- `strategies/` — each strategy is a `bt.Strategy` subclass; `__init__.py` exposes a `REGISTRY` dict. The CLI and dashboard both look strategies up by name through this registry, so adding a new one means: subclass, import in `__init__.py`, add to `REGISTRY`. No engine changes.
- `engine/` — `runner.run_backtest(strategy_cls, df, params=..., cash=..., commission=..., slippage=...) -> BacktestResult` is the single execution path. It wires Cerebro, attaches the four standard analyzers (`SharpeRatio`, `DrawDown`, `TradeAnalyzer`, `TimeReturn`), runs once, and hands the strategy instance to `reporting.metrics` for summarisation. `optimizer.optimize` is a deterministic cartesian-product sweep that calls `run_backtest` per combination.
- `reporting/metrics.py` — converts backtrader analyzer output into plain dicts/Series. **The dashboard's equity-curve chart depends on the `TimeReturn` analyzer being attached in `runner._add_analyzers`.** If you add or remove analyzers, update both files.
- `cli.py` — registered as the `backtester` console script via `pyproject.toml` `[project.scripts]`.

`dashboard/app.py` is a Streamlit front-end over the same engine. It manipulates `sys.path` so the app runs without `pip install -e .` first, and uses `st.cache_data` keyed on `(strategy, symbol, params, cash)` to avoid re-running identical backtests.

`samples/ohlcv/*.csv` are committed; `data_cache/` is gitignored. Tests use the samples — they are part of the public test contract, don't rename or remove without updating fixtures.

## Conventions

- Public functions/classes get type hints; tests do not.
- `ruff` config (in `pyproject.toml`) enforces import sorting and basic lints; CI fails on violations.
- New dependencies go in `pyproject.toml` with version pins/ranges, never as a bare `pip install`.
