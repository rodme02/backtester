# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project actually is

An **in-depth empirical study of ML techniques applied to US equities and crypto**, framed as an honest evaluation: every major model family — linear, tree ensembles, sequence models, LLM-derived sentiment, classical factor — runs through the same purged-walk-forward / deflated-Sharpe / realistic-cost harness, with strong pragmatic discussion of every result.

The point is the survey + the rigour, not any individual signal. Most signals fail under honest evaluation (consistent with 50+ years of empirical-finance literature); the project includes a **positive-control case** (Jegadeesh-Titman 12-1 momentum) to prove the harness can identify real edge when it exists.

When in doubt about whether to add a feature: **does it make the evaluation more honest, or the survey more comprehensive?** Prefer those over feature creep that doesn't serve the study.

## Commands

```bash
pip install -e ".[dev,notebooks,ml,llm]"
pytest                                       # all (>= 60 tests)
pytest tests/test_eval_walkforward.py        # one file
pytest -k deflated                           # by keyword

ruff check .
ruff check . --fix

# Run a notebook end-to-end (uses live APIs unless BACKTESTER_FIXTURE_MODE=1)
jupyter nbconvert --to notebook --execute notebooks/01_gbm_us_equities.ipynb

make test           # convenience target
make notebooks      # execute all notebooks
make all            # full clean run
```

Env vars (loaded automatically from `.env` via `python-dotenv` per call, so `.env` edits don't require kernel restart):
- `FRED_API_KEY` — macro features (VIX, yield-curve slope, credit spreads).
- `GROQ_API_KEY` — free LLM API for the sentiment case study (Llama 3.3 70B).
- `OLLAMA_HOST` — optional, for local-LLM fallback (Apple Silicon Macs benefit from MPS).
- `BACKTESTER_FIXTURE_MODE=1` — substitute bundled `samples/` snapshots for live API calls (used by CI).
- Yahoo Finance and Binance public endpoints need no key.

## Architecture

Three layers, all operating on plain pandas/numpy. No bespoke event loop; the bar-by-bar simulator is intentionally absent (overkill for daily ML signals).

### 1. The honest-eval harness (the project's credibility layer)

`src/backtester/eval/` — operates on plain `pandas.Series` of returns / positions. **Never bake market-specific assumptions in here.**

- `walkforward.walk_forward_splits(n, n_splits, label_horizon, embargo, min_train_size)` — yields `(train_idx, test_idx)` pairs with **purge** (rows whose label window leaks into test) and **embargo** (rows after each test fold are excluded from subsequent training folds). Implements Lopez de Prado AFML §7.4. Tests assert: monotonic train→test ordering, no label-window leakage, embargo rows excluded from any subsequent training fold.
- `statistics.{annualised_sharpe, probabilistic_sharpe_ratio, deflated_sharpe_ratio, dsr_sensitivity, bootstrap_ci, holm_correct}` — pure functions on numpy. DSR uses the Gumbel approximation for the expected max SR across `n_trials`; `trials_sr_var` is the cross-trial Sharpe variance assumption (1.0 = worst case / independent trials, 0.25–0.5 = realistic / correlated trials, 0.1 = strong-overlap regime). **Always report DSR sensitivity across the var grid** — fixing var=1.0 alone reads as p-hacked pessimism.
- `costs.{CostModel, apply_costs, EQUITIES_LIQUID, CRYPTO_PERP}` — costs charged per period proportional to `|Δ position|`. Initial entry counts as turnover. Profiles approximate retail-broker reality.
- `regimes.{trend_regimes, vol_regimes, per_regime_metrics}` — leakage-free regime taggers (200d SMA / expanding-quantile of rolling vol).

**Invariant:** every public eval function takes a numpy/pandas series of returns and is asset-class agnostic.

### 2. The ML modelling layer

`src/backtester/{features,models,portfolio}/`:

- `features/{technical,macro,cross_sectional,crypto,sentiment}.py` — leakage-free feature builders. **Causality invariant:** `f(series[:t+1])[t] == f(series)[t]` — enforced by `tests/test_features_leakage.py` with random `t/k` sampling.
- `models/{linear,random_forest,gbm,sequence}.py` — uniform `fit(X, y) -> self` / `predict_proba(X) -> ndarray` interface so the eval harness drives any model identically. `sequence.py` contains `LSTMClassifier`, `TCNClassifier`, `TransformerClassifier`, all CPU-only (PyTorch via the `[ml]` extra). Adding a model = subclass the interface, drop in `models/__init__.py`'s registry.
- `portfolio/cross_sectional.py` — case-study notebooks consume model probability scores and run them through `long_short_quantile_weights → daily_returns_from_book → apply_book_costs` to produce a daily-return Series fed directly into `eval/`.

### 3. Data layer

`src/backtester/data/`:

- `csv_loader.py` — bundled-sample loader.
- `yfinance.py`, `fred.py`, `binance.py`, `news.py` — live cached pulls. Each respects `BACKTESTER_FIXTURE_MODE` for CI.
- `universe.py` (+ `samples/universe_us_liquid.csv`) — point-in-time-aware ticker eligibility. Use this whenever building cross-sectional features on equities to avoid survivorship bias.
- `llm.py` — Groq client + Ollama fallback + per-headline cache. Returns `[-1, +1]` sentiment scores given (ticker, headline).

All loaders return either a DataFrame indexed by `datetime` with lower-cased OHLCV columns, or a pandas Series indexed by `datetime`. Every loader has its own date-keyed disk cache under `data_cache/<source>/` (gitignored). Samples in `samples/` are committed.

## Conventions

- Public functions/classes get type hints; tests do not.
- `ruff` config (in `pyproject.toml`) enforces import sorting + select lints. CI fails on violations.
- New deps go in `pyproject.toml` with version pins/ranges, never bare `pip install`. Optional extras: `[dev]`, `[notebooks]`, `[ml]` (torch), `[llm]` (groq + feedparser).
- New eval functionality: write the test first, hit a known closed-form or degenerate case for the assertion. The eval layer's credibility is the project's credibility.
- Every case-study notebook follows the same template: hypothesis → data setup → feature construction → walk-forward training → portfolio → DSR sensitivity + bootstrap CI + per-regime breakdown → "if this were production" discussion.
- New case studies live in `notebooks/NN_*.ipynb`, paired with a section in `docs/writeup.md`.
