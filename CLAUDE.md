# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project actually is

An **in-depth empirical study of ML techniques applied to US equities and crypto**, framed as an honest evaluation: every major model family — linear, tree ensembles, sequence models, LLM-derived sentiment, classical factor — runs through the same advanced harness (CPCV with PBO, triple-barrier labels with uniqueness weights, deflated Sharpe with sensitivity, stationary block bootstrap, asset-class-aware costs *with* borrow + funding, regime stratification, MDA importance), with strong pragmatic discussion of every result.

The point is the survey + the rigour, not any individual signal. Most signals fail under honest evaluation; the project includes a **positive-control case** (Jegadeesh-Titman 12-1 momentum) to prove the harness can identify real edge when it exists.

When in doubt about whether to add a feature: **does it make the evaluation more honest, or the survey more comprehensive?** Prefer those over feature creep that doesn't serve the study.

## Commands

```bash
make install           # editable install with all extras + pre-commit hook
make test              # pytest
make lint              # ruff check
make format            # ruff check --fix + ruff format
make notebooks         # execute every notebook end-to-end
make all               # lint + test
make clean             # nuke caches
```

Equivalent raw forms:

```bash
pip install -e ".[dev,notebooks,ml,llm]"
pytest                                       # 90+ tests
pytest tests/test_eval_cpcv.py               # one file
pytest -k deflated                           # by keyword
ruff check . && ruff check . --fix
jupyter nbconvert --to notebook --execute notebooks/01_tabular_equities.ipynb
```

Env vars (loaded automatically from `.env` via `python-dotenv`):

- `FRED_API_KEY` — macro features.
- `GROQ_API_KEY` — Case 4 LLM sentiment.
- `OLLAMA_HOST` — optional local-LLM fallback.
- `BACKTESTER_FIXTURE_MODE=1` — substitute bundled `samples/` snapshots for live API calls. Used by CI.
- Yahoo Finance and Binance public endpoints need no key.

## Architecture

Three layers, all operating on plain pandas/numpy. No bespoke event loop; the bar-by-bar simulator is intentionally absent (overkill for daily ML signals).

### 1. The honest-eval harness (the project's credibility layer)

`src/backtester/eval/` — operates on plain `pandas.Series` of returns / positions. **Never bake market-specific assumptions in here.**

- `walkforward.walk_forward_splits` — purged + embargoed walk-forward CV. The simpler reference path; tests assert no label-window leakage and embargo correctness.
- **`cpcv.{cpcv_splits, group_bounds, n_paths, reconstruct_paths}`** — Combinatorial Purged CV (AFML §12). Partition the timeline into `n_groups`, every `C(n, k_test)` choice yields a train/test split (purge + embargo applied around each test group); per-combo predictions are stitched into `C(n_groups - 1, k_test - 1)` distinct full-length OOS paths. *This is the centrepiece statistical upgrade.*
- `statistics.{annualised_sharpe, probabilistic_sharpe_ratio, deflated_sharpe_ratio, dsr_sensitivity, bootstrap_ci, holm_correct}` — pure numpy. **Always report `dsr_sensitivity` across `var ∈ {1.0, 0.5, 0.25, 0.1}`** — fixing var=1.0 alone reads as p-hacked pessimism. `bootstrap_ci(method="stationary")` (default when `block_size>1`) is Politis-Romano stationary block bootstrap; `method="fixed"` keeps the classical CBB; `method="iid"` is the no-blocks variant.
- **`pbo.probability_of_backtest_overfitting`** — Bailey-Borwein-LdP-Zhu 2017. Given `(T × S)` strategy returns, returns PBO + supporting stats (median logit, performance degradation, partition count).
- **`feature_importance.{mda_sklearn, mda_manual}`** — permutation-based MDA. `mda_sklearn` wraps `sklearn.inspection.permutation_importance`; `mda_manual` accepts any predict callable for sequence models.
- `costs.{CostModel, apply_costs, EQUITIES_LIQUID(_WITH_BORROW), CRYPTO_PERP(_WITH_FUNDING)}` — costs charged per `|Δ position|`, plus optional daily borrow on equity shorts and dynamic funding payments on crypto perps (long pays positive funding, short receives).
- `regimes.{trend_regimes, vol_regimes, per_regime_metrics}` — leakage-free regime taggers.

**Invariant:** every public eval function takes a numpy/pandas series of returns and is asset-class agnostic.

### 2. Labels (`src/backtester/labels/`)

- `triple_barrier.{triple_barrier_events, triple_barrier_labels, avg_uniqueness_weights}` — PT/SL/timeout labelling (AFML §3) and concurrency-based sample weights (AFML §4). Defaults: `pt_mult=2`, `sl_mult=1`, `max_holding=5` for equities, `7` for crypto. Long/short side bias supported.
- **Use both binary and triple-barrier in case studies** so readers see how the choice changes the verdict.

### 3. The ML modelling layer

`src/backtester/{features,models,portfolio}/`:

- `features/{technical,macro,cross_sectional,crypto,sentiment}.py` — leakage-free feature builders. **Causality invariant:** `f(series[:t+1])[t] == f(series)[t]` — enforced by `tests/test_features_leakage.py` with random `t/k` sampling.
- `models/{linear,random_forest,gbm,sequence}.py` — uniform `fit(X, y, sample_weight=None) -> self` / `predict_proba(X) -> ndarray` interface. Adding a model = subclass the interface, register in `models/__init__.py`. `sequence.py` ships `LSTMClassifier`, `TCNClassifier`, `TransformerClassifier`, all CPU-only via the `[ml]` extra.
- `portfolio/cross_sectional.py` — case-study notebooks consume model probability scores and run them through `long_short_quantile_weights → daily_returns_from_book → apply_book_costs` to produce a daily-return Series fed directly into `eval/`.

### 4. Data layer

`src/backtester/data/`:

- `csv_loader.py` — bundled-sample loader.
- `yfinance.py`, `fred.py`, `binance.py`, `news.py` — live cached pulls. Each respects `BACKTESTER_FIXTURE_MODE` for CI.
- `universe.py` (+ `samples/universe_us_liquid.csv`) — point-in-time-aware ticker eligibility. Use this whenever building cross-sectional features on equities to avoid survivorship bias.
- `llm.py` — Groq client + Ollama fallback + per-headline cache. Returns `[-1, +1]` sentiment scores given (ticker, headline).

All loaders return either a DataFrame indexed by `datetime` with lower-cased OHLCV columns, or a pandas Series indexed by `datetime`. Every loader has its own date-keyed disk cache under `data_cache/<source>/` (gitignored).

## Conventions

- Public functions/classes get type hints; tests do not.
- `ruff` config (in `pyproject.toml`) enforces import sorting + select lints. CI fails on violations.
- New deps go in `pyproject.toml` with version pins/ranges, never bare `pip install`. Optional extras: `[dev]`, `[notebooks]`, `[ml]` (torch), `[llm]` (groq + feedparser).
- New eval functionality: write the test first, hit a known closed-form or degenerate case for the assertion. The eval layer's credibility is the project's credibility.
- Every case-study notebook follows the same template: hypothesis → data setup → feature construction → triple-barrier labels (where applicable) → walk-forward + CPCV training → portfolio → DSR sensitivity + bootstrap CI + PBO + per-regime breakdown → "if this were production" discussion.
- New case studies live in `notebooks/NN_*.ipynb`, paired with a section in `docs/writeup.md`.
