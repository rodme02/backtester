# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
uses semantic versioning where versions correspond to harness milestones.

## [Unreleased]

### Changed
- Repository renamed `backtester` → `ml-signals-in-markets` (the Python
  package is still imported as `backtester`).
- `samples/fixtures/` reorganised into per-source subdirectories
  (`yfinance/`, `binance/`, `fred/`).
- `_cache_path()` consolidated into `data/_cache.py`; the four date-keyed
  loaders (`yfinance`, `fred`, `binance`, `news`) all delegate to the
  shared helper.
- Doc-generator scripts moved from `docs/` to `scripts/`.

### Removed
- Empty namespace packages `src/backtester/{engine,reporting,strategies}`.

## [0.2.0] — 2026-04

The "honest-evaluation harness" milestone. Every signal in the survey is
now evaluated through the same advanced statistical machinery.

### Added
- **Combinatorial Purged Cross-Validation** (AFML §12): `eval/cpcv.py`
  generates `C(n,k)` train/test splits with purge + embargo and
  reconstructs `C(n-1, k-1)` distinct full-length OOS paths.
- **Triple-barrier labels** with sample uniqueness weights (AFML §3-4):
  `labels/triple_barrier.py`.
- **Probability of Backtest Overfitting** (Bailey-Borwein-LdP-Zhu 2017):
  `eval/pbo.py`.
- **Deflated Sharpe ratio with sensitivity** across
  `trials_sr_var ∈ {1.0, 0.5, 0.25, 0.1}`: `eval/statistics.py`.
- **Stationary block bootstrap** (Politis-Romano 1994) for autocorrelated
  daily returns.
- **MDA permutation feature importance** (sklearn + manual variants):
  `eval/feature_importance.py`.
- **Asset-class costs with borrow + funding**: equity short-borrow and
  dynamic crypto perpetual funding payments in `eval/costs.py`.
- **Sequence models**: LSTM, TCN, Transformer classifiers in
  `models/sequence.py` (CPU-only via the `[ml]` extra).
- **LLM sentiment factor**: Groq backend with Ollama fallback and
  per-headline cache (`data/llm.py`); per-ticker yfinance news loader
  (`data/news.py`); `features/sentiment.py`.
- **Five empirical case studies** under `notebooks/`:
  1. Tabular ML on US equities (logistic/RF/GBM × binary/triple-barrier)
  2. Crypto signal universe (returns/funding/basis/carry-rank/union)
  3. Sequence models on crypto (LSTM/TCN/Transformer)
  4. LLM sentiment (data-constrained finding documented)
  5. Jegadeesh-Titman 12-1 momentum positive control
- **Cross-cutting writeup** at `docs/writeup.md` (~4,400 words).
- **CI fixtures**: 74 bundled CSVs in `samples/fixtures/`; CI runs all
  notebooks hermetically with `BACKTESTER_FIXTURE_MODE=1`.
- **Pre-commit + ruff + GitHub Actions** matrix on Python 3.10–3.13.

### Changed
- `strategies/` module renamed to `portfolio/`; legacy backtrader
  dependency removed.
- README and `CLAUDE.md` reframed around the honest-evaluation thesis.

## [0.1.0] — Initial

- Walk-forward CV harness with purge + embargo (`eval/walkforward.py`).
- Annualised Sharpe, PSR, bootstrap CI, Holm correction in
  `eval/statistics.py`.
- Per-asset cost model with turnover-based charging (`eval/costs.py`).
- Trend + volatility regime tagging (`eval/regimes.py`).
- Data loaders: yfinance, FRED, Binance public.
- Point-in-time universe loader (`data/universe.py`) to avoid
  survivorship bias.
- Leakage-free feature builders: technical, macro, cross-sectional.
- Baseline models: linear, random forest, GBM.
