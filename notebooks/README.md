# Notebooks

Each notebook is a self-contained case study evaluating one ML-driven trading signal through the harness in `src/backtester/eval/`.

| Notebook | Status | Signal | Asset class |
| --- | --- | --- | --- |
| `01_gbm_us_equities.ipynb` | ✅ executed | Histogram-GBM direction classifier | US equities |
| `02_sequence_models_crypto.ipynb` | ✅ executed | LSTM + TCN side-by-side | Binance USDT perps |
| `03_llm_sentiment.ipynb` | ⏳ week 4 | LLM-derived sentiment factor | News-covered tickers |

## Running locally

```bash
pip install -e ".[dev,notebooks,ml]"  # ml extra adds torch (CPU) for LSTM/TCN
cp .env.example .env  # then add ALPHA_VANTAGE_API_KEY / FRED_API_KEY
jupyter notebook notebooks/01_gbm_us_equities.ipynb
```

`data_cache/` (gitignored) holds yfinance / FRED / Binance pulls keyed by date — same-day re-runs are offline. yfinance times out frequently; the loader retries with backoff, but you may need to re-execute the data-load cell to fill gaps.

## Reading the verdicts

- **Deflated Sharpe Ratio (DSR).** Probability the true Sharpe exceeds the *expected best of `n_trials`* candidates. We use `n_trials=20` by default. **DSR ≥ 0.95** = the strategy clears the multiple-trial significance bar. Anything less is "we cannot reject the null."
- **Bootstrap 95% CI on Sharpe.** A range produced by 20-day-block resampling. **Strictly above zero** = the point estimate isn't a fluke.
- **Per-regime breakdown.** A signal that only earns in bulls or only in low-vol windows hasn't generalised.

A negative result is still a result. Most ML signals don't survive honest evaluation, and that finding *is* the contribution.
