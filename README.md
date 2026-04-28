# Backtester

[![CI](https://github.com/rodrigomedeiros/backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/rodrigomedeiros/backtester/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> Event-driven backtesting framework with a Streamlit lab for trying strategies on real market data.

![dashboard preview](docs/dashboard.png)

> _Add the screenshot above by running the dashboard locally and saving it to `docs/dashboard.png`._

## What it does

- Runs trading strategies through [backtrader](https://www.backtrader.com/) with realistic commission and slippage.
- Ships two reference strategies: a clean SMA crossover with ATR-based risk sizing, and an advanced trend-follower confirmed by RSI / MACD / ADX with split bracket exits.
- Bundles five years of daily OHLCV samples (AAPL, AMZN, MSFT, SPY, TSLA) so it works offline on a fresh clone.
- Streamlit dashboard for interactive parameter tweaks, equity-curve and drawdown charts, and key performance metrics.
- CLI for scripted single runs; programmatic API for parameter sweeps.

## Why it's interesting

- **Event-driven, not just vectorised.** The framework wraps backtrader's `Cerebro`, so stops, take-profits, and partial exits are simulated bar-by-bar — closer to live execution than naive `signal.shift(1) * returns` math.
- **Strategy registry.** Adding a strategy is "subclass `bt.Strategy` and register" — no engine changes.
- **Cached data layer.** Alpha Vantage fetches go through an on-disk cache so a parameter sweep doesn't burn the free-tier quota.
- **Reproducible results.** Tests pin a tiny fixture run to exact metric values so refactors can't silently change the math.

## Tech stack

| Layer       | Choice            | Why                                                    |
| ----------- | ----------------- | ------------------------------------------------------ |
| Engine      | backtrader        | Mature event-driven simulator, bracket orders built in |
| Dashboard   | Streamlit + Plotly| Fastest path from Python to interactive UI            |
| Data        | Alpha Vantage     | Free daily OHLCV; tier-friendly with caching          |
| Tests / lint| pytest + ruff     | Standard, fast, opinionated                           |

## Quickstart

```bash
git clone https://github.com/rodrigomedeiros/backtester.git
cd backtester
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
streamlit run dashboard/app.py
```

CLI:

```bash
backtester run --strategy ma_crossover --symbol AAPL
```

To pull live data, copy `.env.example` to `.env`, set `ALPHAVANTAGE_API_KEY`, and pass `--source alphavantage`.

## Project layout

```
src/backtester/
  data/         # CSV loader + cached Alpha Vantage client
  strategies/   # MaCrossover, AdvancedTrendFollowing, REGISTRY
  engine/       # run_backtest, optimize (grid search)
  reporting/    # Sharpe / drawdown / win-rate / equity curve
  cli.py
dashboard/app.py
tests/
samples/ohlcv/  # Bundled OHLCV CSVs
```

## Roadmap / known limitations

- Daily bars only; intraday timeframes not yet wired.
- Optimiser is sequential single-process; multiprocess sweep is on the roadmap.
- Cost model is flat commission + percentage slippage. No spread/borrow modelling.
- No walk-forward analysis or out-of-sample split helper yet.
- Live trading is out of scope — this is a research tool.

## License

MIT — see [LICENSE](LICENSE).
