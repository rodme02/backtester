"""ML Signals in Markets — an in-depth empirical study harness.

Top-level subpackages:

- ``eval``        — walk-forward / CPCV / deflated Sharpe / PBO / costs / regimes
- ``features``    — leakage-free technical, macro, cross-sectional, crypto, sentiment
- ``labels``      — triple-barrier labels + sample uniqueness weights
- ``models``      — uniform fit/predict_proba wrappers (linear, RF, GBM, MLP, LSTM, TCN, Transformer)
- ``portfolio``   — cross-sectional long/short construction + book costs
- ``data``        — cached free data fetchers (yfinance, FRED, Binance, news, LLM)
"""

__version__ = "0.2.0"
