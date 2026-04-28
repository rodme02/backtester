"""Command-line entry point: ``python -m backtester.cli run --symbol AAPL``."""

from __future__ import annotations

import argparse
import json
import sys

from .data import fetch_daily, load_samples
from .engine import run_backtest
from .strategies import REGISTRY


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backtester")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a single backtest")
    run.add_argument("--strategy", choices=sorted(REGISTRY), default="ma_crossover")
    run.add_argument("--symbol", default="AAPL")
    run.add_argument("--source", choices=("samples", "alphavantage"), default="samples")
    run.add_argument("--cash", type=float, default=100_000.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        data = load_samples(args.symbol) if args.source == "samples" else fetch_daily(args.symbol)
        result = run_backtest(REGISTRY[args.strategy], data, cash=args.cash)
        print(json.dumps({
            "strategy": args.strategy,
            "symbol": args.symbol,
            "final_value": round(result.final_value, 2),
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in result.metrics.items()},
        }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
