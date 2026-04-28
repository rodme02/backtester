"""Plain SMA crossover with ATR-based stop and take-profit."""

from __future__ import annotations

import backtrader as bt


class MaCrossover(bt.Strategy):
    params = (
        ("fast_period", 20),
        ("slow_period", 100),
        ("atr_period", 14),
        ("risk_fraction", 0.02),
        ("stop_atr_multiple", 2.0),
        ("take_profit_multiple", 3.0),
    )

    def __init__(self) -> None:
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self._pending = None

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            self._pending = None

    def next(self) -> None:
        if self._pending or self.position:
            return

        atr = self.atr[0]
        if atr <= 0:
            return

        price = self.data.close[0]
        risk_capital = self.broker.getvalue() * self.p.risk_fraction

        if self.crossover > 0:
            stop = price - atr * self.p.stop_atr_multiple
            target = price + atr * self.p.take_profit_multiple
            size = max(1, int(risk_capital / max(price - stop, 1e-9)))
            self._pending = self.buy_bracket(
                size=size, exectype=bt.Order.Market,
                stopprice=stop, limitprice=target,
            )
        elif self.crossover < 0:
            stop = price + atr * self.p.stop_atr_multiple
            target = price - atr * self.p.take_profit_multiple
            size = max(1, int(risk_capital / max(stop - price, 1e-9)))
            self._pending = self.sell_bracket(
                size=size, exectype=bt.Order.Market,
                stopprice=stop, limitprice=target,
            )
