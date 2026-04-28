"""Trend-following strategy with RSI / MACD / ADX confirmation,
ATR-based position sizing and split bracket exits."""

from __future__ import annotations

import backtrader as bt


class AdvancedTrendFollowing(bt.Strategy):
    params = (
        ("fast_period", 20),
        ("slow_period", 100),
        ("atr_period", 14),
        ("risk_fraction", 0.01),
        ("stop_atr_multiple", 2.0),
        ("take_profit1_multiple", 1.5),
        ("take_profit2_multiple", 3.0),
        ("rsi_period", 14),
        ("rsi_min_long", 50),
        ("rsi_max_short", 50),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("adx_period", 14),
        ("adx_threshold", 20.0),
        ("enable_short", True),
    )

    def __init__(self) -> None:
        d = self.data
        self.fast_ma = bt.ind.SMA(d, period=self.p.fast_period)
        self.slow_ma = bt.ind.SMA(d, period=self.p.slow_period)
        self.crossover = bt.ind.CrossOver(self.fast_ma, self.slow_ma)
        self.atr = bt.ind.ATR(d, period=self.p.atr_period)
        self.rsi = bt.ind.RSI(d, period=self.p.rsi_period)
        self.macd = bt.ind.MACD(
            d,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )
        self.adx = bt.ind.ADX(d, period=self.p.adx_period)
        self._pending = False

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            self._pending = False

    def next(self) -> None:
        if self._pending or self.position:
            return

        atr = self.atr[0]
        if atr <= 0 or self.adx[0] < self.p.adx_threshold:
            return

        price = self.data.close[0]
        risk_capital = self.broker.getvalue() * self.p.risk_fraction

        bull = (
            self.crossover > 0
            and self.rsi[0] >= self.p.rsi_min_long
            and self.macd.macd[0] > self.macd.signal[0]
        )
        bear = (
            self.p.enable_short
            and self.crossover < 0
            and self.rsi[0] <= self.p.rsi_max_short
            and self.macd.macd[0] < self.macd.signal[0]
        )

        if bull:
            stop = price - atr * self.p.stop_atr_multiple
            tp1 = price + atr * self.p.take_profit1_multiple
            tp2 = price + atr * self.p.take_profit2_multiple
            self._enter(bt.Order.Buy, price, stop, tp1, tp2, risk_capital)
        elif bear:
            stop = price + atr * self.p.stop_atr_multiple
            tp1 = price - atr * self.p.take_profit1_multiple
            tp2 = price - atr * self.p.take_profit2_multiple
            self._enter(bt.Order.Sell, price, stop, tp1, tp2, risk_capital)

    def _enter(
        self,
        side: int,
        price: float,
        stop: float,
        tp1: float,
        tp2: float,
        risk_capital: float,
    ) -> None:
        risk_per_share = abs(price - stop)
        if risk_per_share <= 0:
            return
        half = risk_capital / 2
        size1 = int(half / risk_per_share)
        size2 = int(half / risk_per_share)
        if size1 < 1 and size2 < 1:
            return
        bracket = self.buy_bracket if side == bt.Order.Buy else self.sell_bracket
        if size1 > 0:
            bracket(size=size1, exectype=bt.Order.Market, stopprice=stop, limitprice=tp1)
        if size2 > 0:
            bracket(size=size2, exectype=bt.Order.Market, stopprice=stop, limitprice=tp2)
        self._pending = True
