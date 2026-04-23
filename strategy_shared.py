"""Shared strategy utilities for the London Breakout FX runner.

Candle timestamp convention (pinned):
    Hourly bars use **start-of-candle** timestamps. A row timestamped
    2025-01-10T07:00:00Z represents the candle that opens at 07:00 UTC and
    closes at 08:00 UTC. Both live OANDA responses and backtest CSVs follow
    this convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


# =========================
# Daily ATR (Wilder, 1-day lagged)
# =========================

def compute_daily_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return a Series of daily ATR in price units (not pips), indexed by UTC date.

    `df` must have columns: datetime (UTC-naive), high, low, close. The returned
    series is shifted by one day so each date's value uses prior-day bars only.
    """
    daily = (
        df.set_index("datetime")
          .resample("D")
          .agg({"high": "max", "low": "min", "close": "last"})
          .dropna()
    )
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().shift(1)
    atr.index = atr.index.normalize()
    return atr


# =========================
# London Breakout — Asian session range
# =========================

@dataclass
class AsianRange:
    trading_date: str       # YYYY-MM-DD
    asian_high: float
    asian_low: float
    asian_range_pips: float
    atr_price: float        # daily ATR in price units (not pips)


def compute_asian_range(
    df: pd.DataFrame,
    target_date: date,
    *,
    asian_start: int,
    asian_end: int,
    pip_size: float,
    atr_period: int,
) -> Optional[AsianRange]:
    """Extract the Asian high/low (hours asian_start..asian_end inclusive) for
    `target_date` along with the prior-day-shifted daily ATR.

    Returns None if there are no H1 bars inside the Asian window for that date
    or if ATR is NaN (not enough history).
    """
    day_bars = df[
        (df["datetime"].dt.normalize() == pd.Timestamp(target_date))
        & (df["datetime"].dt.hour >= asian_start)
        & (df["datetime"].dt.hour <= asian_end)
    ]
    if day_bars.empty:
        return None

    # Live data includes a `complete` flag — skip unclosed bars so a signal
    # fired before the 06:00 H1 candle finalises doesn't use partial OHLC.
    # Backtest CSVs have no `complete` column (all bars historical/closed).
    if "complete" in day_bars.columns:
        day_bars = day_bars[day_bars["complete"]]
        expected_hours = set(range(asian_start, asian_end + 1))
        got_hours = set(day_bars["datetime"].dt.hour.tolist())
        if not expected_hours.issubset(got_hours):
            return None

    asian_high = float(day_bars["high"].max())
    asian_low = float(day_bars["low"].min())
    if asian_high <= asian_low:
        return None

    atr_series = compute_daily_atr(df, period=atr_period)
    atr_val = atr_series.get(pd.Timestamp(target_date))
    if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
        return None

    return AsianRange(
        trading_date=target_date.isoformat(),
        asian_high=asian_high,
        asian_low=asian_low,
        asian_range_pips=(asian_high - asian_low) / pip_size,
        atr_price=float(atr_val),
    )
