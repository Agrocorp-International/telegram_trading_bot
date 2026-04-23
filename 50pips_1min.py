import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional, List

from strategy_shared import compute_daily_atr


@dataclass
class Trade:
    trade_date: pd.Timestamp
    direction: str
    signal_close: float
    buy_stop_order: float
    sell_stop_order: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    stop_pips: float
    result_pips: float
    outcome: str
    sweep_direction: str   # "up" or "down" — which side was swept before rejection
    sweep_extreme: float   # highest high (up sweep) or lowest low (down sweep) reached


class FiftyPipStrategyBacktester:
    """
    Backtester for the '50-pips-a-day' forex strategy — Sweep → Confirm → Enter.

    Strategy logic:
    - Use 1-minute candles.
    - At 07:59 close (end of the 07:00 hourly bar), define the range:
        buy_entry  = close + entry_offset_pips   (ATR-scaled)
        sell_entry = close - entry_offset_pips
    - Wait for price to sweep one side of the range (false breakout).
    - After the sweep, wait for price to close back inside the range (rejection).
    - Enter OPPOSITE to the sweep direction (trade the trap, not the breakout):
        Upside sweep → reject → SHORT entry at close
        Downside sweep → reject → LONG entry at close
    - Stop loss: beyond the sweep extreme + buffer pips (structural, not ATR).
    - Take profit: signal_close (mean-reversion back to range midpoint).
    - Force-close at EOD if still open.

    CSV expected columns:
        datetime, open, high, low, close
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pip_size: float = 0.0001,
        session_hour: int = 7,
        entry_offset_atr_mult: float = 0.625,
        stop_atr_mult: float = 0.125,    # unused in sweep mode; kept for API compat
        target_atr_mult: float = 0.625,  # unused in sweep mode; kept for API compat
        spread_pips: float = 0.0,
        day_end_hour: int = 23,
        force_exit_eod: bool = True,
        sweep_buffer_pips: float = 2.0,   # extra pips beyond sweep extreme for stop
        sweep_window_bars: int = 120,      # max bars after signal to detect sweep (0 = unlimited)
                                           # 120 bars = 2 hours on 1-min = 08:00–10:00 UTC
        rejection_min_pips: float = 2.0,   # close must be this far back inside range to confirm rejection
        target_mode: str = "opposite_band", # "opposite_band" = sell_entry/buy_entry (asymmetric R:R)
                                            # "midpoint"      = signal_close (conservative)
        htf_trend_bars: int = 60,          # rolling window for HTF trend proxy (0 = disabled)
                                           # 60 bars on 1-min data = ~1 hour = London pre-session trend
        use_htf_filter: bool = True,       # only trade when sweep direction opposes HTF trend
    ):
        self.df = data.copy()
        self.pip_size = pip_size
        self.session_hour = session_hour
        self.entry_offset_atr_mult = entry_offset_atr_mult
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.spread_pips = spread_pips
        self.day_end_hour = day_end_hour
        self.force_exit_eod = force_exit_eod
        self.sweep_buffer_pips = sweep_buffer_pips
        self.sweep_window_bars = sweep_window_bars
        self.rejection_min_pips = rejection_min_pips
        self.target_mode = target_mode
        self.htf_trend_bars = htf_trend_bars
        self.use_htf_filter = use_htf_filter
        self.trades: List[Trade] = []

        self._validate_and_prepare()

    def _validate_and_prepare(self) -> None:
        required_cols = {"datetime", "open", "high", "low", "close", "atr_pips"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.df = self.df.sort_values("datetime").reset_index(drop=True)
        self.df["date"] = self.df["datetime"].dt.date
        self.df["hour"] = self.df["datetime"].dt.hour

        if self.htf_trend_bars > 0:
            # Rolling mean over past N 1-min bars — no lookahead bias (uses only prior closes).
            # NaN for first htf_trend_bars rows; handled in run_backtest via pd.isna() check.
            self.df["htf_trend"] = self.df["close"].rolling(self.htf_trend_bars).mean()

    def _pips_to_price(self, pips: float) -> float:
        return pips * self.pip_size

    def _entry_prices(self, close_price: float, entry_offset_pips: float) -> tuple[float, float]:
        offset = self._pips_to_price(entry_offset_pips)
        buy_entry = close_price + offset
        sell_entry = close_price - offset
        return buy_entry, sell_entry

    def _apply_spread_to_entry(self, price: float, direction: str) -> float:
        spread = self._pips_to_price(self.spread_pips)
        if direction == "long":
            return price + spread / 2
        return price - spread / 2

    def _apply_spread_to_exit(self, price: float, direction: str) -> float:
        spread = self._pips_to_price(self.spread_pips)
        if direction == "long":
            return price - spread / 2
        return price + spread / 2

    def _build_trade_levels(
        self,
        entry_price: float,
        direction: str,
        stop_pips: float,
        target_pips: float,
    ) -> tuple[float, float]:
        stop_dist = self._pips_to_price(stop_pips)
        target_dist = self._pips_to_price(target_pips)

        if direction == "long":
            stop_price = entry_price - stop_dist
            target_price = entry_price + target_dist
        else:
            stop_price = entry_price + stop_dist
            target_price = entry_price - target_dist

        return stop_price, target_price

    def _find_entry_after_signal(
        self,
        day_data: pd.DataFrame,
        signal_idx: int,
        buy_entry: float,
        sell_entry: float,
    ) -> Optional[dict]:
        """
        Finds which pending order triggers first after the signal candle closes.

        Conservative tie-break rule:
        - If both buy and sell entries are crossed within the same candle,
          assume no trade for that bar because intrabar sequence is unknown.
        """
        after_signal = day_data.iloc[signal_idx + 1 :].copy()
        if after_signal.empty:
            return None

        for _, row in after_signal.iterrows():
            hit_buy = row["high"] >= buy_entry
            hit_sell = row["low"] <= sell_entry

            if hit_buy and hit_sell:
                # Ambiguous same-bar dual trigger; conservative: skip trade
                return None
            if hit_buy:
                entry_price = self._apply_spread_to_entry(buy_entry, "long")
                return {
                    "direction": "long",
                    "entry_time": row["datetime"],
                    "entry_price": entry_price,
                }
            if hit_sell:
                entry_price = self._apply_spread_to_entry(sell_entry, "short")
                return {
                    "direction": "short",
                    "entry_time": row["datetime"],
                    "entry_price": entry_price,
                }

        return None

    def _find_sweep_entry(
        self,
        day_data: pd.DataFrame,
        signal_idx: int,
        buy_entry: float,
        sell_entry: float,
        signal_close: float,
    ) -> Optional[dict]:
        """
        Sweep → Confirm → Enter (with time window + rejection strength filters).

        Phase 1 — sweep detection (within sweep_window_bars only):
            Scan up to sweep_window_bars bars after signal. First bar whose
            high >= buy_entry = "up sweep"; low <= sell_entry = "down sweep".
            Simultaneous → skip. Window expired with no sweep → no trade.

        Phase 2 — rejection confirmation (no time limit after sweep):
            Track the extreme. Enter when close is rejection_min_pips back inside
            the range — not just barely crossing the line.

        Stop:   sweep_extreme ± sweep_buffer_pips (structural, beyond the trap).
        Target: opposite_band mode → sell_entry (short) / buy_entry (long)
                midpoint mode       → signal_close
        """
        after_signal = day_data.iloc[signal_idx + 1:].copy()
        if after_signal.empty:
            return None

        rejection_dist = self._pips_to_price(self.rejection_min_pips)
        sweep_direction = None
        sweep_extreme = None

        for bar_num, (_, row) in enumerate(after_signal.iterrows()):
            high = row["high"]
            low = row["low"]

            if sweep_direction is None:
                # Phase 1: sweep detection — honour time window
                if self.sweep_window_bars > 0 and bar_num >= self.sweep_window_bars:
                    break  # window expired, no sweep → skip day
                hit_buy = high >= buy_entry
                hit_sell = low <= sell_entry
                if hit_buy and hit_sell:
                    return None  # simultaneous ambiguous sweep → skip
                elif hit_buy:
                    sweep_direction = "up"
                    sweep_extreme = high
                elif hit_sell:
                    sweep_direction = "down"
                    sweep_extreme = low

            else:
                # Phase 2: rejection confirmation — update extreme, check strong close
                if sweep_direction == "up":
                    sweep_extreme = max(sweep_extreme, high)
                    # Require close meaningfully back inside (not just barely below buy_entry)
                    if row["close"] < buy_entry - rejection_dist:
                        stop_raw = sweep_extreme + self._pips_to_price(self.sweep_buffer_pips)
                        entry_price = self._apply_spread_to_entry(row["close"], "short")
                        stop_pips = (stop_raw - entry_price) / self.pip_size
                        if self.target_mode == "opposite_band":
                            target_price = sell_entry
                        else:
                            target_price = signal_close
                        target_pips = (entry_price - target_price) / self.pip_size
                        return {
                            "direction": "short",
                            "entry_time": row["datetime"],
                            "entry_price": entry_price,
                            "stop_price": stop_raw,
                            "stop_pips": max(stop_pips, 0.1),
                            "target_price": target_price,
                            "target_pips": max(target_pips, 0.1),
                            "sweep_direction": sweep_direction,
                            "sweep_extreme": sweep_extreme,
                        }

                else:  # sweep_direction == "down"
                    sweep_extreme = min(sweep_extreme, low)
                    # Require close meaningfully back inside
                    if row["close"] > sell_entry + rejection_dist:
                        stop_raw = sweep_extreme - self._pips_to_price(self.sweep_buffer_pips)
                        entry_price = self._apply_spread_to_entry(row["close"], "long")
                        stop_pips = (entry_price - stop_raw) / self.pip_size
                        if self.target_mode == "opposite_band":
                            target_price = buy_entry
                        else:
                            target_price = signal_close
                        target_pips = (target_price - entry_price) / self.pip_size
                        return {
                            "direction": "long",
                            "entry_time": row["datetime"],
                            "entry_price": entry_price,
                            "stop_price": stop_raw,
                            "stop_pips": max(stop_pips, 0.1),
                            "target_price": target_price,
                            "target_pips": max(target_pips, 0.1),
                            "sweep_direction": sweep_direction,
                            "sweep_extreme": sweep_extreme,
                        }

        return None

    def _find_exit(
        self,
        remaining_data: pd.DataFrame,
        direction: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
    ) -> dict:
        """
        Finds exit after entry.

        Conservative same-bar handling:
        - Long:
            if both stop and target hit in same bar -> assume stop first
        - Short:
            if both stop and target hit in same bar -> assume stop first
        """
        for _, row in remaining_data.iterrows():
            high_ = row["high"]
            low_ = row["low"]

            if direction == "long":
                hit_stop = low_ <= stop_price
                hit_target = high_ >= target_price

                if hit_stop and hit_target:
                    return {
                        "exit_time": row["datetime"],
                        "exit_price": stop_price,
                        "outcome": "stop_same_bar",
                    }
                if hit_stop:
                    return {
                        "exit_time": row["datetime"],
                        "exit_price": stop_price,
                        "outcome": "stop",
                    }
                if hit_target:
                    return {
                        "exit_time": row["datetime"],
                        "exit_price": target_price,
                        "outcome": "target",
                    }

            else:  # short
                hit_stop = high_ >= stop_price
                hit_target = low_ <= target_price

                if hit_stop and hit_target:
                    return {
                        "exit_time": row["datetime"],
                        "exit_price": stop_price,
                        "outcome": "stop_same_bar",
                    }
                if hit_stop:
                    return {
                        "exit_time": row["datetime"],
                        "exit_price": stop_price,
                        "outcome": "stop",
                    }
                if hit_target:
                    return {
                        "exit_time": row["datetime"],
                        "exit_price": target_price,
                        "outcome": "target",
                    }

        # No stop/target hit before data ends
        last_row = remaining_data.iloc[-1]
        return {
            "exit_time": last_row["datetime"],
            "exit_price": last_row["close"],
            "outcome": "forced_close_end_of_data",
        }

    def run_backtest(self) -> pd.DataFrame:
        self.trades = []

        for trade_date, day_data in self.df.groupby("date"):
            day_data = day_data.reset_index(drop=True)

            signal_rows = day_data[day_data["hour"] == self.session_hour]
            if signal_rows.empty:
                continue

            # Use the LAST candle of the signal hour (e.g. 07:59 on 1-min data).
            # This equals the close of the hourly bar, matching live trading behaviour.
            # For hourly data there is only one candle per hour, so [-1] == [0].
            signal_idx = signal_rows.index[-1]
            signal_row = day_data.loc[signal_idx]

            atr_pips = signal_row["atr_pips"]
            if pd.isna(atr_pips):
                continue

            entry_offset_pips = atr_pips * self.entry_offset_atr_mult

            signal_close = signal_row["close"]
            buy_entry, sell_entry = self._entry_prices(signal_close, entry_offset_pips)

            entry_info = self._find_sweep_entry(
                day_data=day_data,
                signal_idx=signal_idx,
                buy_entry=buy_entry,
                sell_entry=sell_entry,
                signal_close=signal_close,
            )

            if entry_info is None:
                continue

            # HTF trend filter: only take trades where the sweep is AGAINST the trend.
            # Up sweep → short: only if signal_close < htf_trend (bearish heading into London)
            # Down sweep → long: only if signal_close > htf_trend (bullish heading into London)
            if self.htf_trend_bars > 0 and self.use_htf_filter:
                htf_val = signal_row.get("htf_trend", np.nan)
                if not pd.isna(htf_val):
                    if entry_info["direction"] == "short" and signal_close >= htf_val:
                        continue  # trend is up; up-sweep is with-trend, not a trap → skip
                    if entry_info["direction"] == "long" and signal_close <= htf_val:
                        continue  # trend is down; down-sweep is with-trend, not a trap → skip

            direction = entry_info["direction"]
            entry_time = entry_info["entry_time"]
            entry_price = entry_info["entry_price"]
            stop_price = entry_info["stop_price"]
            stop_pips = entry_info["stop_pips"]
            target_price = entry_info["target_price"]

            post_entry = day_data[day_data["datetime"] > entry_time].copy()

            if self.force_exit_eod:
                post_entry = post_entry[post_entry["hour"] <= self.day_end_hour]

            if post_entry.empty:
                continue

            exit_info = self._find_exit(
                remaining_data=post_entry,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
            )

            exit_price = self._apply_spread_to_exit(exit_info["exit_price"], direction)
            exit_time = exit_info["exit_time"]
            outcome = exit_info["outcome"]

            if direction == "long":
                result_pips = (exit_price - entry_price) / self.pip_size
            else:
                result_pips = (entry_price - exit_price) / self.pip_size

            self.trades.append(
                Trade(
                    trade_date=pd.Timestamp(trade_date),
                    direction=direction,
                    signal_close=signal_close,
                    buy_stop_order=buy_entry,
                    sell_stop_order=sell_entry,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    stop_pips=stop_pips,
                    result_pips=result_pips,
                    outcome=outcome,
                    sweep_direction=entry_info["sweep_direction"],
                    sweep_extreme=entry_info["sweep_extreme"],
                )
            )

        return pd.DataFrame([asdict(t) for t in self.trades])

    @staticmethod
    def performance_summary(trades_df: pd.DataFrame) -> pd.Series:
        if trades_df.empty:
            return pd.Series(
                {
                    "trades": 0,
                    "win_rate": np.nan,
                    "total_pips": 0.0,
                    "avg_pips": np.nan,
                    "median_pips": np.nan,
                    "profit_factor": np.nan,
                    "max_win_pips": np.nan,
                    "max_loss_pips": np.nan,
                    "avg_R": np.nan,
                    "median_R": np.nan,
                    "std_R": np.nan,
                    "win_rate_R": np.nan,
                    "expectancy_R": np.nan,
                }
            )

        wins = trades_df.loc[trades_df["result_pips"] > 0, "result_pips"]
        losses = trades_df.loc[trades_df["result_pips"] < 0, "result_pips"]

        gross_profit = wins.sum() if not wins.empty else 0.0
        gross_loss = abs(losses.sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

        span_days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).days
        years = span_days / 365.25 if span_days > 0 else np.nan
        total_pips = trades_df["result_pips"].sum()
        annualized_pips = total_pips / years if years and not np.isnan(years) else np.nan

        # R-multiples: normalize each trade's result by the risk it was taken with.
        valid_r = trades_df[trades_df["stop_pips"] > 0]
        if valid_r.empty:
            avg_r = median_r = std_r = win_rate_r = np.nan
        else:
            r_series = valid_r["result_pips"] / valid_r["stop_pips"]
            avg_r = r_series.mean()
            median_r = r_series.median()
            std_r = r_series.std()
            win_rate_r = (r_series > 0).mean()

        return pd.Series(
            {
                "trades": len(trades_df),
                "win_rate": (trades_df["result_pips"] > 0).mean(),
                "total_pips": total_pips,
                "avg_pips": trades_df["result_pips"].mean(),
                "median_pips": trades_df["result_pips"].median(),
                "profit_factor": profit_factor,
                "max_win_pips": trades_df["result_pips"].max(),
                "max_loss_pips": trades_df["result_pips"].min(),
                "years": years,
                "annualized_pips": annualized_pips,
                "avg_R": avg_r,
                "median_R": median_r,
                "std_R": std_r,
                "win_rate_R": win_rate_r,
                "expectancy_R": avg_r,
            }
        )


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"time": "datetime", "timestamp_utc": "datetime", "timestamp_gmt": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df


def build_equity_curve(
    trades_df: pd.DataFrame,
    initial_capital: float,
    dollars_per_pip: float,
) -> pd.DataFrame:
    """
    Returns a daily equity curve:
        index: calendar days from first entry to last exit
        columns: pnl (per-day $), equity, drawdown (% from running peak)
    Trades are realized on exit date.
    """
    curve = trades_df.sort_values("exit_time").copy()
    curve["pnl"] = curve["result_pips"] * dollars_per_pip
    curve["exit_date"] = curve["exit_time"].dt.normalize()

    daily_pnl = curve.groupby("exit_date")["pnl"].sum()

    start = curve["entry_time"].min().normalize()
    end = curve["exit_time"].max().normalize()
    idx = pd.date_range(start, end, freq="D")

    daily = daily_pnl.reindex(idx, fill_value=0.0).rename("pnl").to_frame()
    daily["equity"] = initial_capital + daily["pnl"].cumsum()
    running_max = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / running_max - 1.0
    return daily


def account_summary(
    trades_df: pd.DataFrame,
    initial_capital: float,
    dollars_per_pip: float,
) -> pd.Series:
    if trades_df.empty:
        return pd.Series({
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "cagr_pct": np.nan,
            "ann_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
        })

    daily = build_equity_curve(trades_df, initial_capital, dollars_per_pip)
    final_equity = daily["equity"].iloc[-1]
    years = len(daily) / 365.25

    total_return = final_equity / initial_capital - 1.0
    cagr = (final_equity / initial_capital) ** (1 / years) - 1.0 if years > 0 and final_equity > 0 else np.nan

    daily_ret = daily["equity"].pct_change().fillna(0.0)
    ann_vol = daily_ret.std() * np.sqrt(365.25)
    sharpe = (daily_ret.mean() * 365.25) / ann_vol if ann_vol > 0 else np.nan
    max_dd = daily["drawdown"].min()

    return pd.Series({
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100 if not np.isnan(cagr) else np.nan,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
    })


def build_compounding_curve(
    trades_df: pd.DataFrame,
    initial_capital: float,
    risk_per_trade: float,
) -> pd.DataFrame:
    """
    Fixed-fractional sizing: each trade risks `risk_per_trade` of current equity,
    with $/pip = (equity * risk_per_trade) / trade.stop_pips. Equity compounds between trades.

    Returns a daily curve with columns: pnl, equity, drawdown.
    """
    ordered = trades_df.sort_values("exit_time").copy()

    equity = initial_capital
    exit_dates: List[pd.Timestamp] = []
    pnls: List[float] = []

    for _, trade in ordered.iterrows():
        trade_stop_pips = trade["stop_pips"]
        if trade_stop_pips <= 0:
            raise ValueError("stop_pips must be > 0 for every trade in compounding sizing")
        dollars_per_pip = (equity * risk_per_trade) / trade_stop_pips
        pnl = trade["result_pips"] * dollars_per_pip
        equity += pnl
        exit_dates.append(trade["exit_time"].normalize())
        pnls.append(pnl)

    trade_pnl = pd.Series(pnls, index=pd.DatetimeIndex(exit_dates))
    daily_pnl = trade_pnl.groupby(trade_pnl.index).sum()

    start = ordered["entry_time"].min().normalize()
    end = ordered["exit_time"].max().normalize()
    idx = pd.date_range(start, end, freq="D")

    daily = daily_pnl.reindex(idx, fill_value=0.0).rename("pnl").to_frame()
    daily["equity"] = initial_capital + daily["pnl"].cumsum()
    running_max = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / running_max - 1.0
    return daily


def compounding_account_summary(
    trades_df: pd.DataFrame,
    initial_capital: float,
    risk_per_trade: float,
) -> pd.Series:
    if trades_df.empty:
        return pd.Series({
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "cagr_pct": np.nan,
            "ann_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
            "risk_per_trade_pct": risk_per_trade * 100,
        })

    daily = build_compounding_curve(trades_df, initial_capital, risk_per_trade)
    final_equity = daily["equity"].iloc[-1]
    years = len(daily) / 365.25

    total_return = final_equity / initial_capital - 1.0
    cagr = (final_equity / initial_capital) ** (1 / years) - 1.0 if years > 0 and final_equity > 0 else np.nan

    daily_ret = daily["equity"].pct_change().fillna(0.0)
    ann_vol = daily_ret.std() * np.sqrt(365.25)
    sharpe = (daily_ret.mean() * 365.25) / ann_vol if ann_vol > 0 else np.nan
    max_dd = daily["drawdown"].min()

    return pd.Series({
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100 if not np.isnan(cagr) else np.nan,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "risk_per_trade_pct": risk_per_trade * 100,
    })


def plot_equity_curve(
    trades_df: pd.DataFrame,
    initial_capital: float,
    dollars_per_pip: float,
    risk_per_trade: Optional[float] = None,
) -> None:
    """
    Plots fixed-sizing equity curve. If risk_per_trade is given, overlays the
    compounding (fixed-fractional) curve using per-trade stop_pips.
    """
    if trades_df.empty:
        print("No trades to plot.")
        return

    fixed = build_equity_curve(trades_df, initial_capital, dollars_per_pip)

    compounding = None
    if risk_per_trade is not None:
        compounding = build_compounding_curve(trades_df, initial_capital, risk_per_trade)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(fixed.index, fixed["equity"], color="#1f77b4",
             label=f"Fixed ${dollars_per_pip:g}/pip")
    if compounding is not None:
        ax1.plot(compounding.index, compounding["equity"], color="#2ca02c",
                 label=f"Compounding {risk_per_trade*100:g}% risk/trade")
    ax1.axhline(initial_capital, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.set_ylabel("Equity ($)")
    ax1.set_yscale("log" if compounding is not None else "linear")
    ax1.set_title(f"50-Pip Strategy (1-min) — Equity Curve (start ${initial_capital:,.0f})")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, which="both")

    ax2.fill_between(fixed.index, fixed["drawdown"] * 100, 0,
                     color="#1f77b4", alpha=0.3, label="Fixed DD")
    if compounding is not None:
        ax2.fill_between(compounding.index, compounding["drawdown"] * 100, 0,
                         color="#2ca02c", alpha=0.3, label="Compounding DD")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ============================================
    # CHANGE THESE SETTINGS
    # ============================================
    CSV_FILE = "data/eurusd_1min_2026.csv"  # 1-minute EURUSD CSV
    PIP_SIZE = 0.0001            # EURUSD/GBPUSD usually 0.0001, JPY pairs usually 0.01
    SESSION_HOUR = 7             # signal = close of the 07:00 hourly bar (last 1-min candle = 07:59)
    ATR_PERIOD = 20
    ENTRY_OFFSET_ATR_MULT = 0.481
    STOP_ATR_MULT = 0.096
    TARGET_ATR_MULT = 0.481
    SPREAD_PIPS = 1.3              # OANDA Standard: ~1.0-1.4 pip spread + stop-order slippage
    DAY_END_HOUR = 20            # force-close by 20:59 UTC — avoids OANDA 21:00/22:00 UTC rollover year-round
    FORCE_EXIT_EOD = True
    SWEEP_BUFFER_PIPS = 2.0      # extra pips beyond sweep extreme for structural stop
    SWEEP_WINDOW_BARS = 120      # max 1-min bars after signal to detect sweep (120 = 2h = 08:00–10:00 UTC)
    REJECTION_MIN_PIPS = 2.0     # close must be this far back inside range to confirm rejection
    TARGET_MODE = "opposite_band" # "opposite_band" = range low/high (asymmetric R:R)
                                  # "midpoint"       = signal_close (conservative)
    HTF_TREND_BARS = 60          # rolling-mean window for trend proxy (0 = disable filter)
                                 # 60 bars on 1-min = ~1 hour = London pre-session trend
    USE_HTF_FILTER = True        # True: only trade when sweep is against the HTF trend

    INITIAL_CAPITAL = 1_000.0    # starting account balance in $
    DOLLARS_PER_PIP = 1.0        # 0.1 = micro, 1.0 = mini, 10.0 = standard lot (EURUSD)
    RISK_PER_TRADE = 0.01        # compounding mode: fraction of equity risked per trade

    START_DATE = "2024-01-01"    # inclusive; None for no lower bound
    END_DATE = None              # inclusive; None for no upper bound

    df = load_data("data/eurusd_1min_2024.csv")
    df = compute_daily_atr(df, pip_size=PIP_SIZE, period=ATR_PERIOD)

    if START_DATE:
        df = df[df["datetime"] >= pd.Timestamp(START_DATE)]
    if END_DATE:
        df = df[df["datetime"] <= pd.Timestamp(END_DATE) + pd.Timedelta(days=1)]
    print(f"Data range: {df['datetime'].min()} to {df['datetime'].max()}  ({len(df):,} bars)")

    # ATR sanity / regime check
    atr_daily = df.set_index("datetime")["atr_pips"].resample("D").first().dropna()
    print("\n=== DAILY ATR (pips) - regime check ===")
    print(atr_daily.describe().to_string(float_format=lambda x: f"{x:,.2f}"))

    bt = FiftyPipStrategyBacktester(
        data=df,
        pip_size=PIP_SIZE,
        session_hour=SESSION_HOUR,
        entry_offset_atr_mult=ENTRY_OFFSET_ATR_MULT,
        stop_atr_mult=STOP_ATR_MULT,
        target_atr_mult=TARGET_ATR_MULT,
        spread_pips=SPREAD_PIPS,
        day_end_hour=DAY_END_HOUR,
        force_exit_eod=FORCE_EXIT_EOD,
        sweep_buffer_pips=SWEEP_BUFFER_PIPS,
        sweep_window_bars=SWEEP_WINDOW_BARS,
        rejection_min_pips=REJECTION_MIN_PIPS,
        target_mode=TARGET_MODE,
        htf_trend_bars=HTF_TREND_BARS,
        use_htf_filter=USE_HTF_FILTER,
    )

    trades = bt.run_backtest()
    summary = bt.performance_summary(trades)

    print("\n=== PERFORMANCE SUMMARY (pips + R) ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    if not trades.empty:
        acct = account_summary(trades, INITIAL_CAPITAL, DOLLARS_PER_PIP)
        print(f"\n=== FIXED SIZING (${INITIAL_CAPITAL:,.0f} start, ${DOLLARS_PER_PIP:g}/pip) ===")
        print(acct.to_string(float_format=lambda x: f"{x:,.4f}"))

        comp = compounding_account_summary(trades, INITIAL_CAPITAL, RISK_PER_TRADE)
        mean_stop = trades["stop_pips"].mean()
        print(f"\n=== COMPOUNDING ({RISK_PER_TRADE*100:g}% risk/trade, per-trade ATR stop, mean ~ {mean_stop:.1f} pips) ===")
        print(comp.to_string(float_format=lambda x: f"{x:,.4f}"))

        print("\n=== FIRST 10 TRADES ===")
        print(trades.head(10))

        trades.to_csv("50_pip_strategy_trades_1min.csv", index=False)
        print("\nSaved trades to 50_pip_strategy_trades_1min.csv")

        # Enriched fixed-sizing trade log.
        # Adds $-P&L, R, running equity, drawdown. Matches the column layout of
        # the hourly backtest export so both files can be compared side-by-side.
        EXPORT_FROM = pd.Timestamp(START_DATE) if START_DATE else trades["entry_time"].min().normalize()
        export = (
            trades[trades["entry_time"] >= EXPORT_FROM]
            .sort_values("exit_time")
            .copy()
        )
        if export.empty:
            print(f"\nNo trades with entry_time >= {EXPORT_FROM.date()} to export.")
        else:
            export["pnl_usd"] = export["result_pips"] * DOLLARS_PER_PIP
            export["R"] = np.where(
                export["stop_pips"] > 0,
                export["result_pips"] / export["stop_pips"],
                np.nan,
            )
            export["cum_pnl_usd"] = export["pnl_usd"].cumsum()
            export["equity_usd"] = INITIAL_CAPITAL + export["cum_pnl_usd"]
            export["drawdown_pct"] = export["equity_usd"] / export["equity_usd"].cummax() - 1.0

            export_path = "50_pip_trades_1min_fixed_sizing.csv"
            export.to_csv(export_path, index=False)
            print(
                f"\nExported enriched fixed-sizing log -> {export_path} "
                f"({len(export)} trades from {EXPORT_FROM.date()}, "
                f"sum pnl_usd={export['pnl_usd'].sum():,.2f}, "
                f"final equity_usd={export['equity_usd'].iloc[-1]:,.2f})"
            )

        # Check intra-bar ambiguity: stop_same_bar count should be much lower than hourly
        same_bar = (trades["outcome"] == "stop_same_bar").sum()
        print(f"\nstop_same_bar outcomes: {same_bar} / {len(trades)} ({100*same_bar/len(trades):.1f}%)")
        print("(on 1-min data this should be near zero vs hourly where it can be significant)")

        plot_equity_curve(trades, INITIAL_CAPITAL, DOLLARS_PER_PIP, RISK_PER_TRADE)
    else:
        print("\nNo trades found.")
