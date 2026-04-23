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


class FiftyPipStrategyBacktester:
    """
    Backtester for the '50-pips-a-day' forex strategy.

    Strategy logic:
    - Use 1-hour candles.
    - When the 07:00 candle closes, place:
        buy stop  = close + entry_offset_pips
        sell stop = close - entry_offset_pips
    - Once one order triggers, cancel the other.
    - For the triggered trade:
        stop loss  = stop_pips
        take profit = target_pips
    - Exit when stop or target is hit.
    - If still open by the end of the trading day, optionally close at cutoff.

    CSV expected columns:
        datetime, open, high, low, close

    Example:
        datetime,open,high,low,close
        2024-01-01 00:00:00,1.2730,1.2740,1.2720,1.2735
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pip_size: float = 0.0001,
        session_hour: int = 7,
        entry_offset_atr_mult: float = 0.625,
        stop_atr_mult: float = 0.125,
        target_atr_mult: float = 0.625,
        spread_pips: float = 0.0,
        day_end_hour: int = 23,
        force_exit_eod: bool = True,
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
        Finds which pending order triggers first after the 07:00 candle closes.

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

            signal_idx = signal_rows.index[0]
            signal_row = day_data.loc[signal_idx]

            atr_pips = signal_row["atr_pips"]
            if pd.isna(atr_pips):
                continue

            entry_offset_pips = atr_pips * self.entry_offset_atr_mult
            stop_pips = atr_pips * self.stop_atr_mult
            target_pips = atr_pips * self.target_atr_mult

            signal_close = signal_row["close"]
            buy_entry, sell_entry = self._entry_prices(signal_close, entry_offset_pips)

            entry_info = self._find_entry_after_signal(
                day_data=day_data,
                signal_idx=signal_idx,
                buy_entry=buy_entry,
                sell_entry=sell_entry,
            )

            if entry_info is None:
                continue

            direction = entry_info["direction"]
            entry_time = entry_info["entry_time"]
            entry_price = entry_info["entry_price"]

            stop_price, target_price = self._build_trade_levels(
                entry_price, direction, stop_pips, target_pips
            )

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
    df = df.rename(columns={"time": "datetime", "timestamp_utc": "datetime"})
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
    ax1.set_title(f"50-Pip Strategy — Equity Curve (start ${initial_capital:,.0f})")
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
    CSV_FILE = "eurusd_1h.csv"   # your CSV file
    PIP_SIZE = 0.0001            # EURUSD/GBPUSD usually 0.0001, JPY pairs usually 0.01
    SESSION_HOUR = 7             # 07:00 candle
    # ATR-adaptive distances. Calibrated to measured pre-2016 EURUSD daily ATR(20) ~ 104 pips:
    # 50/10/50 pip reference setup -> 0.481 / 0.096 / 0.481 multipliers so pre-2016 trades
    # approximately match the original fixed-pip backtest and post-2016 auto-scales down.
    ATR_PERIOD = 20
    ENTRY_OFFSET_ATR_MULT = 0.481
    STOP_ATR_MULT = 0.096
    TARGET_ATR_MULT = 0.481
    SPREAD_PIPS = 2           # OANDA Standard: ~1.0-1.4 pip spread + stop-order slippage
    DAY_END_HOUR = 20            # force-close by 20:59 UTC — avoids OANDA 21:00/22:00 UTC rollover year-round
    FORCE_EXIT_EOD = True

    INITIAL_CAPITAL = 1_000.0    # starting account balance in $
    DOLLARS_PER_PIP = 1.0        # 0.1 = micro, 1.0 = mini, 10.0 = standard lot (EURUSD)
    RISK_PER_TRADE = 0.01      # compounding mode: fraction of equity risked per trade (0.25%)

    START_DATE = "2025-01-01"    # inclusive; None for no lower bound
    END_DATE = None              # inclusive; None for no upper bound

    df = load_data("data/eurusd_hourly.csv")
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
    split = pd.Timestamp("2016-01-01")
    pre_mean = atr_daily[atr_daily.index < split].mean()
    post_mean = atr_daily[atr_daily.index >= split].mean()
    print(f"Pre-{split.year} mean ATR:  {pre_mean:,.1f} pips")
    print(f"Post-{split.year} mean ATR: {post_mean:,.1f} pips")
    print("(multipliers calibrated to pre-2016 mean ~104 pips; re-calibrate if this figure changes materially)")

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

        trades.to_csv("50_pip_strategy_trades.csv", index=False)
        print("\nSaved trades to 50_pip_strategy_trades.csv")

        # Enriched fixed-sizing trade log for 2025-01-01 onwards.
        # Adds $-P&L, R, running equity, drawdown. Filter is independent of
        # START_DATE so the export window is stable across backtest ranges.
        EXPORT_FROM = pd.Timestamp("2025-01-01")
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
            # Equity starts from INITIAL_CAPITAL at the first exported trade.
            export["equity_usd"] = INITIAL_CAPITAL + export["cum_pnl_usd"]
            export["drawdown_pct"] = export["equity_usd"] / export["equity_usd"].cummax() - 1.0

            export_path = "50_pip_trades_2025_fixed_sizing.csv"
            export.to_csv(export_path, index=False)
            print(
                f"\nExported enriched fixed-sizing log -> {export_path} "
                f"({len(export)} trades from {EXPORT_FROM.date()}, "
                f"sum pnl_usd={export['pnl_usd'].sum():,.2f}, "
                f"final equity_usd={export['equity_usd'].iloc[-1]:,.2f})"
            )

        # ---- Sub-period breakdown around 2016 ----
        SPLIT_DATE = pd.Timestamp("2016-01-01")
        sub_periods = {
            f"PRE-{SPLIT_DATE.year}  (entry < {SPLIT_DATE.date()})": trades[trades["entry_time"] < SPLIT_DATE],
            f"POST-{SPLIT_DATE.year} (entry >= {SPLIT_DATE.date()})": trades[trades["entry_time"] >= SPLIT_DATE],
        }
        sub_summaries: dict = {}
        for label, sub in sub_periods.items():
            print(f"\n######## {label} ########")
            if sub.empty:
                print("No trades in this window.")
                continue
            sub_summary = bt.performance_summary(sub)
            sub_summaries[label] = sub_summary
            print("\n--- performance (pips + R) ---")
            print(sub_summary.to_string(float_format=lambda x: f"{x:,.4f}"))
            print(f"\n--- fixed sizing (${INITIAL_CAPITAL:,.0f}, ${DOLLARS_PER_PIP:g}/pip) ---")
            print(account_summary(sub, INITIAL_CAPITAL, DOLLARS_PER_PIP).to_string(float_format=lambda x: f"{x:,.4f}"))
            sub_mean_stop = sub["stop_pips"].mean()
            print(f"\n--- compounding ({RISK_PER_TRADE*100:g}% risk/trade, mean stop ~ {sub_mean_stop:.1f} pips) ---")
            print(compounding_account_summary(sub, INITIAL_CAPITAL, RISK_PER_TRADE).to_string(float_format=lambda x: f"{x:,.4f}"))

        # Expectancy-R delta: did the strategy's edge (risk-normalized) survive post-2016?
        if len(sub_summaries) == 2:
            (pre_label, pre_sum), (post_label, post_sum) = list(sub_summaries.items())
            pre_r = pre_sum["expectancy_R"]
            post_r = post_sum["expectancy_R"]
            delta_r = post_r - pre_r
            print("\n### R-EXPECTANCY DELTA ###")
            print(f"  {pre_label:40s} expectancy_R: {pre_r:.4f}")
            print(f"  {post_label:40s} expectancy_R: {post_r:.4f}")
            print(f"  DELTA (post - pre):                            {delta_r:+.4f}")
            if pre_r and not pd.isna(pre_r) and not pd.isna(post_r):
                print(f"  Relative change:                               {100 * delta_r / abs(pre_r):+.1f}%")

        plot_equity_curve(trades, INITIAL_CAPITAL, DOLLARS_PER_PIP, RISK_PER_TRADE)
    else:
        print("\nNo trades found.")