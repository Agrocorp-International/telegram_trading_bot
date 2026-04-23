import glob as glob_module
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


class FiftyPipHybridBacktester:
    """
    Hybrid backtester for the '50-pips-a-day' forex strategy.

    Architecture:
        hourly_df  -> signal builder   (07:00 H1 candle, ATR, buy/sell stop levels)
        minute_df  -> execution engine (replay from 08:00, minute-level sequencing)

    Strategy logic:
    - Read the 07:00 H1 candle close from hourly_df.
    - Compute ATR-scaled pending orders:
        buy_stop  = close + entry_offset_pips
        sell_stop = close - entry_offset_pips
    - Scan minute bars from 08:00 onward:
        - First bar where high >= buy_stop  -> long trigger
        - First bar where low  <= sell_stop -> short trigger
        - Both in same bar -> skip (ambiguous)
    - Once in trade, scan minute bars for stop / target exit.
    - Force-close at EOD if still open.

    Timestamp convention (from strategy_shared.py):
        Hourly bars are stamped at candle-open. The row timestamped 07:00
        covers 07:00-08:00 UTC. Minute execution therefore starts at 08:00.

    CSV column requirements:
        hourly_df : datetime, open, high, low, close, atr_pips
        minute_df : datetime, open, high, low, close
    """

    def __init__(
        self,
        hourly_data: pd.DataFrame,
        minute_data: pd.DataFrame,
        pip_size: float = 0.0001,
        session_hour: int = 7,
        entry_offset_atr_mult: float = 0.481,
        stop_atr_mult: float = 0.096,
        target_atr_mult: float = 0.481,
        spread_pips: float = 0.0,
        day_end_hour: int = 20,
        force_exit_eod: bool = True,
    ):
        self.hourly_df = hourly_data.copy()
        self.minute_df = minute_data.copy()
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
        required_hourly = {"datetime", "open", "high", "low", "close", "atr_pips"}
        missing_hourly = required_hourly - set(self.hourly_df.columns)
        if missing_hourly:
            raise ValueError(f"hourly_df missing columns: {missing_hourly}")

        required_minute = {"datetime", "open", "high", "low", "close"}
        missing_minute = required_minute - set(self.minute_df.columns)
        if missing_minute:
            raise ValueError(f"minute_df missing columns: {missing_minute}")

        self.hourly_df["datetime"] = pd.to_datetime(self.hourly_df["datetime"])
        self.hourly_df = self.hourly_df.sort_values("datetime").reset_index(drop=True)
        self.hourly_df["date"] = self.hourly_df["datetime"].dt.date
        self.hourly_df["hour"] = self.hourly_df["datetime"].dt.hour

        self.minute_df["datetime"] = pd.to_datetime(self.minute_df["datetime"])
        self.minute_df = self.minute_df.sort_values("datetime").reset_index(drop=True)
        self.minute_df["date"] = self.minute_df["datetime"].dt.date
        self.minute_df["hour"] = self.minute_df["datetime"].dt.hour

    # ------------------------------------------------------------------
    # Price / pip helpers
    # ------------------------------------------------------------------

    def _pips_to_price(self, pips: float) -> float:
        return pips * self.pip_size

    def _entry_prices(self, close_price: float, entry_offset_pips: float) -> tuple[float, float]:
        offset = self._pips_to_price(entry_offset_pips)
        return close_price + offset, close_price - offset

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
            return entry_price - stop_dist, entry_price + target_dist
        return entry_price + stop_dist, entry_price - target_dist

    # ------------------------------------------------------------------
    # Minute-level execution engine
    # ------------------------------------------------------------------

    def _simulate_trade_on_minutes(
        self,
        minute_data: pd.DataFrame,
        buy_entry: float,
        sell_entry: float,
        stop_pips: float,
        target_pips: float,
    ) -> Optional[dict]:
        """
        Replay minute bars to determine trade outcome.

        Phase 1 (pending): scan for the first stop-order trigger.
            - high >= buy_entry  -> long
            - low  <= sell_entry -> short
            - both in same bar   -> return None (conservative skip)

        Phase 2 (in trade): scan for stop or target.
            - same-bar dual hit  -> stop_same_bar (conservative worst case)

        After loop:
            - In trade but bars exhausted -> forced_close_eod at last close
            - Never triggered             -> return None

        Spread is applied here. run_backtest must NOT apply it again.
        """
        if minute_data.empty:
            return None

        in_trade = False
        direction = None
        entry_time = None
        entry_price = None
        stop_price = None
        target_price = None
        last_row = None

        for _, row in minute_data.iterrows():
            high_ = row["high"]
            low_ = row["low"]
            dt = row["datetime"]
            last_row = row

            if not in_trade:
                hit_buy = high_ >= buy_entry
                hit_sell = low_ <= sell_entry

                if hit_buy and hit_sell:
                    return None  # ambiguous dual trigger in same bar

                if hit_buy:
                    direction = "long"
                    entry_time = dt
                    entry_price = self._apply_spread_to_entry(buy_entry, "long")
                    stop_price, target_price = self._build_trade_levels(
                        entry_price, direction, stop_pips, target_pips
                    )
                    in_trade = True
                    continue

                if hit_sell:
                    direction = "short"
                    entry_time = dt
                    entry_price = self._apply_spread_to_entry(sell_entry, "short")
                    stop_price, target_price = self._build_trade_levels(
                        entry_price, direction, stop_pips, target_pips
                    )
                    in_trade = True
                    continue

            else:
                if direction == "long":
                    hit_stop = low_ <= stop_price
                    hit_target = high_ >= target_price

                    if hit_stop and hit_target:
                        raw_exit = stop_price
                        outcome = "stop_same_bar"
                    elif hit_stop:
                        raw_exit = stop_price
                        outcome = "stop"
                    elif hit_target:
                        raw_exit = target_price
                        outcome = "target"
                    else:
                        continue

                else:  # short
                    hit_stop = high_ >= stop_price
                    hit_target = low_ <= target_price

                    if hit_stop and hit_target:
                        raw_exit = stop_price
                        outcome = "stop_same_bar"
                    elif hit_stop:
                        raw_exit = stop_price
                        outcome = "stop"
                    elif hit_target:
                        raw_exit = target_price
                        outcome = "target"
                    else:
                        continue

                exit_price = self._apply_spread_to_exit(raw_exit, direction)
                result_pips = (
                    (exit_price - entry_price) / self.pip_size
                    if direction == "long"
                    else (entry_price - exit_price) / self.pip_size
                )
                return {
                    "direction": direction,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "exit_time": dt,
                    "exit_price": exit_price,
                    "result_pips": result_pips,
                    "outcome": outcome,
                    "stop_pips": stop_pips,
                }

        # Loop exhausted
        if in_trade and last_row is not None:
            raw_exit = last_row["close"]
            exit_price = self._apply_spread_to_exit(raw_exit, direction)
            result_pips = (
                (exit_price - entry_price) / self.pip_size
                if direction == "long"
                else (entry_price - exit_price) / self.pip_size
            )
            return {
                "direction": direction,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_time": last_row["datetime"],
                "exit_price": exit_price,
                "result_pips": result_pips,
                "outcome": "forced_close_eod",
                "stop_pips": stop_pips,
            }

        return None

    # ------------------------------------------------------------------
    # Main backtest loop
    # ------------------------------------------------------------------

    def run_backtest(self) -> pd.DataFrame:
        self.trades = []

        for trade_date, hour_day_data in self.hourly_df.groupby("date"):
            hour_day_data = hour_day_data.reset_index(drop=True)

            signal_rows = hour_day_data[hour_day_data["hour"] == self.session_hour]
            if signal_rows.empty:
                continue

            signal_row = signal_rows.iloc[0]

            atr_pips = signal_row["atr_pips"]
            if pd.isna(atr_pips):
                continue

            entry_offset_pips = atr_pips * self.entry_offset_atr_mult
            stop_pips = atr_pips * self.stop_atr_mult
            target_pips = atr_pips * self.target_atr_mult

            signal_close = signal_row["close"]
            buy_entry, sell_entry = self._entry_prices(signal_close, entry_offset_pips)

            # Minute execution starts after the 07:00 H1 candle closes (= 08:00)
            signal_end = signal_row["datetime"] + pd.Timedelta(hours=1)
            minute_day = self.minute_df[self.minute_df["date"] == trade_date].copy()
            minute_after_signal = minute_day[minute_day["datetime"] >= signal_end].copy()

            if self.force_exit_eod:
                minute_after_signal = minute_after_signal[
                    minute_after_signal["hour"] <= self.day_end_hour
                ]

            if minute_after_signal.empty:
                continue

            trade_info = self._simulate_trade_on_minutes(
                minute_data=minute_after_signal,
                buy_entry=buy_entry,
                sell_entry=sell_entry,
                stop_pips=stop_pips,
                target_pips=target_pips,
            )

            if trade_info is None:
                continue

            self.trades.append(
                Trade(
                    trade_date=pd.Timestamp(trade_date),
                    direction=trade_info["direction"],
                    signal_close=signal_close,
                    buy_stop_order=buy_entry,
                    sell_stop_order=sell_entry,
                    entry_time=trade_info["entry_time"],
                    exit_time=trade_info["exit_time"],
                    entry_price=trade_info["entry_price"],
                    exit_price=trade_info["exit_price"],
                    stop_price=trade_info["stop_price"],
                    target_price=trade_info["target_price"],
                    stop_pips=trade_info["stop_pips"],
                    result_pips=trade_info["result_pips"],
                    outcome=trade_info["outcome"],
                )
            )

        return pd.DataFrame([asdict(t) for t in self.trades])

    # ------------------------------------------------------------------
    # Performance analytics
    # ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Equity curve / account analytics (module-level, reusable)
# ------------------------------------------------------------------

def build_equity_curve(
    trades_df: pd.DataFrame,
    initial_capital: float,
    dollars_per_pip: float,
) -> pd.DataFrame:
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
    ax1.set_title(f"50-Pip Strategy (Hybrid H1+1min) — Equity Curve (start ${initial_capital:,.0f})")
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


# ------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------

def load_hourly_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"time": "datetime", "timestamp_utc": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df


def load_minute_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"time": "datetime", "timestamp_utc": "datetime", "timestamp_gmt": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df


def load_minute_data_multi(pattern: str) -> pd.DataFrame:
    """Load and concatenate all CSVs matching a glob pattern.

    Example:
        load_minute_data_multi("data/eurusd_1min_*.csv")
    """
    paths = sorted(glob_module.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    print(f"Found {len(paths)} minute CSV file(s):")
    for p in paths:
        print(f"  {p}")
    frames = [load_minute_data(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return combined


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ============================================
    # CHANGE THESE SETTINGS
    # ============================================
    HOURLY_CSV    = "data/eurusd_hourly.csv"
    MINUTE_GLOB   = "data/eurusd_1min_*.csv"   # matches all year files

    PIP_SIZE    = 0.0001
    SESSION_HOUR = 7
    ATR_PERIOD  = 20
    ENTRY_OFFSET_ATR_MULT = 0.481
    STOP_ATR_MULT         = 0.096
    TARGET_ATR_MULT       = 0.481
    SPREAD_PIPS  = 1.3        # OANDA Standard
    DAY_END_HOUR = 20         # force-close by 20:59 UTC
    FORCE_EXIT_EOD = True

    INITIAL_CAPITAL  = 1_000.0
    DOLLARS_PER_PIP  = 1.0
    RISK_PER_TRADE   = 0.01

    START_DATE = "2022-01-01"   # must be within minute data coverage
    END_DATE   = None

    # ---- Load data ----
    print("Loading hourly data...")
    hourly_df = load_hourly_data(HOURLY_CSV)
    hourly_df = compute_daily_atr(hourly_df, pip_size=PIP_SIZE, period=ATR_PERIOD)

    print("Loading minute data...")
    minute_df = load_minute_data_multi(MINUTE_GLOB)
    # No ATR needed on minute data — signal comes entirely from hourly

    # ---- Timestamp alignment check ----
    # Confirm hourly bar stamps and minute bar stamps line up as expected.
    h_sample = hourly_df[hourly_df["datetime"].dt.hour == SESSION_HOUR].head(3)[["datetime", "close"]]
    m_sample = minute_df[(minute_df["datetime"].dt.hour >= SESSION_HOUR) &
                         (minute_df["datetime"].dt.hour <= SESSION_HOUR + 1)].head(6)[["datetime", "open", "close"]]
    print(f"\n--- Hourly 07:00 sample (first 3 rows) ---\n{h_sample.to_string(index=False)}")
    print(f"\n--- Minute 07:xx–08:xx sample (first 6 rows) ---\n{m_sample.to_string(index=False)}")
    print("\nExpected: hourly row at HH:00, minute execution starts at 08:00.")

    # ---- Date range filter ----
    if START_DATE:
        hourly_df = hourly_df[hourly_df["datetime"] >= pd.Timestamp(START_DATE)]
        minute_df = minute_df[minute_df["datetime"] >= pd.Timestamp(START_DATE)]
    if END_DATE:
        hourly_df = hourly_df[hourly_df["datetime"] <= pd.Timestamp(END_DATE) + pd.Timedelta(days=1)]
        minute_df = minute_df[minute_df["datetime"] <= pd.Timestamp(END_DATE) + pd.Timedelta(days=1)]

    print(f"\nHourly data range: {hourly_df['datetime'].min()} to {hourly_df['datetime'].max()}  ({len(hourly_df):,} bars)")
    print(f"Minute data range: {minute_df['datetime'].min()} to {minute_df['datetime'].max()}  ({len(minute_df):,} bars)")

    # ATR regime check
    atr_daily = hourly_df.set_index("datetime")["atr_pips"].resample("D").first().dropna()
    print("\n=== DAILY ATR (pips) — regime check ===")
    print(atr_daily.describe().to_string(float_format=lambda x: f"{x:,.2f}"))

    # ---- Run backtest ----
    bt = FiftyPipHybridBacktester(
        hourly_data=hourly_df,
        minute_data=minute_df,
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
        print(f"\n=== COMPOUNDING ({RISK_PER_TRADE*100:g}% risk/trade, mean stop ~ {mean_stop:.1f} pips) ===")
        print(comp.to_string(float_format=lambda x: f"{x:,.4f}"))

        # Intrabar ambiguity check — should be near-zero with minute data
        same_bar = (trades["outcome"] == "stop_same_bar").sum()
        print(f"\nstop_same_bar outcomes: {same_bar} / {len(trades)} ({100*same_bar/len(trades):.1f}%)")
        print("(should be near-zero; confirms minute data resolves intrabar ambiguity)")

        print("\n=== FIRST 10 TRADES ===")
        print(trades.head(10).to_string())

        trades.to_csv("50_pip_hybrid_trades.csv", index=False)
        print("\nSaved trades to 50_pip_hybrid_trades.csv")

        # Enriched export with P&L, R, running equity
        export = trades.sort_values("exit_time").copy()
        export["pnl_usd"] = export["result_pips"] * DOLLARS_PER_PIP
        export["R"] = np.where(
            export["stop_pips"] > 0,
            export["result_pips"] / export["stop_pips"],
            np.nan,
        )
        export["cum_pnl_usd"] = export["pnl_usd"].cumsum()
        export["equity_usd"] = INITIAL_CAPITAL + export["cum_pnl_usd"]
        export["drawdown_pct"] = export["equity_usd"] / export["equity_usd"].cummax() - 1.0
        export.to_csv("50_pip_hybrid_trades_enriched.csv", index=False)
        print(
            f"Saved enriched log -> 50_pip_hybrid_trades_enriched.csv "
            f"({len(export)} trades, sum pnl_usd={export['pnl_usd'].sum():,.2f}, "
            f"final equity_usd={export['equity_usd'].iloc[-1]:,.2f})"
        )

        plot_equity_curve(trades, INITIAL_CAPITAL, DOLLARS_PER_PIP, RISK_PER_TRADE)
    else:
        print("\nNo trades found. Check that the date range overlaps both hourly and minute data.")
