"""P&L report for LB-tagged closed trades.

Pure compute + rendering — no Telegram or OANDA imports. Takes raw OANDA
closed-trade dicts as input, returns parsed DataFrame, stats dict, and a
PNG as bytes. All math + chart rendering is unit-testable via synthetic data.
"""
from __future__ import annotations

from io import BytesIO
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# Parse raw OANDA trades → DataFrame
# =========================

def build_trades_df(raw_trades: List[dict]) -> pd.DataFrame:
    """Parse OANDA closed-trade records into a tidy DataFrame.

    Columns: trade_id, side (long/short), open_time (UTC), close_time (UTC),
             entry, exit, realized_pl, units.

    Rows are silently skipped if they lack `realizedPL` or `closeTime` —
    those are load-bearing for every downstream stat. All other fields use
    safe fallbacks so a partially malformed record doesn't crash the report.
    """
    rows = []
    for t in raw_trades:
        pl_raw = t.get("realizedPL")
        close_raw = t.get("closeTime")
        if pl_raw is None or not close_raw:
            continue
        try:
            realized_pl = float(pl_raw)
        except (TypeError, ValueError):
            continue

        initial_units = t.get("initialUnits", "0")
        try:
            units = int(initial_units)
        except (TypeError, ValueError):
            units = 0
        side = "long" if units > 0 else ("short" if units < 0 else "flat")

        def _f(x):
            if x in (None, ""):
                return np.nan
            try:
                return float(x)
            except (TypeError, ValueError):
                return np.nan

        rows.append({
            "trade_id": t.get("id", ""),
            "side": side,
            "open_time": pd.to_datetime(t.get("openTime"), utc=True, errors="coerce"),
            "close_time": pd.to_datetime(close_raw, utc=True, errors="coerce"),
            "entry": _f(t.get("price")),
            "exit": _f(t.get("averageClosePrice")),
            "realized_pl": realized_pl,
            "units": abs(units),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["close_time"])
    return df.sort_values("close_time").reset_index(drop=True)


def filter_trades_since(df: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    """Return rows with close_time >= since. `since` must be tz-aware UTC."""
    if since.tzinfo is None:
        raise ValueError("since must be timezone-aware")
    if df.empty:
        return df
    return df[df["close_time"] >= since].reset_index(drop=True)


# =========================
# Stats
# =========================

def compute_stats(df: pd.DataFrame) -> dict:
    """All summary stats, safe on empty / all-win / all-loss DataFrames."""
    if df.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0,
            "total_pl": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "avg_win": None, "avg_loss": None,
            "best_day_pl": None, "worst_day_pl": None,
            "best_day_date": None, "worst_day_date": None,
            "max_drawdown": 0.0,
        }

    wins = df[df["realized_pl"] > 0]
    losses = df[df["realized_pl"] < 0]
    gross_wins = float(wins["realized_pl"].sum()) if not wins.empty else 0.0
    gross_losses = float(-losses["realized_pl"].sum()) if not losses.empty else 0.0

    if gross_wins == 0 and gross_losses == 0:
        pf = None  # all trades break even
    elif gross_losses == 0:
        pf = float("inf")
    else:
        pf = gross_wins / gross_losses

    by_day = (
        df.assign(close_date=df["close_time"].dt.tz_convert("UTC").dt.date)
          .groupby("close_date")["realized_pl"].sum()
          .sort_values()
    )
    best_day = by_day.iloc[-1] if not by_day.empty else None
    worst_day = by_day.iloc[0] if not by_day.empty else None
    best_date = by_day.index[-1] if not by_day.empty else None
    worst_date = by_day.index[0] if not by_day.empty else None

    # Drawdown baseline: pre-trading equity = 0, so an all-loss series
    # reports max_dd = -total_loss (not -0 relative to the first dip).
    cum = df["realized_pl"].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd_series = cum - peak  # always ≤ 0
    max_dd = float(dd_series.min()) if dd_series.size else 0.0

    return {
        "trades": int(len(df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "total_pl": float(df["realized_pl"].sum()),
        "win_rate": float(len(wins) / len(df)),
        "profit_factor": pf,
        "avg_win": float(wins["realized_pl"].mean()) if not wins.empty else None,
        "avg_loss": float(losses["realized_pl"].mean()) if not losses.empty else None,
        "best_day_pl": float(best_day) if best_day is not None else None,
        "worst_day_pl": float(worst_day) if worst_day is not None else None,
        "best_day_date": best_date,
        "worst_day_date": worst_date,
        "max_drawdown": max_dd,   # negative or zero
    }


# =========================
# Chart render
# =========================

def render_pnl_chart(df: pd.DataFrame, label: str = "LB") -> bytes:
    """Return a two-panel PNG as bytes. No text inside the figure — stats
    go in the Telegram caption separately. Uses Agg backend (headless)."""
    fig, (ax_cum, ax_month) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    if df.empty:
        for ax, title in ((ax_cum, "Cumulative P&L"),
                          (ax_month, "Monthly P&L")):
            ax.set_title(title)
            ax.text(0.5, 0.5, "No trades yet", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        # Panel 1: cumulative P&L
        cum = df["realized_pl"].cumsum()
        ax_cum.plot(df["close_time"], cum, color="#2a7ae4", linewidth=1.8)
        ax_cum.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax_cum.fill_between(df["close_time"], cum, 0,
                            where=(cum >= 0), color="#2a7ae4", alpha=0.12)
        ax_cum.fill_between(df["close_time"], cum, 0,
                            where=(cum < 0), color="#d9534f", alpha=0.15)
        ax_cum.set_title(f"{label} Cumulative Realized P&L ($)")
        ax_cum.set_ylabel("$")
        ax_cum.grid(True, alpha=0.3)

        # Panel 2: monthly bars
        monthly = (
            df.set_index("close_time")["realized_pl"]
              .resample("MS")
              .sum()
        )
        colors = ["#3c9c3c" if v >= 0 else "#d9534f" for v in monthly.values]
        bar_labels = [d.strftime("%Y-%m") for d in monthly.index]
        ax_month.bar(range(len(monthly)), monthly.values, color=colors)
        ax_month.set_xticks(range(len(monthly)))
        ax_month.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
        ax_month.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax_month.set_title("Monthly Realized P&L ($)")
        ax_month.set_ylabel("$")
        ax_month.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =========================
# Caption
# =========================

def _fmt_money(v) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.2f}"


def _fmt_pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _fmt_pf(v) -> str:
    if v is None:
        return "n/a"
    if v == float("inf"):
        return "∞"
    return f"{v:.2f}"


def format_stats_caption(
    stats: dict,
    skipped_count: int = 0,
    label: str = "LB",
    since: pd.Timestamp | None = None,
) -> str:
    if stats["trades"] == 0:
        base = f"📭 No {label} closed trades yet."
        if since is not None:
            base += f"\nSince: {since.strftime('%Y-%m-%d %H:%M')} UTC"
        return base

    lines = [f"📊 {label} P&L Summary"]
    if since is not None:
        lines.append(f"Since: {since.strftime('%Y-%m-%d %H:%M')} UTC")
    lines += [
        f"Trades: {stats['trades']}  (W {stats['wins']} / L {stats['losses']})   "
        f"Win rate: {_fmt_pct(stats['win_rate'])}",
        f"Total P&L: {_fmt_money(stats['total_pl'])}   "
        f"Profit factor: {_fmt_pf(stats['profit_factor'])}",
        f"Avg win: {_fmt_money(stats['avg_win'])}   "
        f"Avg loss: {_fmt_money(stats['avg_loss'])}",
    ]
    if stats.get("best_day_date") is not None:
        lines.append(
            f"Best day: {_fmt_money(stats['best_day_pl'])} on {stats['best_day_date']} UTC"
        )
    if stats.get("worst_day_date") is not None:
        lines.append(
            f"Worst day: {_fmt_money(stats['worst_day_pl'])} on {stats['worst_day_date']} UTC"
        )
    lines.append(f"Max drawdown: {_fmt_money(stats['max_drawdown'])}")
    if skipped_count:
        lines.append(
            f"\n⚠ Skipped {skipped_count} closed trades with missing tag metadata."
        )
    return "\n".join(lines)
