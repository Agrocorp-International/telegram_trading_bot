"""Core strategy logic for the London Breakout FX live runner.

Pure-ish layer between OANDA REST and the Telegram bot. Functions here do not
speak to Telegram; they return dicts or raise exceptions, and the bot layer
handles DMs.

Design invariants:
 1. One buy-stop/sell-stop pair per UTC trading date, intraday, EOD-closed.
 2. Every OANDA order/trade carries tag=LB_TAG so reconcile can distinguish
    strategy-owned objects from anything else on the account.
 3. Local state is cleared only after OANDA confirms zero tagged pending
    orders AND zero tagged open trades.
 4. Startup reconciles local state against OANDA.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

import oanda_client as oc
from config import (
    ACCOUNT_CCY_EXPECTED,
    ATR_PERIOD,
    DOLLARS_PER_PIP,
    EVENTS_JOURNAL_FILE,
    INSTRUMENT,
    TRADE_LOG_FILE,
    LB_ASIAN_END,
    LB_ASIAN_START,
    LB_ENTRY_OFFSET_ATR,
    LB_EOD_HOUR_UTC,
    LB_EOD_MINUTE_UTC,
    LB_MIN_ASIAN_RANGE_PIPS,
    LB_SIGNAL_FIRE_HOUR_UTC,
    LB_SIGNAL_FIRE_MINUTE_UTC,
    LB_SIGNAL_FIRE_SECOND_UTC,
    LB_SL_ATR,
    LB_STATE_FILE,
    LB_TAG,
    LB_TP_ATR,
    MAX_MARGIN_CLOSEOUT_PCT,
    PIP_SIZE,
    RISK_PER_TRADE,
    SPREAD_PIPS,
    UNITS_PER_DOLLAR_PER_PIP,
    build_client_id,
)
from strategy_shared import compute_asian_range

log = logging.getLogger("live_core")


# =========================
# Persistence helpers
# =========================

def _atomic_write_json(path: str, data) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def load_state() -> Optional[dict]:
    if not os.path.exists(LB_STATE_FILE):
        return None
    try:
        with open(LB_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to read {LB_STATE_FILE}: {e}")
        return None


def save_state(state: dict) -> None:
    _atomic_write_json(LB_STATE_FILE, state)


def clear_state() -> None:
    if os.path.exists(LB_STATE_FILE):
        os.remove(LB_STATE_FILE)


def journal(event: str, **payload) -> None:
    line = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with open(EVENTS_JOURNAL_FILE, "a") as f:
        f.write(json.dumps(line, default=str) + "\n")


_TRADE_LOG_COLUMNS = [
    "entry_time", "exit_time", "direction",
    "entry_price", "exit_price",
    "stop_price", "target_price",
    "stop_pips", "result_pips", "outcome",
]


def append_trade_log(
    entry_time, exit_time, direction,
    entry_price, exit_price,
    stop_price, target_price,
    stop_pips, result_pips, outcome,
) -> None:
    """Append one completed trade to the CSV trade log (same columns as backtest output)."""
    row = {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "stop_pips": stop_pips,
        "result_pips": result_pips,
        "outcome": outcome,
    }
    file_exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRADE_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    journal("trade_log_appended", **{k: str(v) for k, v in row.items()})


# =========================
# Time helpers
# =========================

_SGT = timezone(timedelta(hours=8))


def fmt_sgt(dt) -> str:
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_SGT).strftime("%Y-%m-%d %H:%M SGT")


def format_session_schedule() -> str:
    """Human-readable summary of today's UTC schedule with SGT + London conversions.

    SGT is UTC+8 year-round (no DST). London is BST (UTC+1) in summer, GMT (UTC+0)
    in winter; zoneinfo picks whichever is active for today's date.
    """
    today = datetime.now(timezone.utc).date()
    london_tz = ZoneInfo("Europe/London")

    def _row(label: str, hour: int, minute: int = 0, second: int = 0, next_day: bool = False) -> str:
        dt_utc = datetime(today.year, today.month, today.day, hour, minute, second, tzinfo=timezone.utc)
        if next_day:
            dt_utc += timedelta(days=1) * 0  # placeholder; session end already on same UTC date
        sgt_dt = dt_utc.astimezone(_SGT)
        london_dt = dt_utc.astimezone(london_tz)
        # Is the SGT clock on a different date than UTC? (happens for late UTC → SGT next-day)
        sgt_rollover = "(+1 day)" if sgt_dt.date() != dt_utc.date() else ""
        time_fmt = "%H:%M:%S" if second else "%H:%M"
        return (
            f"  {label:22s} {dt_utc.strftime(time_fmt)} UTC   "
            f"{sgt_dt.strftime(time_fmt)} SGT {sgt_rollover}   "
            f"{london_dt.strftime(time_fmt)} {london_dt.tzname()}"
        )

    asian_end_hour_exclusive = LB_ASIAN_END + 1  # LB_ASIAN_END=6 means last bar is 06:00-06:59

    return "\n".join([
        "Daily schedule (Mon–Fri):",
        _row("Asian session start:", LB_ASIAN_START, 0),
        _row("Asian session end:", asian_end_hour_exclusive, 0),
        _row("Signal fire:",
             LB_SIGNAL_FIRE_HOUR_UTC, LB_SIGNAL_FIRE_MINUTE_UTC, LB_SIGNAL_FIRE_SECOND_UTC),
        _row("EOD sweep:", LB_EOD_HOUR_UTC, LB_EOD_MINUTE_UTC),
    ])


def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def _eod_cutoff_for(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, LB_EOD_HOUR_UTC, LB_EOD_MINUTE_UTC, 0, tzinfo=timezone.utc)


def size_units(stop_pips: float) -> int:
    """1%-risk compounding sizing. Fetches live NAV from OANDA."""
    acct = oc.get_account_summary()
    nav = float(acct["NAV"])
    dollars_per_pip = (nav * RISK_PER_TRADE) / stop_pips
    return max(1000, int(dollars_per_pip * UNITS_PER_DOLLAR_PER_PIP))


# =========================
# Signal computation
# =========================

@dataclass
class Signal:
    trading_date: str
    asian_high: float
    asian_low: float
    asian_range_pips: float
    atr_price: float
    atr_pips: float
    stop_pips: float
    target_pips: float
    buy_stop: float
    sell_stop: float
    buy_sl: float
    buy_tp: float
    sell_sl: float
    sell_tp: float
    units: int
    gtd_time_iso: str
    spread_pips: float


class SignalUnavailable(Exception):
    """Raised when Asian range is missing, narrower than min, or ATR NaN."""


def compute_signal(
    candles: pd.DataFrame,
    today: Optional[date] = None,
    spread_pips: Optional[float] = None,
) -> Signal:
    """Compute today's London Breakout signal from H1 candles.

    Expects columns: datetime (UTC-naive), open, high, low, close, complete.
    When `spread_pips` is None, falls back to the static config SPREAD_PIPS
    (preserves backtest parity); live callers pass the OANDA quote spread.
    Raises SignalUnavailable if the Asian window is missing, the range is
    too narrow, or ATR is NaN.
    """
    today = today or _today_utc_date()

    rng = compute_asian_range(
        candles, today,
        asian_start=LB_ASIAN_START, asian_end=LB_ASIAN_END,
        pip_size=PIP_SIZE, atr_period=ATR_PERIOD,
    )
    if rng is None:
        raise SignalUnavailable(
            f"Asian range unavailable for {today.isoformat()} "
            f"(missing bars or ATR NaN — need ≥{ATR_PERIOD} daily bars of history)"
        )

    if rng.asian_range_pips < LB_MIN_ASIAN_RANGE_PIPS:
        raise SignalUnavailable(
            f"Asian range {rng.asian_range_pips:.1f}p < min {LB_MIN_ASIAN_RANGE_PIPS:.0f}p — skip day"
        )

    effective_spread_pips = SPREAD_PIPS if spread_pips is None else spread_pips

    atr_price = rng.atr_price
    atr_pips = atr_price / PIP_SIZE
    offset = atr_price * LB_ENTRY_OFFSET_ATR
    stop_dist = atr_price * LB_SL_ATR
    target_dist = atr_price * LB_TP_ATR
    spread = effective_spread_pips * PIP_SIZE

    buy_stop = rng.asian_high + offset + spread / 2
    sell_stop = rng.asian_low - offset - spread / 2
    buy_sl = buy_stop - stop_dist
    buy_tp = buy_stop + target_dist
    sell_sl = sell_stop + stop_dist
    sell_tp = sell_stop - target_dist

    return Signal(
        trading_date=today.isoformat(),
        asian_high=rng.asian_high,
        asian_low=rng.asian_low,
        asian_range_pips=rng.asian_range_pips,
        atr_price=atr_price,
        atr_pips=atr_pips,
        stop_pips=stop_dist / PIP_SIZE,
        target_pips=target_dist / PIP_SIZE,
        buy_stop=buy_stop,
        sell_stop=sell_stop,
        buy_sl=buy_sl,
        buy_tp=buy_tp,
        sell_sl=sell_sl,
        sell_tp=sell_tp,
        units=size_units(stop_dist / PIP_SIZE),
        gtd_time_iso=_eod_cutoff_for(today).isoformat(),
        spread_pips=effective_spread_pips,
    )


def format_signal(sig: Signal) -> str:
    return (
        f"📊 EUR_USD London Breakout — {sig.trading_date}\n"
        f"  Asian H {sig.asian_high:.5f}  L {sig.asian_low:.5f}  "
        f"range {sig.asian_range_pips:.1f}p   ATR {sig.atr_pips:.1f}p\n"
        f"  buy_stop  {sig.buy_stop:.5f}  SL {sig.buy_sl:.5f}  TP {sig.buy_tp:.5f}\n"
        f"  sell_stop {sig.sell_stop:.5f}  SL {sig.sell_sl:.5f}  TP {sig.sell_tp:.5f}\n"
        f"  stop {sig.stop_pips:.1f}p   target {sig.target_pips:.1f}p   spread {sig.spread_pips:.1f}p\n"
        f"  units {sig.units}   GTD {fmt_sgt(sig.gtd_time_iso)}"
    )


# =========================
# Pre-placement guards
# =========================

class GuardBlocked(Exception):
    pass


def run_pre_placement_guards(today: Optional[date] = None) -> None:
    """Raise GuardBlocked with a descriptive reason if any guard trips."""
    today = today or _today_utc_date()

    state = load_state()
    if state and state.get("trading_date") == today.isoformat():
        raise GuardBlocked(f"⚠ already armed for today: pair_id={state.get('pair_id')}")

    today_cids = {build_client_id(today.isoformat(), s) for s in ("buy", "sell")}
    for o in oc.get_pending_orders_by_tag(LB_TAG):
        if o.get("clientExtensions", {}).get("id") in today_cids:
            raise GuardBlocked(
                f"⚠ OANDA has tagged pending order for today "
                f"(id={o['clientExtensions']['id']}, orderID={o.get('id')}) — restart to reconcile"
            )
    for t in oc.get_open_trades_by_tag(LB_TAG):
        if t.get("clientExtensions", {}).get("id") in today_cids:
            raise GuardBlocked(
                f"⚠ OANDA has tagged open trade for today "
                f"(id={t['clientExtensions']['id']}, tradeID={t.get('id')}) — restart to reconcile"
            )

    acct = oc.get_account_summary()
    ccy = acct.get("currency")
    if ccy != ACCOUNT_CCY_EXPECTED:
        raise GuardBlocked(
            f"⛔ account currency is {ccy}, sizing formula requires {ACCOUNT_CCY_EXPECTED}"
        )
    if INSTRUMENT != "EUR_USD":
        raise GuardBlocked(
            f"⛔ instrument is {INSTRUMENT}, fixed-$/pip formula is EUR_USD only"
        )
    try:
        mc_pct = float(acct.get("marginCloseoutPercent", "0") or 0)
    except ValueError:
        mc_pct = 0.0
    if mc_pct > MAX_MARGIN_CLOSEOUT_PCT:
        raise GuardBlocked(
            f"⛔ marginCloseoutPercent={mc_pct:.2%} > {MAX_MARGIN_CLOSEOUT_PCT:.0%}"
        )


# =========================
# Pair placement
# =========================

class PairPlacementError(Exception):
    pass


def place_entry_pair(sig: Signal) -> dict:
    """Submit long + short stop entries atomically-in-spirit."""
    buy_cid = build_client_id(sig.trading_date, "buy")
    sell_cid = build_client_id(sig.trading_date, "sell")
    pair_id = f"lb_EURUSD_{sig.trading_date}"
    comment = f"asian={sig.asian_range_pips:.1f}p atr={sig.atr_pips:.1f}p"
    gtd_dt = datetime.fromisoformat(sig.gtd_time_iso.replace("Z", "+00:00"))

    long_resp = oc.place_stop_entry(
        tag=LB_TAG, client_id=buy_cid, units=+sig.units,
        entry_price=sig.buy_stop, sl_price=sig.buy_sl, tp_price=sig.buy_tp,
        gtd_time=gtd_dt, comment=comment,
    )
    long_order_id = long_resp.get("orderCreateTransaction", {}).get("id")

    try:
        short_resp = oc.place_stop_entry(
            tag=LB_TAG, client_id=sell_cid, units=-sig.units,
            entry_price=sig.sell_stop, sl_price=sig.sell_sl, tp_price=sig.sell_tp,
            gtd_time=gtd_dt, comment=comment,
        )
    except Exception as e:
        journal("pair_placement_partial_failure",
                trading_date=sig.trading_date, long_order_id=long_order_id, error=str(e))
        try:
            oc.cancel_order(long_order_id)
        except Exception as cancel_err:
            journal("pair_placement_rollback_failed",
                    trading_date=sig.trading_date,
                    long_order_id=long_order_id, error=str(cancel_err))
            raise PairPlacementError(
                f"short submit failed ({e}); long rollback ALSO failed ({cancel_err}) — "
                f"MANUAL INTERVENTION REQUIRED for order {long_order_id}"
            ) from e
        raise PairPlacementError(f"short submit failed, long rolled back: {e}") from e

    short_order_id = short_resp.get("orderCreateTransaction", {}).get("id")

    state = {
        "pair_id": pair_id,
        "trading_date": sig.trading_date,
        "asian_high": sig.asian_high,
        "asian_low": sig.asian_low,
        "asian_range_pips": sig.asian_range_pips,
        "atr_pips": sig.atr_pips,
        "stop_pips": sig.stop_pips,
        "target_pips": sig.target_pips,
        "spread_pips": sig.spread_pips,
        "units": sig.units,
        "buy_order_id": long_order_id,
        "sell_order_id": short_order_id,
        "buy_client_id": buy_cid,
        "sell_client_id": sell_cid,
        "buy_stop": sig.buy_stop,
        "sell_stop": sig.sell_stop,
        "buy_sl": sig.buy_sl,
        "buy_tp": sig.buy_tp,
        "sell_sl": sig.sell_sl,
        "sell_tp": sig.sell_tp,
        "filled_side": None,
        "trade_id": None,
        "status": "armed",
    }
    save_state(state)
    journal("pair_placed", **state)
    return state


# =========================
# OCO watchdog
# =========================

def oco_watchdog_tick() -> Optional[dict]:
    """One poll cycle. Returns an action dict if something happened, else None."""
    state = load_state()
    if not state:
        return None
    if state.get("status") == "done":
        return None

    pending = oc.get_pending_orders_by_tag(LB_TAG)
    pending_cids = {p.get("clientExtensions", {}).get("id") for p in pending}
    trades_list = oc.get_open_trades_by_tag(LB_TAG)
    trades_by_cid = {t.get("clientExtensions", {}).get("id"): t for t in trades_list}

    buy_cid = state["buy_client_id"]
    sell_cid = state["sell_client_id"]

    buy_pending = buy_cid in pending_cids
    sell_pending = sell_cid in pending_cids
    buy_trade = trades_by_cid.get(buy_cid)
    sell_trade = trades_by_cid.get(sell_cid)

    if buy_trade and sell_trade:
        worse = min(
            [buy_trade, sell_trade],
            key=lambda t: float(t.get("unrealizedPL", "0") or 0),
        )
        closed = oc.close_trade(worse["id"])
        state["status"] = "both_filled_one_closed"
        save_state(state)
        journal("both_legs_filled_rare",
                kept_side="short" if worse is buy_trade else "long",
                closed_trade_id=worse["id"], close_response=closed)
        return {"action": "both_filled_closed_worse", "closed_trade_id": worse["id"]}

    if buy_trade and sell_pending:
        resp = oc.cancel_order(state["sell_order_id"])
        fill_price = float(buy_trade.get("price", 0) or 0)
        state.update(filled_side="long", trade_id=buy_trade["id"], status="filled",
                     fill_price=fill_price)
        save_state(state)
        journal("sibling_cancelled", filled_side="long",
                trade_id=buy_trade["id"], fill_price=fill_price,
                cancelled_order_id=state["sell_order_id"], cancel_response=resp)
        return {"action": "cancelled_sibling", "filled_side": "long",
                "trade_id": buy_trade["id"], "fill_price": fill_price,
                "sl_price": state.get("buy_sl"), "tp_price": state.get("buy_tp"),
                "stop_pips": state.get("stop_pips"), "target_pips": state.get("target_pips"),
                "units": state.get("units"),
                "spread_pips": state.get("spread_pips") or SPREAD_PIPS, "pip_size": PIP_SIZE}

    if sell_trade and buy_pending:
        resp = oc.cancel_order(state["buy_order_id"])
        fill_price = float(sell_trade.get("price", 0) or 0)
        state.update(filled_side="short", trade_id=sell_trade["id"], status="filled",
                     fill_price=fill_price)
        save_state(state)
        journal("sibling_cancelled", filled_side="short",
                trade_id=sell_trade["id"], fill_price=fill_price,
                cancelled_order_id=state["buy_order_id"], cancel_response=resp)
        return {"action": "cancelled_sibling", "filled_side": "short",
                "trade_id": sell_trade["id"], "fill_price": fill_price,
                "sl_price": state.get("sell_sl"), "tp_price": state.get("sell_tp"),
                "stop_pips": state.get("stop_pips"), "target_pips": state.get("target_pips"),
                "units": state.get("units"),
                "spread_pips": state.get("spread_pips") or SPREAD_PIPS, "pip_size": PIP_SIZE}

    if state.get("status") == "filled" and not buy_trade and not sell_trade:
        state["status"] = "done"
        trade_id = state.get("trade_id")
        side = state.get("filled_side")
        # Fetch realized PnL + account snapshot for the close DM
        entry_price = exit_price = realized_pl = nav = None
        open_time = close_time = None
        try:
            td = oc.get_trade_details(trade_id)
            entry_price = float(td.get("price", 0) or 0) or None
            exit_price = float(td.get("averageClosePrice", 0) or 0) or None
            realized_pl = float(td.get("realizedPL", 0) or 0)
            open_time = td.get("openTime")
            close_time = td.get("closeTime")
        except Exception as e:
            journal("trade_closed_details_fetch_failed", trade_id=trade_id, error=str(e))
        try:
            acct = oc.get_account_summary()
            nav = float(acct.get("NAV", 0) or 0)
        except Exception as e:
            journal("trade_closed_nav_fetch_failed", trade_id=trade_id, error=str(e))

        pips = None
        if entry_price and exit_price:
            pips = ((exit_price - entry_price) if side == "long"
                    else (entry_price - exit_price)) / PIP_SIZE

        save_state(state)
        journal("trade_closed_via_sl_tp",
                filled_side=side, trade_id=trade_id,
                entry_price=entry_price, exit_price=exit_price,
                pips=pips, realized_pl=realized_pl, nav=nav)

        try:
            sl_key = "buy_sl" if side == "long" else "sell_sl"
            tp_key = "buy_tp" if side == "long" else "sell_tp"
            outcome = "target" if (pips is not None and pips > 0) else "stop"
            append_trade_log(
                entry_time=open_time, exit_time=close_time,
                direction=side, entry_price=entry_price, exit_price=exit_price,
                stop_price=state.get(sl_key), target_price=state.get(tp_key),
                stop_pips=state.get("stop_pips"), result_pips=pips, outcome=outcome,
            )
        except Exception as e:
            journal("trade_log_error", trade_id=trade_id, error=str(e))

        return {"action": "trade_closed_sl_tp", "trade_id": trade_id,
                "filled_side": side, "entry_price": entry_price,
                "exit_price": exit_price, "pips": pips,
                "realized_pl": realized_pl, "nav": nav}

    if state.get("status") == "filled" and state.get("filled_side") == "long" and sell_pending:
        resp = oc.cancel_order(state["sell_order_id"])
        journal("sibling_cancel_retry", side="sell",
                order_id=state["sell_order_id"], response=resp)
        return {"action": "retry_sibling_cancel", "side": "sell"}
    if state.get("status") == "filled" and state.get("filled_side") == "short" and buy_pending:
        resp = oc.cancel_order(state["buy_order_id"])
        journal("sibling_cancel_retry", side="buy",
                order_id=state["buy_order_id"], response=resp)
        return {"action": "retry_sibling_cancel", "side": "buy"}

    return None


# =========================
# EOD sweep
# =========================

class EodNotFlat(Exception):
    pass


def eod_sweep(max_close_retries: int = 3) -> dict:
    """STEP A→E: cancel pending → verify → close open trades → verify → clear state."""
    summary = {"cancelled_orders": [], "closed_trades": [], "errors": []}

    state = load_state()
    today_cids = set()
    if state:
        today_cids = {state.get("buy_client_id"), state.get("sell_client_id")}

    for o in oc.get_pending_orders_by_tag(LB_TAG):
        cid = o.get("clientExtensions", {}).get("id")
        if today_cids and cid not in today_cids:
            continue
        try:
            resp = oc.cancel_order(o["id"])
            summary["cancelled_orders"].append({"order_id": o["id"], "client_id": cid})
            journal("eod_cancel", order_id=o["id"], client_id=cid, response=resp)
        except Exception as e:
            summary["errors"].append({"stage": "cancel", "order_id": o["id"], "error": str(e)})

    remaining = [o for o in oc.get_pending_orders_by_tag(LB_TAG)
                 if (not today_cids) or o.get("clientExtensions", {}).get("id") in today_cids]
    if remaining:
        summary["errors"].append({
            "stage": "verify_cancel",
            "surviving_orders": [o["id"] for o in remaining],
        })
        journal("eod_cancel_incomplete", surviving=[o["id"] for o in remaining])
        raise EodNotFlat(f"{len(remaining)} tagged pending orders did not cancel")

    for attempt in range(1, max_close_retries + 1):
        trades_list = oc.get_open_trades_by_tag(LB_TAG)
        if today_cids:
            trades_list = [t for t in trades_list
                           if t.get("clientExtensions", {}).get("id") in today_cids]
        if not trades_list:
            break
        for t in trades_list:
            try:
                resp = oc.close_trade(t["id"])
                journal("eod_close", trade_id=t["id"], attempt=attempt, response=resp)
                # Build close detail from the close_trade response itself. Calling
                # /trades/{id} here races against OANDA's eviction (practice server
                # returns 404 within ~230ms of close), so we parse the response.
                closed_record = {"trade_id": t["id"]}
                try:
                    fill_tx = (resp or {}).get("orderFillTransaction") or {}
                    closes = fill_tx.get("tradesClosed") or [{}]
                    x_price = float(fill_tx.get("price", 0) or 0) or None
                    realized_pl = float(closes[0].get("realizedPL", 0) or 0)
                    side = state.get("filled_side") if state else None
                    e_price = state.get("fill_price") if state else None
                    pips = None
                    if e_price and x_price and side:
                        pips = ((x_price - e_price) if side == "long"
                                else (e_price - x_price)) / PIP_SIZE
                    closed_record.update(
                        filled_side=side,
                        entry_price=e_price,
                        exit_price=x_price,
                        pips=pips,
                        realized_pl=realized_pl,
                    )
                    # Log to trade CSV on first successful close (attempt == 1)
                    if attempt == 1:
                        sl_key = "buy_sl" if side == "long" else "sell_sl"
                        tp_key = "buy_tp" if side == "long" else "sell_tp"
                        append_trade_log(
                            entry_time=None, exit_time=fill_tx.get("time"),
                            direction=side, entry_price=e_price, exit_price=x_price,
                            stop_price=state.get(sl_key) if state else None,
                            target_price=state.get(tp_key) if state else None,
                            stop_pips=state.get("stop_pips") if state else None,
                            result_pips=pips, outcome="forced_close_eod",
                        )
                except Exception as log_err:
                    journal("trade_log_eod_error", trade_id=t["id"], error=str(log_err))
                summary["closed_trades"].append(closed_record)
            except Exception as e:
                summary["errors"].append({
                    "stage": "close", "trade_id": t["id"], "attempt": attempt, "error": str(e),
                })
                journal("eod_close_error", trade_id=t["id"], attempt=attempt, error=str(e))
        time.sleep(1.0)

    remaining_trades = oc.get_open_trades_by_tag(LB_TAG)
    if today_cids:
        remaining_trades = [t for t in remaining_trades
                            if t.get("clientExtensions", {}).get("id") in today_cids]
    if remaining_trades:
        summary["errors"].append({
            "stage": "verify_close",
            "surviving_trades": [t["id"] for t in remaining_trades],
        })
        journal("eod_not_flat_escalation",
                surviving_trades=[t["id"] for t in remaining_trades])
        raise EodNotFlat(f"{len(remaining_trades)} tagged open trades did not close")

    # Populate summary-level P&L fields from the first closed trade (LB only ever
    # has one open trade at a time). Summary aggregates the per-trade records; we
    # no longer call /trades/{id} here — it races with OANDA's eviction and 404s.
    if summary["closed_trades"]:
        first = summary["closed_trades"][0]
        summary["realized_pl"] = first.get("realized_pl")
        summary["result_pips"] = first.get("pips")
        summary["filled_side"] = first.get("filled_side")

    clear_state()
    journal("eod_completed", summary=summary)
    return summary


# =========================
# Startup reconciliation
# =========================

def startup_reconcile() -> dict:
    """Reconcile local state with OANDA on boot."""
    state = load_state()
    today = _today_utc_date().isoformat()
    pending = oc.get_pending_orders_by_tag(LB_TAG)
    trades_list = oc.get_open_trades_by_tag(LB_TAG)

    relevant_pending = [o for o in pending
                        if o.get("clientExtensions", {}).get("id", "").startswith(f"lb_EURUSD_{today}_")]
    relevant_trades = [t for t in trades_list
                       if t.get("clientExtensions", {}).get("id", "").startswith(f"lb_EURUSD_{today}_")]

    if state is None and not relevant_pending and not relevant_trades:
        journal("startup_reconcile", outcome="clean_slate")
        return {"action": "clean_slate"}

    if state is None and (relevant_pending or relevant_trades):
        rebuilt = _rebuild_state_from_oanda(today, relevant_pending, relevant_trades)
        save_state(rebuilt)
        journal("startup_reconcile", outcome="rebuilt_from_oanda", state=rebuilt)
        action = {"action": "rebuilt", "state": rebuilt}
    elif state is not None and not relevant_pending and not relevant_trades:
        journal("startup_reconcile", outcome="archived_stale_state", state=state)
        clear_state()
        action = {"action": "archived_stale_state", "was_state": state}
    else:
        journal("startup_reconcile", outcome="in_sync",
                pair_id=state.get("pair_id") if state else None,
                pending_count=len(relevant_pending), trade_count=len(relevant_trades))
        action = {"action": "in_sync"}

    tick = oco_watchdog_tick()
    if tick:
        action["watchdog_tick"] = tick
    return action


def _rebuild_state_from_oanda(today: str, pending: list, trades_list: list) -> dict:
    buy_cid = build_client_id(today, "buy")
    sell_cid = build_client_id(today, "sell")

    def find(items, cid):
        return next((o for o in items if o.get("clientExtensions", {}).get("id") == cid), None)

    buy_order, sell_order = find(pending, buy_cid), find(pending, sell_cid)
    buy_trade, sell_trade = find(trades_list, buy_cid), find(trades_list, sell_cid)

    filled_side, trade_id, status = None, None, "armed"
    if buy_trade and not sell_trade:
        filled_side, trade_id, status = "long", buy_trade["id"], "filled"
    elif sell_trade and not buy_trade:
        filled_side, trade_id, status = "short", sell_trade["id"], "filled"
    elif buy_trade and sell_trade:
        status = "both_filled"

    def _px(o):
        if o is None:
            return None
        return float(o.get("price", 0) or 0)

    units = (abs(int(buy_order["units"])) if buy_order
             else (abs(int(buy_trade["initialUnits"])) if buy_trade
                   else int(DOLLARS_PER_PIP * UNITS_PER_DOLLAR_PER_PIP)))

    return {
        "pair_id": f"lb_EURUSD_{today}",
        "trading_date": today,
        "asian_high": None, "asian_low": None, "asian_range_pips": None,
        "atr_pips": None, "stop_pips": None, "target_pips": None,
        "spread_pips": None,
        "units": units,
        "buy_order_id": buy_order["id"] if buy_order else None,
        "sell_order_id": sell_order["id"] if sell_order else None,
        "buy_client_id": buy_cid,
        "sell_client_id": sell_cid,
        "buy_stop": _px(buy_order), "sell_stop": _px(sell_order),
        "buy_sl": None, "buy_tp": None, "sell_sl": None, "sell_tp": None,
        "filled_side": filled_side, "trade_id": trade_id,
        "status": status, "reconciled": True,
    }
