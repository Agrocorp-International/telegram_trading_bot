"""Telegram bot + scheduler for the London Breakout FX live runner.

Run: python 50_pips_eurusd/telegram_bot.py from the project root. Requires .env
with OANDA + Telegram vars (see .env.example).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from functools import wraps
from typing import Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from io import BytesIO

import requests

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from pnl_report import (
    build_trades_df,
    compute_stats,
    filter_trades_since,
    format_stats_caption,
    render_pnl_chart,
)

import oanda_client as oc
from config import (
    BOT_LOG_FILE,
    DOLLARS_PER_PIP,
    EVENTS_JOURNAL_FILE,
    LB_EOD_HOUR_UTC,
    LB_EOD_MINUTE_UTC,
    LB_HOURLY_CHECK_MINUTE_UTC,
    LB_MAX_SPREAD_PIPS,
    LB_SIGNAL_FIRE_HOUR_UTC,
    LB_SIGNAL_FIRE_MINUTE_UTC,
    LB_SIGNAL_FIRE_SECOND_UTC,
    LB_TAG,
    MODE_TAG,
    PAUSE_FLAG_FILE,
    PENDING_CONFIRMATIONS_FILE,
    PLACE_ORDERS,
    PNL_SINCE_FILE,
    PROPOSAL_TTL_SECONDS,
    SPREAD_PIPS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_USER_ID,
    WATCHDOG_SECONDS,
)
from live_core import (
    EodNotFlat,
    GuardBlocked,
    PairPlacementError,
    Signal,
    SignalUnavailable,
    clear_state,
    compute_signal,
    eod_sweep,
    fmt_sgt,
    format_session_schedule,
    format_signal,
    hourly_dual_trigger_check,
    journal,
    load_state,
    oco_watchdog_tick,
    place_entry_pair,
    run_pre_placement_guards,
    startup_reconcile,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(BOT_LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("telegram_bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

HEARTBEAT_URL: str = os.environ.get("HEARTBEAT_URL", "").strip()


# =========================
# Pause flag
# =========================

def is_paused() -> bool:
    return os.path.exists(PAUSE_FLAG_FILE)


def set_paused(paused: bool) -> None:
    if paused:
        with open(PAUSE_FLAG_FILE, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    elif os.path.exists(PAUSE_FLAG_FILE):
        os.remove(PAUSE_FLAG_FILE)


# =========================
# Pending confirmations
# =========================

def _load_pending() -> Dict[str, dict]:
    if not os.path.exists(PENDING_CONFIRMATIONS_FILE):
        return {}
    try:
        with open(PENDING_CONFIRMATIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_pending(data: Dict[str, dict]) -> None:
    tmp = f"{PENDING_CONFIRMATIONS_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PENDING_CONFIRMATIONS_FILE)


def _put_pending(token: str, kind: str, payload: dict) -> None:
    data = _load_pending()
    data[token] = {"kind": kind, "created_at": time.time(), "payload": payload}
    _save_pending(data)


def _pop_pending(token: str) -> Optional[dict]:
    data = _load_pending()
    entry = data.pop(token, None)
    _save_pending(data)
    return entry


def _prune_pending() -> int:
    data = _load_pending()
    cutoff = time.time() - PROPOSAL_TTL_SECONDS
    expired = [t for t, e in data.items() if e.get("created_at", 0) < cutoff]
    for t in expired:
        data.pop(t, None)
    if expired:
        _save_pending(data)
    return len(expired)


def _confirm_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Confirm", callback_data=f"cfm:{token}"),
        InlineKeyboardButton("❌ Cancel", callback_data=f"cxl:{token}"),
    ]])


# =========================
# /pnl cutoff marker
# =========================

def _load_pnl_since() -> Optional[datetime]:
    if not os.path.exists(PNL_SINCE_FILE):
        return None
    try:
        with open(PNL_SINCE_FILE, "r") as f:
            data = json.load(f)
        return datetime.fromisoformat(data["since_utc"])
    except Exception:
        return None


def _save_pnl_since(since: datetime) -> None:
    data = {
        "since_utc": since.isoformat(),
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = f"{PNL_SINCE_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PNL_SINCE_FILE)


def _clear_pnl_since() -> bool:
    if os.path.exists(PNL_SINCE_FILE):
        os.remove(PNL_SINCE_FILE)
        return True
    return False


# =========================
# Auth
# =========================

def auth_required(handler):
    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user is None or user.id != TELEGRAM_USER_ID:
            log.warning(
                f"Unauthorized access user_id={getattr(user, 'id', None)} "
                f"username={getattr(user, 'username', None)}"
            )
            return
        return await handler(update, context)
    return wrapper


async def _with_lock(context: ContextTypes.DEFAULT_TYPE, blocking_fn, *args, **kwargs):
    return await _with_lock_app(context.application, blocking_fn, *args, **kwargs)


async def _with_lock_app(application: Application, blocking_fn, *args, **kwargs):
    lock: asyncio.Lock = application.bot_data["oanda_lock"]
    async with lock:
        return await asyncio.to_thread(blocking_fn, *args, **kwargs)


async def safe_send(application: Application, text: str, *, context: str = "") -> bool:
    """Send a Telegram DM that never raises. Logs + journals on failure so a
    broken send can't silently kill the calling async task."""
    try:
        await application.bot.send_message(TELEGRAM_USER_ID, text)
        return True
    except Exception as e:
        log.exception(f"telegram send failed ({context or 'unspecified'}): {e}")
        try:
            journal("telegram_send_failed", context=context, error=str(e),
                    preview=text[:200])
        except Exception:
            pass
        return False


# =========================
# Blocking helpers
# =========================

def _fetch_live_spread_pips() -> tuple[float, Optional[str]]:
    """Return (spread_pips, fallback_warning).

    Hard-skip on malformed response / not-tradeable (raises SignalUnavailable).
    Soft-fallback to static SPREAD_PIPS on transient V20/network errors and
    return a warning string for the async caller to DM.
    """
    from oandapyV20.exceptions import V20Error
    try:
        quote = oc.get_current_price()
    except oc.PricingUnavailable as e:
        raise SignalUnavailable(f"Pricing response malformed: {e}")
    except (V20Error, ConnectionError, TimeoutError, OSError) as e:
        warn = f"Live pricing unavailable ({type(e).__name__}: {e}) — using static {SPREAD_PIPS}p"
        log.warning(warn)
        journal("pricing_transient_failure", error=str(e), fallback_spread_pips=SPREAD_PIPS)
        return SPREAD_PIPS, warn

    if not quote.tradeable:
        raise SignalUnavailable(
            f"EUR_USD not tradeable (bid={quote.bid:.5f} ask={quote.ask:.5f})"
        )
    if quote.spread_pips > LB_MAX_SPREAD_PIPS:
        raise SignalUnavailable(
            f"Spread {quote.spread_pips:.1f}p > cap {LB_MAX_SPREAD_PIPS:.1f}p — skip day"
        )
    journal("pricing_live",
            bid=quote.bid, ask=quote.ask,
            spread_pips=quote.spread_pips, time=quote.time)
    return quote.spread_pips, None


def _fetch_candles_and_signal() -> tuple["Signal", Optional[str]]:
    candles = oc.get_h1_candles()
    spread_pips, warn = _fetch_live_spread_pips()
    return compute_signal(candles, spread_pips=spread_pips), warn


def _signal_job_blocking() -> dict:
    if is_paused():
        journal("signal_skipped_pause")
        return {"outcome": "paused"}
    try:
        run_pre_placement_guards()
    except GuardBlocked as e:
        journal("guard_blocked", reason=str(e))
        return {"outcome": "guard_blocked", "reason": str(e)}
    try:
        sig, pricing_warn = _fetch_candles_and_signal()
    except SignalUnavailable as e:
        journal("signal_unavailable", reason=str(e))
        return {"outcome": "signal_unavailable", "reason": str(e)}
    journal("signal_computed",
            trading_date=sig.trading_date,
            asian_range_pips=sig.asian_range_pips, atr_pips=sig.atr_pips,
            buy_stop=sig.buy_stop, sell_stop=sig.sell_stop,
            stop_pips=sig.stop_pips, target_pips=sig.target_pips,
            spread_pips=sig.spread_pips, units=sig.units)
    if not PLACE_ORDERS:
        return {"outcome": "dry_run", "signal": sig, "pricing_warn": pricing_warn}
    try:
        state = place_entry_pair(sig)
    except PairPlacementError as e:
        return {"outcome": "pair_error", "reason": str(e), "signal": sig,
                "pricing_warn": pricing_warn}
    except Exception as e:
        journal("pair_placement_exception", error=str(e))
        return {"outcome": "pair_error", "reason": f"unexpected: {e}", "signal": sig,
                "pricing_warn": pricing_warn}
    return {"outcome": "placed", "signal": sig, "state": state,
            "pricing_warn": pricing_warn}


def _eod_blocking() -> dict:
    try:
        summary = eod_sweep()
        return {"outcome": "flat", "summary": summary}
    except EodNotFlat as e:
        return {"outcome": "not_flat", "reason": str(e)}
    except Exception as e:
        journal("eod_exception", error=str(e))
        return {"outcome": "error", "reason": str(e)}


def _status_blocking() -> dict:
    state = load_state()
    try:
        acct = oc.get_account_summary()
        reachable = True
    except Exception as e:
        acct = {"error": str(e)}
        reachable = False
    try:
        pending = oc.get_pending_orders_by_tag(LB_TAG)
        trades_list = oc.get_open_trades_by_tag(LB_TAG)
    except Exception:
        pending, trades_list = [], []
    return {
        "reachable": reachable,
        "account": acct,
        "pending_count": len(pending),
        "trades_count": len(trades_list),
        "state": state,
        "paused": is_paused(),
    }


def _cancel_strategy_orders_blocking() -> dict:
    pending = oc.get_pending_orders_by_tag(LB_TAG)
    cancelled, errors = [], []
    for o in pending:
        try:
            oc.cancel_order(o["id"])
            cancelled.append(o["id"])
        except Exception as e:
            errors.append({"order_id": o["id"], "error": str(e)})
    journal("user_cancel_strategy_orders", cancelled=cancelled, errors=errors)
    return {"cancelled": cancelled, "errors": errors}


def _close_strategy_trade_blocking() -> dict:
    trades_list = oc.get_open_trades_by_tag(LB_TAG)
    closed, errors = [], []
    for t in trades_list:
        try:
            resp = oc.close_trade(t["id"])
            closed.append({"trade_id": t["id"], "response": resp})
        except Exception as e:
            errors.append({"trade_id": t["id"], "error": str(e)})
    journal("user_close_strategy_trade",
            closed=[c["trade_id"] for c in closed], errors=errors)
    if not oc.get_open_trades_by_tag(LB_TAG) and not oc.get_pending_orders_by_tag(LB_TAG):
        clear_state()
    return {"closed": closed, "errors": errors}



def _daily_summary_blocking() -> dict:
    """Collect a snapshot of LB strategy for the 04:00 SGT (20:00 UTC) summary DM."""
    result: dict = {}

    # Account
    try:
        acct = oc.get_account_summary()
        result["nav"] = float(acct.get("NAV", 0))
        result["balance"] = float(acct.get("balance", 0))
        result["margin_pct"] = float(acct.get("marginCloseoutPercent") or 0)
        result["ccy"] = acct.get("currency", "USD")
    except Exception as e:
        result["acct_err"] = str(e)

    # LB section
    try:
        result["lb_state"] = load_state()
        result["lb_trades"] = oc.get_open_trades_by_tag(LB_TAG)
        result["lb_orders"] = oc.get_pending_orders_by_tag(LB_TAG)
    except Exception as e:
        result["lb_err"] = str(e)
        result.setdefault("lb_trades", [])
        result.setdefault("lb_orders", [])
    result["lb_paused"] = is_paused()

    return result


def _format_daily_summary(d: dict) -> str:
    _SGT = timezone(timedelta(hours=8))
    now_sgt = datetime.now(tz=_SGT)
    lines = [f"📊 Daily Summary — {now_sgt.strftime('%H:%M SGT (%Y-%m-%d)')}"]

    if "acct_err" in d:
        lines.append(f"Account: ❌ {d['acct_err']}")
    else:
        lines.append(
            f"Account: {d.get('ccy', 'USD')} "
            f"${d.get('nav', 0):,.2f} NAV  |  "
            f"margin {d.get('margin_pct', 0):.4f}%"
        )

    # LB
    lb_pause_tag = "  ⏸ paused" if d.get("lb_paused") else ""
    lines.append(f"\nLondon Breakout (lb_eurusd){lb_pause_tag}:")
    if "lb_err" in d:
        lines.append(f"  ❌ {d['lb_err']}")
    else:
        n_trades = len(d.get("lb_trades", []))
        n_orders = len(d.get("lb_orders", []))
        if n_trades == 0 and n_orders == 0:
            lines.append("  FLAT — no open trades or orders")
        else:
            parts = []
            if n_trades:
                parts.append(f"{n_trades} trade{'s' if n_trades > 1 else ''} open")
            if n_orders:
                parts.append(f"{n_orders} order{'s' if n_orders > 1 else ''} pending")
            lines.append("  " + " | ".join(parts))

    return "\n".join(lines)


# =========================
# Handlers
# =========================

@auth_required
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"London Breakout FX bot online ({MODE_TAG}, PLACE_ORDERS={PLACE_ORDERS}).\n"
        "  📊 Daily summary DM fires at 04:00 SGT (20:00 UTC) Mon–Fri\n"
        "\nLondon Breakout:\n"
        "  /status — OANDA + LB state\n"
        "  /signal — dry-run today's LB signal\n"
        "  /run_now — manually fire LB signal_job\n"
        "  /pause, /resume — block/unblock LB job\n"
        "  /positions — LB tagged open trades\n"
        "  /orders — LB tagged pending orders\n"
        "  /pnl — LB P&L chart + stats\n"
        "  /pnl_reset — hide trades closed before now (confirm-gated)\n"
        "  /pnl_reset_clear — restore full /pnl history\n"
        "  /cancel_strategy_orders — confirm-gated\n"
        "  /close_strategy_trade — confirm-gated\n"
        "  /demo_trade — preview DM flow (no orders)\n"
        "  /test_trade — real round-trip smoke test\n"
        "\n  /help"
    )


@auth_required
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


@auth_required
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching status…")
    try:
        s = await _with_lock(context, _status_blocking)
    except Exception as e:
        await update.message.reply_text(f"❌ status error: {e}")
        return
    acct = s["account"]
    if s["reachable"]:
        acct_line = (
            f"ccy={acct.get('currency')}  NAV=${float(acct.get('NAV', 0)):.2f}  "
            f"margin_close={float(acct.get('marginCloseoutPercent') or 0):.4f}"
        )
    else:
        acct_line = f"❌ unreachable — {acct.get('error')}"
    state = s["state"]
    state_line = (
        f"pair_id={state['pair_id']} status={state['status']} "
        f"filled={state.get('filled_side')}"
    ) if state else "none"
    next_run_raw = context.application.bot_data.get("next_signal_run", "unknown")
    try:
        next_run_display = fmt_sgt(datetime.fromisoformat(str(next_run_raw)))
    except Exception:
        next_run_display = str(next_run_raw)
    await update.message.reply_text(
        f"Bot: ok  mode={MODE_TAG}  PLACE_ORDERS={PLACE_ORDERS}  paused={s['paused']}\n"
        f"OANDA: {acct_line}\n"
        f"Tagged pending: {s['pending_count']}  Tagged open trades: {s['trades_count']}\n"
        f"Local state: {state_line}\n"
        f"Next signal_job: {next_run_display}"
    )


@auth_required
async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching candles…")
    try:
        sig, pricing_warn = await _with_lock(context, _fetch_candles_and_signal)
    except SignalUnavailable as e:
        await update.message.reply_text(f"📭 {e}")
        return
    except Exception as e:
        await update.message.reply_text(f"❌ OANDA/compute error: {e}")
        return
    if pricing_warn:
        await update.message.reply_text(f"⚠️ {pricing_warn}")
    await update.message.reply_text(format_signal(sig))


@auth_required
async def cmd_run_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Running signal_job now…")
    try:
        result = await _with_lock(context, _signal_job_blocking)
    except Exception as e:
        await update.message.reply_text(f"❌ run_now error: {e}")
        return
    warn = result.get("pricing_warn")
    if warn:
        await update.message.reply_text(f"⚠️ {warn}")
    await update.message.reply_text(_format_signal_outcome(result))


@auth_required
async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_paused(True)
    await update.message.reply_text("⏸ PAUSE flag set — signal_job will skip.")


@auth_required
async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_paused(False)
    await update.message.reply_text("▶️ PAUSE flag cleared.")


@auth_required
async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fetch LB-tagged closed trades, render chart + stats, send to Telegram."""
    await update.message.reply_text("📊 Crunching P&L…")

    # Step 1: fetch under OANDA lock (blocking network call)
    def _fetch_blocking():
        return oc.list_closed_trades_by_tag(LB_TAG)

    try:
        raw_trades, skipped = await _with_lock(context, _fetch_blocking)
    except Exception as e:
        await update.message.reply_text(f"❌ /pnl fetch error: {e}")
        return

    # Step 2: compute + render OUTSIDE the lock (pure, no API calls)
    df = build_trades_df(raw_trades)
    since = _load_pnl_since()
    if since is not None:
        df = filter_trades_since(df, since)
    if df.empty:
        msg = "📭 No LB-tagged closed trades yet."
        if since is not None:
            msg = (
                f"📭 No LB-tagged closed trades since "
                f"{since.strftime('%Y-%m-%d %H:%M')} UTC."
            )
        if skipped:
            msg += f"\n  (skipped {skipped} closed trades with missing tag metadata)"
        await update.message.reply_text(msg)
        return

    stats = compute_stats(df)
    try:
        png = await asyncio.to_thread(render_pnl_chart, df)
    except Exception as e:
        await update.message.reply_text(f"❌ /pnl render error: {e}")
        return

    caption = format_stats_caption(stats, skipped, label="LB", since=since)
    await context.bot.send_photo(
        chat_id=TELEGRAM_USER_ID,
        photo=InputFile(BytesIO(png), filename="lb_pnl.png"),
        caption=caption,
    )


@auth_required
async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        trades_list = await _with_lock(context, oc.get_open_trades_by_tag, LB_TAG)
    except Exception as e:
        await update.message.reply_text(f"❌ OANDA error: {e}")
        return
    if not trades_list:
        await update.message.reply_text("📭 No tagged open trades.")
        return
    lines = []
    for t in trades_list:
        lines.append(
            f"• trade {t.get('id')}  {t.get('instrument')}  "
            f"units {t.get('currentUnits')}  avgPx {t.get('price')}  "
            f"uPL {t.get('unrealizedPL')}"
        )
    await update.message.reply_text("Tagged open trades:\n" + "\n".join(lines))


@auth_required
async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        pending = await _with_lock(context, oc.get_pending_orders_by_tag, LB_TAG)
    except Exception as e:
        await update.message.reply_text(f"❌ OANDA error: {e}")
        return
    if not pending:
        await update.message.reply_text("📭 No tagged pending orders.")
        return
    lines = []
    for o in pending:
        lines.append(
            f"• order {o.get('id')}  {o.get('instrument')}  "
            f"{o.get('type')} {o.get('units')}  @ {o.get('price')}  "
            f"GTD {o.get('gtdTime')}  tag={o.get('clientExtensions', {}).get('tag')}"
        )
    await update.message.reply_text("Tagged pending orders:\n" + "\n".join(lines))


@auth_required
async def cmd_test_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Confirm-gated: place a REAL market order, hold 20s, close at market."""
    if not PLACE_ORDERS:
        await update.message.reply_text(
            "❌ PLACE_ORDERS=false in .env. Flip it to true and restart the bot to run a real test."
        )
        return
    token = uuid.uuid4().hex[:8]
    _put_pending(token, "test_trade", {})
    await update.message.reply_text(
        f"⚠️ REAL TEST TRADE on OANDA ({MODE_TAG})\n"
        f"Will place MARKET BUY 10,000 EUR_USD units (tag='test_trade'), "
        f"hold 20 seconds, then close at market.\n"
        f"Uses a separate tag from the strategy — won't touch LB state.\n\n"
        f"Confirm?",
        reply_markup=_confirm_keyboard(token),
    )


async def _execute_test_trade_flow(application: Application):
    """Place → sleep 20s → close. Uses tag='test_trade' (isolated from LB)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cid = f"test_trade_{ts}"

    def _place_blocking():
        resp = oc.place_market_order(
            tag="test_trade", client_id=cid, units=10000,
            comment=f"bot smoke test {ts}",
        )
        fill_tx = resp.get("orderFillTransaction", {}) or {}
        trade_opened = fill_tx.get("tradeOpened") or {}
        return {
            "fill_price": float(fill_tx.get("price", 0) or 0),
            "trade_id": trade_opened.get("tradeID"),
            "client_id": cid,
        }

    try:
        placed = await _with_lock_app(application, _place_blocking)
    except Exception as e:
        await application.bot.send_message(TELEGRAM_USER_ID, f"❌ Test place failed: {e}")
        return

    if not placed.get("trade_id"):
        await application.bot.send_message(
            TELEGRAM_USER_ID,
            f"⚠ Market order placed but no tradeID returned. Response: {placed}"
        )
        return

    await application.bot.send_message(
        TELEGRAM_USER_ID,
        f"🧪 TEST: placed MARKET BUY 10,000 units @ {placed['fill_price']:.5f}  "
        f"tradeID={placed['trade_id']}\n"
        f"   client_id={placed['client_id']}\n"
        f"   Holding 20s, will close at market..."
    )

    await asyncio.sleep(20)

    def _close_blocking():
        oc.close_trade(placed["trade_id"])
        td = oc.get_trade_details(placed["trade_id"])
        acct = oc.get_account_summary()
        return {"td": td, "acct": acct}

    try:
        closed = await _with_lock_app(application, _close_blocking)
    except Exception as e:
        await application.bot.send_message(
            TELEGRAM_USER_ID,
            f"❌ Test close failed: {e}\n"
            f"   tradeID={placed['trade_id']} may still be open — check OANDA."
        )
        return

    td = closed["td"]
    acct = closed["acct"]
    entry = float(td.get("price", 0) or 0)
    exit_px = float(td.get("averageClosePrice", 0) or 0)
    pnl = float(td.get("realizedPL", 0) or 0)
    nav = float(acct.get("NAV", 0) or 0)
    pips = (exit_px - entry) / 0.0001 if entry and exit_px else 0.0

    await application.bot.send_message(
        TELEGRAM_USER_ID,
        f"🧪 TEST: closed  tradeID={placed['trade_id']}\n"
        f"   Entry {entry:.5f} → Exit {exit_px:.5f}   {pips:+.1f}p\n"
        f"   Realized PnL: ${pnl:+,.2f}   |   Account NAV: ${nav:,.2f}\n"
        f"   ✅ Full place→close round-trip succeeded."
    )


@auth_required
async def cmd_demo_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Walk through a simulated trade lifecycle with fake data — no OANDA calls."""
    today = date.today().isoformat()

    fake_sig = Signal(
        trading_date=today,
        asian_high=1.08800,
        asian_low=1.08550,
        asian_range_pips=25.0,
        atr_price=0.0060,
        atr_pips=60.0,
        stop_pips=12.0,
        target_pips=60.0,
        buy_stop=1.08807,
        sell_stop=1.08544,
        buy_sl=1.08687,
        buy_tp=1.09407,
        sell_sl=1.08664,
        sell_tp=1.07944,
        units=10000,
        gtd_time_iso=f"{today}T20:00:00+00:00",
        spread_pips=SPREAD_PIPS,
    )
    fake_state = {
        "pair_id": f"lb_EURUSD_{today}",
        "trading_date": today,
        "buy_order_id": "DEMO-BUY-12345",
        "sell_order_id": "DEMO-SELL-12346",
    }

    await update.message.reply_text(
        "🎬 Demo trade flow starting — 4 DMs over ~5s. "
        "No real orders are placed."
    )

    # 1. Pair placed (07:00:15 UTC in a real day)
    placed_msg = _format_signal_outcome({"outcome": "placed", "signal": fake_sig, "state": fake_state})
    await context.bot.send_message(TELEGRAM_USER_ID, f"[DEMO]\n{placed_msg}")
    await asyncio.sleep(2)

    # 2. Fill detected — long leg triggers, sibling cancelled
    _demo_dpp = fake_sig.units / 10_000
    _demo_max_loss = fake_sig.stop_pips * _demo_dpp
    _demo_max_profit = fake_sig.target_pips * _demo_dpp
    await context.bot.send_message(
        TELEGRAM_USER_ID,
        f"[DEMO]\n🔔 LB fill (long) @ {fake_sig.buy_stop:.5f}  tradeID=DEMO-TRADE-99999\n"
        f"   Spread: {SPREAD_PIPS:.1f}p charged (baked into entry price)\n"
        f"   SL {fake_sig.buy_sl:.5f}  TP {fake_sig.buy_tp:.5f}\n"
        f"   Max loss ${_demo_max_loss:,.2f}  Max profit ${_demo_max_profit:,.2f}\n"
        f"   Sibling order cancelled."
    )
    await asyncio.sleep(2)

    # 3. Trade closed — TP hit with PnL + NAV
    close_action = {
        "action": "trade_closed_sl_tp",
        "trade_id": "DEMO-TRADE-99999",
        "filled_side": "long",
        "entry_price": fake_sig.buy_stop,
        "exit_price": fake_sig.buy_tp,
        "pips": (fake_sig.buy_tp - fake_sig.buy_stop) / 0.0001,
        "realized_pl": 60.00,           # 60 pips * $1/pip
        "nav": 10060.00,                # assume starting NAV 10000
    }
    await context.bot.send_message(TELEGRAM_USER_ID, "[DEMO]\n" + _format_close_dm(close_action))
    await asyncio.sleep(1)

    # 4. EOD summary (20:00 UTC — nothing left to clean up)
    await context.bot.send_message(
        TELEGRAM_USER_ID,
        "[DEMO]\n✅ EOD flat. cancelled_orders=0 closed_trades=1 errors=0"
    )

    await update.message.reply_text(
        "🎬 Demo complete. That's the full DM sequence for a winning trade day."
    )



@auth_required
async def cmd_cancel_strategy_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    token = uuid.uuid4().hex[:8]
    _put_pending(token, "cancel_orders", {})
    await update.message.reply_text(
        "Confirm CANCEL ALL strategy-owned pending orders?",
        reply_markup=_confirm_keyboard(token),
    )


@auth_required
async def cmd_close_strategy_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    token = uuid.uuid4().hex[:8]
    _put_pending(token, "close_trade", {})
    await update.message.reply_text(
        "Confirm CLOSE strategy-owned open trade(s) at market?",
        reply_markup=_confirm_keyboard(token),
    )


@auth_required
async def cmd_pnl_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(timezone.utc)
    token = uuid.uuid4().hex[:8]
    _put_pending(token, "pnl_reset", {})
    await update.message.reply_text(
        f"Confirm reset /pnl view?\n"
        f"Trades closed before {now.strftime('%Y-%m-%d %H:%M')} UTC will be hidden.\n"
        f"Reversible via /pnl_reset_clear.",
        reply_markup=_confirm_keyboard(token),
    )


@auth_required
async def cmd_pnl_reset_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _clear_pnl_since():
        journal("pnl_reset_cleared")
        await update.message.reply_text(
            "✅ PnL cutoff cleared — /pnl now shows full history."
        )
    else:
        await update.message.reply_text("📭 No PnL cutoff was set.")


@auth_required
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if ":" not in data:
        return
    kind_tag, token = data.split(":", 1)
    entry = _pop_pending(token)
    if entry is None:
        await query.edit_message_text(query.message.text + "\n\n⏰ Token expired or already used.")
        return
    age = time.time() - entry.get("created_at", 0)
    if age > PROPOSAL_TTL_SECONDS:
        await query.edit_message_text(query.message.text + f"\n\n⏰ Expired ({int(age)}s).")
        return

    if kind_tag == "cxl":
        await query.edit_message_text(query.message.text + "\n\n❌ Cancelled.")
        return
    if kind_tag != "cfm":
        return

    kind = entry["kind"]
    if kind == "cancel_orders":
        await query.edit_message_text(query.message.text + "\n\n⏳ Cancelling…")
        try:
            result = await _with_lock(context, _cancel_strategy_orders_blocking)
        except Exception as e:
            await context.bot.send_message(TELEGRAM_USER_ID, f"❌ {e}")
            return
        await context.bot.send_message(
            TELEGRAM_USER_ID,
            f"Cancelled {len(result['cancelled'])} strategy orders.\n"
            f"Errors: {len(result['errors'])}"
        )
    elif kind == "close_trade":
        await query.edit_message_text(query.message.text + "\n\n⏳ Closing…")
        try:
            result = await _with_lock(context, _close_strategy_trade_blocking)
        except Exception as e:
            await context.bot.send_message(TELEGRAM_USER_ID, f"❌ {e}")
            return
        await context.bot.send_message(
            TELEGRAM_USER_ID,
            f"Closed {len(result['closed'])} strategy trade(s).\n"
            f"Errors: {len(result['errors'])}"
        )
    elif kind == "test_trade":
        await query.edit_message_text(query.message.text + "\n\n⏳ Running test trade flow…")
        await _execute_test_trade_flow(context.application)
    elif kind == "pnl_reset":
        now = datetime.now(timezone.utc)
        _save_pnl_since(now)
        journal("pnl_reset", since_utc=now.isoformat())
        await query.edit_message_text(query.message.text + "\n\n✅ Reset.")
        await context.bot.send_message(
            TELEGRAM_USER_ID,
            f"✅ /pnl cutoff set to {now.strftime('%Y-%m-%d %H:%M')} UTC.\n"
            f"Use /pnl_reset_clear to restore full history."
        )


# =========================
# Formatting
# =========================

def _format_close_dm(action: dict, outcome_override: Optional[str] = None) -> str:
    """Format the 'trade closed' DM with PnL + account snapshot.
    Pass outcome_override (e.g. "FORCED_EOD") to label closes that aren't SL/TP."""
    trade_id = action.get("trade_id")
    side = action.get("filled_side", "?")
    entry = action.get("entry_price")
    exit_px = action.get("exit_price")
    pips = action.get("pips")
    pnl = action.get("realized_pl")
    nav = action.get("nav")

    if outcome_override:
        outcome = outcome_override
    else:
        outcome = "TP" if (pips is not None and pips > 0) else ("SL" if pips is not None else "SL/TP")
    lines = [f"🔴 LB closed ({side}, {outcome}).  tradeID={trade_id}"]

    if entry and exit_px and pips is not None:
        lines.append(f"   Entry {entry:.5f} → Exit {exit_px:.5f}   {pips:+.1f}p")
    if pnl is not None:
        pnl_line = f"   Realized PnL: ${pnl:+,.2f}"
        if nav is not None:
            pnl_line += f"   |   Account NAV: ${nav:,.2f}"
        lines.append(pnl_line)
    elif nav is not None:
        lines.append(f"   Account NAV: ${nav:,.2f}")

    return "\n".join(lines)



def _format_signal_outcome(result: dict) -> str:
    outcome = result.get("outcome")
    if outcome == "paused":
        return "⏸ paused — signal_job skipped."
    if outcome == "guard_blocked":
        return f"📭 Guard blocked: {result['reason']}"
    if outcome == "signal_unavailable":
        return f"📭 {result['reason']}"
    if outcome == "dry_run":
        return "🧪 DRY RUN (PLACE_ORDERS=false)\n\n" + format_signal(result["signal"])
    if outcome == "pair_error":
        return f"❌ Pair placement error: {result['reason']}"
    if outcome == "placed":
        sig = result["signal"]
        state = result["state"]
        return (
            f"🟢 LB pair placed ({MODE_TAG})  pair_id={state['pair_id']}\n"
            f"  Asian H {sig.asian_high:.5f}  L {sig.asian_low:.5f}  range {sig.asian_range_pips:.1f}p\n"
            f"  buy_stop {sig.buy_stop:.5f}  SL {sig.buy_sl:.5f}  TP {sig.buy_tp:.5f}  "
            f"orderID {state['buy_order_id']}\n"
            f"  sell_stop {sig.sell_stop:.5f}  SL {sig.sell_sl:.5f}  TP {sig.sell_tp:.5f}  "
            f"orderID {state['sell_order_id']}\n"
            f"  units {sig.units}  GTD {fmt_sgt(sig.gtd_time_iso)}"
        )
    return f"Unknown outcome: {outcome}"


# =========================
# Scheduled jobs
# =========================

async def scheduled_signal_job(application: Application):
    log.info("signal_job firing.")
    # OANDA practice v20 occasionally 401s the first request after a long idle
    # (on a clean_slate morning the watchdog is a no-op, so the token can sit
    # cold for ~24 min between startup and 07:00Z). A cheap read warms the
    # session; retry once on auth error, then let the real job proceed.
    from oandapyV20.exceptions import V20Error
    for attempt in (1, 2):
        try:
            await _with_lock_app(application, oc.get_account_summary)
            break
        except V20Error as e:
            msg = str(e)
            if attempt == 1 and "Insufficient authorization" in msg:
                log.warning(f"signal_job warmup 401 (attempt 1), retrying in 2s: {msg}")
                await asyncio.sleep(2)
                continue
            await safe_send(application, f"❌ signal_job error: {e}", context="signal_job_error")
            return
        except Exception as e:
            await safe_send(application, f"❌ signal_job error: {e}", context="signal_job_error")
            return
    try:
        result = await _with_lock_app(application, _signal_job_blocking)
    except Exception as e:
        await safe_send(application, f"❌ signal_job error: {e}", context="signal_job_error")
        return
    warn = result.get("pricing_warn")
    if warn:
        await safe_send(application, f"⚠️ {warn}", context="signal_warn")
    await safe_send(application, _format_signal_outcome(result), context="signal_outcome")
    sched: AsyncIOScheduler = application.bot_data["scheduler"]
    job = sched.get_job("signal_job")
    if job:
        application.bot_data["next_signal_run"] = str(job.next_run_time)


async def scheduled_watchdog(application: Application):
    # --- London Breakout watchdog ---
    if load_state() is not None:
        try:
            action = await _with_lock_app(application, oco_watchdog_tick)
        except Exception as e:
            log.warning(f"LB watchdog tick error: {e}")
            action = None
        if action:
            act = action.get("action")
            if act == "cancelled_sibling":
                fill = action.get("fill_price")
                spread = action.get("spread_pips", 0)
                fill_str = f" @ {fill:.5f}" if fill else ""
                sl = action.get("sl_price")
                tp = action.get("tp_price")
                sl_tp_line = f"\n   SL {sl:.5f}  TP {tp:.5f}" if sl and tp else ""
                units = action.get("units") or 0
                stop_pips = action.get("stop_pips") or 0
                target_pips = action.get("target_pips") or 0
                dpp = units / 10_000
                risk_line = ""
                if dpp > 0 and stop_pips > 0 and target_pips > 0:
                    max_loss = stop_pips * dpp
                    max_profit = target_pips * dpp
                    risk_line = f"\n   Max loss ${max_loss:,.2f}  Max profit ${max_profit:,.2f}"
                await safe_send(
                    application,
                    f"🔔 LB fill ({action['filled_side']}){fill_str}  tradeID={action['trade_id']}\n"
                    f"   Spread: {spread:.1f}p charged (baked into entry price)"
                    f"{sl_tp_line}"
                    f"{risk_line}\n"
                    f"   Sibling order cancelled.",
                    context="watchdog_fill",
                )
            elif act == "both_filled_closed_worse":
                await safe_send(
                    application,
                    f"🚨 Both legs filled within 30s window. "
                    f"Closed worse-P&L trade {action['closed_trade_id']} at market.",
                    context="watchdog_both_filled",
                )
            elif act == "trade_closed_sl_tp":
                await safe_send(application, _format_close_dm(action),
                                context="watchdog_sl_tp_close")



async def scheduled_hourly_candle_check(application: Application):
    if load_state() is None:
        return
    try:
        action = await _with_lock_app(application, hourly_dual_trigger_check)
    except Exception as e:
        log.warning(f"hourly_candle_check error: {e}")
        return
    if action and action.get("action") == "dual_trigger_skip":
        candle_sgt = fmt_sgt(action["candle_time"])
        await safe_send(
            application,
            f"⏭ Dual-trigger H1 candle ({candle_sgt}). "
            f"Both stops spanned — orders cancelled, skipping today. "
            f"cancelled={action['cancelled']}",
            context="hourly_dual_trigger",
        )



async def scheduled_daily_summary_job(application: Application):
    log.info("daily_summary_job firing.")
    try:
        result = await _with_lock_app(application, _daily_summary_blocking)
    except Exception as e:
        await safe_send(application, f"❌ daily_summary error: {e}",
                        context="daily_summary_error")
        return
    await safe_send(application, _format_daily_summary(result),
                    context="daily_summary")


async def scheduled_eod(application: Application):
    log.info("eod_sweep firing.")
    try:
        result = await _with_lock_app(application, _eod_blocking)
    except Exception as e:
        await safe_send(application, f"❌ eod_sweep error: {e}", context="eod_lock_error")
        return
    outcome = result.get("outcome")
    if outcome == "flat":
        s = result["summary"]
        # Per-trade close DMs for each force-closed trade (matches the normal
        # SL/TP close DM style so the user gets consistent notifications).
        for ct in s["closed_trades"]:
            if ct.get("entry_price") and ct.get("exit_price"):
                await safe_send(
                    application,
                    _format_close_dm(ct, outcome_override="FORCED_EOD"),
                    context="eod_trade_close",
                )
            else:
                await safe_send(
                    application,
                    f"🔴 LB force-closed tradeID={ct.get('trade_id')} (detail fetch failed).",
                    context="eod_trade_close_minimal",
                )
        if s.get("realized_pl") is not None:
            side = s.get("filled_side", "?")
            pips = s.get("result_pips")
            pl = s["realized_pl"]
            pips_str = f"  {pips:+.1f}p" if pips is not None else ""
            pnl_line = f"\n  Trade ({side}): ${pl:+,.2f}{pips_str}"
        else:
            pnl_line = "\n  No trades taken"
        await safe_send(
            application,
            f"✅ EOD flat. cancelled_orders={len(s['cancelled_orders'])} "
            f"closed_trades={len(s['closed_trades'])} errors={len(s['errors'])}"
            f"{pnl_line}",
            context="eod_summary",
        )
    elif outcome == "not_flat":
        await safe_send(
            application,
            f"🚨 EOD NOT FLAT: {result['reason']} — state retained, will retry via watchdog.",
            context="eod_not_flat",
        )
    else:
        await safe_send(
            application,
            f"❌ EOD sweep failure: {result.get('reason')}",
            context="eod_failure",
        )


async def scheduled_heartbeat():
    try:
        await asyncio.to_thread(requests.get, HEARTBEAT_URL, timeout=5)
    except Exception as e:
        log.warning(f"heartbeat ping failed: {e}")


# =========================
# Main
# =========================

def main() -> None:
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID == 0:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_USER_ID in .env")

    _prune_pending()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.bot_data["oanda_lock"] = asyncio.Lock()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("signal", cmd_signal))
    application.add_handler(CommandHandler("run_now", cmd_run_now))
    application.add_handler(CommandHandler("pause", cmd_pause))
    application.add_handler(CommandHandler("resume", cmd_resume))
    application.add_handler(CommandHandler("positions", cmd_positions))
    application.add_handler(CommandHandler("orders", cmd_orders))
    application.add_handler(CommandHandler("pnl", cmd_pnl))
    application.add_handler(CommandHandler("pnl_reset", cmd_pnl_reset))
    application.add_handler(CommandHandler("pnl_reset_clear", cmd_pnl_reset_clear))
    application.add_handler(CommandHandler("cancel_strategy_orders", cmd_cancel_strategy_orders))
    application.add_handler(CommandHandler("close_strategy_trade", cmd_close_strategy_trade))
    application.add_handler(CommandHandler("demo_trade", cmd_demo_trade))
    application.add_handler(CommandHandler("test_trade", cmd_test_trade))
    application.add_handler(CallbackQueryHandler(callback_handler))

    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    application.bot_data["scheduler"] = scheduler

    signal_trigger = CronTrigger(
        day_of_week="mon-fri",
        hour=LB_SIGNAL_FIRE_HOUR_UTC,
        minute=LB_SIGNAL_FIRE_MINUTE_UTC,
        second=LB_SIGNAL_FIRE_SECOND_UTC,
        timezone=timezone.utc,
    )
    eod_trigger = CronTrigger(
        day_of_week="mon-fri",
        hour=LB_EOD_HOUR_UTC, minute=LB_EOD_MINUTE_UTC,
        timezone=timezone.utc,
    )
    hourly_check_trigger = CronTrigger(
        day_of_week="mon-fri",
        hour="8-19", minute=LB_HOURLY_CHECK_MINUTE_UTC,
        timezone=timezone.utc,
    )
    watchdog_trigger = IntervalTrigger(seconds=WATCHDOG_SECONDS)
    daily_summary_trigger = CronTrigger(
        day_of_week="mon-fri", hour=20, minute=0, second=0, timezone=timezone.utc
    )

    async def post_init(app: Application):
        schedule = format_session_schedule()
        log.info("\n" + schedule)

        # LB startup reconcile
        try:
            async with app.bot_data["oanda_lock"]:
                result = await asyncio.to_thread(startup_reconcile)
            log.info(f"startup_reconcile (LB): {result.get('action')}")
        except Exception as e:
            log.error(f"LB startup_reconcile failed: {e}")
            result = {"action": f"error: {e}"}

        scheduler.add_job(scheduled_signal_job, signal_trigger, args=[app],
                          id="signal_job", replace_existing=True,
                          misfire_grace_time=300, coalesce=True)
        scheduler.add_job(scheduled_daily_summary_job, daily_summary_trigger, args=[app],
                          id="daily_summary_job", replace_existing=True,
                          misfire_grace_time=300, coalesce=True)
        scheduler.add_job(scheduled_watchdog, watchdog_trigger, args=[app],
                          id="watchdog_job", replace_existing=True)
        scheduler.add_job(scheduled_eod, eod_trigger, args=[app],
                          id="eod_job", replace_existing=True,
                          misfire_grace_time=300, coalesce=True)
        scheduler.add_job(scheduled_hourly_candle_check, hourly_check_trigger, args=[app],
                          id="hourly_candle_check", replace_existing=True,
                          misfire_grace_time=300, coalesce=True)
        if HEARTBEAT_URL:
            scheduler.add_job(scheduled_heartbeat, IntervalTrigger(minutes=5),
                              id="heartbeat_job", replace_existing=True)
            log.info(f"heartbeat job registered (every 5 min) → {HEARTBEAT_URL[:40]}...")
        scheduler.start()

        job = scheduler.get_job("signal_job")
        app.bot_data["next_signal_run"] = str(job.next_run_time) if job else "unknown"
        log.info(
            f"Scheduler started. "
            f"LB signal={LB_SIGNAL_FIRE_HOUR_UTC:02d}:{LB_SIGNAL_FIRE_MINUTE_UTC:02d}Z  "
            f"eod={LB_EOD_HOUR_UTC:02d}:{LB_EOD_MINUTE_UTC:02d}Z  watchdog={WATCHDOG_SECONDS}s"
        )

        await app.bot.send_message(
            TELEGRAM_USER_ID,
            f"🤖 LB bot online ({MODE_TAG}, PLACE_ORDERS={PLACE_ORDERS}).\n"
            f"LB reconcile={result.get('action')}  paused={is_paused()}\n"
            f"  next LB signal: {fmt_sgt(job.next_run_time) if job else 'unknown'}\n\n"
            f"{schedule}",
        )

    async def post_shutdown(app: Application):
        if scheduler.running:
            scheduler.shutdown(wait=False)

    application.post_init = post_init
    application.post_shutdown = post_shutdown

    log.info(f"Bot starting. journal={EVENTS_JOURNAL_FILE}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
