"""Thin oandapyV20 wrapper scoped to the portfolio strategy's needs.

Every write carries clientExtensions with a caller-supplied `tag` so reconciliation
can distinguish LB vs. MR vs. anything else on the account. Read helpers accept
a tag argument so the same wrapper serves both strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import pandas as pd

from config import (
    CANDLE_LOOKBACK_COUNT,
    INSTRUMENT,
    OANDA_ACCOUNT_ID,
    OANDA_API_TOKEN,
    OANDA_ENV,
    PIP_SIZE,
    build_client_id,
)


# Price formatting — EURUSD quotes with 5 decimal places.
PRICE_PRECISION = 5


def _client() -> oandapyV20.API:
    return oandapyV20.API(access_token=OANDA_API_TOKEN, environment=OANDA_ENV)


def _fmt_price(p: float) -> str:
    return f"{p:.{PRICE_PRECISION}f}"


def _fmt_gtd(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


# =========================
# Account
# =========================

def get_account_summary() -> dict:
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    _client().request(r)
    return r.response["account"]


# =========================
# Pricing (live bid/ask)
# =========================

@dataclass(frozen=True)
class LivePriceQuote:
    bid: float
    ask: float
    spread_pips: float
    tradeable: bool
    time: str


class PricingUnavailable(Exception):
    """Raised when OANDA returns no usable quote for INSTRUMENT."""


def get_current_price() -> LivePriceQuote:
    params = {"instruments": INSTRUMENT}
    r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
    _client().request(r)
    prices = r.response.get("prices") or []
    if not prices:
        raise PricingUnavailable("empty prices[] in response")
    p = prices[0]
    if p.get("instrument") != INSTRUMENT:
        raise PricingUnavailable(f"instrument mismatch: got {p.get('instrument')}")
    bids = p.get("bids") or []
    asks = p.get("asks") or []
    if not bids or not asks:
        raise PricingUnavailable("missing bids/asks array")
    bid = float(bids[0]["price"])
    ask = float(asks[0]["price"])
    return LivePriceQuote(
        bid=bid,
        ask=ask,
        spread_pips=(ask - bid) / PIP_SIZE,
        tradeable=bool(p.get("tradeable", False)),
        time=p.get("time", ""),
    )


# =========================
# Candles
# =========================

def get_latest_complete_h1_candle() -> Optional[dict]:
    params = {"granularity": "H1", "price": "M", "count": 3}
    r = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
    _client().request(r)
    for c in reversed(r.response["candles"]):
        if bool(c["complete"]):
            mid = c["mid"]
            return {
                "datetime": pd.to_datetime(c["time"], utc=True).tz_localize(None),
                "open":  float(mid["o"]),
                "high":  float(mid["h"]),
                "low":   float(mid["l"]),
                "close": float(mid["c"]),
            }
    return None


def get_h1_candles(count: int = CANDLE_LOOKBACK_COUNT) -> pd.DataFrame:
    params = {"granularity": "H1", "price": "M", "count": count}
    r = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
    _client().request(r)
    rows = []
    for c in r.response["candles"]:
        mid = c["mid"]
        rows.append({
            "datetime": pd.to_datetime(c["time"], utc=True).tz_localize(None),
            "open":  float(mid["o"]),
            "high":  float(mid["h"]),
            "low":   float(mid["l"]),
            "close": float(mid["c"]),
            "complete": bool(c["complete"]),
        })
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


# =========================
# Orders — write
# =========================

def place_stop_entry(
    *,
    tag: str,
    client_id: str,
    units: int,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    gtd_time: datetime,
    comment: str,
) -> dict:
    """Submit a STOP entry order with SL/TP attached. Used by London Breakout."""
    order_body = {
        "order": {
            "type": "STOP",
            "instrument": INSTRUMENT,
            "units": str(units),
            "price": _fmt_price(entry_price),
            "timeInForce": "GTD",
            "gtdTime": _fmt_gtd(gtd_time),
            "positionFill": "DEFAULT",
            "triggerCondition": "DEFAULT",
            "clientExtensions": {
                "id": client_id,
                "tag": tag,
                "comment": comment,
            },
            "stopLossOnFill": {
                "price": _fmt_price(sl_price),
                "timeInForce": "GTC",
            },
            "takeProfitOnFill": {
                "price": _fmt_price(tp_price),
                "timeInForce": "GTC",
            },
            "tradeClientExtensions": {
                "id": client_id,
                "tag": tag,
                "comment": comment,
            },
        }
    }
    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_body)
    _client().request(r)
    return r.response


def place_market_order(
    *,
    tag: str,
    client_id: str,
    units: int,
    comment: str,
) -> dict:
    """Submit a MARKET order (no SL/TP). Used by /test_trade smoke test."""
    order_body = {
        "order": {
            "type": "MARKET",
            "instrument": INSTRUMENT,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "clientExtensions": {"id": client_id, "tag": tag, "comment": comment},
            "tradeClientExtensions": {"id": client_id, "tag": tag, "comment": comment},
        }
    }
    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_body)
    _client().request(r)
    return r.response


def place_market_order_with_sl(
    *,
    tag: str,
    client_id: str,
    units: int,
    sl_price: float,
    comment: str,
) -> dict:
    """FOK market order with a GTC stop-loss on fill. Used by Mean Reversion strategy."""
    order_body = {
        "order": {
            "type": "MARKET",
            "instrument": INSTRUMENT,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "clientExtensions": {"id": client_id, "tag": tag, "comment": comment},
            "tradeClientExtensions": {"id": client_id, "tag": tag, "comment": comment},
            "stopLossOnFill": {
                "price": _fmt_price(sl_price),
                "timeInForce": "GTC",
            },
        }
    }
    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_body)
    _client().request(r)
    return r.response


def cancel_order(order_specifier: str) -> Optional[dict]:
    from oandapyV20.exceptions import V20Error
    r = orders.OrderCancel(OANDA_ACCOUNT_ID, orderID=order_specifier)
    try:
        _client().request(r)
        return r.response
    except V20Error as e:
        msg = str(e).lower()
        if "does not exist" in msg or "already been cancelled" in msg or "already cancelled" in msg:
            return None
        raise


# =========================
# Orders / trades — read (tag-filtered)
# =========================

def _filter_by_tag(objs: List[dict], tag: str) -> List[dict]:
    return [o for o in objs if o.get("clientExtensions", {}).get("tag") == tag]


def get_pending_orders_by_tag(tag: str) -> List[dict]:
    r = orders.OrdersPending(OANDA_ACCOUNT_ID)
    _client().request(r)
    return _filter_by_tag(r.response.get("orders", []), tag)


def get_open_trades_by_tag(tag: str) -> List[dict]:
    r = trades.OpenTrades(OANDA_ACCOUNT_ID)
    _client().request(r)
    return _filter_by_tag(r.response.get("trades", []), tag)


def close_trade(trade_id: str) -> dict:
    r = trades.TradeClose(OANDA_ACCOUNT_ID, tradeID=trade_id, data={"units": "ALL"})
    _client().request(r)
    return r.response


def get_trade_details(trade_id: str) -> dict:
    """Return the full trade record (open or closed). Includes realizedPL and
    averageClosePrice for closed trades."""
    r = trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id)
    _client().request(r)
    return r.response["trade"]


def list_closed_trades_by_tag(tag: str) -> tuple:
    """Page through TradesList(state=CLOSED) collecting trades tagged with `tag`.

    Returns (matching_trades, skipped_without_tag_count).

    Pagination contract:
    - OANDA returns most-recent-first.
    - Page size = 500 (TradesList batch max).
    - Next page uses beforeID = (oldest_id_on_current_page - 1) as a string.
    - Stop when page returns < 500 rows, or after 50 pages (safety cap).
    - Deduped by trade id across pages.
    """
    MAX_PAGES = 50
    PAGE_SIZE = 500
    client = _client()
    seen_ids = set()
    matches = []
    skipped_no_ext = 0
    before_id = None

    for _ in range(MAX_PAGES):
        params = {
            "state": "CLOSED",
            "instrument": INSTRUMENT,
            "count": PAGE_SIZE,
        }
        if before_id is not None:
            params["beforeID"] = before_id

        r = trades.TradesList(accountID=OANDA_ACCOUNT_ID, params=params)
        client.request(r)
        batch = r.response.get("trades", []) or []

        if not batch:
            break

        page_ids_numeric = []
        for t in batch:
            tid = t.get("id")
            if tid is None or tid in seen_ids:
                continue
            seen_ids.add(tid)
            try:
                page_ids_numeric.append(int(tid))
            except (TypeError, ValueError):
                pass

            ext = t.get("clientExtensions")
            if not ext:
                skipped_no_ext += 1
                continue
            if ext.get("tag") == tag:
                matches.append(t)

        if len(batch) < PAGE_SIZE:
            break  # no more history

        if not page_ids_numeric:
            break  # defensive — shouldn't happen with valid batch
        before_id = str(min(page_ids_numeric))

    return matches, skipped_no_ext


# =========================
# Convenience
# =========================

def reachable() -> bool:
    try:
        get_account_summary()
        return True
    except Exception:
        return False


def describe_account() -> str:
    a = get_account_summary()
    return (
        f"id={a.get('id')} ccy={a.get('currency')} "
        f"balance={a.get('balance')} NAV={a.get('NAV')} "
        f"marginCloseoutPct={a.get('marginCloseoutPercent')}"
    )


__all__ = [
    "get_account_summary",
    "get_current_price",
    "LivePriceQuote",
    "PricingUnavailable",
    "get_h1_candles",
    "get_latest_complete_h1_candle",
    "place_stop_entry",
    "place_market_order",
    "place_market_order_with_sl",
    "cancel_order",
    "get_pending_orders_by_tag",
    "get_open_trades_by_tag",
    "close_trade",
    "get_trade_details",
    "list_closed_trades_by_tag",
    "reachable",
    "describe_account",
    "build_client_id",
]
