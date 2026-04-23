"""Shared configuration for the London Breakout FX live runner + Telegram bot.

Strategy: Asian-session (00-06 UTC) range breakout, buy-stop/sell-stop pair
placed at 07:00 UTC, intraday with EOD force-close at 20:59 UTC.

Params optimised via Scalping/param_sweep.py on EURUSD: SL=0.2×ATR, TP=1.0×ATR,
min Asian range 20 pips.

Secrets loaded from .env (python-dotenv). See .env.example for the template.
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =========================
# OANDA
# =========================
OANDA_API_TOKEN: str = os.environ.get("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID: str = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_ENV: str = os.environ.get("OANDA_ENV", "practice")  # "practice" or "live"

# Instrument and pip conventions — fixed-sizing formula below assumes these.
INSTRUMENT: str = "EUR_USD"
PIP_SIZE: float = 0.0001
SPREAD_PIPS: float = 1.3
ACCOUNT_CCY_EXPECTED: str = "USD"

# =========================
# Shared candles / ATR
# =========================
ATR_PERIOD: int = 14  # matches Scalping/london_breakout.py
CANDLE_LOOKBACK_COUNT: int = 720  # ≈ 30 days of H1 bars — plenty for ATR(14) + RSI(2)

# =========================
# London Breakout (LB)  —  params optimised via Scalping/param_sweep.py
# =========================
LB_ASIAN_START: int = 0              # Asian session start hour UTC
LB_ASIAN_END: int = 6                # Asian session end hour UTC (inclusive)
LB_LONDON_OPEN: int = 7              # London open — place pending orders here
LB_SL_ATR: float = 0.2               # SL = 0.2 × daily ATR
LB_TP_ATR: float = 1.0               # TP = 1.0 × daily ATR
LB_ENTRY_OFFSET_ATR: float = 0.0     # no buffer — enter at Asian high/low
LB_MIN_ASIAN_RANGE_PIPS: float = 20.0
LB_MAX_SPREAD_PIPS: float = 3.0      # skip day if live EUR/USD spread exceeds this
LB_SIGNAL_FIRE_HOUR_UTC: int = 7     # cron fire time for LB signal job
LB_SIGNAL_FIRE_MINUTE_UTC: int = 0
LB_SIGNAL_FIRE_SECOND_UTC: int = 15  # 15s after 07:00 — backtest scans from 07:00:00
LB_EOD_HOUR_UTC: int = 20            # force-close LB intraday trades
LB_EOD_MINUTE_UTC: int = 0           # matches backtest's SESSION_END=20 (bar_time.hour >= 20)
LB_HOURLY_CHECK_MINUTE_UTC: int = 5  # dual-trigger H1 check
LB_TAG: str = "lb_eurusd"

# =========================
# Execution / sizing
# =========================
WATCHDOG_SECONDS: int = 30
DOLLARS_PER_PIP: float = float(os.environ.get("DOLLARS_PER_PIP", "1.0"))
RISK_PER_TRADE: float = float(os.environ.get("RISK_PER_TRADE", "0.01"))
UNITS_PER_DOLLAR_PER_PIP: int = 10_000  # EURUSD + USD account only

# =========================
# Safety
# =========================
PLACE_ORDERS: bool = os.environ.get("PLACE_ORDERS", "false").lower() == "true"
MAX_MARGIN_CLOSEOUT_PCT: float = 0.5


def build_client_id(trading_date: str, side: str) -> str:
    """side in {'buy', 'sell'}, trading_date 'YYYY-MM-DD'."""
    return f"lb_EURUSD_{trading_date}_{side}"


# =========================
# Telegram
# =========================
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_USER_ID: int = int(os.environ.get("TELEGRAM_USER_ID", "0"))
PROPOSAL_TTL_SECONDS: int = 300

# =========================
# Persistence file paths — anchored to FX_DATA_DIR if set (VPS deploy),
# else the 50_pips_eurusd/ dir alongside this file (laptop default, unchanged).
# =========================
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "50_pips_eurusd")
_DATA_DIR: str = os.environ.get("FX_DATA_DIR", _DEFAULT_DATA_DIR)

LB_STATE_FILE: str = os.path.join(_DATA_DIR, "lb_state.json")
TRADE_LOG_FILE: str = os.path.join(_DATA_DIR, "lb_trade_log.csv")
EVENTS_JOURNAL_FILE: str = os.path.join(_DATA_DIR, "fx_events.jsonl")
PENDING_CONFIRMATIONS_FILE: str = os.path.join(_DATA_DIR, "fx_pending_confirmations.json")
PAUSE_FLAG_FILE: str = os.path.join(_DATA_DIR, "fx_pause.flag")
BOT_LOG_FILE: str = os.path.join(_DATA_DIR, "fx_bot.log")
PNL_SINCE_FILE: str = os.path.join(_DATA_DIR, "pnl_since.json")


MODE_TAG: str = "practice" if OANDA_ENV == "practice" else "LIVE"
