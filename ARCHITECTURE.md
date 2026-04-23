# 50_pips_eurusd — Infrastructure, Workflow & Price Derivation

## Context

Reference document for the London Breakout EUR/USD paper-trading bot. Captures the runtime topology (VPS, systemd, data paths, remote), the daily job schedule, and — most importantly — exactly how the signal price levels are derived from OANDA candles. Intended as a "how does this thing actually work" overview.

---

## 1. High-level topology

```
Laptop (Windows, Git Bash)
  │  edit .py → scp to VPS
  ▼
DigitalOcean VPS  (fra1, 1 vCPU / 1 GB, Ubuntu)
  │
  ├─ systemd service `fxbot.service` (user: fxbot, Restart=always)
  │    └─ python venv: /opt/fxbot/app/50_pips_eurusd/.venv
  │         └─ entry: telegram_bot.py  (asyncio + APScheduler + python-telegram-bot)
  │
  ├─ Code dir  : /opt/fxbot/app/50_pips_eurusd/    (deploy target)
  ├─ Data dir  : /var/lib/fxbot/                   (state, logs, trade log)
  └─ Secrets   : /opt/fxbot/.env                   (systemd EnvironmentFile=)
        │
        ▼
  OANDA v20 REST  (api-fxpractice.oanda.com, account 101-003-15208328-003)
        │
        ▼
  Telegram Bot API  (DMs to TELEGRAM_USER_ID only, @auth_required gate)

GitHub mirror (code only, no secrets): github.com/Agrocorp-International/telegram_trading_bot
```

**No database.** All state lives in JSON/CSV on local disk. No message queue. No CI. Deploy is `scp + systemctl restart`.

---

## 2. Infrastructure detail

| Piece | Location / value |
|---|---|
| VPS IP | `159.89.108.207` (DigitalOcean, `fra1`) |
| Hostname | `ubuntu-s-1vcpu-1gb-fra1` |
| SSH | `root@159.89.108.207`, key-only (`PermitRootLogin prohibit-password`) |
| Runtime user | `fxbot` (non-privileged) |
| Service | `/etc/systemd/system/fxbot.service` → `fxbot.service` |
| Interpreter | `/opt/fxbot/app/50_pips_eurusd/.venv/bin/python` |
| Code dir | `/opt/fxbot/app/50_pips_eurusd/` |
| Data dir | `/var/lib/fxbot/` (set via `FX_DATA_DIR=` in systemd env) |
| Env file | `/opt/fxbot/.env` — `OANDA_API_TOKEN`, `OANDA_ACCOUNT_ID`, `OANDA_ENV`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID`, `PLACE_ORDERS`, `DOLLARS_PER_PIP`, `RISK_PER_TRADE`, `HEARTBEAT_URL?` |
| fail2ban | Enabled, default sshd jail, laptop IP `151.192.54.122` whitelisted |
| Git remote | `origin = https://github.com/Agrocorp-International/telegram_trading_bot` |

**Data dir file map** (all under `/var/lib/fxbot/`):
- `lb_state.json` — armed-pair state (JSON). `None` when flat.
- `fx_events.jsonl` — structured journal (one line per event).
- `fx_bot.log` — Python logger output.
- `lb_trade_log.csv` — one row per closed trade (10 cols: entry_time, exit_time, direction, entry_price, exit_price, stop_price, target_price, stop_pips, result_pips, outcome). Written by `append_trade_log()` in [live_core.py:111-136](live_core.py#L111-L136).
- `pnl_since.json` — `/pnl` cutoff timestamp.
- `fx_pending_confirmations.json` — Telegram confirm-button TTL state.
- `fx_pause.flag` — empty file; presence = bot skips `signal_job`.

---

## 3. Process & scheduler topology

Single Python process. Event loop runs python-telegram-bot (long-poll) + an `AsyncIOScheduler` (UTC) registered via `post_init` in [telegram_bot.py:1205-1244](telegram_bot.py#L1205-L1244).

All OANDA calls are serialized by a single `asyncio.Lock` (`application.bot_data["oanda_lock"]`), via `_with_lock` / `_with_lock_app` helpers ([telegram_bot.py:217-222](telegram_bot.py#L217-L222)). Blocking work is dispatched via `asyncio.to_thread(...)`.

| Job | Trigger | Purpose | Code |
|---|---|---|---|
| `signal_job` | Cron Mon-Fri 07:00:15 UTC | Compute & place buy/sell stop pair | [telegram_bot.py:991](telegram_bot.py#L991) → `_signal_job_blocking` at [:285](telegram_bot.py#L285) |
| `watchdog_job` | Interval 30 s | OCO: cancel sibling on fill, detect SL/TP close | `oco_watchdog_tick` [live_core.py:448](live_core.py#L448) |
| `eod_job` | Cron Mon-Fri 20:00 UTC | Force-close open LB trade, cancel stale pending | `eod_sweep` [live_core.py:637](live_core.py#L637) |
| `daily_summary_job` | Cron Mon-Fri 20:00 UTC | DM daily P&L summary | [telegram_bot.py:1079](telegram_bot.py#L1079) |
| `heartbeat_job` | Interval 5 min | Ping `HEARTBEAT_URL` for uptime monitoring (optional) | [telegram_bot.py:1233](telegram_bot.py#L1233) |

At **process start** (`post_init`): `startup_reconcile` ([live_core.py:753](live_core.py#L753)) compares `lb_state.json` against live OANDA (tagged pending + open trades) and either marks `in_sync` or repairs the local state. **Trading state survives restarts** via this reconcile.

---

## 4. Daily workflow (Mon-Fri, UTC)

```
00:00   Asian session start. H1 bars 00, 01, 02, 03, 04, 05, 06 will form the range.
        Watchdog runs every 30s but short-circuits (no armed state → no-op).

07:00:15  signal_job fires:
             0.  Warmup ping → oc.get_account_summary() (retries once on 401)
             1.  run_pre_placement_guards  — not already armed today, USD account,
                                              margin closeout < 50%, no stray
                                              tagged orders/trades
             2.  _fetch_candles_and_signal  — pull 720 H1 bars + live spread
             3.  compute_signal             — derive price levels (see §5)
             4.  place_entry_pair           — submit long + short STOP orders
                                              with SL/TP attached, GTD = 20:00Z
             → Telegram DM with signal details + order IDs, write lb_state.json

07:00:45  watchdog starts polling (state now armed):
             every 30s fetch pendingOrders + openTrades (tagged), decide:
               - neither filled yet → no-op
               - one filled         → cancel sibling, mark filled_side
               - both filled (rare) → close the losing side
               - filled leg gone    → SL/TP hit → append trade log, clear state

20:00     eod_job:  force-close any still-open LB trade, cancel any pending,
                     reconcile state → append to trade log.
20:00     daily_summary_job: DM P&L for the day.
```

Weekends are excluded via `day_of_week="mon-fri"` on every cron trigger.

---

## 5. Price derivation — the full pipeline

End-to-end, where every number on the Telegram signal DM comes from. All code in [live_core.py:238-303](live_core.py#L238-L303) (`compute_signal`) + [strategy_shared.py](strategy_shared.py).

### 5.1 Inputs pulled from OANDA

| Input | OANDA endpoint | How | Notes |
|---|---|---|---|
| H1 candles | `GET /v3/instruments/EUR_USD/candles` | `oc.get_h1_candles(count=720)` | 720 bars ≈ 30 days. **Mid** prices (`price="M"`). Columns: datetime (UTC, start-of-candle), open, high, low, close, complete. |
| Live spread | `GET /v3/accounts/{id}/pricing?instruments=EUR_USD` | `oc.get_current_price()` → `LivePriceQuote` | `spread_pips = (ask - bid) / 0.0001`. Signal aborts if spread > **3.0 pips** (`LB_MAX_SPREAD_PIPS`). |
| Account NAV | `GET /v3/accounts/{id}/summary` | `oc.get_account_summary()` | Used only for position sizing — see 5.5. |

### 5.2 Asian range (`compute_asian_range`, [strategy_shared.py:58-107](strategy_shared.py#L58-L107))

Candle-timestamp convention: **start-of-candle** (a bar stamped `07:00:00Z` opens at 07:00 and closes at 08:00).

```
day_bars = all H1 bars where:
             bar.datetime.date() == today (UTC)
             AND bar.hour in [0..6]        # LB_ASIAN_START..LB_ASIAN_END inclusive
             AND bar.complete == True       # skip the currently-forming 06-07 bar

if expected hours {0,1,2,3,4,5,6} ⊄ got_hours:  abort (None)
if day_bars is empty:                           abort

asian_high       = max(day_bars.high)
asian_low        = min(day_bars.low)
asian_range_pips = (asian_high - asian_low) / 0.0001
```

Abort (raise `SignalUnavailable` → Telegram "skip day") if `asian_range_pips < 20` (`LB_MIN_ASIAN_RANGE_PIPS`).

### 5.3 Daily ATR (`compute_daily_atr`, [strategy_shared.py:22-42](strategy_shared.py#L22-L42))

Wilder's ATR(14), **1-day lagged** (so today's ATR uses only prior-day bars — matches the backtester and avoids look-ahead):

```
daily = H1 → resample("D") → max(high), min(low), last(close)
true_range = max(
    high - low,
    |high - prev_close|,
    |low  - prev_close|,
)
ATR = tr.ewm(alpha = 1/14, adjust=False).mean().shift(1)   # Wilder, lag 1
atr_price  = ATR[today]          # in price units, e.g. 0.00087
atr_pips   = atr_price / 0.0001  # e.g. 8.7 pips
```

Abort if ATR is NaN (insufficient history — need ≥14 prior daily bars).

### 5.4 Entry, SL, TP levels ([live_core.py:271-283](live_core.py#L271-L283))

Constants in [config.py](config.py):
- `LB_ENTRY_OFFSET_ATR = 0.0` — **no** buffer beyond Asian high/low
- `LB_SL_ATR = 0.2` — stop distance = 0.2 × ATR
- `LB_TP_ATR = 1.0` — target distance = 1.0 × ATR

```
offset       = atr_price * 0.0           = 0                     # no buffer
stop_dist    = atr_price * 0.2
target_dist  = atr_price * 1.0           = atr_price
spread_half  = live_spread_pips * 0.0001 / 2

buy_stop     = asian_high + spread_half
sell_stop    = asian_low  - spread_half
buy_sl       = buy_stop  - stop_dist
buy_tp       = buy_stop  + target_dist
sell_sl      = sell_stop + stop_dist
sell_tp      = sell_stop - target_dist

stop_pips    = stop_dist   / 0.0001      # typically ~0.2 × atr_pips
target_pips  = target_dist / 0.0001      # typically ~1.0 × atr_pips (5× R)
```

Half-spread is added to buy side and subtracted from sell side so the STOP trigger price already accounts for the ask/bid offset — entry is "really" at Asian high/low in mid terms.

### 5.5 Position sizing (`size_units`, [live_core.py:201-206](live_core.py#L201-L206))

**Compounding 1%-risk fixed-fractional.** Fetches NAV live per fire:

```
nav              = float(OANDA account summary NAV)
dollars_per_pip  = (nav * RISK_PER_TRADE) / stop_pips     # RISK_PER_TRADE = 0.01
units            = max(1000, int(dollars_per_pip * UNITS_PER_DOLLAR_PER_PIP))
                                                           # UNITS_PER_DOLLAR_PER_PIP = 10_000
                                                           # (EUR_USD, USD account only)
```

The 1000-units floor is OANDA's practice-account minimum. Formula assumes `INSTRUMENT == "EUR_USD"` and `ACCOUNT_CCY == "USD"` — both enforced by `run_pre_placement_guards` in [live_core.py:326-366](live_core.py#L326-L366).

### 5.6 Order submission ([oanda_client.py:145-189](oanda_client.py#L145-L189), [live_core.py:376-440](live_core.py#L376-L440))

Two `OrderCreate` calls, each for a `STOP` order with:
- `instrument: "EUR_USD"`, `units: +N / -N`, `price: buy_stop / sell_stop`
- `timeInForce: "GTD"`, `gtdTime: <today> 20:00:00 UTC`
- `stopLossOnFill` + `takeProfitOnFill` at the SL/TP levels (GTC)
- `clientExtensions.tag = "lb_eurusd"`, `clientExtensions.id = "lb_EURUSD_<date>_{buy|sell}"` — used for reconciliation/idempotency

If the short leg submission fails after the long leg succeeded, `place_entry_pair` **rolls back the long leg via `cancel_order`** to avoid a naked position. Double-failure path is surfaced as `MANUAL INTERVENTION REQUIRED` in Telegram — no silent state.

---

## 6. OCO management (`oco_watchdog_tick`, [live_core.py:448-576](live_core.py#L448-L576))

OANDA has no native OCO, so the bot polls every 30 s and enforces it:

- **Pre-fill** (both pending): no-op.
- **One filled**: cancel the sibling pending order, update `lb_state.json` (`filled_side`, `trade_id`, `fill_price`, `status="filled"`). Telegram DM with SL/TP reminder.
- **Both filled** (rare — gap open): close the leg with the worse unrealized P&L immediately, keep the better one.
- **Filled leg disappears**: the trade closed (SL or TP hit) — fetch `TradeDetails` for realized P&L + average close price, append to `lb_trade_log.csv`, DM the result, set `status="done"`.
- **Idempotent retries**: if a prior cancel silently failed, re-issue it on subsequent ticks.

State transitions: `armed → filled → done`, or `armed → done` (via EOD/whipsaw/cancel).

---

## 7. EOD sweep (`eod_sweep`, [live_core.py:637](live_core.py#L637))

20:00 UTC Mon-Fri: for each tagged open trade, `TradeClose(units=ALL)` with up to 3 retries; cancel any tagged pending orders; verify OANDA shows zero tagged orders/trades before clearing `lb_state.json`. If anything can't be closed (liquidity, network), raises `EodNotFlat` and Telegram alerts — state stays armed so the watchdog and next EOD retry.

---

## 8. Safety guards (`run_pre_placement_guards`, [live_core.py:326-366](live_core.py#L326-L366))

Every signal_job runs these before any order submission. Any failure raises `GuardBlocked` → Telegram alert → no orders placed:

| Guard | Rule |
|---|---|
| Double-arm | Abort if `lb_state.trading_date == today` |
| Stale tagged pending | Abort if OANDA has tagged pending with today's client_id |
| Stale tagged open trade | Abort if OANDA has tagged open trade with today's client_id |
| Account currency | Must be `USD` (sizing formula assumption) |
| Instrument | Must be `EUR_USD` (pip/sizing assumption) |
| Margin closeout | `marginCloseoutPercent ≤ 50%` |
| Pause flag | `fx_pause.flag` present → skip (not via guard; checked at top of `_signal_job_blocking`) |
| `PLACE_ORDERS` flag | When `false`, computes and DMs the signal but does **not** submit orders (dry-run) |

Other guards outside the pre-placement block:
- Live spread > 3 pips → `SignalUnavailable` ("skip day")
- Asian range < 20 pips → `SignalUnavailable`
- Transient 401 on the first OANDA hit after idle → warmup-ping retry in `scheduled_signal_job` (already deployed)

---

## 9. Telegram surface ([telegram_bot.py](telegram_bot.py))

All commands are gated by `@auth_required` (sender must be `TELEGRAM_USER_ID`). Every scheduled event also DMs a summary; failures DM a `❌` line.

| Command | What it does |
|---|---|
| `/status` | Account NAV, open/pending LB orders, next signal time |
| `/orders` | List current tagged pending orders |
| `/run_now` | Fire `_signal_job_blocking` manually (same path as cron) |
| `/pause` / `/resume` | Toggle `fx_pause.flag` |
| `/pnl` | Chart + stats of closed LB trades (since `pnl_since.json` cutoff) |
| `/pnl_reset` / `/pnl_reset_clear` | Move / clear the cutoff |
| `/cancel_strategy_orders` | Cancel all tagged pending (confirm button) |
| `/close_strategy_trade` | Market-close tagged open trade (confirm button) |
| `/test_trade` | 1000-unit FOK market smoke test (PLACE_ORDERS must be true) |
| `/demo_trade` | Like test_trade but with SL attached |

---

## 10. Observability

- **Live**: every scheduled event DMs Telegram. Any unhandled exception in a scheduled coroutine reaches `safe_send(... "❌ <event> error: ...")`.
- **Structured journal**: `fx_events.jsonl` — one JSON line per significant event (`signal_computed`, `pair_placement_exception`, `sibling_cancelled`, `trade_closed_via_sl_tp`, `eod_exception`, …). Grep-friendly for post-mortems.
- **Logger**: `fx_bot.log` (INFO) + systemd `journalctl -u fxbot`.
- **Trade log**: `lb_trade_log.csv` — one row per closed trade, identical columns to the backtester's output so live vs. backtest are directly comparable.
- **Heartbeat**: if `HEARTBEAT_URL` is set, pings every 5 min — wire to cronitor/betterstack/etc. for dead-man alerts.

---

## 11. Deploy & rollback

Full recipe lives in memory file `reference_paper_trading_vps.md`. Summary:

```bash
# On VPS — backup first
cd /opt/fxbot/app/50_pips_eurusd && cp <file> <file>.bak-$(date +%Y%m%d-%H%M%S)

# From laptop Git Bash — upload
scp <file> root@159.89.108.207:/opt/fxbot/app/50_pips_eurusd/

# On VPS — chown, restart, verify
chown fxbot:fxbot /opt/fxbot/app/50_pips_eurusd/<file>
systemctl restart fxbot
journalctl -u fxbot -n 30 --no-pager
```

Verification signal: Telegram receives `🤖 LB bot online (...)` within ~5 s of restart. No green DM → check `journalctl` for tracebacks, restore from `.bak-*`, restart.

**GitHub** is a code mirror only — `git push` does **not** deploy. Deploy is always `scp + systemctl restart`.
