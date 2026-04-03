# CLAUDE.md — BTC/USDT Mean Reversion Trading Bot

## Project Overview

A focused, conservative trading bot that trades BTC/USDT on Binance using a mean reversion strategy on the 1H timeframe. The goal is consistent small gains with strict capital protection. Built in Python, designed to run 24/7 on a VPS or local machine.

---

## Strategy: Mean Reversion on BTC/USDT (1H)

### Core Concept
BTC price regularly deviates from its short-term average and snaps back. We exploit this by entering when price is statistically "stretched" and exiting when it returns to normal.

### Signal Logic
- **Indicator**: Bollinger Bands (20-period SMA, ±2 standard deviations) + RSI (14-period)
- **Long entry**: Price closes below lower Bollinger Band AND RSI < 35
- **Short entry**: Disabled (conservative mode — long-only to avoid shorting in bull markets)
- **Exit**: Price crosses back above the 20-period SMA (middle band) OR hits stop-loss/take-profit

### Entry Filters (reduce false signals)
- Volume on signal candle must be above 20-period average volume (confirms genuine move)
- No entry if price is in a strong downtrend (50-period SMA sloping down > 0.5% over last 10 candles)
- No entry if an open position already exists (one trade at a time)

### Risk Rules (non-negotiable)
- **Position size**: 10% of available balance per trade (never more)
- **Stop-loss**: 2% below entry price (hard, always set as exchange order)
- **Take-profit**: 3% above entry price (1.5:1 reward/risk ratio)
- **Max daily loss**: If balance drops 5% in a calendar day, bot pauses until next UTC midnight
- **Max open time**: If trade is open > 48 hours without hitting TP or SL, close at market

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for trading bots |
| Exchange | Binance (via `python-binance`) | Best API, low fees, high BTC liquidity |
| Indicators | `pandas-ta` | Simple, reliable TA library |
| Data | Binance REST API (candles) | Free, accurate OHLCV data |
| Scheduling | `APScheduler` | Runs strategy check every closed candle |
| Logging | Python `logging` + file rotation | Full audit trail of every decision |
| Config | `.env` file via `python-dotenv` | Keeps API keys out of code |
| Notifications | Telegram Bot API | Sends trade alerts to your phone |

---

## Folder Structure

```
btc-bot/
├── CLAUDE.md                  # This file
├── .env                       # API keys (never commit)
├── .env.example               # Template for .env
├── requirements.txt
├── main.py                    # Entry point — starts the bot loop
├── config.py                  # Loads env vars, defines all constants
├── strategy/
│   ├── __init__.py
│   ├── signals.py             # Bollinger Bands + RSI signal logic
│   └── filters.py             # Entry filters (volume, trend)
├── execution/
│   ├── __init__.py
│   ├── order_manager.py       # Place, cancel, check orders on Binance
│   └── position_tracker.py   # Tracks open position state
├── risk/
│   ├── __init__.py
│   ├── sizing.py              # Position size calculator
│   └── daily_guard.py        # Daily loss limit enforcer
├── data/
│   ├── __init__.py
│   └── fetcher.py             # Fetches OHLCV candles from Binance
├── notifications/
│   ├── __init__.py
│   └── telegram.py            # Sends alerts via Telegram Bot API
├── logs/
│   └── bot.log                # Auto-created, rotated daily
└── tests/
    ├── test_signals.py
    ├── test_sizing.py
    └── test_filters.py
```

---

## Key Files — Responsibilities

### `config.py`
- Loads all env vars
- Defines all strategy constants (BB period, RSI thresholds, SL/TP %, position size %, max daily loss %)
- Single source of truth — no magic numbers anywhere else in the codebase

### `data/fetcher.py`
- `get_candles(symbol, interval, limit)` — returns a pandas DataFrame with OHLCV columns
- Handles Binance API rate limits gracefully (retry with backoff)

### `strategy/signals.py`
- `get_signal(df)` — takes OHLCV DataFrame, returns `"long"`, `"exit"`, or `None`
- Calculates Bollinger Bands and RSI using `pandas-ta`
- Only looks at the last **closed** candle (index -2, not -1)

### `strategy/filters.py`
- `passes_filters(df)` — returns `True/False`
- Checks volume condition and trend condition

### `execution/order_manager.py`
- `place_market_order(side, quantity)`
- `place_oco_order(entry_price, sl_price, tp_price, quantity)` — sets SL and TP simultaneously as OCO order on Binance
- `cancel_all_open_orders(symbol)`

### `execution/position_tracker.py`
- Maintains in-memory state: `{ is_open, entry_price, entry_time, quantity }`
- On bot restart, checks Binance for open orders to restore state

### `risk/sizing.py`
- `calculate_quantity(balance, entry_price)` — returns BTC quantity to buy based on 10% of balance

### `risk/daily_guard.py`
- Tracks starting balance each UTC day
- `is_trading_allowed()` — returns `False` if daily loss > 5%

### `main.py`
- Initialises all modules
- Schedules `run_cycle()` to fire at the close of every 1H candle
- `run_cycle()` flow:
  1. Check `daily_guard.is_trading_allowed()`
  2. Fetch candles
  3. Check `position_tracker` — if position open, check exit signal and max time rule
  4. If no position, check filters + signal
  5. If signal fires, size position, place OCO order, update tracker, send Telegram alert

---

## Environment Variables (`.env.example`)

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TESTNET=true   # Set to false for live trading
```

---

## Development Phases

### Phase 1 — Paper Trading (Week 1–2)
- Set `TESTNET=true`
- Use Binance Testnet (free fake money)
- Run for at least 2 weeks, log every trade
- Goal: confirm signals fire correctly, orders execute, no crashes

### Phase 2 — Live with Minimum Capital (Week 3–6)
- Fund with $100–$200 real capital
- Monitor daily, compare live results to paper results
- Goal: validate real execution (slippage, fees, API latency)

### Phase 3 — Scale Up (Month 2+)
- Only increase capital if Phase 2 shows consistent positive expectancy over 20+ trades
- Never add capital to a losing system — fix the strategy first

---

## Backtesting (Before Phase 1)

Use `backtesting.py` library or manual pandas simulation:
- Pull 1–2 years of BTC/USDT 1H data from Binance
- Simulate signal logic with realistic fees (0.1% per trade)
- Key metrics to check:
  - Win rate (target: > 45%)
  - Average R (reward/risk, target: > 0.8 after fees)
  - Max drawdown (must be < 20%)
  - Sharpe ratio (target: > 1.0)

---

## Important Rules for Claude Code

- Never hardcode API keys
- Always use the last **closed** candle for signals, never the live candle
- All orders must have a stop-loss — never enter without one
- Log every decision with timestamp: signal fired, filter passed/failed, order placed, order filled
- If Binance API returns an error, log it and skip the cycle — never retry blindly in a loop
- All monetary values stored as `Decimal`, never `float`
- Keep each module focused — strategy logic never touches order execution directly