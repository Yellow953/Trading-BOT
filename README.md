# BTC/USDT Mean Reversion Bot

A conservative, long-only trading bot for BTC/USDT on Binance. Enters when price is statistically oversold and exits when it reverts to the mean.

## Strategy

- **Signal**: Price closes below the lower Bollinger Band (20-period, ±2σ) **and** RSI(14) < 35
- **Exit**: Price crosses back above the 20-period SMA (middle band)
- **Filters**: Volume above 20-period average, no strong downtrend (50-SMA slope)
- **Risk**: 10% position size, 2% stop-loss, 3% take-profit (OCO order), 5% max daily loss, 48h max hold

## Setup

**1. Clone and create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure credentials**
```bash
cp .env.example .env
# Edit .env with your keys
```

| Variable | Description |
|---|---|
| `BINANCE_API_KEY` | Binance API key |
| `BINANCE_API_SECRET` | Binance API secret |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (from @BotFather) |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |
| `TESTNET` | `true` for paper trading, `false` for live |

**3. Run**
```bash
python main.py
```

The bot fires at `HH:01 UTC` every hour (one minute after candle close). All decisions are logged to `logs/bot.log`.

## Project Structure

```
├── config.py                  # All constants — single source of truth
├── main.py                    # Entry point and strategy cycle
├── data/
│   └── fetcher.py             # Fetches OHLCV candles from Binance
├── strategy/
│   ├── signals.py             # Bollinger Bands + RSI signal logic
│   └── filters.py             # Volume and trend entry filters
├── execution/
│   ├── order_manager.py       # Binance order placement (market + OCO)
│   └── position_tracker.py   # In-memory position state + restart recovery
├── risk/
│   ├── sizing.py              # Position size calculator (10% of balance)
│   └── daily_guard.py        # Daily loss limit enforcer (5%)
├── notifications/
│   └── telegram.py            # Trade alerts via Telegram
└── tests/
    ├── test_signals.py
    ├── test_filters.py
    └── test_sizing.py
```

## Testing

```bash
pytest tests/ -v
```

## Development Phases

| Phase | Capital | Goal |
|---|---|---|
| 1 — Paper trading | Testnet (fake) | 2 weeks, confirm signals and execution work |
| 2 — Live minimum | $100–$200 real | Validate real execution vs paper results |
| 3 — Scale up | Increase only if profitable | 20+ trades with positive expectancy |

## Important Notes

- Never commit `.env` — it contains your API keys
- The bot is **long-only** — no shorting
- All orders include a stop-loss — the bot will never enter without one
- Monetary values use `Decimal` throughout to avoid floating-point errors
