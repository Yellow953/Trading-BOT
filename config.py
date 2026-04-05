import os
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

# --- Binance ---
BINANCE_API_KEY = os.environ["BINANCE_API_KEY"]
BINANCE_API_SECRET = os.environ["BINANCE_API_SECRET"]
BINANCE_TESTNET = os.getenv("TESTNET", "true").lower() == "true"

# --- Symbol / Timeframe ---
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
CANDLE_LIMIT = 300  # enough for EMA_TREND + ADX + buffer

# Candle duration in hours — used to convert MAX_HOLD_HOURS into candle count
_INTERVAL_HOURS = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12, "1d": 24}
CANDLE_HOURS = _INTERVAL_HOURS.get(INTERVAL, 1)

# --- Strategy: EMA crossover trend-following ---
EMA_FAST = 9        # fast EMA — short-term momentum
EMA_SLOW = 21       # slow EMA — medium-term momentum
EMA_TREND = 50      # trend EMA — only long when price is above this
RSI_PERIOD = 14
RSI_MIN = 40        # entry requires RSI above this (upside momentum present)
RSI_MAX = 70        # entry blocked above this (overbought)
VOLUME_AVG_PERIOD = 20
ADX_PERIOD = 14
ADX_MIN = 25        # minimum ADX — require trend strength (not ranging)

# --- Risk ---
POSITION_SIZE_PCT = Decimal("0.10")   # 10% of available balance
STOP_LOSS_PCT = Decimal("0.02")       # 2% below entry
TAKE_PROFIT_PCT = Decimal("0.06")     # 6% above entry (3:1 reward/risk ratio)
MAX_DAILY_LOSS_PCT = Decimal("0.05")  # pause if balance drops 5% in a UTC day
MAX_HOLD_HOURS = 48                               # maximum hold time in real hours
MAX_HOLD_CANDLES = MAX_HOLD_HOURS // CANDLE_HOURS  # same limit expressed in candles

# --- Binance order precision ---
QUANTITY_STEP = Decimal("0.00001")    # BTC/USDT LOT_SIZE stepSize
PRICE_TICK = Decimal("0.01")          # BTC/USDT PRICE_FILTER tickSize
MIN_NOTIONAL = Decimal("10")          # Binance minimum notional in USDT

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# --- Grid trading ---
GRID_LOWER = Decimal(os.getenv("GRID_LOWER", "75000"))   # lowest buy level (USDT)
GRID_UPPER = Decimal(os.getenv("GRID_UPPER", "95000"))   # highest sell level (USDT)
GRID_COUNT = int(os.getenv("GRID_COUNT", "10"))          # number of grid levels
GRID_CAPITAL_USDT = Decimal(os.getenv("GRID_CAPITAL_USDT", "500"))  # total capital to deploy
GRID_CHECK_SECONDS = int(os.getenv("GRID_CHECK_SECONDS", "60"))     # poll interval
