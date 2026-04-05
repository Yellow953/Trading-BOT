from __future__ import annotations

import time
import logging
from typing import Optional
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import config

logger = logging.getLogger(__name__)

_client = Client(
    config.BINANCE_API_KEY,
    config.BINANCE_API_SECRET,
    testnet=config.BINANCE_TESTNET,
)

_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

_NUMERIC_COLS = ["open", "high", "low", "close", "volume"]


def get_candles(
    symbol: str = config.SYMBOL,
    interval: str = config.INTERVAL,
    limit: int = config.CANDLE_LIMIT,
) -> pd.DataFrame:
    """Return a DataFrame of OHLCV candles, sorted ascending by open_time.

    Retries up to 3 times with exponential backoff on transient errors.
    Raises on persistent failure so the caller can skip the cycle.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            raw = _client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(raw, columns=_KLINE_COLUMNS)
            for col in _NUMERIC_COLS:
                df[col] = df[col].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df.reset_index(drop=True)
            logger.debug("Fetched %d candles for %s %s", len(df), symbol, interval)
            return df
        except BinanceAPIException as exc:
            logger.warning("Binance API error on attempt %d: %s", attempt + 1, exc)
            last_exc = exc
        except Exception as exc:
            logger.warning("Network error on attempt %d: %s", attempt + 1, exc)
            last_exc = exc
        time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch candles after 3 attempts") from last_exc
