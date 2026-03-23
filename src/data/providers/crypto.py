"""CCXT-based crypto data provider (Binance primary, Kraken fallback)."""
import logging
import time
from datetime import datetime, timezone
from typing import List

import ccxt
import pandas as pd

from src.data.providers.base import DataProvider

logger = logging.getLogger(__name__)

_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
_CHUNK_SIZE = 1000


class CryptoProvider(DataProvider):
    """Fetches OHLCV data from Binance via CCXT. Falls back to Kraken on error."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        exchange_id = config.get("markets", {}).get("crypto", {}).get("exchange", "binance")
        self._exchange = self._init_exchange(exchange_id)

    def _init_exchange(self, exchange_id: str):
        try:
            ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            ex.load_markets()
            return ex
        except Exception as e:
            logger.warning("Failed to init %s (%s), falling back to kraken", exchange_id, e)
            ex = ccxt.kraken({"enableRateLimit": True})
            ex.load_markets()
            return ex

    @property
    def name(self) -> str:
        return f"ccxt_{self._exchange.id}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: datetime, until: datetime) -> pd.DataFrame:
        """Paginate CCXT to fetch all candles in [since, until]."""
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        all_rows = []

        cursor = since_ms
        while cursor < until_ms:
            try:
                rows = self._exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=_CHUNK_SIZE)
            except Exception as e:
                logger.error("CCXT fetch error for %s %s: %s", symbol, timeframe, e)
                break

            if not rows:
                break

            all_rows.extend([r for r in rows if r[0] < until_ms])
            last_ts = rows[-1][0]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            if self._exchange.rateLimit:
                time.sleep(self._exchange.rateLimit / 1000)

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        idx = pd.to_datetime([r[0] for r in all_rows], unit="ms", utc=True)
        df = pd.DataFrame({
            "open":   [r[1] for r in all_rows],
            "high":   [r[2] for r in all_rows],
            "low":    [r[3] for r in all_rows],
            "close":  [r[4] for r in all_rows],
            "volume": [r[5] for r in all_rows],
        }, index=idx)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def available_symbols(self) -> list[str]:
        return list(self._exchange.markets.keys()) if self._exchange.markets else []

    def available_timeframes(self) -> list[str]:
        return _TIMEFRAMES
