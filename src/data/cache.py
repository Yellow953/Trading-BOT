"""SQLite cache for OHLCV data. Key: (provider_name, symbol, timeframe, ts)."""
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    provider    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    ts          INTEGER NOT NULL,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    PRIMARY KEY (provider, symbol, timeframe, ts)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    provider    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    since_ts    INTEGER NOT NULL,
    until_ts    INTEGER NOT NULL,
    PRIMARY KEY (provider, symbol, timeframe, since_ts, until_ts)
)
"""


class DataCache:
    """Read-through cache backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get_or_fetch(self, provider, symbol: str, timeframe: str,
                     since: datetime, until: datetime) -> pd.DataFrame:
        """Return cached data; fetch from provider only for missing ranges."""
        # Check if this exact range (or a superset) has already been fetched.
        if self._range_covered(provider.name, symbol, timeframe, since, until):
            logger.debug("Cache hit: %s %s %s", provider.name, symbol, timeframe)
            return self._read(provider.name, symbol, timeframe, since, until)

        cached = self._read(provider.name, symbol, timeframe, since, until)

        if cached.empty:
            logger.info("Cache miss: %s %s %s", provider.name, symbol, timeframe)
            fresh = provider.fetch_ohlcv(symbol, timeframe, since, until)
            self._write(provider.name, symbol, timeframe, fresh)
            self._log_fetch(provider.name, symbol, timeframe, since, until)
            return fresh

        cached_until = cached.index.max().to_pydatetime().replace(tzinfo=timezone.utc)
        fetch_since = max(since, cached_until)
        logger.info("Partial cache miss: fetching from %s", fetch_since)
        fresh = provider.fetch_ohlcv(symbol, timeframe, fetch_since, until)
        if not fresh.empty:
            self._write(provider.name, symbol, timeframe, fresh)
            combined = pd.concat([cached, fresh]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            self._log_fetch(provider.name, symbol, timeframe, since, until)
            return combined[(combined.index >= since) & (combined.index <= until)]

        self._log_fetch(provider.name, symbol, timeframe, since, until)
        return cached[(cached.index >= since) & (cached.index <= until)]

    def _read(self, provider_name: str, symbol: str, timeframe: str,
              since: datetime, until: datetime) -> pd.DataFrame:
        since_ts = int(since.timestamp())
        until_ts = int(until.timestamp())
        rows = self._conn.execute(
            "SELECT ts, open, high, low, close, volume FROM ohlcv "
            "WHERE provider=? AND symbol=? AND timeframe=? AND ts>=? AND ts<=? ORDER BY ts",
            (provider_name, symbol, timeframe, since_ts, until_ts),
        ).fetchall()

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
        return pd.DataFrame({
            "open":   [r[1] for r in rows],
            "high":   [r[2] for r in rows],
            "low":    [r[3] for r in rows],
            "close":  [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        }, index=idx)

    def _range_covered(self, provider_name: str, symbol: str, timeframe: str,
                       since: datetime, until: datetime) -> bool:
        """Return True if a previous fetch already covered [since, until]."""
        since_ts = int(since.timestamp())
        until_ts = int(until.timestamp())
        row = self._conn.execute(
            "SELECT 1 FROM fetch_log WHERE provider=? AND symbol=? AND timeframe=? "
            "AND since_ts<=? AND until_ts>=? LIMIT 1",
            (provider_name, symbol, timeframe, since_ts, until_ts),
        ).fetchone()
        return row is not None

    def _log_fetch(self, provider_name: str, symbol: str, timeframe: str,
                   since: datetime, until: datetime) -> None:
        """Record that [since, until] has been fetched from the provider."""
        self._conn.execute(
            "INSERT OR REPLACE INTO fetch_log (provider, symbol, timeframe, since_ts, until_ts) "
            "VALUES (?,?,?,?,?)",
            (provider_name, symbol, timeframe, int(since.timestamp()), int(until.timestamp())),
        )
        self._conn.commit()

    def _write(self, provider_name: str, symbol: str, timeframe: str,
               df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = [
            (provider_name, symbol, timeframe, int(ts.timestamp()),
             row["open"], row["high"], row["low"], row["close"], row["volume"])
            for ts, row in df.iterrows()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO ohlcv "
            "(provider, symbol, timeframe, ts, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
