"""yfinance-based stocks data provider."""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from src.data.providers.base import DataProvider

logger = logging.getLogger(__name__)

# yfinance native timeframe → fetch interval string
_YF_FETCH_TF = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "1h",   # fetch 1h, resample to 4h
    "1d":  "1d",
}

# yfinance max lookback in days per timeframe (None = unlimited)
_MAX_LOOKBACK_DAYS = {
    "1m":  7,
    "5m":  60,
    "15m": 60,
    "1h":  730,
    "4h":  730,   # uses 1h data under the hood
    "1d":  None,
}

_RESAMPLE_NEEDED = {"4h": "4h"}


class StocksProvider(DataProvider):
    """Fetches US stock OHLCV data via yfinance."""

    @property
    def name(self) -> str:
        return "yfinance_stocks"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: datetime, until: datetime) -> pd.DataFrame:
        yf_tf = _YF_FETCH_TF.get(timeframe)
        if yf_tf is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}' for StocksProvider. "
                f"Supported: {list(_YF_FETCH_TF)}"
            )

        # Clamp `since` to yfinance window limit
        max_days = _MAX_LOOKBACK_DAYS.get(timeframe)
        if max_days is not None:
            earliest_allowed = datetime.now(tz=timezone.utc) - timedelta(days=max_days)
            if since < earliest_allowed:
                logger.warning(
                    "Requested since=%s exceeds yfinance %s limit (%d days). Clamping to %s.",
                    since, timeframe, max_days, earliest_allowed,
                )
                since = earliest_allowed

        raw = yf.download(
            symbol,
            start=since.strftime("%Y-%m-%d"),
            end=until.strftime("%Y-%m-%d"),
            interval=yf_tf,
            auto_adjust=True,
            progress=False,
        )
        df = self._normalize(raw)

        if timeframe in _RESAMPLE_NEEDED and not df.empty:
            rule = _RESAMPLE_NEEDED[timeframe]
            df = df.resample(rule, label="left", closed="left").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

        return df

    def available_symbols(self) -> list[str]:
        return self.config.get("markets", {}).get("stocks", {}).get("symbols", [])

    def available_timeframes(self) -> list[str]:
        return list(_YF_FETCH_TF.keys())

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = df.copy()
        # Handle both flat and MultiIndex columns (yfinance behavior varies by version)
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()
