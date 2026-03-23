"""yfinance-based stocks data provider."""
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from src.data.providers.base import DataProvider

logger = logging.getLogger(__name__)

# yfinance native timeframe strings
_YF_FETCH_TF = {"1h": "1h", "4h": "1h", "1d": "1d"}
_RESAMPLE_NEEDED = {"4h": "4h"}


class StocksProvider(DataProvider):
    """Fetches US stock OHLCV data via yfinance."""

    @property
    def name(self) -> str:
        return "yfinance_stocks"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: datetime, until: datetime) -> pd.DataFrame:
        """Download OHLCV from yfinance, resampling to 4h when needed."""
        yf_tf = _YF_FETCH_TF.get(timeframe)
        if yf_tf is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}' for StocksProvider. "
                f"Supported: {list(_YF_FETCH_TF)}"
            )

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
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()
