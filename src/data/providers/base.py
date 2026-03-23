"""Abstract DataProvider interface."""
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataProvider(ABC):
    """Common interface for all market data sources."""

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame:
        """Return DataFrame with columns [open, high, low, close, volume], UTC datetime index."""

    @abstractmethod
    def available_symbols(self) -> list[str]:
        """Return list of tradeable symbols."""

    @abstractmethod
    def available_timeframes(self) -> list[str]:
        """Return list of supported timeframe strings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string (e.g. 'ccxt_binance', 'yfinance_stocks')."""
