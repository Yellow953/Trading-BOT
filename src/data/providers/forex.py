"""yfinance-based forex data provider."""
import logging
from datetime import datetime

from src.data.providers.stocks import StocksProvider

logger = logging.getLogger(__name__)


class ForexProvider(StocksProvider):
    """
    Forex pairs via yfinance (e.g. EURUSD=X).
    Identical behaviour to StocksProvider — same yfinance API.
    """

    @property
    def name(self) -> str:
        return "yfinance_forex"

    def available_symbols(self) -> list[str]:
        return self.config.get("markets", {}).get("forex", {}).get("symbols", [])
