from __future__ import annotations

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
import config

logger = logging.getLogger(__name__)


class DailyGuard:
    """Enforces the max daily loss rule.

    Tracks the starting balance for each UTC calendar day. If the current
    balance has dropped more than MAX_DAILY_LOSS_PCT from the day's start,
    trading is paused until the next UTC midnight.
    """

    def __init__(self) -> None:
        self._start_balance: Optional[Decimal] = None
        self._day: Optional[date] = None

    def reset(self, balance: Decimal) -> None:
        """Record the starting balance for today. Call at bot startup."""
        self._start_balance = balance
        self._day = datetime.utcnow().date()
        logger.info("Daily guard reset: start balance = %s USDT on %s", balance, self._day)

    def is_trading_allowed(self, current_balance: Decimal) -> bool:
        """Return False if the daily loss limit has been breached."""
        today = datetime.utcnow().date()

        # Auto-reset at UTC midnight
        if self._day is None or today != self._day:
            logger.info("New UTC day — resetting daily guard")
            self.reset(current_balance)
            return True

        if self._start_balance is None or self._start_balance == 0:
            logger.warning("Daily guard has no start balance — allowing trading")
            return True

        loss_pct = (self._start_balance - current_balance) / self._start_balance

        if loss_pct >= config.MAX_DAILY_LOSS_PCT:
            logger.warning(
                "Daily loss limit reached: %.2f%% loss (limit %.2f%%). Trading paused until UTC midnight.",
                float(loss_pct) * 100,
                float(config.MAX_DAILY_LOSS_PCT) * 100,
            )
            return False

        return True
