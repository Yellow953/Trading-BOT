from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from typing import Optional
from binance.exceptions import BinanceAPIException
import config

logger = logging.getLogger(__name__)


@dataclass
class PositionTracker:
    """Tracks the single open position state in memory.

    On bot restart, call restore_from_exchange() to re-sync state
    from any live open orders on Binance.
    """

    is_open: bool = False
    entry_price: Optional[Decimal] = None
    entry_time: Optional[datetime] = None
    quantity: Optional[Decimal] = None
    oco_order_list_id: Optional[int] = None

    def open_position(
        self,
        entry_price: Decimal,
        quantity: Decimal,
        oco_order_list_id: int,
    ) -> None:
        self.is_open = True
        self.entry_price = entry_price
        self.entry_time = datetime.utcnow()
        self.quantity = quantity
        self.oco_order_list_id = oco_order_list_id
        logger.info(
            "Position opened: entry=%.2f qty=%s oco=%s",
            float(entry_price), quantity, oco_order_list_id,
        )

    def close_position(self) -> None:
        logger.info(
            "Position closed: entry=%.2f qty=%s",
            float(self.entry_price) if self.entry_price else 0,
            self.quantity,
        )
        self.is_open = False
        self.entry_price = None
        self.entry_time = None
        self.quantity = None
        self.oco_order_list_id = None

    def is_expired(self) -> bool:
        """Return True if position has been open longer than MAX_HOLD_HOURS."""
        if not self.is_open or self.entry_time is None:
            return False
        elapsed = datetime.utcnow() - self.entry_time
        return elapsed.total_seconds() > config.MAX_HOLD_HOURS * 3600

    def restore_from_exchange(self) -> None:
        """On startup, check Binance for open orders and restore position state.

        If OCO orders are found, infers entry_price from the stop-limit price.
        Sets entry_time conservatively to now - 1h (unknown actual entry time).
        """
        # Import here to avoid circular import at module level
        from binance.client import Client
        client = Client(
            config.BINANCE_API_KEY,
            config.BINANCE_API_SECRET,
            testnet=config.BINANCE_TESTNET,
        )
        try:
            open_orders = client.get_open_orders(symbol=config.SYMBOL)
            if not open_orders:
                logger.info("No open orders found on Binance — no position to restore")
                return

            # Find any stop-limit order to infer entry price
            sl_order = next(
                (o for o in open_orders if o["type"] in ("STOP_LOSS_LIMIT", "STOP_LIMIT_SELL")),
                None,
            )
            if sl_order is None:
                sl_order = next(
                    (o for o in open_orders if "stopPrice" in o and o["stopPrice"] != "0.00000000"),
                    None,
                )

            if sl_order:
                sl_price = Decimal(sl_order["stopPrice"])
                inferred_entry = sl_price / (1 - config.STOP_LOSS_PCT)
                inferred_qty = Decimal(sl_order["origQty"])
                oco_id = sl_order.get("orderListId", -1)

                self.is_open = True
                self.entry_price = inferred_entry
                self.entry_time = datetime.utcnow() - timedelta(hours=1)
                self.quantity = inferred_qty
                self.oco_order_list_id = oco_id

                logger.warning(
                    "Restored position from exchange: inferred entry=%.2f qty=%s oco=%s",
                    float(inferred_entry), inferred_qty, oco_id,
                )
            else:
                logger.warning(
                    "Open orders found but could not infer position — manual review needed: %s",
                    [o["orderId"] for o in open_orders],
                )

        except BinanceAPIException as exc:
            logger.error("Failed to restore position from exchange: %s", exc)
