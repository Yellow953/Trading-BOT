"""Grid trading engine — manages slots, detects fills, places counter-orders."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from decimal import Decimal
from typing import List, Optional

import config
import notifications.telegram as telegram
from execution.order_manager import (
    place_limit_buy,
    place_limit_sell,
    cancel_all_open_orders,
    get_open_order_ids,
    get_current_price,
)
from grid.calculator import compute_levels, quantity_for_level, level_above

logger = logging.getLogger(__name__)

_STATE_FILE = "grid_state.json"


@dataclass
class GridSlot:
    level_price: Decimal
    state: str = "empty"           # "empty" | "buy_placed" | "sell_placed"
    buy_order_id: int = -1
    sell_order_id: int = -1
    quantity: Decimal = Decimal("0")
    buy_price: Decimal = Decimal("0")
    sell_price: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert Decimals to strings for JSON serialisation
        for key in ("level_price", "quantity", "buy_price", "sell_price"):
            d[key] = str(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GridSlot":
        return cls(
            level_price=Decimal(d["level_price"]),
            state=d["state"],
            buy_order_id=int(d["buy_order_id"]),
            sell_order_id=int(d["sell_order_id"]),
            quantity=Decimal(d["quantity"]),
            buy_price=Decimal(d["buy_price"]),
            sell_price=Decimal(d["sell_price"]),
        )


class GridEngine:
    def __init__(self) -> None:
        self.levels: List[Decimal] = compute_levels(
            config.GRID_LOWER, config.GRID_UPPER, config.GRID_COUNT
        )
        # Capital per buy level (count-1 intervals, top level is sell-only)
        self._capital_per_level = config.GRID_CAPITAL_USDT / (config.GRID_COUNT - 1)
        self.slots: List[GridSlot] = []
        self.realized_pnl: Decimal = Decimal("0")
        self.cycles: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load persisted state or do a fresh setup."""
        if self._load_state():
            logger.info("Grid state restored from %s — running check_and_react", _STATE_FILE)
            # Handle any fills that occurred while we were offline
            self.check_and_react()
        else:
            logger.info("No saved state — performing fresh grid setup")
            self.setup()

    def setup(self) -> None:
        """Place limit buy orders at all levels below current price."""
        current = get_current_price()
        logger.info("Setting up grid — current price: %.2f", float(current))

        self.slots = []
        for level in self.levels:
            slot = GridSlot(level_price=level)
            if level < current:
                qty = quantity_for_level(self._capital_per_level, level)
                if qty <= Decimal("0"):
                    logger.warning("Qty rounds to 0 at level %s — skipping", level)
                    self.slots.append(slot)
                    continue
                try:
                    order = place_limit_buy(level, qty)
                    slot.buy_order_id = int(order["orderId"])
                    slot.quantity = qty
                    slot.state = "buy_placed"
                    logger.info("Buy placed @ %s (orderId=%s)", level, slot.buy_order_id)
                except Exception as exc:
                    logger.error("Failed to place buy @ %s: %s", level, exc)
            else:
                logger.debug("Level %s >= current price %s — no buy placed", level, current)
            self.slots.append(slot)

        self._save_state()
        logger.info(
            "Grid setup complete: %d levels, %d buy orders placed",
            len(self.levels),
            sum(1 for s in self.slots if s.state == "buy_placed"),
        )

    def check_and_react(self) -> None:
        """Detect fills (missing order IDs) and place counter-orders."""
        try:
            open_ids = get_open_order_ids()
        except Exception as exc:
            logger.error("Cannot fetch open order IDs — skipping cycle: %s", exc)
            return

        changed = False

        for slot in self.slots:
            if slot.state == "buy_placed" and slot.buy_order_id != -1:
                if slot.buy_order_id not in open_ids:
                    changed = True
                    self._on_buy_filled(slot)

            elif slot.state == "sell_placed" and slot.sell_order_id != -1:
                if slot.sell_order_id not in open_ids:
                    changed = True
                    self._on_sell_filled(slot)

        if changed:
            self._save_state()

    def shutdown(self) -> None:
        """Cancel all open orders and persist final state."""
        logger.info("Grid shutdown — cancelling all open orders")
        try:
            cancel_all_open_orders()
        except Exception as exc:
            logger.error("Error cancelling orders on shutdown: %s", exc)
        self._save_state()
        logger.info("Grid state saved. Realized PnL: %.4f USDT | Cycles: %d",
                    float(self.realized_pnl), self.cycles)

    def daily_summary(self) -> None:
        """Send a Telegram summary of today's grid performance."""
        open_buys = sum(1 for s in self.slots if s.state == "buy_placed")
        open_sells = sum(1 for s in self.slots if s.state == "sell_placed")

        # Unrealized PnL: sum of (current_price - buy_price) * qty for sell_placed slots
        try:
            current = get_current_price()
        except Exception:
            current = Decimal("0")

        unrealized = Decimal("0")
        for slot in self.slots:
            if slot.state == "sell_placed" and slot.buy_price > 0:
                unrealized += (current - slot.buy_price) * slot.quantity

        msg = telegram.fmt_grid_summary(
            realized=self.realized_pnl,
            unrealized=unrealized,
            cycles=self.cycles,
            open_buys=open_buys,
            open_sells=open_sells,
        )
        telegram.send_message(msg)
        logger.info("Daily summary sent")

    # ------------------------------------------------------------------
    # Fill handlers
    # ------------------------------------------------------------------

    def _on_buy_filled(self, slot: GridSlot) -> None:
        """A buy order has been filled — place the corresponding sell one level above."""
        slot.buy_price = slot.level_price  # limit order fills at or better than limit price
        sell_level = level_above(self.levels, slot.level_price)

        if sell_level is None:
            # We are at the top level — nothing above to sell to; mark empty
            logger.warning("Buy filled at top level %s — no sell level above", slot.level_price)
            slot.state = "empty"
            slot.buy_order_id = -1
            return

        slot.sell_price = sell_level
        try:
            order = place_limit_sell(sell_level, slot.quantity)
            slot.sell_order_id = int(order["orderId"])
            slot.state = "sell_placed"
            logger.info(
                "Buy filled @ %s — sell placed @ %s (orderId=%s)",
                slot.level_price, sell_level, slot.sell_order_id,
            )
            telegram.send_message(
                telegram.fmt_grid_buy_fill(slot.level_price, sell_level, slot.quantity)
            )
        except Exception as exc:
            logger.error("Failed to place sell after buy fill @ %s: %s", slot.level_price, exc)
            # Leave state as buy_placed so next cycle doesn't lose track
            # Reset buy_order_id so we don't mis-detect as filled again
            slot.buy_order_id = -1
            slot.state = "empty"

    def _on_sell_filled(self, slot: GridSlot) -> None:
        """A sell order has been filled — record profit and re-place the buy."""
        profit = (slot.sell_price - slot.buy_price) * slot.quantity
        self.realized_pnl += profit
        self.cycles += 1

        logger.info(
            "Sell filled @ %s (buy was @ %s) — profit=%.4f USDT | total_pnl=%.4f USDT | cycles=%d",
            slot.sell_price, slot.buy_price, float(profit),
            float(self.realized_pnl), self.cycles,
        )
        telegram.send_message(
            telegram.fmt_grid_cycle(
                buy_price=slot.buy_price,
                sell_price=slot.sell_price,
                quantity=slot.quantity,
                profit=profit,
                total_pnl=self.realized_pnl,
                cycles=self.cycles,
            )
        )

        # Re-place the buy at this level
        qty = quantity_for_level(self._capital_per_level, slot.level_price)
        slot.sell_order_id = -1
        slot.sell_price = Decimal("0")
        slot.buy_price = Decimal("0")

        try:
            order = place_limit_buy(slot.level_price, qty)
            slot.buy_order_id = int(order["orderId"])
            slot.quantity = qty
            slot.state = "buy_placed"
            logger.info("Re-placed buy @ %s (orderId=%s)", slot.level_price, slot.buy_order_id)
        except Exception as exc:
            logger.error("Failed to re-place buy @ %s: %s", slot.level_price, exc)
            slot.state = "empty"
            slot.buy_order_id = -1

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        data = {
            "realized_pnl": str(self.realized_pnl),
            "cycles": self.cycles,
            "slots": [s.to_dict() for s in self.slots],
        }
        tmp = _STATE_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, _STATE_FILE)
        except Exception as exc:
            logger.error("Failed to save grid state: %s", exc)

    def _load_state(self) -> bool:
        """Load state from disk. Returns True if successful."""
        if not os.path.exists(_STATE_FILE):
            return False
        try:
            with open(_STATE_FILE) as f:
                data = json.load(f)
            self.realized_pnl = Decimal(data["realized_pnl"])
            self.cycles = int(data["cycles"])
            self.slots = [GridSlot.from_dict(s) for s in data["slots"]]
            logger.info(
                "Loaded grid state: %d slots | realized_pnl=%.4f | cycles=%d",
                len(self.slots), float(self.realized_pnl), self.cycles,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load grid state — starting fresh: %s", exc)
            return False
