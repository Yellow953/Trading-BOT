"""Grid level calculations — pure functions, no side effects."""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import List, Optional

import config


def compute_levels(lower: Decimal, upper: Decimal, count: int) -> List[Decimal]:
    """Return `count` evenly-spaced price levels from lower to upper (inclusive).

    Levels are quantized to PRICE_TICK and returned in ascending order.
    Requires count >= 2.
    """
    if count < 2:
        raise ValueError("GRID_COUNT must be >= 2")
    step = (upper - lower) / (count - 1)
    levels = []
    for i in range(count):
        raw = lower + step * i
        levels.append(raw.quantize(config.PRICE_TICK, rounding=ROUND_DOWN))
    return levels


def quantity_for_level(capital_per_level: Decimal, level_price: Decimal) -> Decimal:
    """Return the BTC quantity to buy at a given level.

    capital_per_level = GRID_CAPITAL_USDT / (GRID_COUNT - 1)
    """
    qty = capital_per_level / level_price
    return qty.quantize(config.QUANTITY_STEP, rounding=ROUND_DOWN)


def level_above(levels: List[Decimal], price: Decimal) -> Optional[Decimal]:
    """Return the first level strictly above `price`, or None if at the top."""
    for lvl in sorted(levels):
        if lvl > price:
            return lvl
    return None


def level_below(levels: List[Decimal], price: Decimal) -> Optional[Decimal]:
    """Return the first level strictly below `price`, or None if at the bottom."""
    for lvl in sorted(levels, reverse=True):
        if lvl < price:
            return lvl
    return None
