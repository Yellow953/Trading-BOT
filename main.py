"""Entry point — initialises all modules and starts the bot loop."""

import logging
import logging.handlers
import os
import sys
from decimal import Decimal

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import config
from data.fetcher import get_candles
from strategy.signals import get_signal
from strategy.filters import passes_filters
from risk.sizing import calculate_quantity
from risk.daily_guard import DailyGuard
from execution.order_manager import (
    get_usdt_balance,
    place_market_buy,
    place_market_sell,
    place_oco_sell,
    cancel_all_open_orders,
)
from execution.position_tracker import PositionTracker
import notifications.telegram as telegram

# ---------------------------------------------------------------------------
# Logging setup — rotates daily at UTC midnight, matching the daily guard
# ---------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)

_handler = logging.handlers.TimedRotatingFileHandler(
    "logs/bot.log", when="midnight", utc=True, backupCount=30
)
_handler.setFormatter(
    logging.Formatter("%(asctime)s UTC  %(levelname)-8s  %(name)s — %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler, logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module instances
# ---------------------------------------------------------------------------

daily_guard = DailyGuard()
position_tracker = PositionTracker()


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_cycle() -> None:
    """Execute one strategy cycle. Runs once per closed 1H candle."""
    logger.info("--- Cycle start ---")

    # 1. Check daily loss limit
    try:
        balance = get_usdt_balance()
    except Exception as exc:
        logger.error("Could not fetch balance — skipping cycle: %s", exc)
        return

    if not daily_guard.is_trading_allowed(balance):
        logger.info("Daily loss limit active — skipping cycle")
        return

    # 2. Fetch candles
    try:
        df = get_candles()
    except Exception as exc:
        logger.error("Could not fetch candles — skipping cycle: %s", exc)
        return

    # 3. Manage existing position
    if position_tracker.is_open:
        _manage_open_position(df)
        return

    # 4. Look for a new entry
    _look_for_entry(df, balance)

    logger.info("--- Cycle end ---")


def _manage_open_position(df) -> None:
    """Check exit conditions for the current open position."""
    assert position_tracker.entry_price is not None
    assert position_tracker.quantity is not None

    # Force close if position has been open too long
    if position_tracker.is_expired():  # uses MAX_HOLD_HOURS * 3600 seconds internally
        logger.warning("Position expired (%dh max hold) — forcing market close", config.MAX_HOLD_HOURS)
        try:
            cancel_all_open_orders()
            sell_order = place_market_sell(position_tracker.quantity)
            exit_price = _fill_price(sell_order)
            telegram.send_message(
                telegram.fmt_expired_close(position_tracker.entry_price, position_tracker.quantity)
            )
        except Exception as exc:
            logger.error("Failed to force-close expired position: %s", exc)
            return
        position_tracker.close_position()
        return

    # Exit is handled entirely by OCO (TP/SL) and max hold timeout.
    # Death cross signal exits were net negative in backtesting — disabled.
    logger.info("Position open — holding (waiting for OCO TP/SL or timeout)")


def _look_for_entry(df, balance: Decimal) -> None:
    """Evaluate filters and signal for a new long entry."""
    if not passes_filters(df):
        logger.info("Filters not passed — no entry")
        return

    signal = get_signal(df)
    if signal != "long":
        logger.info("No long signal — no entry")
        return

    # Calculate position size using the closed candle close as price estimate
    est_price = Decimal(str(df.iloc[-2]["close"]))
    try:
        quantity = calculate_quantity(balance, est_price)
    except ValueError as exc:
        logger.error("Position sizing error — skipping entry: %s", exc)
        return

    # Place market buy — use actual fill price for SL/TP calculation
    try:
        buy_order = place_market_buy(quantity)
    except Exception as exc:
        logger.error("Market buy failed — skipping entry: %s", exc)
        return

    actual_fill = _fill_price(buy_order)
    actual_qty = _fill_qty(buy_order)

    # Calculate SL/TP from actual fill price
    tp_price = actual_fill * (1 + config.TAKE_PROFIT_PCT)
    sl_trigger = actual_fill * (1 - config.STOP_LOSS_PCT)
    sl_limit = sl_trigger * Decimal("0.999")  # 0.1% below trigger to guarantee fill

    # Place OCO sell (SL + TP simultaneously)
    try:
        oco_order = place_oco_sell(actual_qty, tp_price, sl_trigger, sl_limit)
    except Exception as exc:
        logger.error(
            "OCO order failed after market buy — attempting emergency market sell: %s", exc
        )
        try:
            place_market_sell(actual_qty)
        except Exception as sell_exc:
            logger.critical("Emergency sell also failed — manual intervention required: %s", sell_exc)
        return

    oco_id = oco_order.get("orderListId", -1)
    position_tracker.open_position(actual_fill, actual_qty, oco_id)
    daily_guard.reset(balance)  # refresh start-of-day balance after entering trade

    telegram.send_message(telegram.fmt_entry(actual_fill, sl_trigger, tp_price, actual_qty))
    logger.info(
        "ENTRY: fill=%.2f qty=%s sl=%.2f tp=%.2f oco=%s",
        float(actual_fill), actual_qty, float(sl_trigger), float(tp_price), oco_id,
    )


def _fill_price(order: dict) -> Decimal:
    """Extract the average fill price from a FULL market order response."""
    fills = order.get("fills", [])
    if fills:
        return Decimal(fills[0]["price"])
    # Fallback: use cummulativeQuoteQty / executedQty
    executed = Decimal(order.get("executedQty", "0"))
    cumulative = Decimal(order.get("cummulativeQuoteQty", "0"))
    if executed > 0:
        return cumulative / executed
    raise ValueError(f"Cannot determine fill price from order: {order}")


def _fill_qty(order: dict) -> Decimal:
    """Extract the executed quantity from a market order response."""
    return Decimal(order.get("executedQty", "0"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mode = "TESTNET" if config.BINANCE_TESTNET else "LIVE"
    logger.info("Starting BTC/USDT mean reversion bot — mode: %s", mode)

    # Restore position state from exchange on restart
    position_tracker.restore_from_exchange()

    # Initialise daily guard with current balance
    try:
        balance = get_usdt_balance()
        daily_guard.reset(balance)
    except Exception as exc:
        logger.critical("Cannot fetch initial balance — aborting: %s", exc)
        sys.exit(1)

    logger.info("Initial balance: %.2f USDT", float(balance))

    # Schedule run_cycle at HH:01 UTC (1 minute after candle close to avoid race conditions)
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_cycle,
        CronTrigger(minute=1, timezone="UTC"),
        id="strategy_cycle",
        name="1H strategy cycle",
        max_instances=1,  # prevent overlapping runs
        coalesce=True,    # skip missed runs rather than catching up
    )

    logger.info("Scheduler started — firing at minute=1 of every hour (UTC)")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")


if __name__ == "__main__":
    main()
