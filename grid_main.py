"""Entry point for the BTC/USDT grid trading bot."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import config
from grid.engine import GridEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)

_handler = logging.handlers.TimedRotatingFileHandler(
    "logs/grid.log", when="midnight", utc=True, backupCount=30
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    mode = "TESTNET" if config.BINANCE_TESTNET else "LIVE"
    logger.info("Starting BTC/USDT grid bot — mode: %s", mode)
    logger.info(
        "Grid config: lower=%.2f upper=%.2f levels=%d capital=%.2f USDT",
        float(config.GRID_LOWER), float(config.GRID_UPPER),
        config.GRID_COUNT, float(config.GRID_CAPITAL_USDT),
    )

    engine = GridEngine()
    engine.start()

    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        engine.check_and_react,
        IntervalTrigger(seconds=config.GRID_CHECK_SECONDS),
        id="grid_check",
        name="Grid fill checker",
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        engine.daily_summary,
        CronTrigger(hour=0, minute=1, timezone="UTC"),
        id="daily_summary",
        name="Daily grid summary",
        max_instances=1,
    )

    logger.info(
        "Scheduler started — checking every %ds, daily summary at 00:01 UTC",
        config.GRID_CHECK_SECONDS,
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Grid bot stopped by user")
    finally:
        scheduler.shutdown(wait=False)
        engine.shutdown()


if __name__ == "__main__":
    main()
