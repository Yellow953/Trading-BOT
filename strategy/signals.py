from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pandas_ta_classic as ta

import config

logger = logging.getLogger(__name__)


def get_signal(df: pd.DataFrame) -> Optional[str]:
    """EMA crossover trend-following signal.

    Evaluates the last closed candle (df.iloc[-2]) and the one before it
    (df.iloc[-3]) to detect EMA crossovers.

    Returns:
        "long"  — golden cross: fast EMA crosses above slow EMA, price above trend EMA
        "exit"  — death cross: fast EMA crosses below slow EMA, OR price drops below trend EMA
        None    — no actionable signal
    """
    if len(df) < config.EMA_TREND + 3:
        logger.warning("Not enough candles for EMA signal (need %d, got %d)",
                       config.EMA_TREND + 3, len(df))
        return None

    ema_fast = ta.ema(df["close"], length=config.EMA_FAST)
    ema_slow = ta.ema(df["close"], length=config.EMA_SLOW)
    ema_trend = ta.ema(df["close"], length=config.EMA_TREND)

    if ema_fast is None or ema_slow is None or ema_trend is None:
        logger.warning("Could not compute EMAs")
        return None

    # Signal candle = iloc[-2] (last closed), previous = iloc[-3]
    fast_now  = ema_fast.iloc[-2]
    fast_prev = ema_fast.iloc[-3]
    slow_now  = ema_slow.iloc[-2]
    slow_prev = ema_slow.iloc[-3]
    trend_now = ema_trend.iloc[-2]
    close_now = df["close"].iloc[-2]

    if any(pd.isna(v) for v in [fast_now, fast_prev, slow_now, slow_prev, trend_now]):
        logger.warning("NaN in EMA values — skipping signal")
        return None

    golden_cross = fast_prev <= slow_prev and fast_now > slow_now
    above_trend  = close_now > trend_now
    death_cross  = fast_prev >= slow_prev and fast_now < slow_now
    below_trend  = close_now < trend_now

    logger.debug(
        "EMA signal check — fast=%.2f slow=%.2f trend=%.2f close=%.2f",
        fast_now, slow_now, trend_now, close_now,
    )

    if golden_cross and above_trend:
        logger.info("LONG signal: golden cross (fast=%.2f > slow=%.2f), price above trend=%.2f",
                    fast_now, slow_now, trend_now)
        return "long"

    if death_cross or below_trend:
        logger.info("EXIT signal: %s",
                    "death cross" if death_cross else "price below trend EMA")
        return "exit"

    return None
