from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pandas_ta_classic as ta

import config

logger = logging.getLogger(__name__)

_ADX_COL = f"ADX_{config.ADX_PERIOD}"
_RSI_COL = f"RSI_{config.RSI_PERIOD}"


def passes_filters(
    df: pd.DataFrame,
    adx_min: Optional[float] = None,
) -> bool:
    """Entry filters for EMA crossover strategy.

    Checks performed on df.iloc[-2] (last closed candle):
      1. RSI in range [RSI_MIN, RSI_MAX] — trending up, not overbought
      2. Volume above 20-period average — confirms genuine move
      3. ADX above minimum — market must be trending, not ranging

    Args:
        df:      OHLCV DataFrame.
        adx_min: Override for config.ADX_MIN (used by optimizer).
    """
    min_rows = config.ADX_PERIOD + config.VOLUME_AVG_PERIOD + 5
    if len(df) < min_rows:
        logger.warning("Not enough candles to evaluate filters")
        return False

    candle = df.iloc[-2]

    # --- 1. RSI range filter ---
    rsi = ta.rsi(df["close"], length=config.RSI_PERIOD)
    if rsi is None or pd.isna(rsi.iloc[-2]):
        logger.warning("Could not compute RSI — skipping RSI filter")
        return False

    rsi_val = float(rsi.iloc[-2])
    if not (float(config.RSI_MIN) <= rsi_val <= float(config.RSI_MAX)):
        logger.info("RSI filter FAILED: RSI=%.1f not in [%.0f, %.0f]",
                    rsi_val, config.RSI_MIN, config.RSI_MAX)
        return False

    logger.debug("RSI filter passed: %.1f", rsi_val)

    # --- 2. Volume filter ---
    vol_window = df["volume"].iloc[-config.VOLUME_AVG_PERIOD - 2 : -2]
    avg_volume = vol_window.mean()
    signal_volume = candle["volume"]

    if signal_volume <= avg_volume:
        logger.info("Volume filter FAILED: signal vol=%.2f <= avg=%.2f",
                    signal_volume, avg_volume)
        return False

    logger.debug("Volume filter passed: %.2f > %.2f", signal_volume, avg_volume)

    # --- 3. ADX minimum — market must be trending ---
    adx_threshold = adx_min if adx_min is not None else float(config.ADX_MIN)

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=config.ADX_PERIOD)
    if adx_df is None or _ADX_COL not in adx_df.columns or pd.isna(adx_df[_ADX_COL].iloc[-2]):
        logger.warning("Could not compute ADX — skipping ADX filter")
        return False

    adx_val = float(adx_df[_ADX_COL].iloc[-2])

    if adx_val < adx_threshold:
        logger.info("ADX filter FAILED: ADX=%.1f < %.1f (market is ranging)",
                    adx_val, adx_threshold)
        return False

    logger.debug("ADX filter passed: ADX=%.1f >= %.1f", adx_val, adx_threshold)
    return True
