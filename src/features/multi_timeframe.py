"""Multi-timeframe features — RSI, MACD, ADX computed at higher timeframes."""
import logging

import pandas as pd

from src.features.technical import _rsi, _atr, _adx

logger = logging.getLogger(__name__)

_HIGHER_TF_MAP = {
    "1h": ["4h", "1d"],
    "4h": ["1d"],
    "1d": [],
}

_RESAMPLE_RULE = {
    "4h": "4h",
    "1d": "1D",
}


def add_multi_timeframe_features(df: pd.DataFrame, base_timeframe: str = "1h") -> pd.DataFrame:
    """
    For each higher timeframe, resample, compute indicators, then left-join to
    base timeframe using shift(1) + ffill only. No lookahead.
    """
    higher_tfs = _HIGHER_TF_MAP.get(base_timeframe, [])

    for tf in higher_tfs:
        rule = _RESAMPLE_RULE[tf]
        resampled = _resample_ohlcv(df, rule)
        if resampled.empty:
            continue

        close = resampled["close"]
        high = resampled["high"]
        low = resampled["low"]

        indicators = pd.DataFrame(index=resampled.index)
        indicators[f"tf_{tf}_rsi_14"] = _rsi(close, 14)
        atr = _atr(high, low, close, 14)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
        indicators[f"tf_{tf}_macd_hist_norm"] = macd_hist / atr.replace(0, float("nan"))
        indicators[f"tf_{tf}_adx_14"] = _adx(high, low, close, 14)

        # Drop indicator columns that are entirely NaN (insufficient resampled rows)
        indicators = indicators.dropna(axis=1, how="all")
        if indicators.empty:
            logger.debug("Skipping timeframe %s — all indicators are NaN (insufficient data)", tf)
            continue

        # shift(1): completed candle at t is available at t+1 — prevents lookahead
        indicators = indicators.shift(1)

        # reindex to base timeframe — forward-fill only (NOT backward-fill)
        indicators = indicators.reindex(df.index, method="ffill")

        for col in indicators.columns:
            df[col] = indicators[col]

    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample base OHLCV to a higher timeframe."""
    return df.resample(rule, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
