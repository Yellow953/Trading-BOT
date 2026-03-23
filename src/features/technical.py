"""Technical indicator features — all normalized/relative, no raw prices."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Warmup rows needed before all indicators are valid (longest lookback: EMA200 + Ichimoku shift 26)
WARMUP_PERIODS = 230


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized technical indicator columns to df.
    Input: OHLCV DataFrame. Returns df with additional feature columns appended.
    All values are relative — no raw price scale.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- RSI ---
    df["rsi_14"] = _rsi(close, 14)
    df["rsi_28"] = _rsi(close, 28)

    # --- ATR (used for normalizing MACD histogram) ---
    atr = _atr(high, low, close, 14)

    # --- MACD histogram normalized by ATR ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (macd_line - signal_line) / atr.replace(0, np.nan)

    # --- ADX ---
    df["adx_14"] = _adx(high, low, close, 14)

    # --- Aroon oscillator ---
    df["aroon_osc"] = _aroon_oscillator(high, low, 25)

    # --- SMA/EMA crossover ratios ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df["ema_20_50_ratio"] = (ema20 / ema50) - 1.0
    df["ema_50_200_ratio"] = (ema50 / ema200) - 1.0

    # --- Ichimoku cloud distance ---
    df["ichimoku_cloud_dist"] = _ichimoku_cloud_distance(high, low, close)

    return df


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed ADX (0–100 scale)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def _aroon_oscillator(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    aroon_up = high.rolling(period + 1).apply(
        lambda x: x.argmax() / period * 100, raw=True
    )
    aroon_down = low.rolling(period + 1).apply(
        lambda x: x.argmin() / period * 100, raw=True
    )
    return aroon_up - aroon_down


def _ichimoku_cloud_distance(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Distance of close from Ichimoku cloud midpoint, as % of price."""
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    cloud_mid = (senkou_a + senkou_b) / 2
    return (close - cloud_mid) / close.replace(0, np.nan)
