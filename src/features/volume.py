"""Volume-based features — OBV slope, volume ratio, VWAP distance, CMF."""
import numpy as np
import pandas as pd


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append volume feature columns. All normalized/relative."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # OBV (cumulative), then 20-period linear regression slope, normalized
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    df["obv_slope_norm"] = _linreg_slope(obv, 20) / (close * 0.001).replace(0, np.nan)

    # Volume ratio: current / 20-period SMA
    vol_sma = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_sma.replace(0, np.nan)

    # Rolling VWAP distance: (close - vwap) / close
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).rolling(20).sum()
    cum_vol = volume.rolling(20).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap_dist"] = (close - vwap) / close.replace(0, np.nan)

    # Chaikin Money Flow (20-period)
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    df["cmf_20"] = (clv * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    return df


def _linreg_slope(series: pd.Series, period: int) -> pd.Series:
    """Rolling linear regression slope over `period` bars."""
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def slope(window: np.ndarray) -> float:
        y_mean = window.mean()
        return float(((x - x_mean) * (window - y_mean)).sum() / x_var)

    return series.rolling(period).apply(slope, raw=True)
