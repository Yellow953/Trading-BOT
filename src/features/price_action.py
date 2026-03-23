"""Price action features — returns, candle patterns, distance from extremes."""
import numpy as np
import pandas as pd


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append price action feature columns. All values are relative — no raw price scale.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    # Log returns at multiple lookbacks
    for n in [1, 5, 10, 20]:
        df[f"log_ret_{n}"] = np.log(close / close.shift(n))

    # Candle geometry ratios — clipped to [0, 1] to guard against open outside [low, high]
    hl_range = (high - low).replace(0, np.nan)
    body_high = pd.concat([open_, close], axis=1).max(axis=1).clip(upper=high)
    body_low = pd.concat([open_, close], axis=1).min(axis=1).clip(lower=low)
    df["candle_body_ratio"] = ((body_high - body_low) / hl_range).clip(0.0, 1.0)
    df["upper_shadow_ratio"] = ((high - body_high) / hl_range).clip(0.0, 1.0)
    df["lower_shadow_ratio"] = ((body_low - low) / hl_range).clip(0.0, 1.0)

    # Higher-high / lower-low count over N periods (as fraction)
    for n in [5, 10, 20]:
        df[f"hh_count_{n}"] = (high > high.shift(1)).rolling(n).sum() / n
        df[f"ll_count_{n}"] = (low < low.shift(1)).rolling(n).sum() / n

    # Distance from N-period high/low as % of price
    for n in [20, 50]:
        df[f"dist_from_high_{n}"] = (close - high.rolling(n).max()) / close.replace(0, np.nan)
        df[f"dist_from_low_{n}"] = (close - low.rolling(n).min()) / close.replace(0, np.nan)

    return df
