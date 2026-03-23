"""Feature pipeline: orchestrates all feature modules and generates targets."""
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd

from src.features.technical import add_technical_features, WARMUP_PERIODS
from src.features.price_action import add_price_action_features
from src.features.volume import add_volume_features
from src.features.multi_timeframe import add_multi_timeframe_features

logger = logging.getLogger(__name__)

_OHLCV_COLS = {"open", "high", "low", "close", "volume"}
_MIN_MOVE_THRESHOLD = 0.005  # 0.5%
_TARGET_HORIZONS = [6, 12, 24, 48]


def build_features(
    df: pd.DataFrame,
    timeframe: str = "1h",
    include_targets: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transform raw OHLCV into a feature matrix.

    Args:
        df: OHLCV DataFrame from preprocessor.
        timeframe: Base timeframe string (used for multi-TF selection).
        include_targets: If True, append binary target columns.
            WARNING: Must ONLY be True during training label generation.
            At inference (backtest, paper trading), use the default False.

    Returns:
        (X, feature_names): Raw (unscaled) feature DataFrame, list of feature column names.
        StandardScaler is NOT applied here — trainer.py owns it.
    """
    df = df.copy()

    df = add_technical_features(df)
    df = add_price_action_features(df)
    df = add_volume_features(df)
    df = add_multi_timeframe_features(df, base_timeframe=timeframe)

    # Add targets BEFORE dropping warmup rows (need future close prices).
    # WARNING: include_targets=True must ONLY be called during training label generation.
    # At inference time (backtest, paper trading), always use include_targets=False.
    # NOTE FOR CALLERS: When include_targets=True, the last max(horizons)=48 rows will have
    # NaN targets (no future data). The pipeline does NOT drop these rows. The caller
    # (trainer.py) must call df.dropna(subset=target_cols) before fitting the model.
    if include_targets:
        close = df["close"]
        n = len(df)
        for horizon in _TARGET_HORIZONS:
            future_close = close.shift(-horizon)
            fwd_return = (future_close - close) / close
            df[f"target_{horizon}"] = (fwd_return > _MIN_MOVE_THRESHOLD).astype(float)
            # NaN the last `horizon` rows — no future data available
            if n > horizon:
                df.iloc[-horizon:, df.columns.get_loc(f"target_{horizon}")] = np.nan
            else:
                df[f"target_{horizon}"] = np.nan

    # Drop warmup rows
    df = df.iloc[WARMUP_PERIODS:].copy()

    feature_cols = [
        c for c in df.columns
        if c not in _OHLCV_COLS and not c.startswith("target_")
    ]

    # Drop any remaining NaN in feature columns (e.g., from multi-TF warmup)
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.debug("Dropped %d rows with NaN in feature columns after warmup", dropped)

    target_cols = [f"target_{h}" for h in _TARGET_HORIZONS] if include_targets else []
    return df[feature_cols + target_cols], feature_cols
