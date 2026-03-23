"""Tests for feature engineering modules."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with enough rows to cover all indicator warmups."""
    np.random.seed(42)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    price = 40000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "open":   price + np.random.randn(n) * 50,
        "high":   price + np.abs(np.random.randn(n)) * 150,
        "low":    price - np.abs(np.random.randn(n)) * 150,
        "close":  price,
        "volume": np.abs(np.random.randn(n)) * 1000 + 500,
    }, index=idx)


def test_technical_features_no_raw_prices():
    """Technical feature columns contain no raw price-scale values."""
    from src.features.technical import add_technical_features

    df = _make_ohlcv()
    result = add_technical_features(df.copy())

    added_cols = [c for c in result.columns if c not in {"open", "high", "low", "close", "volume"}]
    assert len(added_cols) > 0, "No feature columns were added"

    for col in added_cols:
        col_max = result[col].abs().max()
        assert col_max < 1000, f"Column '{col}' has suspiciously large values (max={col_max:.1f})"


def test_technical_features_no_nan_after_warmup():
    """After dropping the warmup period, technical features contain no NaN."""
    from src.features.technical import add_technical_features, WARMUP_PERIODS

    df = _make_ohlcv(500)
    result = add_technical_features(df.copy())
    trimmed = result.iloc[WARMUP_PERIODS:]

    added_cols = [c for c in result.columns if c not in {"open", "high", "low", "close", "volume"}]
    nan_counts = trimmed[added_cols].isna().sum()
    assert nan_counts.sum() == 0, f"NaNs found after warmup:\n{nan_counts[nan_counts > 0]}"


def test_price_action_no_raw_prices():
    """Price action features are all relative (no raw price-scale values)."""
    from src.features.price_action import add_price_action_features

    df = _make_ohlcv()
    result = add_price_action_features(df.copy())

    added_cols = [c for c in result.columns if c not in {"open", "high", "low", "close", "volume"}]
    for col in added_cols:
        assert result[col].abs().max() < 100, f"Column '{col}' exceeds expected range"


def test_price_action_candle_ratios_bounded():
    """Candle body and shadow ratios are bounded in [0, 1]."""
    from src.features.price_action import add_price_action_features

    df = _make_ohlcv(200)
    result = add_price_action_features(df.copy())
    result = result.dropna()

    for col in ["candle_body_ratio", "upper_shadow_ratio", "lower_shadow_ratio"]:
        assert (result[col] >= 0).all(), f"{col} has negative values"
        assert (result[col] <= 1.0).all(), f"{col} exceeds 1.0"


def test_volume_features_no_raw_prices():
    """Volume features contain no raw price-scale values."""
    from src.features.volume import add_volume_features

    df = _make_ohlcv(300)
    result = add_volume_features(df.copy())

    added_cols = [c for c in result.columns if c not in {"open", "high", "low", "close", "volume"}]
    for col in added_cols:
        assert result[col].abs().max() < 1000, f"Column '{col}' may contain raw price-scale values"


def test_multi_timeframe_no_lookahead():
    """Higher-timeframe values at time t use only data available at or before t."""
    from src.features.multi_timeframe import add_multi_timeframe_features

    df = _make_ohlcv(200)
    result = add_multi_timeframe_features(df.copy(), base_timeframe="1h")

    mtf_cols = [c for c in result.columns if c.startswith("tf_")]
    assert len(mtf_cols) > 0, "No multi-timeframe columns found"

    # The first warmup row of a multi-TF column must be NaN (not filled from future)
    assert pd.isna(result[mtf_cols[0]].iloc[0])


def test_pipeline_output_is_nan_free():
    """Pipeline output (after warmup drop) contains no NaN in feature columns."""
    from src.features.pipeline import build_features

    df = _make_ohlcv(500)
    X, feature_names = build_features(df, timeframe="1h")

    assert X.isna().sum().sum() == 0, "NaN values remain after pipeline"
    assert len(feature_names) == len(X.columns)


def test_pipeline_generates_targets():
    """Pipeline generates binary target columns for each horizon."""
    from src.features.pipeline import build_features

    df = _make_ohlcv(500)
    X, _ = build_features(df, timeframe="1h", include_targets=True)

    for horizon in [6, 12, 24, 48]:
        col = f"target_{horizon}"
        assert col in X.columns, f"Missing target column: {col}"
        assert set(X[col].dropna().unique()).issubset({0, 1}), f"{col} has non-binary values"


def test_pipeline_target_threshold():
    """Target = 1 only when forward return exceeds 0.5%."""
    from src.features.pipeline import build_features
    import pandas as pd

    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    price = 42000.0
    df = pd.DataFrame({
        "open": price, "high": price * 1.001, "low": price * 0.999,
        "close": price, "volume": 500.0,
    }, index=idx)

    X, _ = build_features(df, timeframe="1h", include_targets=True)
    if "target_6" in X.columns and not X.empty:
        assert X["target_6"].max() == 0


def test_pipeline_no_targets_at_inference():
    """With include_targets=False (default), no target columns appear."""
    from src.features.pipeline import build_features

    df = _make_ohlcv(300)
    X, feature_names = build_features(df, timeframe="1h", include_targets=False)

    target_cols = [c for c in X.columns if c.startswith("target_")]
    assert len(target_cols) == 0


def test_pipeline_output_is_unscaled():
    """Pipeline output is raw (not standardized). StandardScaler is owned by trainer.py."""
    from src.features.pipeline import build_features

    df = _make_ohlcv(400)
    X, feature_names = build_features(df, timeframe="1h")

    rsi_cols = [c for c in feature_names if "rsi" in c]
    assert len(rsi_cols) > 0

    for col in rsi_cols:
        # RSI is 0-100. A StandardScaler would compress this to ~[-3, 3].
        assert X[col].max() > 10, f"RSI column '{col}' looks scaled (max={X[col].max():.2f})"


def test_pipeline_reproducible():
    """Same input data produces identical feature matrix."""
    from src.features.pipeline import build_features
    import pandas as pd

    df = _make_ohlcv(300)
    X1, names1 = build_features(df.copy(), timeframe="1h")
    X2, names2 = build_features(df.copy(), timeframe="1h")

    pd.testing.assert_frame_equal(X1, X2)
    assert names1 == names2
