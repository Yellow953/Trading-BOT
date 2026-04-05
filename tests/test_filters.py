"""Tests for strategy/filters.py — no mocking required."""

from __future__ import annotations

import sys
import types
from decimal import Decimal
from typing import Optional, List

import numpy as np
import pandas as pd
import pytest

# Minimal config stub
_cfg = types.ModuleType("config")
_cfg.BB_PERIOD = 20
_cfg.BB_STD = Decimal("2.0")
_cfg.RSI_PERIOD = 14
_cfg.RSI_OVERSOLD = Decimal("35")
_cfg.TREND_SMA_PERIOD = 50
_cfg.TREND_SLOPE_CANDLES = 10
_cfg.TREND_SLOPE_PCT = Decimal("0.005")
_cfg.VOLUME_AVG_PERIOD = 20
sys.modules["config"] = _cfg

from strategy.filters import passes_filters


def _make_df(closes: List[float], volumes: Optional[List[float]] = None) -> pd.DataFrame:
    n = len(closes)
    if volumes is None:
        volumes = [1000.0] * n
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
        "volume": volumes,
    })


def _sufficient_rows() -> int:
    """Minimum rows needed: TREND_SMA_PERIOD + TREND_SLOPE_CANDLES + signal candle buffer."""
    return 50 + 10 + 5  # = 65


class TestPassesFilters:
    def test_fails_with_insufficient_data(self):
        df = _make_df([50000.0] * 30)
        assert passes_filters(df) is False

    def test_passes_with_strong_uptrend_and_high_volume(self):
        n = _sufficient_rows()
        # Gradual uptrend so SMA slopes positively
        closes = list(np.linspace(40000, 50000, n))
        # Signal candle at -2 has very high volume
        volumes = [500.0] * n
        volumes[-2] = 5000.0  # 10x the average — well above threshold
        df = _make_df(closes, volumes)
        assert passes_filters(df) is True

    def test_fails_volume_filter(self):
        n = _sufficient_rows()
        closes = list(np.linspace(40000, 50000, n))
        # Signal candle volume is below average
        volumes = [1000.0] * n
        volumes[-2] = 100.0  # well below the 1000 average
        df = _make_df(closes, volumes)
        assert passes_filters(df) is False

    def test_fails_trend_filter_on_strong_downtrend(self):
        n = _sufficient_rows()
        # Sharp downtrend: SMA slope will be well below -0.5%
        closes = list(np.linspace(60000, 40000, n))
        volumes = [500.0] * n
        volumes[-2] = 5000.0  # pass volume filter
        df = _make_df(closes, volumes)
        assert passes_filters(df) is False

    def test_passes_mild_downtrend_below_slope_threshold(self):
        n = _sufficient_rows()
        # Very mild downtrend: less than 0.5% over 10 candles
        # Start at 50000, end at 49900 — 0.2% drop total
        closes = list(np.linspace(50000, 49900, n))
        volumes = [500.0] * n
        volumes[-2] = 5000.0
        df = _make_df(closes, volumes)
        # Slope is small enough to pass the trend filter
        assert passes_filters(df) is True

    def test_volume_average_excludes_signal_candle(self):
        # Ensure the average is computed from candles before the signal candle (-2),
        # not including the signal candle itself.
        n = _sufficient_rows()
        closes = list(np.linspace(40000, 50000, n))
        # Make signal candle volume == average of previous 20 (not above)
        base_vol = 1000.0
        volumes = [base_vol] * n
        volumes[-2] = base_vol  # exactly equal to average, should FAIL
        df = _make_df(closes, volumes)
        assert passes_filters(df) is False
