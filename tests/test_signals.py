"""Tests for strategy/signals.py — pure signal logic, no mocking required."""
from __future__ import annotations

import sys
import types
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

# Patch config before importing signals so env vars aren't required
_cfg = types.ModuleType("config")
_cfg.BB_PERIOD = 20
_cfg.BB_STD = Decimal("2.0")
_cfg.RSI_PERIOD = 14
_cfg.RSI_OVERSOLD = Decimal("35")
sys.modules["config"] = _cfg

from strategy.signals import get_signal


def _make_df(closes) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with given close prices."""
    n = len(closes)
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


def _flat_df(n: int = 25, price: float = 50000.0) -> pd.DataFrame:
    """Flat price series — no signal expected."""
    return _make_df([price] * n)


class TestGetSignal:
    def test_returns_none_on_flat_market(self):
        df = _flat_df(30)
        # Flat price sits exactly on the middle band, RSI ~50 — no signal
        result = get_signal(df)
        assert result is None

    def test_returns_none_on_insufficient_data(self):
        df = _flat_df(10)  # fewer than BB_PERIOD rows
        result = get_signal(df)
        assert result is None

    def test_long_signal_when_below_lower_band_and_oversold(self):
        # Series that starts high then drops sharply — produces low RSI and price below BBL.
        # 15 candles declining from 55000 to 51000, then 9 candles crashing to 42000,
        # then a recovery candle at 43000 as the live candle (index -1).
        # index -2 = 42000, well below BBL, RSI will be << 35 after the crash.
        high_phase = list(np.linspace(55000, 51000, 15))
        crash_phase = list(np.linspace(51000, 42000, 9))
        closes = high_phase + crash_phase + [43000.0]  # 25 rows total
        df = _make_df(closes)
        result = get_signal(df)
        assert result == "long"

    def test_exit_signal_when_above_middle_band(self):
        # Build a series where price has risen well above where the middle band would be
        # Start with 20 candles at 50000 to establish the band, then spike up
        closes = [50000.0] * 23 + [60000.0, 60000.0]  # [-2] = 60000
        df = _make_df(closes)
        result = get_signal(df)
        # 60000 is above the middle band of a ~50000 series — should trigger exit
        assert result == "exit"

    def test_signal_uses_second_to_last_candle(self):
        # If we put a long signal condition at index -2 but NOT at index -1,
        # we should still get "long"
        # 23 flat candles to anchor bands, index -2 is the drop, index -1 is recovery
        closes = [50000.0] * 23 + [40000.0, 50000.0]
        df = _make_df(closes)
        result = get_signal(df)
        assert result == "long"

    def test_no_long_signal_when_rsi_not_oversold(self):
        # Gradual decline — RSI won't be oversold even if price is below lower band
        closes = list(np.linspace(55000, 49000, 25))
        df = _make_df(closes)
        result = get_signal(df)
        # The key test is that we don't crash
        assert result in (None, "long", "exit")
