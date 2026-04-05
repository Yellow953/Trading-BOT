"""Tests for risk/sizing.py — pure arithmetic, no mocking required."""

import sys
import types
from decimal import Decimal

# Minimal config stub
_cfg = types.ModuleType("config")
_cfg.POSITION_SIZE_PCT = Decimal("0.10")
_cfg.QUANTITY_STEP = Decimal("0.00001")
_cfg.MIN_NOTIONAL = Decimal("10")
sys.modules["config"] = _cfg

import pytest
from risk.sizing import calculate_quantity


class TestCalculateQuantity:
    def test_normal_case(self):
        # 10% of 10000 USDT at 60000 USDT/BTC = 0.01666... -> truncated to 0.01666
        balance = Decimal("10000")
        price = Decimal("60000")
        qty = calculate_quantity(balance, price)
        assert qty == Decimal("0.01666")

    def test_truncation_not_rounding(self):
        # 10% of 1000 USDT at 3000 USDT/BTC = 0.03333... -> should truncate to 0.03333, not 0.03334
        balance = Decimal("1000")
        price = Decimal("3000")
        qty = calculate_quantity(balance, price)
        assert qty == Decimal("0.03333")

    def test_result_is_decimal(self):
        qty = calculate_quantity(Decimal("5000"), Decimal("50000"))
        assert isinstance(qty, Decimal)

    def test_quantity_step_respected(self):
        # Result must be a multiple of 0.00001
        qty = calculate_quantity(Decimal("7777"), Decimal("43219"))
        remainder = qty % Decimal("0.00001")
        assert remainder == Decimal("0")

    def test_below_min_notional_raises(self):
        # 10% of $50 at $50000 = 0.0001 BTC = $5 notional < $10 minimum
        with pytest.raises(ValueError, match="notional"):
            calculate_quantity(Decimal("50"), Decimal("50000"))

    def test_exactly_at_min_notional_passes(self):
        # 10% of $100 at $50000 = 0.0002 BTC = $10 notional — exactly at limit
        qty = calculate_quantity(Decimal("100"), Decimal("50000"))
        notional = qty * Decimal("50000")
        assert notional >= Decimal("10")
