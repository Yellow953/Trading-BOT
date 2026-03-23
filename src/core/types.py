"""Shared data contracts for the ML trading bot."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Tuple


@dataclass
class OHLCV:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    direction: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0.0 – 1.0
    strategy_name: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    side: Literal["LONG", "SHORT"]
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    entry_time: datetime


@dataclass
class Trade:
    """Fully closed trade. Created by portfolio.py when a Position is closed."""
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    fees: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: Literal["stop_loss", "take_profit", "trailing_stop", "signal", "circuit_breaker"]


@dataclass
class PortfolioState:
    cash: float
    equity: float
    open_positions: List[Position] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
