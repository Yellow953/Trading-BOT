"""Backtester for BTC/USDT mean reversion strategy.

Reuses strategy/signals.py and strategy/filters.py without modification.
Simulates the exact same logic as main.py's run_cycle().

Usage:
    python backtest.py                  # 2 years, $10k starting balance
    python backtest.py --years 1        # 1 year
    python backtest.py --balance 5000   # custom starting balance
    python backtest.py --csv            # save trade-by-trade CSV
    python backtest.py --verbose        # print each trade
"""
from __future__ import annotations

# Set dummy env vars before importing config (config hard-fails on missing keys).
# Real keys are loaded from .env by dotenv inside config.py — these are fallbacks
# so the backtest works even without a Telegram token.
import os
os.environ.setdefault("BINANCE_API_KEY", "backtest")
os.environ.setdefault("BINANCE_API_SECRET", "backtest")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "backtest")
os.environ.setdefault("TELEGRAM_CHAT_ID", "0")

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from binance.client import Client

import config
from strategy.signals import get_signal
from strategy.filters import passes_filters

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEE_RATE = 0.001          # 0.1% per trade side (Binance standard)
WARMUP = 65               # fallback warmup used by compute_metrics hourly return array
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_historical_data(years: int = 2) -> pd.DataFrame:
    """Download 1H OHLCV candles from Binance. Auto-paginates."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    # Always use live Binance for historical data — testnet has no price history.
    client = Client(api_key, api_secret, testnet=False)

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=365 * years)

    print(f"Fetching {years}y of {config.SYMBOL} {config.INTERVAL} candles "
          f"({start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')})...")

    raw = client.get_historical_klines(
        config.SYMBOL,
        config.INTERVAL,
        start_str=start_dt.strftime("%d %b, %Y"),
        end_str=end_dt.strftime("%d %b, %Y"),
    )

    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.reset_index(drop=True)

    # Drop the last row — it's the still-forming candle
    df = df.iloc[:-1].reset_index(drop=True)

    print(f"Loaded {len(df)} closed candles.")
    return df


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class BacktestState:
    initial_balance: float
    balance: float = 0.0
    peak_balance: float = 0.0
    position_open: bool = False
    entry_price: float = 0.0
    entry_candle_idx: int = 0
    entry_notional: float = 0.0    # dollars risked at entry time
    sl_price: float = 0.0
    tp_price: float = 0.0
    day_start_balance: float = 0.0
    current_day: Optional[object] = None
    trades: list = field(default_factory=list)

    def __post_init__(self):
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.day_start_balance = self.initial_balance


# ---------------------------------------------------------------------------
# Daily guard (calendar-based, no datetime.utcnow())
# ---------------------------------------------------------------------------

def _update_daily_guard(state: BacktestState, candle_time: pd.Timestamp) -> None:
    candle_day = candle_time.date()
    if state.current_day is None or candle_day != state.current_day:
        state.current_day = candle_day
        state.day_start_balance = state.balance


def _daily_loss_exceeded(state: BacktestState) -> bool:
    if state.day_start_balance == 0:
        return False
    loss_pct = (state.day_start_balance - state.balance) / state.day_start_balance
    return loss_pct >= float(config.MAX_DAILY_LOSS_PCT)


# ---------------------------------------------------------------------------
# Intracandle SL/TP resolution
# ---------------------------------------------------------------------------

def _check_sl_tp(
    low: float, high: float, sl_price: float, tp_price: float
) -> Optional[tuple]:
    """Return (reason, exit_price) if SL or TP triggered, else None.
    SL is checked before TP (conservative — assume worst case if both hit)."""
    if low <= sl_price:
        return ("sl", sl_price)
    if high >= tp_price:
        return ("tp", tp_price)
    return None


# ---------------------------------------------------------------------------
# Trade close helper
# ---------------------------------------------------------------------------

def _close_trade(
    state: BacktestState,
    exit_price: float,
    reason: str,
    exit_candle_idx: int,
    df: pd.DataFrame,
) -> None:
    price_return = (exit_price - state.entry_price) / state.entry_price
    gross_pnl = state.entry_notional * price_return
    fee_exit = state.entry_notional * FEE_RATE
    net_pnl = gross_pnl - fee_exit

    one_r_dollar = state.entry_notional * float(config.STOP_LOSS_PCT)
    pnl_r = net_pnl / one_r_dollar if one_r_dollar else 0.0

    state.balance += net_pnl
    state.peak_balance = max(state.peak_balance, state.balance)
    state.position_open = False

    state.trades.append({
        "entry_idx": state.entry_candle_idx,
        "exit_idx": exit_candle_idx,
        "entry_time": df.iloc[state.entry_candle_idx]["open_time"],
        "exit_time": df.iloc[exit_candle_idx]["open_time"],
        "entry_price": round(state.entry_price, 2),
        "exit_price": round(exit_price, 2),
        "exit_reason": reason,
        "pos_notional": round(state.entry_notional, 2),
        "gross_pnl": round(gross_pnl, 4),
        "net_pnl": round(net_pnl, 4),
        "pnl_r": round(pnl_r, 4),
        "hold_candles": exit_candle_idx - state.entry_candle_idx,
    })


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    ema_fast: int = config.EMA_FAST,
    ema_slow: int = config.EMA_SLOW,
    ema_trend: int = config.EMA_TREND,
    sl_pct: float = float(config.STOP_LOSS_PCT),
    tp_pct: float = float(config.TAKE_PROFIT_PCT),
    adx_min: Optional[float] = float(config.ADX_MIN),
) -> BacktestState:
    """Run the backtest with overrideable strategy parameters.

    Uses EMA crossover trend-following logic:
      Long:  fast EMA crosses above slow EMA AND price above trend EMA
      Exit:  fast EMA crosses below slow EMA OR price drops below trend EMA
    """
    import pandas_ta_classic as ta_local

    warmup = ema_trend + 5  # need enough candles for all EMAs

    def _local_signal(window: pd.DataFrame) -> Optional[str]:
        if len(window) < ema_trend + 3:
            return None
        f = ta_local.ema(window["close"], length=ema_fast)
        s = ta_local.ema(window["close"], length=ema_slow)
        t = ta_local.ema(window["close"], length=ema_trend)
        if f is None or s is None or t is None:
            return None
        fn, fp = f.iloc[-2], f.iloc[-3]
        sn, sp = s.iloc[-2], s.iloc[-3]
        tn = t.iloc[-2]
        close = window["close"].iloc[-2]
        if any(pd.isna(v) for v in [fn, fp, sn, sp, tn]):
            return None
        if fp <= sp and fn > sn and close > tn:
            return "long"
        if (fp >= sp and fn < sn) or close < tn:
            return "exit"
        return None

    state = BacktestState(initial_balance=initial_balance)

    for i in range(warmup, len(df)):
        signal_candle = df.iloc[i - 1]
        _update_daily_guard(state, signal_candle["open_time"])

        # --- Manage open position ---
        if state.position_open:
            exec_candle = df.iloc[i]

            # 1. Intracandle SL/TP check
            result = _check_sl_tp(
                exec_candle["low"], exec_candle["high"],
                state.sl_price, state.tp_price,
            )
            if result:
                reason, exit_price = result
                _close_trade(state, exit_price, reason, i, df)
                continue

            # 2. Max hold time
            hold_candles = i - state.entry_candle_idx
            if hold_candles >= config.MAX_HOLD_CANDLES:
                exit_price = signal_candle["close"]
                _close_trade(state, exit_price, "timeout", i, df)
                continue

            # Signal exits disabled — backtesting showed death cross exits were net negative.
            # TP/SL OCO and timeout handle all exits.
            continue

        # --- Look for new entry ---
        if _daily_loss_exceeded(state):
            continue

        window = df.iloc[:i + 1]
        if _local_signal(window) != "long":
            continue
        if not passes_filters(window, adx_min=adx_min):
            continue

        entry_price = signal_candle["close"]
        notional = state.balance * float(config.POSITION_SIZE_PCT)
        fee_entry = notional * FEE_RATE

        state.balance -= fee_entry
        state.peak_balance = max(state.peak_balance, state.balance)
        state.position_open = True
        state.entry_price = entry_price
        state.entry_candle_idx = i
        state.entry_notional = notional
        state.sl_price = entry_price * (1 - sl_pct)
        state.tp_price = entry_price * (1 + tp_pct)

    return state


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(state: BacktestState, df: pd.DataFrame) -> dict:
    trades = state.trades
    n = len(trades)
    initial = state.initial_balance

    if n == 0:
        return {
            "total_trades": 0,
            "total_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_r": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_hold_hours": 0.0,
            "final_balance": round(state.balance, 2),
            "wins": 0,
            "losses": 0,
        }

    total_return_pct = (state.balance - initial) / initial * 100
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    win_rate = wins / n * 100
    avg_r = sum(t["pnl_r"] for t in trades) / n
    avg_hold = sum(t["hold_candles"] for t in trades) / n  # hours on 1H data

    # Max drawdown — walk the equity curve
    running = initial
    peak = initial
    max_dd = 0.0
    for t in trades:
        running += t["net_pnl"]
        peak = max(peak, running)
        dd = (peak - running) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Sharpe ratio — hourly returns over full backtest window
    n_candles = len(df) - WARMUP
    hourly_returns = np.zeros(n_candles)
    for t in trades:
        idx = t["exit_idx"] - (WARMUP - 1)
        if 0 <= idx < n_candles:
            # Return relative to capital deployed at entry
            hourly_returns[idx] = t["net_pnl"] / t["pos_notional"]

    mean_r = hourly_returns.mean()
    std_r = hourly_returns.std(ddof=1)
    sharpe = (mean_r / std_r * math.sqrt(8760)) if std_r > 0 else 0.0

    return {
        "total_trades": n,
        "total_return_pct": round(total_return_pct, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_r": round(avg_r, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "avg_hold_hours": round(avg_hold, 1),
        "final_balance": round(state.balance, 2),
        "wins": wins,
        "losses": n - wins,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(metrics: dict, years: int) -> None:
    target = {
        "win_rate_pct":     ("≥ 45%",  lambda v: v >= 45),
        "avg_r":            ("≥ 0.8R", lambda v: v >= 0.8),
        "max_drawdown_pct": ("< 20%",  lambda v: v < 20),
        "sharpe_ratio":     ("≥ 1.0",  lambda v: v >= 1.0),
    }

    def mark(key, val):
        if key not in target:
            return ""
        label, check = target[key]
        return f"  {'PASS' if check(val) else 'FAIL'}  (target {label})"

    print()
    print("=" * 60)
    print(f"  BACKTEST — BTCUSDT 4H EMA Trend Following  ({years}y)")
    print("=" * 60)
    print(f"  Total trades       : {metrics['total_trades']}  "
          f"({metrics['wins']}W / {metrics['losses']}L)")
    print(f"  Total return       : {metrics['total_return_pct']:>+8.2f}%")
    print(f"  Final balance      : {metrics['final_balance']:>10.2f} USDT")
    print(f"  Win rate           : {metrics['win_rate_pct']:>8.1f}%"
          f"{mark('win_rate_pct', metrics['win_rate_pct'])}")
    print(f"  Average R          : {metrics['avg_r']:>8.3f}R"
          f"{mark('avg_r', metrics['avg_r'])}")
    print(f"  Max drawdown       : {metrics['max_drawdown_pct']:>8.2f}%"
          f"{mark('max_drawdown_pct', metrics['max_drawdown_pct'])}")
    print(f"  Sharpe ratio       : {metrics['sharpe_ratio']:>8.3f}"
          f"{mark('sharpe_ratio', metrics['sharpe_ratio'])}")
    print(f"  Avg hold           : {metrics['avg_hold_hours']:>8.1f}h")
    print("=" * 60)
    print()


def print_trades(trades: list) -> None:
    print(f"  {'Entry time':<22} {'Exit time':<22} {'Reason':<12} "
          f"{'Entry':>8} {'Exit':>8} {'PnL $':>8} {'R':>6}")
    print("  " + "-" * 90)
    for t in trades:
        sign = "+" if t["net_pnl"] >= 0 else ""
        print(
            f"  {str(t['entry_time'])[:19]:<22} {str(t['exit_time'])[:19]:<22} "
            f"{t['exit_reason']:<12} {t['entry_price']:>8.0f} {t['exit_price']:>8.0f} "
            f"{sign}{t['net_pnl']:>7.2f} {t['pnl_r']:>+6.2f}R"
        )
    print()


def save_trades_csv(trades: list, path: str = "backtest_trades.csv") -> None:
    pd.DataFrame(trades).to_csv(path, index=False)
    print(f"Trade log saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC/USDT mean reversion backtester")
    parser.add_argument("--years",   type=int,   default=2,      help="Lookback in years (default: 2)")
    parser.add_argument("--balance", type=float, default=10_000, help="Starting USDT balance (default: 10000)")
    parser.add_argument("--csv",     action="store_true",        help="Save trade log to backtest_trades.csv")
    parser.add_argument("--verbose", action="store_true",        help="Print each trade")
    args = parser.parse_args()

    df = fetch_historical_data(years=args.years)

    print("Running backtest...")
    state = run_backtest(df, initial_balance=args.balance)

    metrics = compute_metrics(state, df)
    print_summary(metrics, args.years)

    if args.verbose and state.trades:
        print_trades(state.trades)

    if args.csv:
        save_trades_csv(state.trades)
