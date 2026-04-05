"""Grid trading backtester for BTC/USDT.

Simulates the exact logic from grid/engine.py on historical OHLCV data.

Fill detection per candle:
  - Bullish candle (close >= open): low reached first → buy fills before sell fills
  - Bearish candle (close < open):  high reached first → sell fills before buy fills

Usage:
    python backtest_grid.py                          # 2 years, default grid
    python backtest_grid.py --years 3
    python backtest_grid.py --lower 70000 --upper 100000 --levels 15
    python backtest_grid.py --capital 1000 --verbose
    python backtest_grid.py --csv
"""
from __future__ import annotations

import os
os.environ.setdefault("BINANCE_API_KEY", "backtest")
os.environ.setdefault("BINANCE_API_SECRET", "backtest")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "backtest")
os.environ.setdefault("TELEGRAM_CHAT_ID", "0")

import argparse
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import List, Optional

import pandas as pd

import config
from backtest import fetch_historical_data
from grid.calculator import compute_levels, quantity_for_level, level_above

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEE_RATE = 0.001  # 0.1% per fill side


# ---------------------------------------------------------------------------
# Simulation slot (mirrors grid/engine.py GridSlot, but uses float for speed)
# ---------------------------------------------------------------------------

@dataclass
class SimSlot:
    level_price: float
    state: str = "empty"          # "empty" | "buy_placed" | "sell_placed"
    quantity: float = 0.0
    buy_price: float = 0.0
    sell_price: float = 0.0


# ---------------------------------------------------------------------------
# Backtest state
# ---------------------------------------------------------------------------

@dataclass
class GridBacktestState:
    capital: float                 # total capital deployed
    cycles: List[dict] = field(default_factory=list)  # completed buy→sell pairs
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    slots: List[SimSlot] = field(default_factory=list)
    # Equity curve: (timestamp, cumulative_pnl) per completed cycle
    equity_events: List[tuple] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grid initialisation
# ---------------------------------------------------------------------------

def _build_slots(
    levels: List[Decimal],
    capital_per_level: Decimal,
    current_price: float,
) -> List[SimSlot]:
    """Create slots and mark buy_placed for all levels below current price."""
    slots = []
    for lvl in levels:
        qty = quantity_for_level(capital_per_level, lvl)
        slot = SimSlot(
            level_price=float(lvl),
            quantity=float(qty),
        )
        if float(lvl) < current_price:
            slot.state = "buy_placed"
            slot.buy_price = float(lvl)
        slots.append(slot)
    return slots


# ---------------------------------------------------------------------------
# Intra-candle fill logic
# ---------------------------------------------------------------------------

def _process_candle(
    slots: List[SimSlot],
    levels_float: List[float],
    open_: float,
    high: float,
    low: float,
    candle_time,
    state: GridBacktestState,
    verbose: bool,
) -> None:
    """Detect fills within a single candle and place counter-orders."""
    # Determine price path direction
    bullish = open_ <= (high + low) / 2  # low first, then high

    if bullish:
        _process_buys(slots, levels_float, low, candle_time, state, verbose)
        _process_sells(slots, high, candle_time, state, verbose)
    else:
        _process_sells(slots, high, candle_time, state, verbose)
        _process_buys(slots, levels_float, low, candle_time, state, verbose)


def _process_buys(
    slots: List[SimSlot],
    levels_float: List[float],
    low: float,
    candle_time,
    state: GridBacktestState,
    verbose: bool,
) -> None:
    for slot in slots:
        if slot.state != "buy_placed":
            continue
        if low <= slot.level_price:
            # Buy filled — compute fee and find the sell level above
            fee = slot.quantity * slot.buy_price * FEE_RATE
            state.total_fees += fee

            sell_lvl = _level_above_float(levels_float, slot.level_price)
            if sell_lvl is None:
                # Top level — no sell target; reset to empty
                slot.state = "empty"
                continue

            slot.sell_price = sell_lvl
            slot.state = "sell_placed"

            if verbose:
                print(f"  {candle_time}  BUY  filled @ {slot.level_price:.2f}  "
                      f"sell → {sell_lvl:.2f}  qty={slot.quantity:.5f}")


def _process_sells(
    slots: List[SimSlot],
    high: float,
    candle_time,
    state: GridBacktestState,
    verbose: bool,
) -> None:
    for slot in slots:
        if slot.state != "sell_placed":
            continue
        if high >= slot.sell_price:
            # Sell filled — record cycle profit
            gross = (slot.sell_price - slot.buy_price) * slot.quantity
            fee = slot.quantity * slot.sell_price * FEE_RATE
            net = gross - fee
            state.total_fees += fee
            state.realized_pnl += net

            state.cycles.append({
                "time": candle_time,
                "buy_price": round(slot.buy_price, 2),
                "sell_price": round(slot.sell_price, 2),
                "quantity": round(slot.quantity, 5),
                "gross_pnl": round(gross, 4),
                "fee": round(fee, 4),
                "net_pnl": round(net, 4),
            })
            state.equity_events.append((candle_time, round(state.realized_pnl, 4)))

            if verbose:
                print(f"  {candle_time}  SELL filled @ {slot.sell_price:.2f}  "
                      f"(buy was {slot.buy_price:.2f})  profit={net:+.4f} USDT")

            # Re-place the buy at this level
            slot.state = "buy_placed"
            slot.buy_price = slot.level_price
            slot.sell_price = 0.0


def _level_above_float(levels: List[float], price: float) -> Optional[float]:
    for lvl in sorted(levels):
        if lvl > price:
            return lvl
    return None


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_grid_backtest(
    df: pd.DataFrame,
    lower: float,
    upper: float,
    count: int,
    capital: float,
    verbose: bool = False,
) -> GridBacktestState:
    """Simulate grid trading on historical OHLCV data.

    Grid is initialised at the first candle's open price and held fixed
    throughout (static grid — same as the live bot).
    """
    levels = compute_levels(Decimal(str(lower)), Decimal(str(upper)), count)
    capital_per_level = Decimal(str(capital)) / (count - 1)
    levels_float = [float(l) for l in levels]

    state = GridBacktestState(capital=capital)

    # Initialise slots at the first candle open price
    first_open = float(df.iloc[0]["open"])
    state.slots = _build_slots(levels, capital_per_level, first_open)

    buy_slots = sum(1 for s in state.slots if s.state == "buy_placed")
    print(f"Grid initialised at {first_open:.2f} — {buy_slots}/{count} buy orders placed")
    print(f"Price range in data: {df['low'].min():.2f} — {df['high'].max():.2f}")
    print(f"Grid range:          {lower:.2f} — {upper:.2f}")
    print()

    for i, row in df.iterrows():
        _process_candle(
            state.slots,
            levels_float,
            open_=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            candle_time=row["open_time"],
            state=state,
            verbose=verbose,
        )

    return state


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_grid_metrics(state: GridBacktestState, df: pd.DataFrame, years: int) -> dict:
    cycles = state.cycles
    n = len(cycles)

    if n == 0:
        return {
            "total_cycles": 0,
            "realized_pnl": 0.0,
            "total_fees": 0.0,
            "return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "avg_profit_per_cycle": 0.0,
            "max_drawdown_pct": 0.0,
            "cycles_per_month": 0.0,
            "monthly_avg_pnl": 0.0,
        }

    total_pnl = state.realized_pnl
    return_pct = total_pnl / state.capital * 100
    annualized = ((1 + return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    avg_profit = total_pnl / n

    # Monthly breakdown
    cycle_df = pd.DataFrame(cycles)
    cycle_df["month"] = pd.to_datetime(cycle_df["time"]).dt.to_period("M")
    monthly = cycle_df.groupby("month")["net_pnl"].sum()
    monthly_avg = monthly.mean() if len(monthly) > 0 else 0.0
    months_total = (pd.to_datetime(df["open_time"].iloc[-1]) -
                    pd.to_datetime(df["open_time"].iloc[0])).days / 30.44
    cycles_per_month = n / months_total if months_total > 0 else 0.0

    # Max drawdown on the equity curve
    peak = 0.0
    max_dd_abs = 0.0
    running = 0.0
    for c in cycles:
        running += c["net_pnl"]
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd_abs:
            max_dd_abs = dd
    max_dd_pct = max_dd_abs / state.capital * 100

    return {
        "total_cycles": n,
        "realized_pnl": round(total_pnl, 4),
        "total_fees": round(state.total_fees, 4),
        "return_pct": round(return_pct, 2),
        "annualized_return_pct": round(annualized, 2),
        "avg_profit_per_cycle": round(avg_profit, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "cycles_per_month": round(cycles_per_month, 1),
        "monthly_avg_pnl": round(monthly_avg, 2),
        "monthly": monthly,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(
    metrics: dict,
    lower: float,
    upper: float,
    count: int,
    capital: float,
    years: int,
) -> None:
    print()
    print("=" * 62)
    print(f"  GRID BACKTEST — BTCUSDT {config.INTERVAL}  ({years}y)")
    print(f"  Range: {lower:.0f}–{upper:.0f}  |  Levels: {count}  |  Capital: {capital:.0f} USDT")
    print("=" * 62)
    print(f"  Total cycles         : {metrics['total_cycles']}")
    print(f"  Cycles / month       : {metrics['cycles_per_month']:.1f}")
    print(f"  Avg profit / cycle   : {metrics['avg_profit_per_cycle']:>+8.4f} USDT")
    print(f"  Realized PnL         : {metrics['realized_pnl']:>+10.2f} USDT")
    print(f"  Total fees paid      : {metrics['total_fees']:>10.2f} USDT")
    print(f"  Return on capital    : {metrics['return_pct']:>+8.2f}%")
    print(f"  Annualized return    : {metrics['annualized_return_pct']:>+8.2f}%  / year")
    print(f"  Monthly avg PnL      : {metrics['monthly_avg_pnl']:>+8.2f} USDT")
    print(f"  Max drawdown         : {metrics['max_drawdown_pct']:>8.2f}%")
    print("=" * 62)

    if "monthly" in metrics and len(metrics["monthly"]) > 0:
        print()
        print("  Monthly PnL breakdown:")
        print("  " + "-" * 30)
        for period, pnl in metrics["monthly"].items():
            bar = "+" * max(0, int(pnl / max(abs(metrics["monthly"].max()), 0.01) * 20))
            sign = "+" if pnl >= 0 else ""
            print(f"  {period}   {sign}{pnl:>8.2f} USDT  {bar}")
        print()


def print_cycles(cycles: list) -> None:
    print(f"\n  {'Time':<22} {'Buy':>8} {'Sell':>8} {'Qty':>9} {'Net PnL':>10}")
    print("  " + "-" * 62)
    for c in cycles:
        sign = "+" if c["net_pnl"] >= 0 else ""
        print(
            f"  {str(c['time'])[:19]:<22} {c['buy_price']:>8.2f} {c['sell_price']:>8.2f} "
            f"{c['quantity']:>9.5f} {sign}{c['net_pnl']:>9.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC/USDT grid trading backtester")
    parser.add_argument("--years",   type=int,   default=2,
                        help="Historical lookback in years (default: 2)")
    parser.add_argument("--lower",   type=float, default=float(config.GRID_LOWER),
                        help=f"Grid lower bound in USDT (default: {config.GRID_LOWER})")
    parser.add_argument("--upper",   type=float, default=float(config.GRID_UPPER),
                        help=f"Grid upper bound in USDT (default: {config.GRID_UPPER})")
    parser.add_argument("--levels",  type=int,   default=config.GRID_COUNT,
                        help=f"Number of grid levels (default: {config.GRID_COUNT})")
    parser.add_argument("--capital", type=float, default=float(config.GRID_CAPITAL_USDT),
                        help=f"Capital to deploy in USDT (default: {config.GRID_CAPITAL_USDT})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each individual fill")
    parser.add_argument("--csv",     action="store_true",
                        help="Save cycle log to grid_backtest_cycles.csv")
    args = parser.parse_args()

    df = fetch_historical_data(years=args.years)

    print("Running grid backtest...")
    state = run_grid_backtest(
        df,
        lower=args.lower,
        upper=args.upper,
        count=args.levels,
        capital=args.capital,
        verbose=args.verbose,
    )

    metrics = compute_grid_metrics(state, df, years=args.years)
    print_summary(metrics, args.lower, args.upper, args.levels, args.capital, args.years)

    if args.verbose and state.cycles:
        print_cycles(state.cycles)

    if args.csv and state.cycles:
        out = "grid_backtest_cycles.csv"
        pd.DataFrame(state.cycles).to_csv(out, index=False)
        print(f"Cycle log saved to {out}")
