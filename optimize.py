"""Parameter optimizer for BTC/USDT EMA crossover strategy.

Grid searches EMA periods, SL/TP ratios, and ADX minimum threshold.

Usage:
    python optimize.py               # default grid
    python optimize.py --years 5     # 5 years of data
    python optimize.py --full        # larger grid (slower)
"""
from __future__ import annotations

import os
os.environ.setdefault("BINANCE_API_KEY", "backtest")
os.environ.setdefault("BINANCE_API_SECRET", "backtest")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "backtest")
os.environ.setdefault("TELEGRAM_CHAT_ID", "0")

import argparse
import itertools
from typing import List

import pandas as pd

import config
from backtest import fetch_historical_data, run_backtest, compute_metrics


def run_grid(
    df: pd.DataFrame,
    ema_fast_values: List[int],
    ema_slow_values: List[int],
    ema_trend_values: List[int],
    sl_values: List[float],
    tp_values: List[float],
    adx_min_values: List[float],
) -> List[dict]:
    combos = [
        (f, s, t, sl, tp, adx)
        for f, s, t, sl, tp, adx in itertools.product(
            ema_fast_values, ema_slow_values, ema_trend_values,
            sl_values, tp_values, adx_min_values,
        )
        if f < s < t and tp > sl  # sanity checks
    ]
    total = len(combos)
    results = []

    for idx, (f, s, t, sl, tp, adx) in enumerate(combos, 1):
        print(
            f"  [{idx}/{total}] EMA {f}/{s}/{t}  SL={sl*100:.1f}%  "
            f"TP={tp*100:.1f}%  ADX≥{adx:.0f}  ...",
            end="\r", flush=True,
        )

        state = run_backtest(
            df,
            ema_fast=f, ema_slow=s, ema_trend=t,
            sl_pct=sl, tp_pct=tp, adx_min=adx,
        )
        m = compute_metrics(state, df)

        results.append({
            "ema_fast": f, "ema_slow": s, "ema_trend": t,
            "sl_pct": f"{sl*100:.1f}%",
            "tp_pct": f"{tp*100:.1f}%",
            "adx_min": adx,
            **m,
        })

    print()
    return results


def print_results(results: List[dict], top_n: int = 20) -> None:
    filtered = [r for r in results if r["total_trades"] >= 15]
    sorted_results = sorted(filtered, key=lambda r: r["avg_r"], reverse=True)
    top = sorted_results[:top_n]

    if not top:
        print("No combinations produced >= 15 trades.")
        # Show best regardless
        top = sorted(results, key=lambda r: r["avg_r"], reverse=True)[:top_n]

    targets = {
        "win_rate_pct":     lambda v: v >= 45,
        "avg_r":            lambda v: v >= 0.3,   # realistic target for trend-following
        "max_drawdown_pct": lambda v: v < 20,
        "sharpe_ratio":     lambda v: v >= 1.0,
    }

    header = (
        f"  {'EMA':<12} {'SL':<6} {'TP':<6} {'ADX≥':<6} "
        f"{'Trades':<8} {'Return':<9} {'WinRate':<9} {'AvgR':<8} {'MaxDD':<8} {'Sharpe':<8}"
    )
    print()
    print("=" * len(header))
    print(f"  TOP {top_n} COMBINATIONS BY AVERAGE R  (min 15 trades)")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in top:
        ema_label = f"{r['ema_fast']}/{r['ema_slow']}/{r['ema_trend']}"
        marks = {k: ("✓" if targets[k](r[k]) else " ") for k in targets}

        print(
            f"  {ema_label:<12} {r['sl_pct']:<6} {r['tp_pct']:<6} {r['adx_min']:<6.0f} "
            f"{r['total_trades']:<8} {r['total_return_pct']:>+7.1f}%  "
            f"{r['win_rate_pct']:>6.1f}%{marks['win_rate_pct']} "
            f"{r['avg_r']:>6.3f}R{marks['avg_r']} "
            f"{r['max_drawdown_pct']:>6.1f}%{marks['max_drawdown_pct']} "
            f"{r['sharpe_ratio']:>7.3f}{marks['sharpe_ratio']}"
        )

    print("=" * len(header))
    print("  ✓ = meets target  |  sorted by Avg R")

    best = top[0]
    print()
    print("  Best combination:")
    print(f"    EMA_FAST        = {best['ema_fast']}")
    print(f"    EMA_SLOW        = {best['ema_slow']}")
    print(f"    EMA_TREND       = {best['ema_trend']}")
    print(f"    STOP_LOSS_PCT   = {best['sl_pct']}")
    print(f"    TAKE_PROFIT_PCT = {best['tp_pct']}")
    print(f"    ADX_MIN         = {best['adx_min']:.0f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMA crossover strategy optimizer")
    parser.add_argument("--years", type=int, default=5, help="Historical data lookback (default: 5)")
    parser.add_argument("--full",  action="store_true",  help="Run full grid (slower)")
    args = parser.parse_args()

    df = fetch_historical_data(years=args.years)

    if args.full:
        ema_fast_values  = [9, 12, 15]
        ema_slow_values  = [21, 26, 30]
        ema_trend_values = [50, 100, 200]
        sl_values        = [0.015, 0.02, 0.025, 0.03]
        tp_values        = [0.04, 0.05, 0.06, 0.08]
        adx_min_values   = [15.0, 20.0, 25.0]
    else:
        ema_fast_values  = [9, 12]
        ema_slow_values  = [21, 26]
        ema_trend_values = [50, 100]
        sl_values        = [0.02, 0.025]
        tp_values        = [0.05, 0.06]
        adx_min_values   = [15.0, 20.0, 25.0]

    # Count valid combos (f < s < t, tp > sl)
    all_combos = list(itertools.product(
        ema_fast_values, ema_slow_values, ema_trend_values,
        sl_values, tp_values, adx_min_values,
    ))
    valid = sum(1 for f, s, t, sl, tp, _ in all_combos if f < s < t and tp > sl)
    print(f"Running {valid} valid parameter combinations on {len(df)} candles...")
    print()

    results = run_grid(
        df, ema_fast_values, ema_slow_values, ema_trend_values,
        sl_values, tp_values, adx_min_values,
    )
    print_results(results, top_n=20)

    out_path = "optimize_results.csv"
    pd.DataFrame(results).sort_values("avg_r", ascending=False).to_csv(out_path, index=False)
    print(f"Full results saved to {out_path}")
