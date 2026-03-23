# ML Trading Bot — Design Spec

**Date:** 2026-03-23
**Status:** Approved
**Approach:** Option C — Layered with shared contracts first

---

## Overview

An ML-powered trading bot that learns patterns from historical data across crypto, stocks, and forex. Runs multiple competing strategies, selects the best performer per market, and auto-selects optimal timeframes. Built for backtesting first, paper trading second.

**Philosophy:** The bot does the learning, not the user. Human decides risk. Bot decides how to trade.

**Implementation approach:** Define shared data contracts first, then build each phase in order with tests before advancing. Defer hardening/edge-case handling to Phase 5.

---

## Section 1: Shared Contracts & Project Scaffold

### Data Contracts

All shared types live in `src/core/types.py` as dataclasses:

```python
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
    confidence: float          # 0.0 – 1.0
    strategy_name: str
    timestamp: datetime
    metadata: dict             # feature values that drove the signal

@dataclass
class Trade:
    """Represents a fully closed trade. Created by portfolio.py when a Position is closed."""
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    fees: float                    # total fees for both legs
    entry_time: datetime
    exit_time: datetime
    pnl: float                     # net P&L after fees
    pnl_pct: float                 # pnl / (entry_price * size)
    exit_reason: Literal["stop_loss", "take_profit", "trailing_stop", "signal", "circuit_breaker"]

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
class PortfolioState:
    cash: float
    equity: float
    open_positions: list[Position]
    equity_curve: list[tuple[datetime, float]]
    closed_trades: list[Trade]
```

### Project Scaffold

```
ml-trading-bot/
├── CLAUDE.md
├── pyproject.toml
├── requirements.txt
├── config.yaml
├── main.py                     # click CLI — all commands stubbed at scaffold time
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── types.py            # All shared dataclasses
│   ├── data/
│   │   ├── __init__.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── crypto.py
│   │   │   ├── stocks.py
│   │   │   └── forex.py
│   │   ├── cache.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py
│   │   ├── price_action.py
│   │   ├── volume.py
│   │   ├── multi_timeframe.py
│   │   └── pipeline.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ml_trend.py
│   │   ├── ml_mean_reversion.py
│   │   ├── ml_momentum.py
│   │   ├── ml_volatility.py
│   │   └── ensemble.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── selector.py
│   │   └── persistence.py
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── portfolio.py
│   │   ├── risk.py
│   │   └── metrics.py
│   ├── paper_trading/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   └── journal.py
│   └── reporting/
│       ├── __init__.py
│       ├── terminal.py
│       └── charts.py
├── data/
│   ├── cache/
│   └── models/
├── output/
│   ├── reports/
│   └── charts/
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_strategies.py
    └── test_backtest.py
```

---

## Section 2: Data Layer (Phase 1)

### DataProvider Interface (`src/data/providers/base.py`)

```python
class DataProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: datetime, limit: int) -> pd.DataFrame:
        """Return DataFrame: columns [open, high, low, close, volume], UTC datetime index."""

    @abstractmethod
    def available_symbols(self) -> list[str]: ...

    @abstractmethod
    def available_timeframes(self) -> list[str]: ...
```

### Implementations

- **`CryptoProvider`** — CCXT/Binance public API. Fallback: Kraken. Paginates large date ranges in chunks. Timeframes: 1m, 5m, 15m, 1h, 4h, 1d.
- **`StocksProvider`** — yfinance. Enforces yfinance limits: 1m → max 7d, 5m → max 60d, 1h → max 730d, 1d → full history.
- **`ForexProvider`** — yfinance forex pairs (e.g. `EURUSD=X`). Same limits as stocks.

### Cache (`src/data/cache.py`)

- SQLite at `data/cache/market_data.db`
- Key: `(provider, symbol, timeframe, date)`
- On fetch: check cache first, pull only missing date ranges, write back
- Never re-fetch what is already cached

### Preprocessor (`src/data/preprocess.py`)

- Validates columns: `[open, high, low, close, volume]` present and numeric
- Drops rows with NaN in any OHLCV column
- Forward-fills gaps of ≤3 consecutive candles; logs warning for larger gaps
- Normalizes index to UTC datetime, sorts ascending
- Removes duplicate timestamps

### Deliverable Gate

`python main.py fetch --market crypto --symbol BTC/USDT` fetches, caches, and prints row count + date range.

### Tests (`tests/test_data.py`)

- Cache hit returns identical data, no API call made
- Cache miss triggers API fetch and writes to cache
- Gap filling: ≤3 candle gaps filled, >3 logged as warning
- All three providers return correct DataFrame schema
- Preprocessor rejects data with missing OHLCV columns

---

## Section 3: Feature Engineering (Phase 1)

### Design Principles

- Every feature is relative/normalized — no raw prices
- Each module is a pure function: `(df: pd.DataFrame, **params) -> pd.DataFrame`
- Features appended as new columns; original OHLCV columns preserved
- Warmup rows (NaN from indicator lookback) dropped by pipeline

### Feature Modules

**`technical.py`**
- RSI(14), RSI(28) — as 0–100 values
- MACD histogram (normalized by ATR)
- ADX(14) — 0–100
- Aroon oscillator
- SMA/EMA crossover ratios: `ema_20/ema_50`, `ema_50/ema_200`
- Ichimoku: distance from cloud as % of price

**`price_action.py`**
- Returns: 1, 5, 10, 20-period log returns
- Candle body ratio: `|close-open| / (high-low)`
- Upper shadow ratio: `(high - max(open,close)) / (high-low)`
- Lower shadow ratio: `(min(open,close) - low) / (high-low)`
- Higher-high count, lower-low count over N periods
- Distance from N-period high/low as % of price

**`volume.py`**
- OBV slope (20-period linear regression slope, normalized)
- Volume ratio: `volume / volume_sma_20`
- VWAP distance: `(close - vwap) / close`
- Chaikin Money Flow (20-period)

**`multi_timeframe.py`**
- Resamples base OHLCV to higher timeframes using this fixed map:
  - Base 1h → higher timeframes: 4h, 1d
  - Base 4h → higher timeframes: 1d
  - Base 1d → no higher timeframes (skip multi-timeframe features)
- Computes RSI, MACD histogram, ADX at each higher timeframe
- Left-joins back to base timeframe index (forward-fill only — no lookahead)
- Column prefix: `tf_4h_rsi`, `tf_1d_adx`, etc.

**`pipeline.py`**
- Orchestrates all modules in order
- Drops warmup NaN rows
- Returns raw (unscaled) `(X: pd.DataFrame, feature_names: list[str])` — scaling is NOT applied here
- Generates target columns: `target_6`, `target_12`, `target_24`, `target_48`
  - Binary: 1 if close at t+N > close at t by > 0.5%, else 0

**StandardScaler ownership:** `trainer.py` owns the scaler. It fits `StandardScaler` on `X_train` after `pipeline.py` returns the raw feature matrix, then transforms both `X_train` and `X_test`. The fitted scaler is saved alongside the model in `persistence.py`. At inference time (backtest engine, paper trading runner), the saved scaler is loaded and applied to the raw pipeline output before calling `generate_signal`. This makes the leakage boundary explicit: pipeline = raw features, trainer/persistence = scaled features.

### Tests (`tests/test_features.py`)

- No raw price values in output columns
- No NaN in output after warmup period
- Multi-timeframe alignment: higher-timeframe value at t uses only data available at t
- Target generation: correct binary labels with threshold applied
- Pipeline output is reproducible (same data → same features)

---

## Section 4: Model Training & Walk-Forward Validation (Phase 2)

### Walk-Forward Protocol

Anchored expanding window:
- Min training window: 6 months
- Test window: 1 month
- Step: 1 month forward per fold
- Model retrained at every fold

### Per-Fold Training (`src/models/trainer.py`)

1. Slice train/test windows
2. Fit `StandardScaler` on train X only
3. Train `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20)`
4. Train `XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)`
5. Combine: `VotingClassifier([rf, xgb], voting='soft')`
6. Evaluate on test window → simulate trades using the trained model's signals on the test period (same backtest engine logic, single pass). Compute Sharpe from the resulting per-trade P&L series: `sharpe = mean(trade_returns) / std(trade_returns) * sqrt(annualization_factor)` where `annualization_factor = trading_periods_per_year / avg_trades_per_period`. If fewer than 10 trades in test window, Sharpe = 0 (insufficient data).
7. Prune features with importance < 0.01
8. Flag overfit if `test_sharpe < 0.8 * train_sharpe`

### Timeframe Auto-Selection

For each `(strategy, symbol)`:
- Run full walk-forward on 1h, 4h, 1d
- Select timeframe with highest average out-of-sample Sharpe

### Model Persistence (`src/models/persistence.py`)

- Save: `data/models/<strategy>_<symbol>_<timeframe>.pkl` (joblib)
- Save alongside: scaler, feature list, training metadata (date, Sharpe scores)
- Load: validate metadata version before returning model

### Strategy Selector (`src/models/selector.py`)

- After all strategies train, ranks by out-of-sample Sharpe per symbol
- Produces initial capital allocation weights for the ensemble at startup: `weight_i = max(sharpe_i, 0) / sum(max(sharpe_j, 0) for all j)`. If all strategies have non-positive Sharpe, fall back to equal weights (0.25 each).
- These selector-derived weights serve as the **initial state** for the ensemble's rolling performance window before any live/paper trades have accumulated.

---

## Section 5: Backtesting Engine (Phase 2)

### Core Loop (`src/backtesting/engine.py`)

```
for each candle t in test period:
    1. Append candle to rolling feature window (no future data)
    2. Each strategy.generate_signal(features_up_to_t) → Signal
    3. Ensemble.route(signals, portfolio_state) → weighted signal
    4. risk.validate(signal, portfolio_state) → sized order or None
    5. Queue valid orders for execution at t+1
    6. At t+1: fill at open + 0.05% slippage + configured fee
    7. Check all open positions: stop hit? TP hit? Trailing stop?
    8. Update PortfolioState
    9. Append to trade journal
```

### Position → Trade Lifecycle

`portfolio.py` owns the transition from open `Position` to closed `Trade`:
1. **Open:** `portfolio.open_position(signal, fill_price, size, stop_loss, take_profit)` → creates `Position`, deducts cash, appends to `open_positions`.
2. **Close:** `portfolio.close_position(position, exit_price, exit_reason)` → creates `Trade` from `Position` fields + exit info, computes pnl, appends to `closed_trades`, returns cash, removes from `open_positions`.
3. The backtest engine calls `close_position` when: stop/TP price is breached at candle t's high/low, trailing stop moves and is hit, a SELL signal fires against a LONG position, or the drawdown circuit breaker triggers.

### Anti-Bias Rules (Hard Constraints)

- Execution at t+1 open — enforced in engine, never current candle close
- StandardScaler fitted only on past data — enforced in trainer
- Multi-timeframe features forward-fill only — enforced in feature pipeline
- Gap fills never look forward — enforced in preprocessor

### Risk Module (`src/backtesting/risk.py`)

- **Position sizing:** Half-Kelly formula: `f = 0.5 * (p/a - q/b)` where `p` = win rate, `q` = 1-p, `b` = avg win (as % of entry), `a` = avg loss (as % of entry). `f` is clamped to [0, 1] and applied to current portfolio equity to get dollar risk. Computed from strategy's rolling 30-trade history; defaults to `f=0.1` (10%) if fewer than 10 trades available. The 3% per-trade cap is applied AFTER Kelly — use `min(kelly_size, 0.03 * equity)`.
- **Per-trade cap:** Never risk > 3% of current portfolio equity
- **Stop-loss:** 1.5× ATR from entry (ATR computed at entry candle)
- **Take-profit:** 2:1 minimum R:R from entry; or trailing stop at 2× ATR if configured
- **Max open positions:** 5 total
- **Correlation guard:** Skip if any existing position in asset with rolling 30d correlation > 0.7
- **Drawdown circuit breaker:** If equity < 0.85 × peak equity, halt all new entries, log alert

### Metrics (`src/backtesting/metrics.py`)

Computed per walk-forward test window, then averaged:
- Total return %
- Sharpe ratio (annualized, risk-free = 0)
- Sortino ratio
- Maximum drawdown %
- Win rate %
- Profit factor (gross profit / gross loss)
- Average hold time
- Total fees paid

### Reporting

- Rich terminal table: strategy competition with all metrics + Buy & Hold baseline
- Matplotlib: equity curve chart, drawdown chart — saved to `output/charts/`

### Deliverable Gate

`python main.py backtest --market crypto --symbol BTC/USDT --strategy trend` runs a single-strategy (ML Trend Following) walk-forward backtest and produces a report with metrics for that strategy + Buy & Hold baseline. The full multi-strategy competition table is the Phase 3 deliverable (requires all 4 strategies + ensemble).

---

## Section 6: All Strategies + Ensemble (Phase 3)

### Strategy Interface (`src/strategies/base.py`)

```python
class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, features: pd.DataFrame) -> Signal: ...

    @abstractmethod
    def get_feature_subset(self) -> list[str]:
        """Return list of feature column names this strategy uses.
        The backtest engine filters the full feature matrix to only these columns
        before calling generate_signal. This prevents strategies from accidentally
        consuming features outside their declared scope."""

    @property
    @abstractmethod
    def name(self) -> str: ...
```

**Feature subset enforcement:** The backtest engine (and paper trading runner) call `strategy.get_feature_subset()` at initialization and slice `X = full_features[strategy.get_feature_subset()]` before each `generate_signal` call. Strategies must only reference columns returned by their own `get_feature_subset()`. Tests verify each strategy's model was trained on the same column set it declares.

### Strategy Implementations

**`ml_trend.py` — ML Trend Following**
- Feature subset: ADX, EMA crossover ratios, MACD histogram, multi-timeframe trend alignment
- Logic: enter when ADX > 25 (trend confirmed) + model confidence > threshold
- Exit: trend exhaustion signal (ADX falling + MACD cross)

**`ml_mean_reversion.py` — ML Mean Reversion**
- Feature subset: Bollinger %B, RSI extremes, VWAP distance, return z-score
- Logic: enter at extremes (RSI < 30 or > 70, BB %B < 0 or > 1)
- Exit: price returns to mean (VWAP distance ~0, RSI ~50)

**`ml_momentum.py` — ML Momentum**
- Feature subset: ROC(5), ROC(20), volume surge ratio, breakout flags, relative strength
- Logic: enter on breakout from consolidation with volume confirmation
- Exit: momentum waning (ROC divergence) or trailing stop

**`ml_volatility.py` — ML Volatility Breakout**
- Feature subset: BB width (squeeze detection), ATR expansion ratio, Keltner/BB squeeze, volume expansion
- Logic: enter when squeeze fires (ATR expanding after compression)
- Exit: volatility contracting back to baseline

### Ensemble Meta-Strategy (`src/strategies/ensemble.py`)

- Does NOT trade independently
- Initialised with weights from `selector.py` output (see Section 4)
- Maintains rolling 30-trade performance window per strategy; updates weights after each batch of 30 trades
- **Weight update formula:** `weight_i = max(rolling_sharpe_i, 0) / sum(max(rolling_sharpe_j, 0) for all j)`. If all rolling Sharpes ≤ 0, revert to equal weights (0.25 each).
- **Regime detection:** trending = ADX > 25; volatile = ATR > 1.5× its 30-day SMA; ranging = else. Regime-adjustment multiplies non-suited strategy weights by 0.5 before renormalization (e.g. in trending regime, mean-reversion weight is halved then weights renormalized to sum to 1.0).
- **Signal blending:** weighted soft vote: `combined_confidence = sum(signal_i.confidence * weight_i)`. Direction taken from the majority-weighted direction; HOLD if combined_confidence < `confidence_threshold` config.
- Re-evaluates weights every 30 trades or 30 days (whichever comes first)

### Deliverable Gate

`python main.py compare` shows all 4 strategies + ensemble + buy-and-hold in competition table, with regime analysis column showing dominant strategy per market period.

---

## Section 7: Paper Trading (Phase 4)

### Runner (`src/paper_trading/runner.py`)

Polling loop:
1. Fetch latest candles since last poll via CCXT/yfinance
2. Append to rolling feature window (same pipeline as backtest)
3. Run trained model → Signal
4. Risk manager validates (identical rules as backtest)
5. Log full decision reasoning to journal
6. Update virtual portfolio (simulated fill at last close price + 0.05% slippage + configured fee — matching backtest fill model)
7. Check divergence against backtest expectations
8. Sleep until `poll_interval_seconds` elapses

### Trade Journal (`src/paper_trading/journal.py`)

Every decision logged to `output/paper_journal.json`:
- Timestamp, signal direction, confidence score
- Top feature values that drove the signal
- Simulated entry/exit prices
- Portfolio state snapshot

Also writes human-readable summary to `output/paper_journal.txt`.

### Divergence Detection

- After each simulated trade closes: compare paper P&L vs backtest expected P&L for same signal type
- Rolling divergence metric: `|paper_pnl - expected_pnl| / expected_pnl`
- If rolling divergence > 30%: log `DIVERGENCE_WARNING`, flag for retraining
- `python main.py status` shows: open positions, equity, recent decisions, divergence score

### Configuration

Paper trading off by default in `config.yaml`:
```yaml
paper_trading:
  enabled: false
  poll_interval_seconds: 300
  journal_path: output/paper_journal.json
```

### Deliverable Gate

`python main.py paper` starts the loop, logs decisions. `python main.py status` prints virtual portfolio state.

---

## Section 8: Hardening (Phase 5)

### Data Resilience

- **API failures:** Retry with exponential backoff (3 attempts, 2s/4s/8s delays). On final failure, use cached data and log warning.
- **Small gaps:** ≤3 consecutive missing candles → forward-fill, log info.
- **Large gaps:** >3 consecutive → skip symbol for that window, log warning.
- **Zero-volume candles:** Detected in preprocessor, skipped in feature computation.
- **Market halts:** If gap price is past stop-loss level, exit at gap price (not stop price).

### Model Staleness

- On startup: check model file mtime vs `retrain_interval_days` config (default 30)
- If stale: auto-retrain before paper trading or backtesting
- Log staleness warning with days elapsed since last train

### Correlation Guard (Multi-Asset)

- Compute rolling 30-day correlation matrix across all open + candidate positions
- Block new position if `|corr(new, existing)| > 0.7`
- Applied in `risk.py` before position sizing

### Test Suite

**`tests/test_data.py`**
- Cache hit/miss behavior
- Provider schema validation (all three providers)
- Gap handling (fill vs skip thresholds)
- API retry logic (mocked)

**`tests/test_features.py`**
- No raw price values in output
- NaN-free output after warmup
- Multi-timeframe no-lookahead guarantee
- Target label correctness with threshold

**`tests/test_strategies.py`**
- Signal output: direction in {BUY, SELL, HOLD}, confidence in [0, 1]
- Feature subset: only expected features used per strategy
- Ensemble weights sum to 1.0

**`tests/test_backtest.py`**
- Execution at t+1 (no lookahead in trade fills)
- Stop-loss triggers correctly
- Take-profit triggers correctly
- Drawdown circuit breaker halts entries at 15% drawdown
- Slippage and fee math correct
- Correlation guard blocks correlated positions

### Performance Target

Backtest on 2 years of 1h BTC/USDT data completes in < 60 seconds. Profile with `cProfile` if slow; vectorize feature computation where possible using pandas/numpy operations.

---

## CLI Summary

```bash
python main.py fetch --market crypto --symbol BTC/USDT        # Phase 1 gate
python main.py backtest --market crypto --symbol BTC/USDT     # Phase 2 gate
python main.py backtest --start 2023-01-01 --end 2025-01-01   # Custom date range
python main.py train --market crypto                           # Retrain models
python main.py compare                                         # Phase 3 gate
python main.py paper                                           # Phase 4 gate
python main.py status                                          # Paper trading status
python main.py report --format terminal                        # Export report
python main.py report --format csv
```

---

## Tech Stack

| Concern | Library |
|---|---|
| Language | Python 3.11+ |
| Crypto data | CCXT (Binance public, Kraken fallback) |
| Stocks/Forex data | yfinance |
| ML | scikit-learn, XGBoost |
| Technical indicators | ta (ta-lib wrapper), pandas-ta fallback |
| Data processing | pandas, numpy |
| Visualization | matplotlib |
| Terminal output | rich |
| CLI | click |
| Config | PyYAML |
| Model persistence | joblib |
| Cache | SQLite (stdlib sqlite3) |
| Testing | pytest |

---

## Risk Warnings

- Past performance does not predict future results.
- Never trade real money until paper trading matches backtest results for ≥30 days.
- Start with tiny amounts if going live — money you can afford to lose completely.
- Markets change. Regular retraining is mandatory.
- The ensemble is the real strategy. Individual strategies cycle through good and bad periods.
