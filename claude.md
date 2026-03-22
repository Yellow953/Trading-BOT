# ML Trading Bot

## Project Overview

An ML-powered trading bot that learns patterns from historical data across crypto, stocks, and forex. It runs multiple competing strategies, selects the best performer per market, and finds its own optimal timeframe. Built for backtesting first, paper trading second, live trading (maybe) last.

**Philosophy:** The bot does the learning, not the user. It engineers features, trains models across multiple strategies and timeframes, backtests with walk-forward validation, and reports what works. Human decides how much risk to take — the bot decides how to trade.

**Risk profile:** Balanced risk/reward. Position sizing via Kelly Criterion (half-Kelly for safety). 3% max loss per trade, 2:1 minimum reward/risk ratio target. Never risk more than 10% of portfolio on correlated positions.

## Tech Stack

- **Language:** Python 3.11+
- **Data — Crypto:** CCXT (exchange-agnostic, free public data, no API key needed for historical/backtesting)
- **Data — Stocks:** yfinance (free Yahoo Finance data, no API key needed)
- **Data — Forex:** yfinance forex pairs (e.g., EURUSD=X) for starting; upgrade path to OANDA API later
- **ML:** scikit-learn (Random Forest, Gradient Boosting, ensemble voting), XGBoost
- **Technical Indicators:** ta (ta-lib python wrapper), pandas-ta as fallback
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib (terminal-friendly charts saved to files)
- **Config:** YAML (config.yaml for all user-tunable params)
- **CLI:** click

## Architecture

```
ml-trading-bot/
├── CLAUDE.md
├── pyproject.toml
├── requirements.txt
├── config.yaml                     # All tunable parameters
├── main.py                         # CLI entrypoint
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract DataProvider interface
│   │   │   ├── crypto.py           # CCXT implementation
│   │   │   ├── stocks.py           # yfinance implementation
│   │   │   └── forex.py            # yfinance forex / future OANDA
│   │   ├── cache.py                # Local SQLite cache to avoid re-fetching
│   │   └── preprocess.py           # Clean, normalize, handle gaps/missing data
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py            # RSI, MACD, Bollinger, ATR, ADX, OBV, etc.
│   │   ├── price_action.py         # Returns, candle patterns, support/resistance
│   │   ├── volume.py               # Volume profile, VWAP, volume ratios
│   │   ├── multi_timeframe.py      # Same indicators computed at multiple timeframes
│   │   └── pipeline.py             # Feature pipeline: raw data → feature matrix
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract Strategy interface
│   │   ├── ml_trend.py             # ML trend-following (Random Forest + Gradient Boosting)
│   │   ├── ml_mean_reversion.py    # ML mean-reversion (detect overextended moves)
│   │   ├── ml_momentum.py          # ML momentum (ride strong moves)
│   │   ├── ml_volatility.py        # ML volatility breakout (trade expansions)
│   │   └── ensemble.py             # Meta-strategy: combines top performers dynamically
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Train/retrain models with walk-forward validation
│   │   ├── selector.py             # Compare strategies, pick best per market/regime
│   │   └── persistence.py          # Save/load trained models (joblib)
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py               # Core backtest loop: signals → trades → portfolio
│   │   ├── portfolio.py            # Track positions, cash, equity curve
│   │   ├── risk.py                 # Position sizing, stop-loss, take-profit, max drawdown
│   │   └── metrics.py              # Sharpe, Sortino, max drawdown, win rate, profit factor
│   ├── paper_trading/
│   │   ├── __init__.py
│   │   ├── runner.py               # Live loop: fetch latest data → predict → log decision
│   │   └── journal.py              # Trade journal: log every decision + reasoning
│   └── reporting/
│       ├── __init__.py
│       ├── terminal.py             # Rich terminal output: tables, colored metrics
│       └── charts.py               # Save matplotlib charts to files (equity curve, drawdown, etc.)
├── data/                           # Auto-created: cached data, saved models
│   ├── cache/
│   └── models/
├── output/                         # Auto-created: backtest reports, charts
│   ├── reports/
│   └── charts/
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_strategies.py
    └── test_backtest.py
```

## Data Provider Interface

Every market (crypto, stocks, forex) implements the same interface so the rest of the system is market-agnostic:

```python
class DataProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: datetime, limit: int) -> pd.DataFrame:
        """Return DataFrame with columns: open, high, low, close, volume, indexed by datetime."""
        pass

    @abstractmethod
    def available_symbols(self) -> list[str]:
        pass

    @abstractmethod
    def available_timeframes(self) -> list[str]:
        pass
```

- **Crypto (CCXT):** Supports 1m, 5m, 15m, 1h, 4h, 1d. Default exchange: Binance (public data, no key). Fallback: Kraken.
- **Stocks (yfinance):** Supports 1m (last 7d), 5m (last 60d), 1h (last 730d), 1d (full history). Symbols like "AAPL", "MSFT".
- **Forex (yfinance):** Pairs like "EURUSD=X", "GBPJPY=X". Same timeframe limits as stocks.

All fetched data is cached in a local SQLite database (`data/cache/market_data.db`) keyed by (provider, symbol, timeframe, date). Never re-fetch what we already have.

## Feature Engineering

The feature pipeline transforms raw OHLCV into a feature matrix. Every feature is computed as a **relative/normalized value** (not raw price) so models generalize across assets and time periods.

### Feature Groups

1. **Trend:** SMA/EMA crossovers (as ratios), MACD histogram, ADX, Aroon, Ichimoku cloud distance
2. **Momentum:** RSI, Stochastic %K/%D, Williams %R, ROC at multiple lookbacks, CCI
3. **Volatility:** Bollinger %B and bandwidth, ATR (as % of price), Keltner channel position, historical volatility ratio
4. **Volume:** OBV slope, volume ratio (current/SMA), VWAP distance, Chaikin Money Flow
5. **Price Action:** N-period returns, candle body/shadow ratios, higher-high/lower-low counts, distance from N-period high/low
6. **Multi-Timeframe:** Key indicators (RSI, MACD, ADX) computed at 2-3 timeframes above the trading timeframe for context

### Target Variable

Binary classification: **will price be higher or lower N candles from now?**

N is not fixed — the bot tests multiple horizons (e.g., 6, 12, 24, 48 candles ahead) and picks what works best per strategy. A minimum move threshold (e.g., must move > 0.5% to count as "up") filters out noise.

## Strategies

Each strategy is an independent ML pipeline with its own model, feature subset, and trading logic. They all produce the same output: a **Signal** (BUY / SELL / HOLD) with a confidence score (0-1).

### Strategy Descriptions

1. **ML Trend Following (`ml_trend.py`):**
   - Learns to identify and ride established trends
   - Key features: ADX, SMA/EMA crossover ratios, MACD, multi-timeframe trend alignment
   - Enters when trend is confirmed + pullback detected, exits on trend exhaustion signals
   - Works best in trending markets

2. **ML Mean Reversion (`ml_mean_reversion.py`):**
   - Learns to detect overextended moves that are likely to snap back
   - Key features: Bollinger %B, RSI extremes, distance from VWAP, z-score of returns
   - Enters at extremes, exits at mean
   - Works best in ranging/choppy markets

3. **ML Momentum (`ml_momentum.py`):**
   - Learns to ride strong directional moves early
   - Key features: ROC, volume surge detection, breakout from consolidation, relative strength
   - Enters on breakout confirmation, tight trailing stop
   - Works best during high-volatility expansions

4. **ML Volatility Breakout (`ml_volatility.py`):**
   - Learns to trade volatility expansions after periods of compression
   - Key features: BB width squeeze, ATR expansion, Keltner/Bollinger squeeze, volume expansion
   - Enters when squeeze fires, exits on volatility contraction
   - Works best at regime transitions

5. **Ensemble Meta-Strategy (`ensemble.py`):**
   - Does NOT trade on its own
   - Tracks recent performance (rolling 30-trade window) of all 4 strategies
   - Allocates capital to top performers, reduces allocation to underperformers
   - Detects market regime (trending vs ranging vs volatile) using ADX + volatility metrics
   - Routes trades to the strategy best suited to the current regime

### Strategy Competition

Every backtest run pits all strategies against each other on the same data. The report shows:
- Individual strategy P&L, Sharpe, drawdown, win rate
- Ensemble vs individual performance
- Which strategy dominated in which time periods (regime analysis)

## Model Training

### Walk-Forward Validation (NOT simple train/test split)

Simple train/test splits overfit. We use anchored walk-forward:

```
[=== Train Window 1 ===][Test 1]
[====== Train Window 2 ========][Test 2]
[========= Train Window 3 ===========][Test 3]
```

- Minimum training window: 6 months of data
- Test window: 1 month
- Step forward: 1 month
- Model is retrained at each step (simulates real-world retraining)
- Final metrics are averaged across ALL test windows

### Models Per Strategy

Each strategy trains an ensemble:
- **RandomForestClassifier** (n_estimators=200, good at capturing non-linear patterns)
- **GradientBoostingClassifier** or **XGBClassifier** (good at sequential pattern learning)
- **VotingClassifier** (soft voting) combines both

### Overfitting Prevention

- Features are standardized (StandardScaler) per training window
- Feature importance is tracked — features with near-zero importance are pruned
- Minimum 1000 training samples required
- Regularization: max_depth capped, min_samples_leaf enforced
- If test performance degrades >20% vs training, flag as overfit and simplify

### Timeframe Auto-Selection

For each (strategy, symbol) pair, the trainer runs walk-forward on multiple timeframes (1h, 4h, 1d, etc.) and picks the timeframe that maximizes **out-of-sample Sharpe ratio**, not accuracy. A strategy might be 60% accurate on 1h but have a better Sharpe on 4h because of fewer false signals.

## Backtesting Engine

### Core Loop

```
for each candle in test_period:
    1. Update feature matrix with latest candle
    2. Each strategy generates Signal (BUY/SELL/HOLD + confidence)
    3. Ensemble routes to active strategy or blends signals
    4. Risk manager validates: position sizing, stop-loss, max exposure
    5. If valid trade: execute at next candle's open (NO lookahead bias)
    6. Update portfolio: check stops, take-profits, trailing stops
    7. Log everything to trade journal
```

### Critical Anti-Bias Rules

- **No lookahead bias:** Trades execute at NEXT candle's open, never current close
- **No survivorship bias:** If using multi-asset, include delisted/dead coins
- **Slippage simulation:** 0.05% slippage added to every entry/exit
- **Fee simulation:** Configurable per market (default: 0.1% crypto, 0% stocks via most brokers, spread-based for forex)
- **Gap handling:** If market gaps past stop-loss, exit at gap price, not stop price

### Risk Management

- **Position sizing:** Half-Kelly Criterion based on strategy's historical win rate and avg win/loss
- **Per-trade risk:** Never risk more than 3% of portfolio on a single trade
- **Stop-loss:** ATR-based (1.5x ATR from entry by default), adjustable in config
- **Take-profit:** 2:1 minimum reward/risk ratio, or trailing stop (configurable)
- **Max open positions:** 5 (prevents overexposure)
- **Correlation guard:** Don't open multiple positions in highly correlated assets (e.g., BTC + ETH)
- **Max drawdown circuit breaker:** If portfolio drops 15% from peak, halt all trading and alert

## Paper Trading Mode

Once backtesting shows a viable strategy:

1. Runs on a cron/loop (configurable interval matching the trading timeframe)
2. Fetches latest candle data via CCXT/yfinance
3. Runs the trained model, generates signal
4. Logs the decision + reasoning to a trade journal (JSON + human-readable)
5. Tracks a virtual portfolio with simulated fills
6. Does NOT place real orders — purely observational
7. After N days, compare paper results to backtest expectations

Paper trading is the **mandatory gate** before any real capital. If paper trading results diverge significantly (>30%) from backtest expectations, the model needs retraining or the strategy is overfit.

## Configuration (config.yaml)

```yaml
general:
  initial_capital: 10000
  base_currency: USD
  log_level: INFO

markets:
  crypto:
    enabled: true
    provider: ccxt
    exchange: binance
    symbols: [BTC/USDT, ETH/USDT, SOL/USDT]
    fee_pct: 0.1
  stocks:
    enabled: true
    provider: yfinance
    symbols: [AAPL, MSFT, TSLA, SPY, QQQ]
    fee_pct: 0.0
  forex:
    enabled: true
    provider: yfinance
    symbols: [EURUSD=X, GBPUSD=X, USDJPY=X]
    fee_pct: 0.002

strategies:
  enabled: [trend, mean_reversion, momentum, volatility, ensemble]
  timeframes_to_test: [1h, 4h, 1d]
  prediction_horizons: [6, 12, 24, 48]
  confidence_threshold: 0.55

risk:
  max_risk_per_trade_pct: 3.0
  reward_risk_ratio: 2.0
  stop_loss_atr_multiplier: 1.5
  max_open_positions: 5
  max_portfolio_drawdown_pct: 15.0
  use_trailing_stop: true
  trailing_stop_atr_multiplier: 2.0

training:
  min_training_months: 6
  walk_forward_test_months: 1
  min_samples: 1000
  retrain_interval_days: 30
  feature_importance_threshold: 0.01

paper_trading:
  enabled: false
  poll_interval_seconds: 300
  journal_path: output/paper_journal.json
```

## CLI Interface

```bash
# Full backtest across all enabled markets and strategies
python main.py backtest

# Backtest specific market/symbol
python main.py backtest --market crypto --symbol BTC/USDT

# Backtest with custom date range
python main.py backtest --start 2023-01-01 --end 2025-01-01

# Show strategy competition results
python main.py compare

# Train/retrain models
python main.py train --market crypto

# Start paper trading
python main.py paper

# Show portfolio status during paper trading
python main.py status

# Export results
python main.py report --format terminal
python main.py report --format csv
```

## Terminal Output Style

No dashboard — clean, dense terminal output. Use rich library for tables and colored metrics. Example:

```
══════════════════════════════════════════════════════
  BACKTEST RESULTS — BTC/USDT — 2023-01-01 to 2025-01-01
══════════════════════════════════════════════════════

Strategy Competition:
┌──────────────────┬──────────┬────────┬──────────┬───────────┬──────────────┐
│ Strategy         │ Return % │ Sharpe │ Win Rate │ Max DD %  │ Profit Factor│
├──────────────────┼──────────┼────────┼──────────┼───────────┼──────────────┤
│ Trend Following  │  +42.3%  │  1.82  │  58.2%   │  -8.4%    │     1.91     │
│ Mean Reversion   │  +18.7%  │  1.24  │  62.1%   │  -6.2%    │     1.45     │
│ Momentum         │  +31.5%  │  1.55  │  51.8%   │  -12.1%   │     1.67     │
│ Vol Breakout     │  +22.1%  │  1.38  │  54.3%   │  -9.8%    │     1.52     │
│ Ensemble         │  +51.8%  │  2.01  │  57.4%   │  -6.9%    │     2.12     │
│ Buy & Hold       │  +35.2%  │  0.91  │   —      │  -22.4%   │      —       │
├──────────────────┼──────────┼────────┼──────────┼───────────┼──────────────┤
│ WINNER           │ Ensemble │  2.01  │          │           │              │
└──────────────────┴──────────┴────────┴──────────┴───────────┴──────────────┘

Best Timeframe: 4h (auto-selected, tested: 1h, 4h, 1d)
Total Trades: 187  |  Avg Hold Time: 18.4h  |  Fees Paid: $423.12

Charts saved to: output/charts/btc_usdt_backtest.png
```

## Build Phases

### Phase 1: Data Layer + Feature Engine
- Implement DataProvider for all three markets (CCXT, yfinance)
- Build SQLite cache
- Build feature pipeline with all indicator groups
- Write tests to verify data integrity and feature correctness
- **Deliverable:** `python main.py fetch --market crypto --symbol BTC/USDT` works and caches data

### Phase 2: Single Strategy + Backtester
- Implement the ML Trend Following strategy (simplest starting point)
- Build the backtest engine with proper anti-bias rules
- Build portfolio tracker, risk manager, metrics calculator
- Walk-forward validation for training
- Terminal reporting
- **Deliverable:** `python main.py backtest --market crypto --symbol BTC/USDT` produces a full report

### Phase 3: All Strategies + Competition
- Implement remaining 3 strategies (mean reversion, momentum, volatility)
- Build ensemble meta-strategy
- Strategy comparison reporting
- Timeframe auto-selection
- Multi-asset backtesting
- **Deliverable:** `python main.py compare` shows all strategies competing

### Phase 4: Paper Trading
- Live data polling loop
- Trade journal
- Virtual portfolio tracking
- Divergence detection (paper vs backtest)
- **Deliverable:** `python main.py paper` runs continuously, logs decisions

### Phase 5: Hardening
- Edge case handling (missing data, API failures, market halts)
- Model staleness detection + auto-retrain
- Correlation guard for multi-asset
- Comprehensive test suite
- Performance profiling (backtests should run fast)

## Code Style

- Type hints everywhere
- Dataclasses or Pydantic for structured data (Signal, Trade, Position, PortfolioState)
- ABC for interfaces (DataProvider, Strategy)
- No print statements — use Python logging module
- Docstrings on public methods (keep them short)
- Tests use pytest
- Use rich library for terminal formatting

## Important Warnings

- **This is an experiment, not a money printer.** Past performance does not predict future results.
- **Never trade real money** until paper trading matches backtest results for at least 30 days.
- **Start with tiny amounts** if you ever go live — money you can afford to lose completely.
- **Markets change.** A model that works today may stop working tomorrow. Regular retraining is mandatory.
- **The ensemble is the real strategy.** Individual strategies will cycle through good and bad periods. The ensemble's job is to adapt.