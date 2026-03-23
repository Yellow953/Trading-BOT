# ML Trading Bot — Phase 1: Data Layer & Feature Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project scaffold, shared data contracts, three market data providers with SQLite caching, a data preprocessor, and the full feature engineering pipeline — ending with a working `python main.py fetch` command.

**Architecture:** Layered from contracts outward. `src/core/types.py` defines shared dataclasses first. Data providers implement a common ABC. Feature modules are pure functions that append normalized columns to DataFrames. A pipeline orchestrator composes them and generates binary classification targets.

**Tech Stack:** Python 3.11+, CCXT, yfinance, ta (technical indicators), pandas, numpy, SQLite (stdlib), click, PyYAML, rich, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-ml-trading-bot-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata and dependency declarations |
| `requirements.txt` | Pinned dependencies for reproducibility |
| `config.yaml` | All user-tunable parameters |
| `main.py` | click CLI — all commands (fetch, backtest, train, compare, paper, status, report) |
| `src/core/types.py` | Shared dataclasses: OHLCV, Signal, Position, Trade, PortfolioState |
| `src/data/providers/base.py` | DataProvider ABC |
| `src/data/providers/crypto.py` | CCXT/Binance with Kraken fallback |
| `src/data/providers/stocks.py` | yfinance stocks |
| `src/data/providers/forex.py` | yfinance forex pairs |
| `src/data/cache.py` | SQLite cache, keyed by (provider, symbol, timeframe, date) |
| `src/data/preprocess.py` | Clean, normalize, fill gaps, validate schema |
| `src/features/technical.py` | RSI, MACD, ADX, Aroon, SMA/EMA ratios, Ichimoku |
| `src/features/price_action.py` | Log returns, candle body/shadow ratios, distance from N-period high/low |
| `src/features/volume.py` | OBV slope, volume ratio, VWAP distance, Chaikin Money Flow |
| `src/features/multi_timeframe.py` | Resample + compute RSI/MACD/ADX at higher timeframes |
| `src/features/pipeline.py` | Orchestrate all modules, drop warmup NaNs, generate targets |
| `tests/test_data.py` | Cache, provider schema, gap handling, preprocessor |
| `tests/test_features.py` | No raw prices, NaN-free, no-lookahead, target labels |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `main.py`
- Create: `src/__init__.py`, `src/core/__init__.py`, `src/data/__init__.py`, `src/data/providers/__init__.py`, `src/features/__init__.py`, `src/strategies/__init__.py`, `src/models/__init__.py`, `src/backtesting/__init__.py`, `src/paper_trading/__init__.py`, `src/reporting/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/cache/.gitkeep`, `data/models/.gitkeep`, `output/reports/.gitkeep`, `output/charts/.gitkeep`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "ml-trading-bot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "ccxt>=4.0.0",
    "yfinance>=0.2.36",
    "scikit-learn>=1.4.0",
    "xgboost>=2.0.0",
    "ta>=0.11.0",
    "pandas>=2.1.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "rich>=13.7.0",
    "click>=8.1.7",
    "PyYAML>=6.0.1",
    "joblib>=1.3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-cov>=4.1.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `requirements.txt`**

```
ccxt>=4.0.0
yfinance>=0.2.36
scikit-learn>=1.4.0
xgboost>=2.0.0
ta>=0.11.0
pandas>=2.1.0
numpy>=1.26.0
matplotlib>=3.8.0
rich>=13.7.0
click>=8.1.7
PyYAML>=6.0.1
joblib>=1.3.0
pytest>=8.0.0
pytest-cov>=4.1.0
```

- [ ] **Step 3: Create `config.yaml`**

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

cache:
  db_path: data/cache/market_data.db
```

- [ ] **Step 4: Create `main.py` with stubbed commands**

```python
"""ML Trading Bot CLI."""
import logging
import click
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
@click.option("--config", default="config.yaml", help="Path to config file.")
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """ML Trading Bot."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    log_level = ctx.obj["config"]["general"].get("log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))


@cli.command()
@click.option("--market", required=True, type=click.Choice(["crypto", "stocks", "forex"]))
@click.option("--symbol", required=True)
@click.option("--timeframe", default="1d")
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
@click.pass_context
def fetch(ctx: click.Context, market: str, symbol: str, timeframe: str, start: str | None, end: str | None) -> None:
    """Fetch and cache market data."""
    from src.data.providers.crypto import CryptoProvider
    from src.data.providers.stocks import StocksProvider
    from src.data.providers.forex import ForexProvider
    from src.data.cache import DataCache
    from src.data.preprocess import preprocess
    from datetime import datetime, timezone
    import rich.console

    console = rich.console.Console()
    config = ctx.obj["config"]
    cache = DataCache(config["cache"]["db_path"])

    providers = {
        "crypto": CryptoProvider,
        "stocks": StocksProvider,
        "forex": ForexProvider,
    }
    provider = providers[market](config)

    since = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else datetime(2023, 1, 1, tzinfo=timezone.utc)
    until = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)

    df = cache.get_or_fetch(provider, symbol, timeframe, since, until)
    df = preprocess(df)

    console.print(f"[green]Fetched {len(df)} rows for {symbol} ({timeframe})[/green]")
    console.print(f"Date range: {df.index[0]} → {df.index[-1]}")


@cli.command()
@click.option("--market", default=None, type=click.Choice(["crypto", "stocks", "forex"]))
@click.option("--symbol", default=None)
@click.option("--strategy", default=None)
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.pass_context
def backtest(ctx: click.Context, market: str | None, symbol: str | None, strategy: str | None, start: str | None, end: str | None) -> None:
    """Run walk-forward backtest. (Phase 2)"""
    click.echo("Backtest not yet implemented (Phase 2).")


@cli.command()
@click.option("--market", default=None, type=click.Choice(["crypto", "stocks", "forex"]))
@click.pass_context
def train(ctx: click.Context, market: str | None) -> None:
    """Train/retrain models. (Phase 2)"""
    click.echo("Train not yet implemented (Phase 2).")


@cli.command()
@click.pass_context
def compare(ctx: click.Context) -> None:
    """Show strategy competition results. (Phase 3)"""
    click.echo("Compare not yet implemented (Phase 3).")


@cli.command()
@click.pass_context
def paper(ctx: click.Context) -> None:
    """Start paper trading loop. (Phase 4)"""
    click.echo("Paper trading not yet implemented (Phase 4).")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show paper trading portfolio status. (Phase 4)"""
    click.echo("Status not yet implemented (Phase 4).")


@cli.command()
@click.option("--format", "fmt", default="terminal", type=click.Choice(["terminal", "csv"]))
@click.pass_context
def report(ctx: click.Context, fmt: str) -> None:
    """Export results. (Phase 2+)"""
    click.echo("Report not yet implemented (Phase 2+).")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 5: Create all `__init__.py` files and directory structure**

```bash
mkdir -p src/core src/data/providers src/features src/strategies src/models src/backtesting src/paper_trading src/reporting tests data/cache data/models output/reports output/charts
touch src/__init__.py src/core/__init__.py src/data/__init__.py src/data/providers/__init__.py src/features/__init__.py src/strategies/__init__.py src/models/__init__.py src/backtesting/__init__.py src/paper_trading/__init__.py src/reporting/__init__.py tests/__init__.py
touch data/cache/.gitkeep data/models/.gitkeep output/reports/.gitkeep output/charts/.gitkeep
```

- [ ] **Step 6: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: Packages install without errors.

- [ ] **Step 7: Verify CLI loads**

```bash
python main.py --help
```

Expected: Shows `fetch`, `backtest`, `train`, `compare`, `paper`, `status`, `report` commands.

- [ ] **Step 8: Commit scaffold**

```bash
git add pyproject.toml requirements.txt config.yaml main.py src/ tests/ data/ output/
git commit -m "feat: project scaffold with CLI stubs and directory structure"
```

---

## Task 2: Shared Data Contracts

**Files:**
- Create: `src/core/types.py`
- Test: `tests/test_data.py` (partial — just imports)

- [ ] **Step 1: Write the failing import test**

Create `tests/test_data.py`:

```python
"""Tests for data layer: providers, cache, preprocessor."""
import pytest


def test_types_importable():
    """Shared contracts are importable and well-formed."""
    from src.core.types import OHLCV, Signal, Position, Trade, PortfolioState
    from datetime import datetime, timezone

    ohlcv = OHLCV(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=42000.0, high=43000.0, low=41000.0, close=42500.0, volume=1234.5,
    )
    assert ohlcv.close == 42500.0

    signal = Signal(direction="BUY", confidence=0.75, strategy_name="trend",
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert signal.direction == "BUY"
    assert signal.metadata == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_data.py::test_types_importable -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `src/core/types.py`**

```python
"""Shared data contracts for the ML trading bot."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


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
    open_positions: list[Position] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    closed_trades: list[Trade] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_data.py::test_types_importable -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/types.py tests/test_data.py
git commit -m "feat: shared data contracts (OHLCV, Signal, Position, Trade, PortfolioState)"
```

---

## Task 3: DataProvider ABC

**Files:**
- Create: `src/data/providers/base.py`

> **Spec deviation (deliberate improvement):** The spec defines `fetch_ohlcv(symbol, timeframe, since, limit: int)`. This plan uses `fetch_ohlcv(symbol, timeframe, since, until: datetime)` instead. A range-based interface (`since`/`until`) is safer for cache integration — `limit` varies per timeframe and exchange, making cache key consistency impossible. All three provider implementations use this signature.

- [ ] **Step 1: Add provider interface test to `tests/test_data.py`**

```python
def test_provider_interface():
    """DataProvider ABC cannot be instantiated directly."""
    from src.data.providers.base import DataProvider
    import pytest

    with pytest.raises(TypeError):
        DataProvider({})  # ABC — must subclass
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_data.py::test_provider_interface -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `src/data/providers/base.py`**

```python
"""Abstract DataProvider interface."""
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataProvider(ABC):
    """Common interface for all market data sources."""

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame:
        """Return DataFrame with columns [open, high, low, close, volume], UTC datetime index."""

    @abstractmethod
    def available_symbols(self) -> list[str]:
        """Return list of tradeable symbols."""

    @abstractmethod
    def available_timeframes(self) -> list[str]:
        """Return list of supported timeframe strings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string (e.g. 'ccxt_binance', 'yfinance_stocks')."""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_data.py::test_provider_interface -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/providers/base.py tests/test_data.py
git commit -m "feat: DataProvider ABC"
```

---

## Task 4: SQLite Cache

**Files:**
- Create: `src/data/cache.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write cache tests**

Add to `tests/test_data.py`:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import tempfile
import os


def _make_ohlcv_df(n: int = 10, start: datetime | None = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for testing."""
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open":   np.random.uniform(40000, 45000, n),
        "high":   np.random.uniform(45000, 50000, n),
        "low":    np.random.uniform(35000, 40000, n),
        "close":  np.random.uniform(40000, 45000, n),
        "volume": np.random.uniform(100, 1000, n),
    }, index=idx)


def test_cache_miss_fetches_and_stores(tmp_path):
    """Cache miss calls provider and writes result to SQLite."""
    from src.data.cache import DataCache
    from src.data.providers.base import DataProvider

    db_path = str(tmp_path / "test.db")
    cache = DataCache(db_path)

    mock_provider = MagicMock(spec=DataProvider)
    mock_provider.name = "test_provider"
    mock_df = _make_ohlcv_df(10)
    mock_provider.fetch_ohlcv.return_value = mock_df

    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 1, 1, 10, tzinfo=timezone.utc)

    result = cache.get_or_fetch(mock_provider, "BTC/USDT", "1h", since, until)

    assert mock_provider.fetch_ohlcv.call_count == 1
    assert len(result) == 10
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


def test_cache_hit_avoids_api_call(tmp_path):
    """Cache hit returns stored data without calling provider."""
    from src.data.cache import DataCache
    from src.data.providers.base import DataProvider

    db_path = str(tmp_path / "test.db")
    cache = DataCache(db_path)

    mock_provider = MagicMock(spec=DataProvider)
    mock_provider.name = "test_provider"
    mock_df = _make_ohlcv_df(5)
    mock_provider.fetch_ohlcv.return_value = mock_df

    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 1, 1, 5, tzinfo=timezone.utc)

    # First call — cache miss
    cache.get_or_fetch(mock_provider, "BTC/USDT", "1h", since, until)
    # Second call — same range should hit cache
    cache.get_or_fetch(mock_provider, "BTC/USDT", "1h", since, until)

    assert mock_provider.fetch_ohlcv.call_count == 1  # not called again
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py::test_cache_miss_fetches_and_stores tests/test_data.py::test_cache_hit_avoids_api_call -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `src/data/cache.py`**

```python
"""SQLite cache for OHLCV data. Key: (provider_name, symbol, timeframe, date)."""
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    provider    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    ts          INTEGER NOT NULL,   -- Unix timestamp (seconds, UTC)
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    PRIMARY KEY (provider, symbol, timeframe, ts)
)
"""


class DataCache:
    """Read-through cache backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get_or_fetch(
        self,
        provider,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame:
        """Return cached data; fetch from provider only for missing ranges."""
        cached = self._read(provider.name, symbol, timeframe, since, until)

        if cached.empty:
            logger.info("Cache miss: %s %s %s — fetching from provider", provider.name, symbol, timeframe)
            fresh = provider.fetch_ohlcv(symbol, timeframe, since, until)
            self._write(provider.name, symbol, timeframe, fresh)
            return fresh

        # Check if cached range fully covers the request
        cached_since = cached.index.min().to_pydatetime().replace(tzinfo=timezone.utc)
        cached_until = cached.index.max().to_pydatetime().replace(tzinfo=timezone.utc)

        if cached_since <= since and cached_until >= until:
            logger.debug("Cache hit: %s %s %s", provider.name, symbol, timeframe)
            return cached[(cached.index >= since) & (cached.index <= until)]

        # Partial miss — fetch the uncovered tail
        fetch_since = max(since, cached_until)
        logger.info("Partial cache miss: fetching %s %s %s from %s", provider.name, symbol, timeframe, fetch_since)
        fresh = provider.fetch_ohlcv(symbol, timeframe, fetch_since, until)
        if not fresh.empty:
            self._write(provider.name, symbol, timeframe, fresh)
            combined = pd.concat([cached, fresh]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            return combined[(combined.index >= since) & (combined.index <= until)]

        return cached[(cached.index >= since) & (cached.index <= until)]

    def _read(self, provider_name: str, symbol: str, timeframe: str, since: datetime, until: datetime) -> pd.DataFrame:
        since_ts = int(since.timestamp())
        until_ts = int(until.timestamp())
        rows = self._conn.execute(
            "SELECT ts, open, high, low, close, volume FROM ohlcv "
            "WHERE provider=? AND symbol=? AND timeframe=? AND ts>=? AND ts<=? ORDER BY ts",
            (provider_name, symbol, timeframe, since_ts, until_ts),
        ).fetchall()

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
        data = {
            "open":   [r[1] for r in rows],
            "high":   [r[2] for r in rows],
            "low":    [r[3] for r in rows],
            "close":  [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        }
        return pd.DataFrame(data, index=idx)

    def _write(self, provider_name: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = [
            (provider_name, symbol, timeframe, int(ts.timestamp()), row["open"], row["high"], row["low"], row["close"], row["volume"])
            for ts, row in df.iterrows()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO ohlcv (provider, symbol, timeframe, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
        logger.debug("Cached %d rows for %s %s %s", len(rows), provider_name, symbol, timeframe)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py::test_cache_miss_fetches_and_stores tests/test_data.py::test_cache_hit_avoids_api_call -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/cache.py tests/test_data.py
git commit -m "feat: SQLite data cache with read-through and partial miss handling"
```

---

## Task 5: Data Preprocessor

**Files:**
- Create: `src/data/preprocess.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write preprocessor tests**

Add to `tests/test_data.py`:

```python
def test_preprocessor_rejects_missing_columns():
    """Preprocessor raises if required OHLCV columns are missing."""
    from src.data.preprocess import preprocess
    import pytest

    bad_df = pd.DataFrame({"open": [1.0], "close": [2.0]})  # missing high, low, volume
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess(bad_df)


def test_preprocessor_drops_ohlcv_nans():
    """Rows with NaN in any OHLCV column are dropped."""
    from src.data.preprocess import preprocess

    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": [1.0, None, 3.0, 4.0, 5.0],
        "high": [2.0, 2.5, 3.5, 4.5, 5.5],
        "low":  [0.5, 0.8, 2.5, 3.5, 4.5],
        "close":[1.5, 2.0, 3.2, 4.2, 5.2],
        "volume":[100, 200, 300, 400, 500],
    }, index=idx)

    result = preprocess(df)
    assert len(result) == 4
    assert result.isna().sum().sum() == 0


def test_preprocessor_fills_small_gaps():
    """Gaps of ≤3 consecutive missing candles are forward-filled."""
    from src.data.preprocess import preprocess

    # Build complete index, then remove 2 rows to simulate gaps
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": 42000.0, "high": 43000.0, "low": 41000.0,
        "close": 42500.0, "volume": 500.0,
    }, index=idx)
    # Remove rows 3 and 4 (gap of 2 — should be filled)
    df_with_gap = df.drop(df.index[[3, 4]])

    result = preprocess(df_with_gap)
    # With rows 3 and 4 removed from a 10-row 1h sequence, the mode diff is 1h
    # (6 of 7 remaining intervals = 1h, 1 = 2h), so freq inference is reliable here.
    assert len(result) == 10  # gaps filled back to 10 rows


def test_preprocessor_large_gap_not_filled():
    """Gaps of >3 consecutive candles are NOT filled (left as gaps, warning logged)."""
    from src.data.preprocess import preprocess
    import logging

    idx = pd.date_range("2024-01-01", periods=15, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": 42000.0, "high": 43000.0, "low": 41000.0,
        "close": 42500.0, "volume": 500.0,
    }, index=idx)
    # Remove rows 3-8 (gap of 6 — exceeds fill threshold)
    df_with_gap = df.drop(df.index[3:9])

    result = preprocess(df_with_gap)

    # Large gap should NOT be filled — row count stays at 9 (not 15)
    assert len(result) < 15, "Large gap should NOT be filled to full row count"


def test_preprocessor_removes_duplicate_timestamps():
    """Duplicate timestamps are removed (last value kept)."""
    from src.data.preprocess import preprocess

    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),  # duplicate
        pd.Timestamp("2024-01-01 01:00", tz="UTC"),
    ])
    df = pd.DataFrame({
        "open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5],
        "low": [0.5, 1.5, 2.5], "close": [1.2, 2.2, 3.2], "volume": [10.0, 20.0, 30.0],
    }, index=idx)

    result = preprocess(df)
    assert len(result) == 2
    assert result.index.is_unique
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py::test_preprocessor_rejects_missing_columns tests/test_data.py::test_preprocessor_drops_ohlcv_nans tests/test_data.py::test_preprocessor_fills_small_gaps tests/test_data.py::test_preprocessor_removes_duplicate_timestamps -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/data/preprocess.py`**

```python
"""Clean, normalize, and validate OHLCV DataFrames."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
_MAX_FILL_GAP = 3  # forward-fill gaps of at most this many consecutive candles


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, clean, and normalize an OHLCV DataFrame.

    Returns a DataFrame with:
    - UTC datetime index, sorted ascending
    - No duplicate timestamps
    - No NaN in OHLCV columns
    - Small gaps (≤3 candles) forward-filled; larger gaps logged as warnings
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Normalize index to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()

    # Remove duplicates (keep last)
    df = df[~df.index.duplicated(keep="last")]

    # Drop rows with NaN in any OHLCV column
    before = len(df)
    df = df.dropna(subset=list(_REQUIRED_COLUMNS))
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with NaN OHLCV values", dropped)

    # Detect and fill small gaps by reindexing to expected uniform frequency
    if len(df) >= 2:
        inferred_freq = _infer_freq(df)
        if inferred_freq:
            full_idx = pd.date_range(df.index[0], df.index[-1], freq=inferred_freq, tz="UTC")
            n_missing = len(full_idx) - len(df)
            if n_missing > 0:
                # Check max consecutive gap length before filling
                df_reindexed = df.reindex(full_idx)
                gap_lengths = df_reindexed["close"].isna().astype(int)
                max_gap = _max_consecutive(gap_lengths)
                if max_gap <= _MAX_FILL_GAP:
                    df = df_reindexed.ffill()
                    logger.info("Forward-filled %d gap candles (max consecutive: %d)", n_missing, max_gap)
                else:
                    logger.warning("Gap of %d consecutive candles detected — too large to fill, leaving as-is", max_gap)

    return df[list(_REQUIRED_COLUMNS)]


def _infer_freq(df: pd.DataFrame) -> str | None:
    """Infer the dominant frequency from the first 50 intervals."""
    diffs = df.index[:50].to_series().diff().dropna()
    if diffs.empty:
        return None
    mode_diff = diffs.mode()[0]
    seconds = int(mode_diff.total_seconds())
    freq_map = {60: "1min", 300: "5min", 900: "15min", 3600: "1h", 14400: "4h", 86400: "1D"}
    return freq_map.get(seconds)


def _max_consecutive(series: pd.Series) -> int:
    """Return the length of the longest consecutive run of 1s in a boolean/int series."""
    max_run = current = 0
    for val in series:
        if val:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py::test_preprocessor_rejects_missing_columns tests/test_data.py::test_preprocessor_drops_ohlcv_nans tests/test_data.py::test_preprocessor_fills_small_gaps tests/test_data.py::test_preprocessor_removes_duplicate_timestamps -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocess.py tests/test_data.py
git commit -m "feat: data preprocessor with gap filling, dedup, and schema validation"
```

---

## Task 6: CryptoProvider (CCXT/Binance)

**Files:**
- Create: `src/data/providers/crypto.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write CryptoProvider tests**

Add to `tests/test_data.py`:

```python
def test_crypto_provider_schema(tmp_path):
    """CryptoProvider returns DataFrame with correct columns when CCXT call is mocked."""
    from src.data.providers.crypto import CryptoProvider
    from unittest.mock import patch

    config = {"markets": {"crypto": {"exchange": "binance", "fee_pct": 0.1}}}

    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 1, 1, 10, tzinfo=timezone.utc)

    # CCXT returns list of [ts_ms, open, high, low, close, volume]
    fake_ohlcv = [
        [1704067200000 + i * 3600000, 42000.0, 43000.0, 41000.0, 42500.0, 100.0]
        for i in range(5)
    ]

    with patch("ccxt.binance") as mock_exchange_cls:
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = fake_ohlcv
        mock_exchange_cls.return_value = mock_ex

        provider = CryptoProvider(config)
        df = provider.fetch_ohlcv("BTC/USDT", "1h", since, until)

    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert df.index.tz is not None  # UTC tz-aware
    assert len(df) == 5


def test_crypto_provider_available_timeframes():
    """CryptoProvider lists expected timeframes."""
    from src.data.providers.crypto import CryptoProvider

    config = {"markets": {"crypto": {"exchange": "binance", "fee_pct": 0.1}}}
    provider = CryptoProvider(config)
    tfs = provider.available_timeframes()
    assert "1h" in tfs
    assert "4h" in tfs
    assert "1d" in tfs
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py::test_crypto_provider_schema tests/test_data.py::test_crypto_provider_available_timeframes -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/data/providers/crypto.py`**

```python
"""CCXT-based crypto data provider (Binance primary, Kraken fallback)."""
import logging
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from src.data.providers.base import DataProvider

logger = logging.getLogger(__name__)

_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
_CHUNK_SIZE = 1000  # max candles per API call


class CryptoProvider(DataProvider):
    """Fetches OHLCV data from Binance via CCXT. Falls back to Kraken on error."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        exchange_id = config.get("markets", {}).get("crypto", {}).get("exchange", "binance")
        self._exchange = self._init_exchange(exchange_id)

    def _init_exchange(self, exchange_id: str):
        try:
            ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            ex.load_markets()
            return ex
        except Exception as e:
            logger.warning("Failed to init %s (%s), falling back to kraken", exchange_id, e)
            ex = ccxt.kraken({"enableRateLimit": True})
            ex.load_markets()
            return ex

    @property
    def name(self) -> str:
        return f"ccxt_{self._exchange.id}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: datetime, until: datetime) -> pd.DataFrame:
        """Paginate CCXT to fetch all candles in [since, until]."""
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        all_rows = []

        cursor = since_ms
        while cursor < until_ms:
            try:
                rows = self._exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=_CHUNK_SIZE)
            except Exception as e:
                logger.error("CCXT fetch error for %s %s: %s", symbol, timeframe, e)
                break

            if not rows:
                break

            all_rows.extend([r for r in rows if r[0] < until_ms])
            last_ts = rows[-1][0]
            if last_ts <= cursor:
                break  # no progress — stop
            cursor = last_ts + 1
            time.sleep(self._exchange.rateLimit / 1000)

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        idx = pd.to_datetime([r[0] for r in all_rows], unit="ms", utc=True)
        df = pd.DataFrame(
            {"open": [r[1] for r in all_rows], "high": [r[2] for r in all_rows],
             "low": [r[3] for r in all_rows], "close": [r[4] for r in all_rows],
             "volume": [r[5] for r in all_rows]},
            index=idx,
        )
        return df[~df.index.duplicated(keep="last")].sort_index()

    def available_symbols(self) -> list[str]:
        return list(self._exchange.markets.keys()) if self._exchange.markets else []

    def available_timeframes(self) -> list[str]:
        return _TIMEFRAMES
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py::test_crypto_provider_schema tests/test_data.py::test_crypto_provider_available_timeframes -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/providers/crypto.py tests/test_data.py
git commit -m "feat: CryptoProvider (CCXT/Binance with Kraken fallback)"
```

---

## Task 7: StocksProvider and ForexProvider

**Files:**
- Create: `src/data/providers/stocks.py`
- Create: `src/data/providers/forex.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write provider schema tests**

Add to `tests/test_data.py`:

```python
def test_stocks_provider_schema():
    """StocksProvider returns correct DataFrame schema when yfinance is mocked."""
    from src.data.providers.stocks import StocksProvider
    from unittest.mock import patch

    config = {"markets": {"stocks": {"fee_pct": 0.0}}}
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 6, 1, tzinfo=timezone.utc)

    fake_df = _make_ohlcv_df(100, start=since)
    fake_df.columns = [c.capitalize() for c in fake_df.columns]  # yfinance uses Title Case

    with patch("yfinance.download", return_value=fake_df):
        provider = StocksProvider(config)
        df = provider.fetch_ohlcv("AAPL", "1d", since, until)

    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert df.index.tz is not None


def test_forex_provider_schema():
    """ForexProvider returns correct DataFrame schema when yfinance is mocked."""
    from src.data.providers.forex import ForexProvider
    from unittest.mock import patch

    config = {"markets": {"forex": {"fee_pct": 0.002}}}
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 6, 1, tzinfo=timezone.utc)

    fake_df = _make_ohlcv_df(100, start=since)
    fake_df.columns = [c.capitalize() for c in fake_df.columns]

    with patch("yfinance.download", return_value=fake_df):
        provider = ForexProvider(config)
        df = provider.fetch_ohlcv("EURUSD=X", "1d", since, until)

    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py::test_stocks_provider_schema tests/test_data.py::test_forex_provider_schema -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/data/providers/stocks.py`**

```python
"""yfinance-based stocks data provider."""
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from src.data.providers.base import DataProvider

logger = logging.getLogger(__name__)

# yfinance native timeframe strings → max lookback in days (None = unlimited)
_TIMEFRAME_LIMITS = {
    "1m": 7, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "90m": 60, "1d": None, "1wk": None, "1mo": None,
}
# For timeframes not native in yfinance, fetch at finer granularity and resample.
# "4h" is not a yfinance interval — fetch at 1h and resample to 4h.
_YF_FETCH_TF = {"1h": "1h", "4h": "1h", "1d": "1d"}
_RESAMPLE_NEEDED = {"4h": "4h"}  # timeframes that need resampling after fetch


class StocksProvider(DataProvider):
    """Fetches US stock OHLCV data via yfinance."""

    @property
    def name(self) -> str:
        return "yfinance_stocks"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: datetime, until: datetime) -> pd.DataFrame:
        yf_tf = _YF_FETCH_TF.get(timeframe)
        if yf_tf is None:
            raise ValueError(f"Unsupported timeframe '{timeframe}' for StocksProvider. Supported: {list(_YF_FETCH_TF)}")

        raw = yf.download(
            symbol, start=since.strftime("%Y-%m-%d"), end=until.strftime("%Y-%m-%d"),
            interval=yf_tf, auto_adjust=True, progress=False,
        )
        df = self._normalize(raw)

        # Resample if the requested timeframe requires it (e.g. 4h from 1h bars)
        if timeframe in _RESAMPLE_NEEDED and not df.empty:
            rule = _RESAMPLE_NEEDED[timeframe]
            df = df.resample(rule, label="left", closed="left").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

        return df

    def available_symbols(self) -> list[str]:
        return self.config.get("markets", {}).get("stocks", {}).get("symbols", [])

    def available_timeframes(self) -> list[str]:
        return list(_TIMEFRAME_LIMITS.keys())

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()
```

- [ ] **Step 4: Create `src/data/providers/forex.py`**

```python
"""yfinance-based forex data provider."""
import logging
from datetime import datetime

import pandas as pd

from src.data.providers.stocks import StocksProvider

logger = logging.getLogger(__name__)


class ForexProvider(StocksProvider):
    """
    Forex pairs via yfinance (e.g. EURUSD=X).
    Identical behaviour to StocksProvider — same yfinance API.
    """

    @property
    def name(self) -> str:
        return "yfinance_forex"

    def available_symbols(self) -> list[str]:
        return self.config.get("markets", {}).get("forex", {}).get("symbols", [])
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_data.py::test_stocks_provider_schema tests/test_data.py::test_forex_provider_schema -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/data/providers/stocks.py src/data/providers/forex.py tests/test_data.py
git commit -m "feat: StocksProvider and ForexProvider (yfinance)"
```

---

## Task 8: Technical Features

**Files:**
- Create: `src/features/technical.py`
- Create: `tests/test_features.py` (initial)

- [ ] **Step 1: Create `tests/test_features.py` with initial test**

```python
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

    # The original OHLCV columns are preserved — check ONLY added columns
    added_cols = [c for c in result.columns if c not in {"open", "high", "low", "close", "volume"}]
    assert len(added_cols) > 0, "No feature columns were added"

    for col in added_cols:
        col_max = result[col].abs().max()
        assert col_max < 1000, f"Column '{col}' has suspiciously large values (max={col_max:.1f}) — may be raw price"


def test_technical_features_no_nan_after_warmup():
    """After dropping the warmup period, technical features contain no NaN."""
    from src.features.technical import add_technical_features, WARMUP_PERIODS

    df = _make_ohlcv(500)
    result = add_technical_features(df.copy())
    trimmed = result.iloc[WARMUP_PERIODS:]

    added_cols = [c for c in result.columns if c not in {"open", "high", "low", "close", "volume"}]
    nan_counts = trimmed[added_cols].isna().sum()
    assert nan_counts.sum() == 0, f"NaNs found after warmup:\n{nan_counts[nan_counts > 0]}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/features/technical.py`**

```python
"""Technical indicator features — all normalized/relative, no raw prices."""
import logging

import numpy as np
import pandas as pd

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    import pandas_ta as pta  # fallback

logger = logging.getLogger(__name__)

# Warmup rows needed before all indicators are valid (longest lookback: EMA200)
WARMUP_PERIODS = 210


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized technical indicator columns to df.
    Input: OHLCV DataFrame (columns: open, high, low, close, volume).
    Returns: df with additional feature columns appended.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- RSI ---
    df["rsi_14"] = _rsi(close, 14)
    df["rsi_28"] = _rsi(close, 28)

    # --- ATR (used for normalizing MACD histogram) ---
    atr = _atr(high, low, close, 14)

    # --- MACD histogram normalized by ATR ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (macd_line - signal_line) / atr.replace(0, np.nan)

    # --- ADX ---
    df["adx_14"] = _adx(high, low, close, 14)

    # --- Aroon oscillator ---
    df["aroon_osc"] = _aroon_oscillator(high, low, 25)

    # --- SMA/EMA crossover ratios ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df["ema_20_50_ratio"] = (ema20 / ema50) - 1.0   # near 0 = aligned
    df["ema_50_200_ratio"] = (ema50 / ema200) - 1.0

    # --- Ichimoku cloud distance ---
    df["ichimoku_cloud_dist"] = _ichimoku_cloud_distance(high, low, close)

    return df


# --- Indicator helpers ---

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed ADX (0–100 scale)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def _aroon_oscillator(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    aroon_up = high.rolling(period + 1).apply(lambda x: x.argmax() / period * 100, raw=True)
    aroon_down = low.rolling(period + 1).apply(lambda x: x.argmin() / period * 100, raw=True)
    return aroon_up - aroon_down


def _ichimoku_cloud_distance(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Distance of close from the midpoint of the Ichimoku cloud, as % of price."""
    nine_high = high.rolling(9).max()
    nine_low = low.rolling(9).min()
    tenkan = (nine_high + nine_low) / 2

    twenty_six_high = high.rolling(26).max()
    twenty_six_low = low.rolling(26).min()
    kijun = (twenty_six_high + twenty_six_low) / 2

    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    cloud_mid = (senkou_a + senkou_b) / 2

    return (close - cloud_mid) / close.replace(0, np.nan)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/technical.py tests/test_features.py
git commit -m "feat: technical indicator features (RSI, MACD, ADX, Aroon, EMA ratios, Ichimoku)"
```

---

## Task 9: Price Action Features

**Files:**
- Create: `src/features/price_action.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Add price action tests**

Add to `tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py::test_price_action_no_raw_prices tests/test_features.py::test_price_action_candle_ratios_bounded -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/features/price_action.py`**

```python
"""Price action features — returns, candle patterns, distance from extremes."""
import numpy as np
import pandas as pd


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append price action feature columns.
    All values are relative — no raw price scale.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    # Log returns at multiple lookbacks
    for n in [1, 5, 10, 20]:
        df[f"log_ret_{n}"] = np.log(close / close.shift(n))

    # Candle geometry ratios
    hl_range = (high - low).replace(0, np.nan)
    df["candle_body_ratio"] = (close - open_).abs() / hl_range
    df["upper_shadow_ratio"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / hl_range
    df["lower_shadow_ratio"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / hl_range

    # Higher-high / lower-low count over N periods
    for n in [5, 10, 20]:
        df[f"hh_count_{n}"] = (high > high.shift(1)).rolling(n).sum() / n
        df[f"ll_count_{n}"] = (low < low.shift(1)).rolling(n).sum() / n

    # Distance from N-period high/low as % of price
    for n in [20, 50]:
        df[f"dist_from_high_{n}"] = (close - high.rolling(n).max()) / close.replace(0, np.nan)
        df[f"dist_from_low_{n}"] = (close - low.rolling(n).min()) / close.replace(0, np.nan)

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py::test_price_action_no_raw_prices tests/test_features.py::test_price_action_candle_ratios_bounded -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/price_action.py tests/test_features.py
git commit -m "feat: price action features (returns, candle geometry, distance from extremes)"
```

---

## Task 10: Volume Features

**Files:**
- Create: `src/features/volume.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Add volume feature tests**

Add to `tests/test_features.py`:

```python
def test_volume_features_no_raw_prices():
    """Volume features contain no raw price-scale values."""
    from src.features.volume import add_volume_features

    df = _make_ohlcv(300)
    result = add_volume_features(df.copy())

    added_cols = [c for c in result.columns if c not in {"open", "high", "low", "close", "volume"}]
    for col in added_cols:
        assert result[col].abs().max() < 1000, f"Column '{col}' may contain raw price-scale values"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_features.py::test_volume_features_no_raw_prices -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/features/volume.py`**

```python
"""Volume-based features — OBV slope, volume ratio, VWAP distance, CMF."""
import numpy as np
import pandas as pd


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append volume feature columns. All normalized/relative."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # OBV (cumulative, then take 20-period linear regression slope, normalized by ATR)
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    df["obv_slope_norm"] = _linreg_slope(obv, 20) / (close * 0.001).replace(0, np.nan)

    # Volume ratio: current / 20-period SMA
    vol_sma = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_sma.replace(0, np.nan)

    # VWAP distance: (close - vwap) / close
    # Rolling VWAP over 20 periods
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).rolling(20).sum()
    cum_vol = volume.rolling(20).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap_dist"] = (close - vwap) / close.replace(0, np.nan)

    # Chaikin Money Flow (20-period)
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    df["cmf_20"] = (clv * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    return df


def _linreg_slope(series: pd.Series, period: int) -> pd.Series:
    """Rolling linear regression slope over `period` bars."""
    x = np.arange(period)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def slope(window):
        y_mean = window.mean()
        return ((x - x_mean) * (window - y_mean)).sum() / x_var

    return series.rolling(period).apply(slope, raw=True)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_features.py::test_volume_features_no_raw_prices -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/volume.py tests/test_features.py
git commit -m "feat: volume features (OBV slope, volume ratio, VWAP distance, CMF)"
```

---

## Task 11: Multi-Timeframe Features

**Files:**
- Create: `src/features/multi_timeframe.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Add multi-timeframe no-lookahead test**

Add to `tests/test_features.py`:

```python
def test_multi_timeframe_no_lookahead():
    """Higher-timeframe values at time t use only data available at or before t."""
    from src.features.multi_timeframe import add_multi_timeframe_features

    # Build a 1h OHLCV dataset
    df = _make_ohlcv(200)
    result = add_multi_timeframe_features(df.copy(), base_timeframe="1h")

    # Multi-TF columns should exist
    mtf_cols = [c for c in result.columns if c.startswith("tf_")]
    assert len(mtf_cols) > 0, "No multi-timeframe columns found"

    # For each 4h bar boundary: the value at bar t should equal the value at bar t-1
    # (because new 4h bar won't close until its 4th 1h candle)
    # We verify no NaN is introduced from right-side (future) data
    # A simple proxy: the first mtf_col value should be NaN (not filled from future)
    first_valid = result[mtf_cols[0]].first_valid_index()
    assert first_valid is not None
    assert pd.isna(result[mtf_cols[0]].iloc[0])  # warmup = NaN, not filled from future
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_features.py::test_multi_timeframe_no_lookahead -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/features/multi_timeframe.py`**

```python
"""Multi-timeframe features — RSI, MACD, ADX computed at higher timeframes."""
import logging

import pandas as pd

from src.features.technical import _rsi, _atr, _adx

logger = logging.getLogger(__name__)

# Base timeframe → list of higher timeframes to compute
_HIGHER_TF_MAP = {
    "1h": ["4h", "1d"],
    "4h": ["1d"],
    "1d": [],  # no higher timeframe available
}

_RESAMPLE_RULE = {
    "4h": "4h",
    "1d": "1D",
}


def add_multi_timeframe_features(df: pd.DataFrame, base_timeframe: str = "1h") -> pd.DataFrame:
    """
    For each higher timeframe, resample the OHLCV data, compute indicators,
    then left-join back to the base timeframe using forward-fill only
    (no lookahead: a completed 4h candle's value only appears at the NEXT 1h bar).
    """
    higher_tfs = _HIGHER_TF_MAP.get(base_timeframe, [])

    for tf in higher_tfs:
        rule = _RESAMPLE_RULE[tf]
        resampled = _resample_ohlcv(df, rule)

        # Compute indicators at this higher timeframe
        close = resampled["close"]
        high = resampled["high"]
        low = resampled["low"]

        indicators = pd.DataFrame(index=resampled.index)
        indicators[f"tf_{tf}_rsi_14"] = _rsi(close, 14)
        atr = _atr(high, low, close, 14)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
        indicators[f"tf_{tf}_macd_hist_norm"] = macd_hist / atr.replace(0, float("nan"))
        indicators[f"tf_{tf}_adx_14"] = _adx(high, low, close, 14)

        # Shift by 1 to avoid lookahead: the completed candle at t is available at t+1
        indicators = indicators.shift(1)

        # Reindex to base timeframe — forward-fill (NOT backward-fill)
        indicators = indicators.reindex(df.index, method="ffill")

        for col in indicators.columns:
            df[col] = indicators[col]

    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample base OHLCV to a higher timeframe using OHLC aggregation."""
    return df.resample(rule, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_features.py::test_multi_timeframe_no_lookahead -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/multi_timeframe.py tests/test_features.py
git commit -m "feat: multi-timeframe features with forward-fill only (no lookahead)"
```

---

## Task 12: Feature Pipeline

**Files:**
- Create: `src/features/pipeline.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Add pipeline tests**

Add to `tests/test_features.py`:

```python
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

    # Build a DataFrame with known prices
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Flat price — all forward returns = 0, so all targets should be 0
    price = 42000.0
    df = pd.DataFrame({
        "open": price, "high": price * 1.001, "low": price * 0.999,
        "close": price, "volume": 500.0,
    }, index=idx)

    X, _ = build_features(df, timeframe="1h", include_targets=True)
    if "target_6" in X.columns:
        # Flat price means target should never be 1 (no 0.5% move)
        assert X["target_6"].max() == 0


def test_pipeline_reproducible():
    """Same input data produces identical feature matrix."""
    from src.features.pipeline import build_features

    df = _make_ohlcv(300)
    X1, names1 = build_features(df.copy(), timeframe="1h")
    X2, names2 = build_features(df.copy(), timeframe="1h")

    pd.testing.assert_frame_equal(X1, X2)
    assert names1 == names2


def test_pipeline_no_targets_at_inference():
    """With include_targets=False (default), no target columns appear — safe for inference."""
    from src.features.pipeline import build_features

    df = _make_ohlcv(300)
    X, feature_names = build_features(df, timeframe="1h", include_targets=False)

    target_cols = [c for c in X.columns if c.startswith("target_")]
    assert len(target_cols) == 0, f"Target columns should not be present at inference: {target_cols}"


def test_pipeline_output_is_unscaled():
    """Pipeline output is raw (not standardized). StandardScaler is owned by trainer.py.

    Verifies the leakage boundary: RSI values (0-100 scale) appear in output, proving
    no StandardScaler was applied in the pipeline.
    """
    from src.features.pipeline import build_features

    df = _make_ohlcv(400)
    X, feature_names = build_features(df, timeframe="1h")

    rsi_cols = [c for c in feature_names if "rsi" in c]
    assert len(rsi_cols) > 0, "Expected RSI columns in output"

    for col in rsi_cols:
        # RSI is 0-100. A StandardScaler would compress this to ~[-3, 3].
        assert X[col].max() > 10, f"RSI column '{col}' looks scaled (max={X[col].max():.2f})"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py::test_pipeline_output_is_nan_free tests/test_features.py::test_pipeline_generates_targets tests/test_features.py::test_pipeline_target_threshold tests/test_features.py::test_pipeline_reproducible -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/features/pipeline.py`**

```python
"""Feature pipeline: orchestrates all feature modules and generates targets."""
import logging

import numpy as np
import pandas as pd

from src.features.technical import add_technical_features, WARMUP_PERIODS
from src.features.price_action import add_price_action_features
from src.features.volume import add_volume_features
from src.features.multi_timeframe import add_multi_timeframe_features

logger = logging.getLogger(__name__)

_OHLCV_COLS = {"open", "high", "low", "close", "volume"}
_MIN_MOVE_THRESHOLD = 0.005  # 0.5%
_TARGET_HORIZONS = [6, 12, 24, 48]


def build_features(
    df: pd.DataFrame,
    timeframe: str = "1h",
    include_targets: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform raw OHLCV into a feature matrix.

    Args:
        df: OHLCV DataFrame from preprocessor.
        timeframe: Base timeframe string (used for multi-TF selection).
        include_targets: If True, append binary target columns.

    Returns:
        (X, feature_names): Raw (unscaled) feature DataFrame, list of feature column names.
        Note: StandardScaler is owned by trainer.py, not here.
    """
    df = df.copy()

    # Build feature groups
    df = add_technical_features(df)
    df = add_price_action_features(df)
    df = add_volume_features(df)
    df = add_multi_timeframe_features(df, base_timeframe=timeframe)

    # Add targets before dropping warmup (need future close prices).
    # WARNING: include_targets=True must ONLY be called during training label generation.
    # At inference time (backtest engine, paper trading runner), always use include_targets=False
    # (the default). Calling with include_targets=True at inference would introduce lookahead bias.
    if include_targets:
        close = df["close"]
        n = len(df)
        for horizon in _TARGET_HORIZONS:
            future_close = close.shift(-horizon)
            fwd_return = (future_close - close) / close
            df[f"target_{horizon}"] = (fwd_return > _MIN_MOVE_THRESHOLD).astype(float)
            # NaN the last `horizon` rows — those have no future data available.
            # Use iloc-based slicing (not negative index) for safety with short DataFrames.
            if n > horizon:
                df.iloc[-horizon:, df.columns.get_loc(f"target_{horizon}")] = np.nan
            else:
                df[f"target_{horizon}"] = np.nan

    # Drop warmup rows (NaN from indicator lookback)
    df = df.iloc[WARMUP_PERIODS:].copy()

    # Identify feature columns (exclude OHLCV and targets)
    feature_cols = [c for c in df.columns if c not in _OHLCV_COLS and not c.startswith("target_")]

    # Drop any remaining NaN in feature columns
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.debug("Dropped %d rows with NaN in feature columns", dropped)

    return df[feature_cols + ([f"target_{h}" for h in _TARGET_HORIZONS] if include_targets else [])], feature_cols
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/pipeline.py tests/test_features.py
git commit -m "feat: feature pipeline with all modules, target generation, and warmup handling"
```

---

## Task 13: Run Full Test Suite & Fix Failures

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass. If any fail, fix before proceeding.

- [ ] **Step 2: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve test failures from full test run"
```

---

## Task 14: Wire CLI `fetch` Command

At this point `main.py` already has a `fetch` stub (Task 1 Step 4). Verify it works end-to-end.

- [ ] **Step 1: Run the fetch command (requires internet — Binance public API)**

```bash
python main.py fetch --market crypto --symbol BTC/USDT --timeframe 1d --start 2024-01-01 --end 2024-02-01
```

Expected output (approximately):
```
Fetched 31 rows for BTC/USDT (1d)
Date range: 2024-01-01 00:00:00+00:00 → 2024-01-31 00:00:00+00:00
```

- [ ] **Step 2: Run fetch again to verify cache hit (faster, no API calls)**

```bash
python main.py fetch --market crypto --symbol BTC/USDT --timeframe 1d --start 2024-01-01 --end 2024-02-01
```

Expected: Same output, faster (cache hit logged at DEBUG level).

- [ ] **Step 3: Test stocks fetch**

```bash
python main.py fetch --market stocks --symbol AAPL --timeframe 1d --start 2024-01-01 --end 2024-06-01
```

- [ ] **Step 4: Commit Phase 1 gate**

```bash
git add -A
git commit -m "feat: Phase 1 complete — data layer, feature pipeline, CLI fetch working"
```

---

## Phase 1 Completion

At this point:
- `python main.py fetch --market crypto --symbol BTC/USDT` works and caches data
- All data provider tests pass
- All feature engineering tests pass
- Feature pipeline produces a clean, NaN-free, normalized feature matrix

**Deferred tests (Phase 5):** The spec's Section 8 test list for `test_data.py` includes "API retry logic (mocked)". These tests require the retry/backoff implementation from Phase 5 hardening and are intentionally not included here.

**Next:** Write the Phase 2 plan (`docs/superpowers/plans/2026-03-23-phase2-strategy-backtester.md`) covering:
- ML Trend Following strategy
- Walk-forward model trainer
- Backtesting engine (portfolio, risk, metrics)
- Terminal reporter + equity chart
- CLI `backtest` command

---

*Spec reference: `docs/superpowers/specs/2026-03-23-ml-trading-bot-design.md` Sections 2–3*
