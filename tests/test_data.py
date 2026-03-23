"""Tests for data layer: providers, cache, preprocessor."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock


def test_types_importable():
    """Shared contracts are importable and well-formed."""
    from src.core.types import OHLCV, Signal, Position, Trade, PortfolioState

    ohlcv = OHLCV(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=42000.0, high=43000.0, low=41000.0, close=42500.0, volume=1234.5,
    )
    assert ohlcv.close == 42500.0

    signal = Signal(direction="BUY", confidence=0.75, strategy_name="trend",
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert signal.direction == "BUY"
    assert signal.metadata == {}


def test_provider_interface():
    """DataProvider ABC cannot be instantiated directly."""
    from src.data.providers.base import DataProvider

    with pytest.raises(TypeError):
        DataProvider({})


def _make_ohlcv_df(n=10, start=None):
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

    cache.get_or_fetch(mock_provider, "BTC/USDT", "1h", since, until)
    cache.get_or_fetch(mock_provider, "BTC/USDT", "1h", since, until)

    assert mock_provider.fetch_ohlcv.call_count == 1


def test_preprocessor_rejects_missing_columns():
    """Preprocessor raises if required OHLCV columns are missing."""
    from src.data.preprocess import preprocess

    bad_df = pd.DataFrame({"open": [1.0], "close": [2.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess(bad_df)


def test_preprocessor_drops_ohlcv_nans():
    """Rows with NaN in any OHLCV column are dropped."""
    from src.data.preprocess import preprocess

    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open":   [1.0, None, 3.0, 4.0, 5.0],
        "high":   [2.0, 2.5, 3.5, 4.5, 5.5],
        "low":    [0.5, 0.8, 2.5, 3.5, 4.5],
        "close":  [1.5, 2.0, 3.2, 4.2, 5.2],
        "volume": [100, 200, 300, 400, 500],
    }, index=idx)

    result = preprocess(df)
    assert len(result) == 4
    assert result.isna().sum().sum() == 0


def test_preprocessor_fills_small_gaps():
    """Gaps of ≤3 consecutive missing candles are forward-filled."""
    from src.data.preprocess import preprocess

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
    assert len(result) == 10


def test_preprocessor_large_gap_not_filled():
    """Gaps of >3 consecutive candles are NOT filled."""
    from src.data.preprocess import preprocess

    idx = pd.date_range("2024-01-01", periods=15, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": 42000.0, "high": 43000.0, "low": 41000.0,
        "close": 42500.0, "volume": 500.0,
    }, index=idx)
    # Remove rows 3-8 (gap of 6 — exceeds fill threshold)
    df_with_gap = df.drop(df.index[3:9])

    result = preprocess(df_with_gap)
    assert len(result) < 15


def test_preprocessor_removes_duplicate_timestamps():
    """Duplicate timestamps are removed (last value kept)."""
    from src.data.preprocess import preprocess

    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 01:00", tz="UTC"),
    ])
    df = pd.DataFrame({
        "open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5],
        "low": [0.5, 1.5, 2.5], "close": [1.2, 2.2, 3.2], "volume": [10.0, 20.0, 30.0],
    }, index=idx)

    result = preprocess(df)
    assert len(result) == 2
    assert result.index.is_unique


def test_cache_partial_miss_fetches_only_missing_range(tmp_path):
    """Partial cache miss fetches only the uncovered tail, not the full range."""
    from src.data.cache import DataCache
    from src.data.providers.base import DataProvider

    db_path = str(tmp_path / "test.db")
    cache = DataCache(db_path)

    mock_provider = MagicMock(spec=DataProvider)
    mock_provider.name = "test_provider"

    # First fetch: hours 0-4 (5 rows)
    first_df = _make_ohlcv_df(5, start=datetime(2024, 1, 1, 0, tzinfo=timezone.utc))
    mock_provider.fetch_ohlcv.return_value = first_df
    since1 = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    until1 = datetime(2024, 1, 1, 4, tzinfo=timezone.utc)
    cache.get_or_fetch(mock_provider, "BTC/USDT", "1h", since1, until1)

    assert mock_provider.fetch_ohlcv.call_count == 1

    # Second fetch: hours 0-9 (10 rows) — first 5 cached, last 5 need fetching
    second_df = _make_ohlcv_df(6, start=datetime(2024, 1, 1, 4, tzinfo=timezone.utc))
    mock_provider.fetch_ohlcv.return_value = second_df
    since2 = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    until2 = datetime(2024, 1, 1, 9, tzinfo=timezone.utc)
    result = cache.get_or_fetch(mock_provider, "BTC/USDT", "1h", since2, until2)

    # Provider should have been called exactly twice total (once per miss)
    assert mock_provider.fetch_ohlcv.call_count == 2
    # Combined result should have 10 unique rows
    assert len(result) == 10
    assert result.index.is_unique


def test_crypto_provider_schema():
    """CryptoProvider returns DataFrame with correct columns when CCXT call is mocked."""
    from src.data.providers.crypto import CryptoProvider
    from unittest.mock import patch, MagicMock

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
        mock_ex.rateLimit = 0
        mock_ex.markets = {}
        mock_ex.load_markets.return_value = None
        mock_exchange_cls.return_value = mock_ex

        provider = CryptoProvider(config)
        df = provider.fetch_ohlcv("BTC/USDT", "1h", since, until)

    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert df.index.tz is not None
    assert len(df) == 5


def test_crypto_provider_available_timeframes():
    """CryptoProvider lists expected timeframes."""
    from src.data.providers.crypto import CryptoProvider
    from unittest.mock import patch, MagicMock

    config = {"markets": {"crypto": {"exchange": "binance", "fee_pct": 0.1}}}

    with patch("ccxt.binance") as mock_exchange_cls:
        mock_ex = MagicMock()
        mock_ex.markets = {}
        mock_ex.load_markets.return_value = None
        mock_exchange_cls.return_value = mock_ex

        provider = CryptoProvider(config)

    tfs = provider.available_timeframes()
    assert "1h" in tfs
    assert "4h" in tfs
    assert "1d" in tfs


def test_stocks_provider_schema():
    """StocksProvider returns correct DataFrame schema when yfinance is mocked."""
    from src.data.providers.stocks import StocksProvider
    from unittest.mock import patch

    config = {"markets": {"stocks": {"fee_pct": 0.0}}}
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 6, 1, tzinfo=timezone.utc)

    fake_df = _make_ohlcv_df(100, start=since)
    # yfinance returns Title Case columns
    fake_df.columns = [c.capitalize() for c in fake_df.columns]

    with patch("yfinance.download", return_value=fake_df):
        provider = StocksProvider(config)
        df = provider.fetch_ohlcv("AAPL", "1d", since, until)

    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert df.index.tz is not None


def test_stocks_provider_4h_resamples_from_1h():
    """StocksProvider requesting 4h resamples from 1h bars (4h not native in yfinance)."""
    from src.data.providers.stocks import StocksProvider
    from unittest.mock import patch

    config = {"markets": {"stocks": {"fee_pct": 0.0}}}
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 1, 8, tzinfo=timezone.utc)

    # 168 hourly bars (7 days)
    fake_df = _make_ohlcv_df(168, start=since)
    fake_df.columns = [c.capitalize() for c in fake_df.columns]

    with patch("yfinance.download", return_value=fake_df):
        provider = StocksProvider(config)
        df = provider.fetch_ohlcv("AAPL", "4h", since, until)

    # 168 hourly bars / 4 = 42 four-hour bars
    assert len(df) == 42
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}


def test_stocks_provider_rejects_unsupported_timeframe():
    """StocksProvider raises ValueError for unsupported timeframes."""
    from src.data.providers.stocks import StocksProvider

    config = {"markets": {"stocks": {"fee_pct": 0.0}}}
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 6, 1, tzinfo=timezone.utc)

    provider = StocksProvider(config)
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        provider.fetch_ohlcv("AAPL", "3d", since, until)


def test_stocks_provider_includes_1m_and_5m_timeframes():
    """StocksProvider supports 1m and 5m timeframes."""
    from src.data.providers.stocks import StocksProvider

    config = {"markets": {"stocks": {"symbols": ["AAPL"]}}}
    provider = StocksProvider(config)
    tfs = provider.available_timeframes()
    assert "1m" in tfs
    assert "5m" in tfs


def test_stocks_provider_clamps_since_for_limited_timeframes():
    """StocksProvider clamps since to the yfinance window limit and logs a warning."""
    from src.data.providers.stocks import StocksProvider
    from unittest.mock import patch

    config = {"markets": {"stocks": {"fee_pct": 0.0}}}
    # Request data from 2 years ago for a 1m timeframe (limit is 7 days)
    since = datetime(2022, 1, 1, tzinfo=timezone.utc)
    until = datetime(2022, 1, 2, tzinfo=timezone.utc)

    fake_df = _make_ohlcv_df(5, start=datetime.now(timezone.utc) - timedelta(days=1))
    fake_df.columns = [c.capitalize() for c in fake_df.columns]

    captured_calls = []

    def mock_download(*args, **kwargs):
        captured_calls.append(kwargs.get("start"))
        return fake_df

    with patch("yfinance.download", side_effect=mock_download):
        provider = StocksProvider(config)
        provider.fetch_ohlcv("AAPL", "1m", since, until)

    # The start date passed to yfinance should NOT be 2022-01-01
    assert captured_calls[0] != "2022-01-01", "since was not clamped"


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
    assert df.index.tz is not None
