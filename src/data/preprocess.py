"""Clean, normalize, and validate OHLCV DataFrames."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
_MAX_FILL_GAP = 3


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

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Identify timestamps that are genuinely absent (gaps) vs present-with-NaN.
    # We do this before dropping NaN rows so we can distinguish the two cases.
    before_drop = set(df.index)

    before = len(df)
    df = df.dropna(subset=list(_REQUIRED_COLUMNS))
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with NaN OHLCV values", dropped)
    # Timestamps that were present but had NaN — these should NOT be gap-filled.
    nan_dropped_timestamps = before_drop - set(df.index)

    if len(df) >= 2:
        inferred_freq = _infer_freq(df)
        if inferred_freq:
            full_idx = pd.date_range(df.index[0], df.index[-1], freq=inferred_freq, tz="UTC")
            # Only consider timestamps that were never in the original data as gaps.
            truly_missing_idx = full_idx.difference(pd.DatetimeIndex(list(before_drop)))
            n_missing = len(truly_missing_idx)
            if n_missing > 0:
                df_reindexed = df.reindex(full_idx)
                # Mask out nan_dropped positions so we measure max gap among true gaps only.
                truly_missing_mask = df_reindexed["close"].isna() & ~df_reindexed.index.isin(
                    pd.DatetimeIndex(list(nan_dropped_timestamps))
                )
                max_gap = _max_consecutive(truly_missing_mask.astype(int))
                if max_gap <= _MAX_FILL_GAP:
                    # Only forward-fill truly missing positions, not NaN-dropped ones.
                    df_reindexed = df_reindexed.ffill()
                    # Re-drop positions that had NaN in original data.
                    df_reindexed = df_reindexed.drop(
                        index=pd.DatetimeIndex(list(nan_dropped_timestamps)).intersection(df_reindexed.index),
                        errors="ignore",
                    )
                    df = df_reindexed
                    logger.info("Forward-filled %d gap candles (max consecutive: %d)", n_missing, max_gap)
                else:
                    logger.warning("Gap of %d consecutive candles — too large to fill", max_gap)

    return df[list(_REQUIRED_COLUMNS)]


def _infer_freq(df: pd.DataFrame):
    diffs = df.index[:50].to_series().diff().dropna()
    if diffs.empty:
        return None
    mode_diff = diffs.mode()[0]
    seconds = int(mode_diff.total_seconds())
    freq_map = {60: "1min", 300: "5min", 900: "15min", 3600: "1h", 14400: "4h", 86400: "1D"}
    return freq_map.get(seconds)


def _max_consecutive(series: pd.Series) -> int:
    max_run = current = 0
    for val in series:
        if val:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
