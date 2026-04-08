"""Data fetcher — OHLCV and market data acquisition.

Multi-timeframe OHLCV via CCXT + Fear & Greed Index.
Skill ref: ohlcv-processing.md, coingecko-api.md
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import aiohttp
import numpy as np
import pandas as pd

from hfi.exchange.client import ExchangeClient

logger = logging.getLogger(__name__)


async def fetch_ohlcv_df(
    client: ExchangeClient,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
    since: int | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data and return as cleaned DataFrame."""
    raw = await client.fetch_ohlcv(symbol, timeframe, limit, since)

    if not raw:
        logger.warning("No OHLCV data for %s %s", symbol, timeframe)
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Validate OHLC relationships
    df = _validate_ohlcv(df)

    # Drop duplicates
    df = df[~df.index.duplicated(keep="last")]

    # Sort by time
    df = df.sort_index()

    return df


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLCV data."""
    # High must be >= Open, Close, Low
    invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
    if invalid_high.any():
        logger.warning("Fixing %d invalid high values", invalid_high.sum())
        df.loc[invalid_high, "high"] = df.loc[invalid_high, ["open", "close"]].max(axis=1)

    # Low must be <= Open, Close, High
    invalid_low = df["low"] > df[["open", "close"]].min(axis=1)
    if invalid_low.any():
        logger.warning("Fixing %d invalid low values", invalid_low.sum())
        df.loc[invalid_low, "low"] = df.loc[invalid_low, ["open", "close"]].min(axis=1)

    # Drop zero/negative volume
    zero_vol = df["volume"] <= 0
    if zero_vol.any():
        logger.warning("Dropping %d zero-volume candles", zero_vol.sum())
        df = df[~zero_vol]

    # Drop zero price
    zero_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if zero_price.any():
        logger.warning("Dropping %d zero-price candles", zero_price.sum())
        df = df[~zero_price]

    return df


async def fetch_multi_timeframe(
    client: ExchangeClient,
    symbol: str,
    timeframes: list[str],
    limit: int = 500,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple timeframes."""
    result = {}
    for tf in timeframes:
        df = await fetch_ohlcv_df(client, symbol, tf, limit)
        if not df.empty:
            result[tf] = df
        else:
            logger.warning("Empty data for %s %s", symbol, tf)
    return result


async def fetch_fear_greed_index() -> dict:
    """Fetch current Fear & Greed Index from alternative.me (free)."""
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    fng = data["data"][0]
                    return {
                        "value": int(fng["value"]),
                        "classification": fng["value_classification"],
                        "timestamp": datetime.fromtimestamp(
                            int(fng["timestamp"]), tz=timezone.utc
                        ),
                    }
    except Exception as e:
        logger.warning("Failed to fetch Fear & Greed Index: %s", e)

    return {"value": 50, "classification": "Neutral", "timestamp": datetime.now(timezone.utc)}


def detect_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Detect gaps in OHLCV data."""
    tf_seconds = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "1d": 86400,
    }
    expected_delta = pd.Timedelta(seconds=tf_seconds.get(timeframe, 3600))

    gaps = df.index.to_series().diff()
    gap_mask = gaps > expected_delta * 1.5  # Allow 50% tolerance
    if gap_mask.any():
        gap_locations = df.index[gap_mask]
        logger.warning(
            "Found %d gaps in data at: %s",
            len(gap_locations),
            gap_locations.tolist()[:5],
        )

    return df
