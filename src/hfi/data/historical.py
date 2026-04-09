"""Historical data acquisition — pagination + CoinGecko daily.

Fetches extended OHLCV history beyond exchange per-call limits.
Uses CCXT `since` parameter for pagination, stores in SQLite cache.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

import aiohttp
import pandas as pd

from hfi.data.cache import CandleCache
from hfi.exchange.client import ExchangeClient

logger = logging.getLogger(__name__)

# Timeframe to milliseconds
TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
    "1d": 86_400_000,
}

# CCXT symbol to CoinGecko ID mapping
COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "AVAX": "avalanche-2", "DOGE": "dogecoin", "LINK": "chainlink",
    "SUI": "sui", "WIF": "dogwifcoin",
}


async def fetch_historical_ohlcv(
    client: ExchangeClient,
    symbol: str,
    timeframe: str = "1h",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    days: int = 365,
    cache: CandleCache | None = None,
) -> pd.DataFrame:
    """Fetch extended historical OHLCV by paginating CCXT calls.

    Args:
        client: Connected ExchangeClient
        symbol: Trading pair (e.g. "BTC/USDT:USDT")
        timeframe: Candle timeframe
        start_date: Start of range (or calculated from `days`)
        end_date: End of range (default: now)
        days: Number of days to fetch if start_date not given
        cache: Optional CandleCache for persistence

    Returns:
        DataFrame with OHLCV data covering the requested range.
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    since_ms = int(start_date.timestamp() * 1000)
    until_ms = int(end_date.timestamp() * 1000)
    bar_ms = TF_MS.get(timeframe, 3_600_000)

    # Check cache for existing data
    if cache:
        latest = await cache.get_latest_timestamp(symbol, timeframe)
        if latest and latest >= since_ms:
            since_ms = latest + bar_ms  # start after last cached bar

    total_expected = (until_ms - since_ms) // bar_ms
    total_fetched = 0
    all_rows = []

    logger.info(
        "Fetching %s %s from %s to %s (~%d bars expected)",
        symbol, timeframe,
        datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
        datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
        total_expected,
    )

    current_since = since_ms
    consecutive_empty = 0

    while current_since < until_ms:
        try:
            ohlcv = await client.fetch_ohlcv(
                symbol, timeframe, limit=1000, since=current_since
            )
        except Exception as e:
            logger.warning("Fetch error at %d: %s. Retrying in 2s...", current_since, e)
            await asyncio.sleep(2)
            try:
                ohlcv = await client.fetch_ohlcv(
                    symbol, timeframe, limit=1000, since=current_since
                )
            except Exception as e2:
                logger.error("Retry failed: %s. Stopping.", e2)
                break

        if not ohlcv:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                logger.info("No more data available. Stopping.")
                break
            current_since += bar_ms * 500
            await asyncio.sleep(0.5)
            continue

        consecutive_empty = 0
        all_rows.extend(ohlcv)
        total_fetched += len(ohlcv)

        # Move forward past last received timestamp
        last_ts = ohlcv[-1][0]
        current_since = last_ts + bar_ms

        # Store batch in cache
        if cache:
            batch_df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], unit="ms", utc=True)
            batch_df = batch_df.set_index("timestamp")
            await cache.store(symbol, timeframe, batch_df)

        # Progress
        progress = min(100, total_fetched / max(total_expected, 1) * 100)
        if total_fetched % 3000 == 0 or len(ohlcv) < 1000:
            logger.info("Progress: %d bars fetched (%.0f%%)", total_fetched, progress)

        # Rate limiting
        await asyncio.sleep(0.3)

        if len(ohlcv) < 1000:
            break  # No more data

    if not all_rows:
        logger.warning("No data fetched for %s %s", symbol, timeframe)
        return pd.DataFrame()

    # Build DataFrame, deduplicate
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp").sort_index()

    # Filter to requested range
    df = df[start_date:end_date]

    logger.info(
        "Fetched %d bars for %s %s (%s to %s)",
        len(df), symbol, timeframe,
        df.index[0].strftime("%Y-%m-%d") if len(df) > 0 else "N/A",
        df.index[-1].strftime("%Y-%m-%d") if len(df) > 0 else "N/A",
    )
    return df


async def fetch_daily_coingecko(
    symbol: str,
    days: int = 365,
) -> pd.DataFrame:
    """Fetch daily OHLC from CoinGecko (free API, no key needed).

    Used for market condition labeling (bull/bear/ranging/crash).
    CoinGecko free tier: 30 calls/min.
    """
    # Extract base currency from CCXT symbol
    base = symbol.split("/")[0] if "/" in symbol else symbol
    coin_id = COINGECKO_IDS.get(base)
    if not coin_id:
        logger.warning("Unknown CoinGecko ID for %s", base)
        return pd.DataFrame()

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": str(days)}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    logger.warning("CoinGecko rate limited. Wait and retry.")
                    await asyncio.sleep(60)
                    async with session.get(url, params=params) as retry_resp:
                        if retry_resp.status != 200:
                            logger.error("CoinGecko retry failed: %d", retry_resp.status)
                            return pd.DataFrame()
                        data = await retry_resp.json()
                elif resp.status != 200:
                    logger.error("CoinGecko error: %d", resp.status)
                    return pd.DataFrame()
                else:
                    data = await resp.json()
    except Exception as e:
        logger.error("CoinGecko fetch failed: %s", e)
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.drop_duplicates()

    # Add synthetic volume (CoinGecko OHLC doesn't include volume)
    df["volume"] = 0.0

    logger.info("CoinGecko: %d daily bars for %s (%s to %s)", len(df), symbol,
                df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))
    return df
