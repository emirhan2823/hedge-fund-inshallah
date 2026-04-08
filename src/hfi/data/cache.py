"""SQLite-based OHLCV candle cache.

Avoids re-fetching historical data from exchange.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/ohlcv/candles.db")


class CandleCache:
    """SQLite cache for OHLCV candles."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_lookup
                ON candles (symbol, timeframe, timestamp)
            """)
            await db.commit()

    async def store(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Store OHLCV DataFrame into cache. Returns number of rows upserted."""
        if df.empty:
            return 0

        rows = []
        for ts, row in df.iterrows():
            rows.append((
                symbol, timeframe, int(ts.timestamp() * 1000),
                row["open"], row["high"], row["low"], row["close"], row["volume"],
            ))

        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(
                """INSERT OR REPLACE INTO candles
                   (symbol, timeframe, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await db.commit()

        logger.debug("Cached %d candles for %s %s", len(rows), symbol, timeframe)
        return len(rows)

    async def load(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load cached candles as DataFrame."""
        query = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol = ? AND timeframe = ?"
        params: list = [symbol, timeframe]

        if since_ms is not None:
            query += " AND timestamp >= ?"
            params.append(since_ms)
        if until_ms is not None:
            query += " AND timestamp <= ?"
            params.append(until_ms)

        query += " ORDER BY timestamp ASC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df

    async def get_latest_timestamp(self, symbol: str, timeframe: str) -> int | None:
        """Get the latest cached timestamp for a symbol/timeframe pair."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT MAX(timestamp) FROM candles WHERE symbol = ? AND timeframe = ?",
                [symbol, timeframe],
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row and row[0] else None

    async def count(self, symbol: str, timeframe: str) -> int:
        """Count cached candles."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?",
                [symbol, timeframe],
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
