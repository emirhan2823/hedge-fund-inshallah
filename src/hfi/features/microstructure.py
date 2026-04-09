"""Market microstructure data — what other traders are doing.

Fetches real-time order flow, funding rate, open interest, and crowd positioning.
All via CCXT public endpoints — NO API KEY needed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from hfi.exchange.client import ExchangeClient

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureSnapshot:
    """Point-in-time microstructure data."""
    funding_rate: float | None = None
    open_interest_change_pct: float | None = None
    long_short_ratio: float | None = None
    orderbook_imbalance: float | None = None
    buy_sell_ratio: float | None = None
    large_trade_pct: float | None = None
    timestamp: datetime | None = None


class MicrostructureCollector:
    """Fetches microstructure data from exchange public APIs."""

    def __init__(self) -> None:
        self._prev_oi: dict[str, float] = {}

    async def fetch_all(
        self, client: ExchangeClient, symbol: str,
    ) -> MicrostructureSnapshot:
        """Fetch all microstructure data concurrently.

        All calls are public endpoints — no API key required.
        Each call is wrapped in try/except — failures return None, never crash pipeline.
        """
        results = await asyncio.gather(
            self._fetch_funding_rate(client, symbol),
            self._fetch_open_interest(client, symbol),
            self._fetch_long_short_ratio(client, symbol),
            self._fetch_orderbook_imbalance(client, symbol),
            self._fetch_trade_flow(client, symbol),
            return_exceptions=True,
        )

        # Unpack (any exception becomes None)
        funding = results[0] if not isinstance(results[0], Exception) else None
        oi_change = results[1] if not isinstance(results[1], Exception) else None
        ls_ratio = results[2] if not isinstance(results[2], Exception) else None
        ob_imbalance = results[3] if not isinstance(results[3], Exception) else None
        trade_flow = results[4] if not isinstance(results[4], Exception) else None

        buy_sell, large_pct = (None, None)
        if isinstance(trade_flow, tuple):
            buy_sell, large_pct = trade_flow

        return MicrostructureSnapshot(
            funding_rate=funding,
            open_interest_change_pct=oi_change,
            long_short_ratio=ls_ratio,
            orderbook_imbalance=ob_imbalance,
            buy_sell_ratio=buy_sell,
            large_trade_pct=large_pct,
            timestamp=datetime.now(timezone.utc),
        )

    async def _fetch_funding_rate(self, client: ExchangeClient, symbol: str) -> float | None:
        """Current funding rate. Positive = longs pay shorts (crowded long)."""
        try:
            ticker = await client.fetch_ticker(symbol)
            # Bybit includes fundingRate in ticker info
            info = ticker.get("info", {})
            fr = info.get("fundingRate")
            if fr is not None:
                return float(fr)
            # Fallback: try fetch_funding_rate
            rates = await client.exchange.fetch_funding_rate(symbol)
            return float(rates.get("fundingRate", 0))
        except Exception as e:
            logger.debug("Funding rate fetch failed for %s: %s", symbol, e)
            return None

    async def _fetch_open_interest(self, client: ExchangeClient, symbol: str) -> float | None:
        """Open interest change % vs previous check."""
        try:
            oi = await client.exchange.fetch_open_interest(symbol)
            current_oi = float(oi.get("openInterestAmount", oi.get("openInterest", 0)))

            prev = self._prev_oi.get(symbol)
            self._prev_oi[symbol] = current_oi

            if prev is None or prev == 0:
                return 0.0

            change_pct = (current_oi - prev) / prev
            return change_pct
        except Exception as e:
            logger.debug("OI fetch failed for %s: %s", symbol, e)
            return None

    async def _fetch_long_short_ratio(self, client: ExchangeClient, symbol: str) -> float | None:
        """Account-based long/short ratio."""
        try:
            data = await client.exchange.fetch_long_short_ratio_history(symbol, limit=1)
            if data and len(data) > 0:
                return float(data[0].get("longShortRatio", 1.0))
            return None
        except Exception as e:
            logger.debug("L/S ratio fetch failed for %s: %s", symbol, e)
            return None

    async def _fetch_orderbook_imbalance(
        self, client: ExchangeClient, symbol: str, depth: int = 10,
    ) -> float | None:
        """Order book imbalance from top N levels.

        Returns [-1, 1]: positive = more bid volume (bullish), negative = more ask volume.
        """
        try:
            ob = await client.exchange.fetch_order_book(symbol, limit=depth)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            bid_vol = sum(b[1] for b in bids[:depth])
            ask_vol = sum(a[1] for a in asks[:depth])
            total = bid_vol + ask_vol

            if total == 0:
                return 0.0

            imbalance = (bid_vol - ask_vol) / total
            return imbalance
        except Exception as e:
            logger.debug("Orderbook fetch failed for %s: %s", symbol, e)
            return None

    async def _fetch_trade_flow(
        self, client: ExchangeClient, symbol: str, limit: int = 100,
    ) -> tuple[float, float] | None:
        """Recent trade flow analysis.

        Returns:
            (buy_sell_ratio, large_trade_pct)
            buy_sell_ratio: fraction of volume from buy-side takers [0, 1]
            large_trade_pct: fraction of volume from trades > 2x average size [0, 1]
        """
        try:
            trades = await client.exchange.fetch_trades(symbol, limit=limit)
            if not trades or len(trades) < 10:
                return None

            buy_vol = 0.0
            sell_vol = 0.0
            sizes = []

            for t in trades:
                amount = float(t.get("amount", 0))
                side = t.get("side", "")
                sizes.append(amount)
                if side == "buy":
                    buy_vol += amount
                else:
                    sell_vol += amount

            total_vol = buy_vol + sell_vol
            buy_sell_ratio = buy_vol / total_vol if total_vol > 0 else 0.5

            # Large trade detection (> 2x average)
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            large_vol = sum(s for s in sizes if s > avg_size * 2)
            total_size = sum(sizes)
            large_trade_pct = large_vol / total_size if total_size > 0 else 0.0

            return (buy_sell_ratio, large_trade_pct)
        except Exception as e:
            logger.debug("Trade flow fetch failed for %s: %s", symbol, e)
            return None
