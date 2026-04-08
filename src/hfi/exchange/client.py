"""Async CCXT exchange client wrapper.

Handles Bybit Futures connectivity: OHLCV, orders, positions, leverage.
Skill ref: dev-skills/fintech/crypto-bot-architect.md
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt

from hfi.core.config import ExchangeConfig

logger = logging.getLogger(__name__)


class ExchangeClient:
    """Async wrapper around CCXT for Bybit Futures."""

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config
        self._exchange: ccxt.Exchange | None = None

    async def connect(self) -> None:
        """Initialize exchange connection."""
        exchange_class = getattr(ccxt, self._config.id)
        self._exchange = exchange_class({
            "apiKey": self._config.api_key,
            "secret": self._config.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": self._config.type,
                "adjustForTimeDifference": True,
            },
        })
        if self._config.testnet:
            self._exchange.set_sandbox_mode(True)

        await self._exchange.load_markets()
        logger.info("Connected to %s (%s)", self._config.id, self._config.type)

    async def close(self) -> None:
        """Close exchange connection."""
        if self._exchange:
            await self._exchange.close()
            logger.info("Disconnected from %s", self._config.id)

    @property
    def exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            raise RuntimeError("Exchange not connected. Call connect() first.")
        return self._exchange

    # --- Market Data ---

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: int | None = None,
    ) -> list[list[float]]:
        """Fetch OHLCV candles. Returns [[timestamp, O, H, L, C, V], ...]."""
        return await self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=limit, since=since
        )

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker data."""
        return await self.exchange.fetch_ticker(symbol)

    async def fetch_tickers(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch tickers for multiple symbols."""
        return await self.exchange.fetch_tickers(symbols)

    # --- Account ---

    async def fetch_balance(self) -> dict[str, Any]:
        """Fetch account balance."""
        return await self.exchange.fetch_balance()

    async def get_usdt_balance(self) -> float:
        """Get USDT balance as float."""
        balance = await self.fetch_balance()
        return float(balance.get("USDT", {}).get("free", 0.0))

    async def get_equity(self) -> float:
        """Get total equity (balance + unrealized PnL)."""
        balance = await self.fetch_balance()
        return float(balance.get("USDT", {}).get("total", 0.0))

    # --- Positions ---

    async def fetch_positions(self, symbols: list[str] | None = None) -> list[dict]:
        """Fetch open positions."""
        positions = await self.exchange.fetch_positions(symbols)
        return [p for p in positions if float(p.get("contracts", 0)) > 0]

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        try:
            await self.exchange.set_leverage(leverage, symbol)
            logger.info("Set leverage %dx for %s", leverage, symbol)
        except ccxt.ExchangeError as e:
            # Some exchanges silently accept if already set
            logger.warning("Set leverage warning for %s: %s", symbol, e)

    # --- Orders ---

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a market order. Amount in base currency units."""
        params = params or {}
        order = await self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params=params,
        )
        logger.info(
            "Market %s %s %.6f %s @ market | order_id=%s",
            side, symbol, amount, symbol.split("/")[0], order["id"],
        )
        return order

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a limit order."""
        params = params or {}
        order = await self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price,
            params=params,
        )
        logger.info(
            "Limit %s %s %.6f @ %.4f | order_id=%s",
            side, symbol, amount, price, order["id"],
        )
        return order

    async def create_stop_loss(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
    ) -> dict[str, Any]:
        """Create a stop-loss order (market trigger)."""
        # Bybit uses stopLoss param in position or conditional order
        params = {
            "stopLoss": {"triggerPrice": stop_price, "type": "market"},
        }
        # For Bybit, use set_trading_stop or conditional order
        order = await self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params={
                "triggerPrice": stop_price,
                "triggerDirection": 2 if side == "sell" else 1,  # 1=rise, 2=fall
                "orderType": "Market",
                "reduceOnly": True,
            },
        )
        logger.info(
            "Stop loss %s %s %.6f trigger@%.4f | order_id=%s",
            side, symbol, amount, stop_price, order["id"],
        )
        return order

    async def create_take_profit(
        self,
        symbol: str,
        side: str,
        amount: float,
        tp_price: float,
    ) -> dict[str, Any]:
        """Create a take-profit order (market trigger)."""
        order = await self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params={
                "triggerPrice": tp_price,
                "triggerDirection": 1 if side == "sell" else 2,
                "orderType": "Market",
                "reduceOnly": True,
            },
        )
        logger.info(
            "Take profit %s %s %.6f trigger@%.4f | order_id=%s",
            side, symbol, amount, tp_price, order["id"],
        )
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an open order."""
        result = await self.exchange.cancel_order(order_id, symbol)
        logger.info("Cancelled order %s for %s", order_id, symbol)
        return result

    async def cancel_all_orders(self, symbol: str) -> list[dict]:
        """Cancel all open orders for a symbol."""
        result = await self.exchange.cancel_all_orders(symbol)
        logger.info("Cancelled all orders for %s", symbol)
        return result

    # --- Utilities ---

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Round amount to exchange precision."""
        return float(self.exchange.amount_to_precision(symbol, amount))

    def price_to_precision(self, symbol: str, price: float) -> float:
        """Round price to exchange precision."""
        return float(self.exchange.price_to_precision(symbol, price))

    def get_min_amount(self, symbol: str) -> float:
        """Get minimum order amount for a symbol."""
        market = self.exchange.market(symbol)
        return float(market.get("limits", {}).get("amount", {}).get("min", 0.0))

    def get_min_cost(self, symbol: str) -> float:
        """Get minimum order cost (in USDT) for a symbol."""
        market = self.exchange.market(symbol)
        return float(market.get("limits", {}).get("cost", {}).get("min", 1.0))
