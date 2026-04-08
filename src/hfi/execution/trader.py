"""Live/Paper trading loop — main asyncio event loop.

Runs the full pipeline on each candle close.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Any

from hfi.core.config import HFIConfig, load_config
from hfi.data.fetcher import fetch_ohlcv_df
from hfi.exchange.client import ExchangeClient
from hfi.execution.paper_trader import PaperTrader
from hfi.features.builder import build_features
from hfi.pipeline.runner import Pipeline
from hfi.risk.circuit_breaker import CircuitBreaker
from hfi.risk.manager import RiskManager

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(
        self,
        config: HFIConfig,
        live: bool = False,
    ) -> None:
        self._config = config
        self._live = live
        self._running = False

        # Components
        self._exchange = ExchangeClient(config.exchange)
        self._risk_manager = RiskManager(config.risk)
        self._circuit_breaker = CircuitBreaker(
            reduce_at=config.risk.consecutive_loss_reduce,
            minimum_at=config.risk.consecutive_loss_min,
            halt_at=config.risk.consecutive_loss_halt,
        )
        self._pipeline = Pipeline(config, self._risk_manager)
        self._paper_trader = PaperTrader(config.backtest.initial_capital)

        # Alerts (lazy import)
        self._alerts: Any = None

    async def start(self) -> None:
        """Start the trading bot."""
        mode = "LIVE" if self._live else "PAPER"
        logger.info("Starting HFI Trading Bot in %s mode", mode)

        if self._live and not self._config.exchange.api_key:
            logger.error("API key required for live trading. Set BYBIT_API_KEY in .env")
            return

        await self._exchange.connect()

        # Set leverage for all pairs
        leverage = self._config.leverage.get_leverage(
            self._paper_trader._balance if not self._live else 100
        )
        for pair in self._config.exchange.pairs:
            try:
                await self._exchange.set_leverage(pair, leverage)
            except Exception as e:
                logger.warning("Could not set leverage for %s: %s", pair, e)

        # Setup alerts
        if self._config.telegram.enabled:
            try:
                from hfi.monitor.alerts import TelegramAlerts
                self._alerts = TelegramAlerts(self._config.telegram)
                await self._alerts.send(f"HFI Bot started in {mode} mode")
            except Exception as e:
                logger.warning("Telegram alerts failed to initialize: %s", e)

        self._running = True

        # Register signal handlers
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGINT, self._shutdown)
            loop.add_signal_handler(signal.SIGTERM, self._shutdown)

        try:
            await self._trading_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            await self._cleanup()

    async def _trading_loop(self) -> None:
        """Main trading loop — runs pipeline on each interval."""
        # Primary timeframe interval in seconds
        tf_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "4h": 14400,
        }
        primary_tf = self._config.trend_follower.timeframe
        interval = tf_seconds.get(primary_tf, 3600)

        logger.info("Trading loop started. Interval: %ds (%s)", interval, primary_tf)

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("Error in trading tick: %s", e, exc_info=True)
                if self._alerts:
                    await self._alerts.send(f"Error: {e}")

            # Wait for next candle
            now = datetime.now(timezone.utc)
            seconds_to_next = interval - (int(now.timestamp()) % interval)
            logger.debug("Next tick in %d seconds", seconds_to_next)

            try:
                await asyncio.sleep(seconds_to_next + 5)  # +5s buffer for candle to close
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        """Single trading tick — process all pairs."""
        logger.info("=== TICK %s ===", datetime.now(timezone.utc).isoformat())

        # Get current prices for stop checking
        current_prices: dict[str, float] = {}
        for pair in self._config.exchange.pairs:
            try:
                ticker = await self._exchange.fetch_ticker(pair)
                current_prices[pair] = float(ticker["last"])
            except Exception as e:
                logger.warning("Failed to fetch ticker for %s: %s", pair, e)

        # Check stops
        closed = self._paper_trader.check_and_close_stops(current_prices)
        for trade in closed:
            self._circuit_breaker.record_trade(trade["pnl"])
            self._risk_manager.record_daily_pnl(trade["pnl"])
            if self._alerts:
                emoji = "+" if trade["pnl"] > 0 else ""
                await self._alerts.send(
                    f"Trade closed: {trade['symbol']} {trade['side']} "
                    f"PnL=${emoji}{trade['pnl']:.2f} ({trade['exit_reason']})"
                )

        # Process each pair
        portfolio = self._paper_trader.get_portfolio_state(current_prices)

        for pair in self._config.exchange.pairs:
            if pair not in current_prices:
                continue

            try:
                await self._process_pair(pair, current_prices[pair], portfolio)
            except Exception as e:
                logger.error("Error processing %s: %s", pair, e)

    async def _process_pair(
        self, symbol: str, current_price: float, portfolio: Any,
    ) -> None:
        """Process a single trading pair through the pipeline."""
        # Skip if already have position in this symbol
        existing = [
            p for p in self._paper_trader.order_manager.open_positions
            if p.symbol == symbol
        ]
        if existing:
            # Update trailing stops
            for pos in existing:
                atr_pct = 0.02  # simplified
                if pos.side == "long":
                    new_stop = current_price * (1 - atr_pct * 2)
                    self._paper_trader.order_manager.update_trailing_stop(pos.id, new_stop)
                else:
                    new_stop = current_price * (1 + atr_pct * 2)
                    self._paper_trader.order_manager.update_trailing_stop(pos.id, new_stop)
            return

        # Fetch OHLCV
        tf = self._config.trend_follower.timeframe
        df = await fetch_ohlcv_df(self._exchange, symbol, tf, limit=200)
        if df.empty or len(df) < 60:
            return

        # Build features
        features_list = build_features(df, symbol)
        if not features_list:
            return

        latest = features_list[-1]

        # Run pipeline
        decision = self._pipeline.run(
            features=latest,
            close_price=current_price,
            portfolio=portfolio,
            open_positions=self._paper_trader.order_manager.get_positions_as_dicts(),
        )

        # Execute
        if decision.action != "skip":
            pos_id = self._paper_trader.execute_decision(decision, current_price)
            if pos_id and self._alerts:
                sig = decision.signal
                sz = decision.sizing
                await self._alerts.send(
                    f"Trade opened: {sig.bias.upper()} {sig.symbol}\n"
                    f"Price: ${current_price:.4f}\n"
                    f"Size: ${sz.position_size_usd:.2f} ({sz.leverage}x)\n"
                    f"SL: ${sz.stop_loss_price:.4f} | TP: ${sz.take_profit_price:.4f}\n"
                    f"Engine: {sig.engine}\n"
                    f"Reason: {sig.reason}"
                )

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutdown requested")
        self._running = False

    async def _cleanup(self) -> None:
        """Cleanup on exit."""
        logger.info("Cleaning up...")
        if self._alerts:
            portfolio = self._paper_trader.get_portfolio_state()
            await self._alerts.send(
                f"HFI Bot stopped.\n"
                f"Balance: ${portfolio.balance_usd:.2f}\n"
                f"Open positions: {portfolio.open_positions}"
            )
        await self._exchange.close()
        logger.info("Cleanup complete")


async def run_bot(config_dir: str = "config", live: bool = False) -> None:
    """Entry point for the trading bot."""
    config = load_config(config_dir)
    bot = TradingBot(config, live=live)
    await bot.start()
