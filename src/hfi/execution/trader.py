"""Live/Paper trading loop — main asyncio event loop.

Runs the full pipeline on each candle close.
Integrates: dashboard, snowball tracker, microstructure, trailing stops.
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
from hfi.features.builder import build_features, build_features_df, merge_microstructure
from hfi.features.microstructure import MicrostructureCollector
from hfi.monitor.dashboard import render_dashboard
from hfi.monitor.snowball import SnowballTracker
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

        # Snowball tracker
        self._snowball = SnowballTracker(
            milestones=config.snowball.milestones,
            monthly_injection=config.snowball.monthly_injection,
        )

        # Microstructure collector (live order flow data)
        self._micro = MicrostructureCollector()

        # Engine-pair mapping from config
        self._engine_pairs: dict[str, list[str]] = getattr(
            config, "engine_pairs", {}
        )

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
        leverage = self._config.leverage.get_leverage(self._paper_trader._balance)
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

        # Register signal handlers (Unix only)
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
                    try:
                        await self._alerts.send(f"Error: {e}")
                    except Exception:
                        pass

            # Wait for next candle
            now = datetime.now(timezone.utc)
            seconds_to_next = interval - (int(now.timestamp()) % interval)
            logger.debug("Next tick in %d seconds", seconds_to_next)

            try:
                await asyncio.sleep(seconds_to_next + 5)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        """Single trading tick — process all pairs."""
        logger.info("=== TICK %s ===", datetime.now(timezone.utc).isoformat())

        # 1. Get current prices
        current_prices: dict[str, float] = {}
        for pair in self._config.exchange.pairs:
            try:
                ticker = await self._exchange.fetch_ticker(pair)
                current_prices[pair] = float(ticker["last"])
            except Exception as e:
                logger.warning("Failed to fetch ticker for %s: %s", pair, e)

        # 2. Check stops on existing positions
        closed = self._paper_trader.check_and_close_stops(current_prices)
        for trade in closed:
            self._circuit_breaker.record_trade(trade["pnl"])
            self._risk_manager.record_daily_pnl(trade["pnl"])

            # Check milestones after trade close
            milestone = self._snowball.check_milestones(self._paper_trader._balance)
            if milestone and self._alerts:
                try:
                    await self._alerts.send_milestone(milestone, self._paper_trader._balance)
                except Exception:
                    pass

            if self._alerts:
                try:
                    emoji = "+" if trade["pnl"] > 0 else ""
                    await self._alerts.send(
                        f"Trade closed: {trade['symbol']} {trade['side']} "
                        f"PnL=${emoji}{trade['pnl']:.2f} ({trade['exit_reason']})"
                    )
                except Exception:
                    pass

        # 3. Monthly injection check
        if self._snowball.should_inject():
            amount = self._config.snowball.monthly_injection
            self._paper_trader.inject_capital(amount)
            self._snowball.record_injection(amount, self._paper_trader._balance)
            logger.info("Monthly injection: +$%.2f -> balance=$%.2f", amount, self._paper_trader._balance)

        # 4. Get portfolio state
        portfolio = self._paper_trader.get_portfolio_state(current_prices)

        # 5. Process each pair
        for pair in self._config.exchange.pairs:
            if pair not in current_prices:
                continue
            try:
                await self._process_pair(pair, current_prices[pair], portfolio)
            except Exception as e:
                logger.error("Error processing %s: %s", pair, e)

        # 6. Render dashboard
        try:
            render_dashboard(
                portfolio=portfolio,
                positions=self._paper_trader.order_manager.open_positions,
                current_prices=current_prices,
                milestones=self._config.snowball.milestones,
            )
        except Exception as e:
            logger.debug("Dashboard render failed: %s", e)

    async def _process_pair(
        self, symbol: str, current_price: float, portfolio: Any,
    ) -> None:
        """Process a single trading pair through the pipeline."""
        # Check existing positions for this symbol
        existing = [
            p for p in self._paper_trader.order_manager.open_positions
            if p.symbol == symbol
        ]

        # Fetch OHLCV (always — needed for trailing stops AND new signals)
        tf = self._config.trend_follower.timeframe
        df = await fetch_ohlcv_df(self._exchange, symbol, tf, limit=200)
        if df.empty or len(df) < 60:
            return

        # Build features
        features_list = build_features(df, symbol)
        if not features_list:
            return
        latest = features_list[-1]

        # Fetch microstructure (optional — never crash pipeline)
        try:
            micro = await self._micro.fetch_all(self._exchange, symbol)
            latest = merge_microstructure(latest, micro)
        except Exception as e:
            logger.debug("Microstructure fetch failed for %s: %s", symbol, e)

        # Update trailing stops for existing positions (using real ATR)
        if existing:
            atr = latest.atr_14
            atr_pct = latest.atr_14_pct
            for pos in existing:
                stop_dist = atr_pct * self._config.trend_follower.atr_stop_mult
                if pos.side == "long":
                    new_stop = current_price * (1 - stop_dist)
                else:
                    new_stop = current_price * (1 + stop_dist)
                self._paper_trader.order_manager.update_trailing_stop(pos.id, new_stop)
            # Don't open new position if we already have one on this symbol
            return

        # Run pipeline for new signal
        decision = self._pipeline.run(
            features=latest,
            close_price=current_price,
            portfolio=portfolio,
            open_positions=self._paper_trader.order_manager.get_positions_as_dicts(),
        )

        # Execute if action is not skip
        if decision.action != "skip":
            pos_id = self._paper_trader.execute_decision(decision, current_price)
            if pos_id and self._alerts:
                sig = decision.signal
                sz = decision.sizing
                try:
                    await self._alerts.send_trade_open(
                        symbol=sig.symbol, side=sig.bias,
                        price=current_price, size=sz.position_size_usd,
                        leverage=sz.leverage, sl=sz.stop_loss_price,
                        tp=sz.take_profit_price, engine=sig.engine,
                        reason=sig.reason,
                    )
                except Exception:
                    pass
        else:
            logger.debug(
                "%s: SKIP (%s)", symbol,
                decision.skip_reason[:80] if decision.skip_reason else "no signal",
            )

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutdown requested")
        self._running = False

    async def _cleanup(self) -> None:
        """Cleanup on exit."""
        logger.info("Cleaning up...")
        if self._alerts:
            try:
                portfolio = self._paper_trader.get_portfolio_state()
                growth = self._snowball.get_growth_summary(portfolio.balance_usd)
                await self._alerts.send(
                    f"HFI Bot stopped.\n"
                    f"Balance: ${portfolio.balance_usd:.2f}\n"
                    f"Open positions: {portfolio.open_positions}\n"
                    f"ROI: {growth['roi']:.1%}"
                )
            except Exception:
                pass
        await self._exchange.close()
        logger.info("Cleanup complete")


async def run_bot(config_dir: str = "config", live: bool = False) -> None:
    """Entry point for the trading bot."""
    config = load_config(config_dir)
    bot = TradingBot(config, live=live)
    await bot.start()
