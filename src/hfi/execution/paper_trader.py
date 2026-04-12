"""Paper trader — simulated execution for testing.

Same interface as live trader but with virtual fills.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from hfi.core.types import PortfolioState
from hfi.execution.order_manager import OrderManager, Position
from hfi.pipeline.runner import PipelineDecision

logger = logging.getLogger(__name__)

SLIPPAGE_PCT = 0.0005  # 0.05% simulated slippage
FEE_PCT = 0.0006       # 0.06% Bybit taker fee


class PaperTrader:
    """Simulated trading execution."""

    def __init__(self, initial_balance: float = 100.0) -> None:
        self._balance = initial_balance
        self._max_equity = initial_balance
        self._order_manager = OrderManager()
        self._total_trades = 0
        self._winning_trades = 0
        self._consecutive_losses = 0
        self._daily_pnl = 0.0

    @property
    def order_manager(self) -> OrderManager:
        return self._order_manager

    def get_portfolio_state(self, current_prices: dict[str, float] | None = None) -> PortfolioState:
        """Get current portfolio snapshot."""
        unrealized = 0.0
        if current_prices:
            for pos in self._order_manager.open_positions:
                price = current_prices.get(pos.symbol, pos.entry_price)
                unrealized += pos.unrealized_pnl(price)

        equity = self._balance + unrealized
        self._max_equity = max(self._max_equity, equity)
        drawdown = (self._max_equity - equity) / self._max_equity if self._max_equity > 0 else 0.0

        return PortfolioState(
            balance_usd=self._balance,
            equity_usd=equity,
            unrealized_pnl=unrealized,
            open_positions=self._order_manager.position_count,
            daily_pnl=self._daily_pnl,
            daily_pnl_pct=self._daily_pnl / self._balance if self._balance > 0 else 0.0,
            max_equity=self._max_equity,
            drawdown_pct=drawdown,
            total_trades=self._total_trades,
            winning_trades=self._winning_trades,
            consecutive_losses=self._consecutive_losses,
        )

    def execute_decision(
        self, decision: PipelineDecision, current_price: float,
    ) -> str | None:
        """Execute a pipeline decision (simulated).

        Returns position_id if trade opened, None if skipped.
        """
        if decision.action == "skip" or decision.signal is None or decision.sizing is None:
            return None

        sizing = decision.sizing
        signal = decision.signal

        # Apply slippage
        if signal.bias == "long":
            fill_price = current_price * (1 + SLIPPAGE_PCT)
        else:
            fill_price = current_price * (1 - SLIPPAGE_PCT)

        # Calculate amount in base currency
        amount = sizing.position_size_usd / fill_price

        # Validate minimum position size
        if amount <= 0 or sizing.position_size_usd < 1.0:
            logger.warning("Position too small: $%.2f (min $1.00)", sizing.position_size_usd)
            return None

        # Note: fees are handled in order_manager.close_position() (entry + exit combined)
        # Do NOT deduct fees here to avoid double-charging

        # Create position
        pos_id = self._order_manager.generate_position_id()
        position = Position(
            id=pos_id,
            symbol=signal.symbol,
            side=signal.bias,
            entry_price=fill_price,
            amount=amount,
            position_size_usd=sizing.position_size_usd,
            leverage=sizing.leverage,
            stop_loss=sizing.stop_loss_price,
            take_profit=sizing.take_profit_price,
            engine=signal.engine,
            entry_time=datetime.now(timezone.utc).isoformat(),
        )
        self._order_manager.add_position(position)

        logger.info(
            "[PAPER] Opened %s %s %s @ %.4f size=$%.2f lev=%dx",
            pos_id, signal.bias, signal.symbol,
            fill_price, sizing.position_size_usd, sizing.leverage,
        )

        return pos_id

    def check_and_close_stops(self, current_prices: dict[str, float]) -> list[dict]:
        """Check stops and close triggered positions."""
        triggers = self._order_manager.check_stops(current_prices)
        closed_trades = []

        for pos_id, trigger_price, reason in triggers:
            # Apply exit slippage
            pos = self._order_manager._positions.get(pos_id)
            if pos is None:
                continue

            if pos.side == "long":
                exit_price = trigger_price * (1 - SLIPPAGE_PCT)
            else:
                exit_price = trigger_price * (1 + SLIPPAGE_PCT)

            trade = self._order_manager.close_position(pos_id, exit_price, reason)
            if trade:
                self._balance += trade["pnl"]
                self._total_trades += 1
                self._daily_pnl += trade["pnl"]

                if trade["pnl"] > 0:
                    self._winning_trades += 1
                    self._consecutive_losses = 0
                else:
                    self._consecutive_losses += 1

                closed_trades.append(trade)

                logger.info(
                    "[PAPER] Closed %s PnL=$%.2f (%.2f%%) reason=%s",
                    pos_id, trade["pnl"], trade["pnl_pct"] * 100, reason,
                )

        return closed_trades

    def inject_capital(self, amount: float) -> None:
        """Add monthly injection."""
        self._balance += amount
        logger.info("[PAPER] Capital injection: +$%.2f → balance=$%.2f", amount, self._balance)
