"""Portfolio-level risk manager.

Enforces: max positions, drawdown limits, daily loss limits, correlation checks.
Skill ref: risk-management.md
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from hfi.core.config import RiskConfig
from hfi.core.types import EngineSignal, PortfolioState

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio-level risk controls."""

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._daily_reset_date: str = ""
        self._weekly_reset_date: str = ""
        self._halted: bool = False
        self._halt_reason: str = ""
        self._days_hitting_daily_limit: int = 0

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def check_can_trade(
        self,
        signal: EngineSignal,
        portfolio: PortfolioState,
        open_positions: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Check if a new trade is allowed by risk rules.

        Returns (can_trade, reason).
        """
        # 0. Check halt
        if self._halted:
            return False, f"HALTED: {self._halt_reason}"

        # 1. Max drawdown check
        if portfolio.drawdown_pct >= self._config.max_drawdown:
            self._halt("Max drawdown exceeded: {:.1f}%".format(portfolio.drawdown_pct * 100))
            return False, self._halt_reason

        # 2. Daily loss limit
        self._update_daily_tracking(portfolio)
        if self._daily_pnl <= -self._config.daily_loss_limit * portfolio.balance_usd:
            return False, f"Daily loss limit hit: ${self._daily_pnl:.2f}"

        # 3. Weekly loss limit
        if self._weekly_pnl <= -self._config.weekly_loss_limit * portfolio.balance_usd:
            self._halt("Weekly loss limit exceeded")
            return False, self._halt_reason

        # 4. Max concurrent positions
        if portfolio.open_positions >= self._config.max_concurrent:
            return False, f"Max concurrent positions ({self._config.max_concurrent}) reached"

        # 5. Same-direction limit
        same_dir_count = sum(
            1 for p in open_positions
            if _position_bias(p) == signal.bias
        )
        if same_dir_count >= self._config.max_same_direction:
            return False, f"Max same-direction positions ({self._config.max_same_direction}) reached"

        # 6. Circuit breaker (consecutive losses)
        if portfolio.consecutive_losses >= self._config.consecutive_loss_halt:
            self._halt(f"Circuit breaker: {portfolio.consecutive_losses} consecutive losses")
            return False, self._halt_reason

        return True, "OK"

    def record_daily_pnl(self, pnl: float) -> None:
        """Record realized PnL for daily tracking."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            # Check if previous day hit limit
            if self._daily_pnl < 0:
                self._days_hitting_daily_limit += 1
            else:
                self._days_hitting_daily_limit = 0

            # 3 days hitting daily limit → weekly halt
            if self._days_hitting_daily_limit >= 3:
                self._halt("Daily limit hit 3 consecutive days")

            self._daily_pnl = 0.0
            self._daily_reset_date = today

        week = datetime.now(timezone.utc).strftime("%Y-W%W")
        if self._weekly_reset_date != week:
            self._weekly_pnl = 0.0
            self._weekly_reset_date = week

        self._daily_pnl += pnl
        self._weekly_pnl += pnl

    def reset_halt(self) -> None:
        """Manual halt reset (requires human intervention)."""
        logger.warning("HALT RESET by operator. Previous reason: %s", self._halt_reason)
        self._halted = False
        self._halt_reason = ""
        self._days_hitting_daily_limit = 0

    def get_drawdown_multiplier(self, portfolio: PortfolioState) -> float:
        """Get current drawdown-based sizing multiplier."""
        dd = portfolio.drawdown_pct
        if dd >= self._config.max_drawdown:
            return 0.0
        elif dd >= self._config.drawdown_levels[2] if len(self._config.drawdown_levels) > 2 else 0.15:
            return self._config.drawdown_multipliers[2] if len(self._config.drawdown_multipliers) > 2 else 0.0
        elif dd >= self._config.drawdown_levels[1] if len(self._config.drawdown_levels) > 1 else 0.10:
            return self._config.drawdown_multipliers[1] if len(self._config.drawdown_multipliers) > 1 else 0.25
        elif dd >= self._config.drawdown_levels[0] if len(self._config.drawdown_levels) > 0 else 0.05:
            return self._config.drawdown_multipliers[0] if len(self._config.drawdown_multipliers) > 0 else 0.50
        return 1.0

    def _halt(self, reason: str) -> None:
        """Halt all trading."""
        self._halted = True
        self._halt_reason = reason
        logger.critical("TRADING HALTED: %s", reason)

    def _update_daily_tracking(self, portfolio: PortfolioState) -> None:
        """Ensure daily counters are fresh."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_reset_date = today


def _position_bias(position: dict[str, Any]) -> str:
    """Extract bias from position dict."""
    side = position.get("side", "").lower()
    if side in ("long", "buy"):
        return "long"
    return "short"
