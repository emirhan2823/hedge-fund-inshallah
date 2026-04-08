"""Circuit breaker — consecutive loss tracking and halt logic.

Skill ref: risk-management.md, crypto-bot-architect.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = Path("data/circuit_breaker_state.json")


class CircuitBreaker:
    """Tracks consecutive losses and enforces progressive size reduction.

    Levels:
    - 3 consecutive losses → reduce size 50%
    - 5 consecutive losses → minimum size (25%)
    - 7 consecutive losses → HALT (0%)
    """

    def __init__(
        self,
        reduce_at: int = 3,
        minimum_at: int = 5,
        halt_at: int = 7,
        state_file: Path | None = None,
    ) -> None:
        self._reduce_at = reduce_at
        self._minimum_at = minimum_at
        self._halt_at = halt_at
        self._state_file = state_file or STATE_FILE
        self._consecutive_losses: int = 0
        self._total_trades: int = 0
        self._halted: bool = False
        self._halt_time: datetime | None = None
        self._load_state()

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def size_multiplier(self) -> float:
        """Get current size multiplier based on consecutive losses."""
        if self._halted or self._consecutive_losses >= self._halt_at:
            return 0.0
        elif self._consecutive_losses >= self._minimum_at:
            return 0.25
        elif self._consecutive_losses >= self._reduce_at:
            return 0.50
        return 1.0

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade result."""
        self._total_trades += 1

        if pnl < 0:
            self._consecutive_losses += 1
            logger.info(
                "Loss recorded. Consecutive losses: %d (mult: %.2f)",
                self._consecutive_losses, self.size_multiplier,
            )

            if self._consecutive_losses >= self._halt_at:
                self._halted = True
                self._halt_time = datetime.now(timezone.utc)
                logger.critical(
                    "CIRCUIT BREAKER HALT: %d consecutive losses",
                    self._consecutive_losses,
                )
        else:
            if self._consecutive_losses > 0:
                logger.info(
                    "Win after %d consecutive losses. Streak reset.",
                    self._consecutive_losses,
                )
            self._consecutive_losses = 0

        self._save_state()

    def reset(self) -> None:
        """Manual reset (requires human intervention)."""
        logger.warning(
            "Circuit breaker RESET by operator. Was at %d consecutive losses.",
            self._consecutive_losses,
        )
        self._consecutive_losses = 0
        self._halted = False
        self._halt_time = None
        self._save_state()

    def _save_state(self) -> None:
        """Persist state to disk."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "consecutive_losses": self._consecutive_losses,
            "total_trades": self._total_trades,
            "halted": self._halted,
            "halt_time": self._halt_time.isoformat() if self._halt_time else None,
        }
        self._state_file.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        """Load state from disk."""
        if self._state_file.exists():
            try:
                state = json.loads(self._state_file.read_text())
                self._consecutive_losses = state.get("consecutive_losses", 0)
                self._total_trades = state.get("total_trades", 0)
                self._halted = state.get("halted", False)
                ht = state.get("halt_time")
                self._halt_time = datetime.fromisoformat(ht) if ht else None
                logger.info(
                    "Loaded circuit breaker state: %d consecutive losses, halted=%s",
                    self._consecutive_losses, self._halted,
                )
            except Exception as e:
                logger.warning("Failed to load circuit breaker state: %s", e)
