"""Snowball tracker — milestone tracking, auto-compound, monthly injections."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = Path("data/snowball_state.json")


class SnowballTracker:
    """Track account growth milestones and monthly injections."""

    def __init__(
        self,
        milestones: list[float] | None = None,
        monthly_injection: float = 100.0,
    ) -> None:
        self._milestones = milestones or [100, 500, 1000, 5000, 10000]
        self._monthly_injection = monthly_injection
        self._reached_milestones: list[float] = []
        self._injection_history: list[dict] = []
        self._start_balance: float = 0.0
        self._start_date: str = ""
        self._load_state()

    @property
    def reached_milestones(self) -> list[float]:
        return self._reached_milestones

    @property
    def next_milestone(self) -> float | None:
        for m in self._milestones:
            if m not in self._reached_milestones:
                return m
        return None

    def check_milestones(self, balance: float) -> float | None:
        """Check if a new milestone was reached. Returns milestone value or None."""
        for m in self._milestones:
            if balance >= m and m not in self._reached_milestones:
                self._reached_milestones.append(m)
                self._save_state()
                logger.info("MILESTONE REACHED: $%.0f (balance=$%.2f)", m, balance)
                return m
        return None

    def should_inject(self) -> bool:
        """Check if monthly injection is due.

        Simple rule: inject if it's been 30+ days since last injection.
        """
        if not self._injection_history:
            return True

        last = self._injection_history[-1]
        last_date = datetime.fromisoformat(last["date"])
        days_since = (datetime.now(timezone.utc) - last_date).days
        return days_since >= 30

    def record_injection(self, amount: float, balance_after: float) -> None:
        """Record a capital injection."""
        self._injection_history.append({
            "date": datetime.now(timezone.utc).isoformat(),
            "amount": amount,
            "balance_after": balance_after,
        })
        self._save_state()
        logger.info(
            "Capital injection: +$%.2f → balance=$%.2f (total injections: %d)",
            amount, balance_after, len(self._injection_history),
        )

    def get_total_injected(self) -> float:
        """Total capital injected (initial + monthly)."""
        return sum(i["amount"] for i in self._injection_history) + self._start_balance

    def get_roi(self, current_balance: float) -> float:
        """Return on total invested capital."""
        invested = self.get_total_injected()
        if invested <= 0:
            return 0.0
        return (current_balance - invested) / invested

    def get_growth_summary(self, current_balance: float) -> dict:
        """Get snowball growth summary."""
        invested = self.get_total_injected()
        profit = current_balance - invested
        roi = self.get_roi(current_balance)

        return {
            "current_balance": current_balance,
            "total_invested": invested,
            "profit": profit,
            "roi": roi,
            "milestones_reached": len(self._reached_milestones),
            "milestones_total": len(self._milestones),
            "next_milestone": self.next_milestone,
            "injections": len(self._injection_history),
            "months_active": len(self._injection_history),
        }

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "reached_milestones": self._reached_milestones,
            "injection_history": self._injection_history,
            "start_balance": self._start_balance,
            "start_date": self._start_date,
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                self._reached_milestones = state.get("reached_milestones", [])
                self._injection_history = state.get("injection_history", [])
                self._start_balance = state.get("start_balance", 0.0)
                self._start_date = state.get("start_date", "")
            except Exception as e:
                logger.warning("Failed to load snowball state: %s", e)
