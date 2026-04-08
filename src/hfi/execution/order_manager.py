"""Order manager — bracket orders, trailing stops, fill tracking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

POSITIONS_FILE = Path("data/positions.json")
TRADE_JOURNAL_FILE = Path("data/trade_journal.jsonl")


@dataclass
class Position:
    """Tracked open position."""

    id: str
    symbol: str
    side: str                # "long" | "short"
    entry_price: float
    amount: float            # in base currency
    position_size_usd: float
    leverage: int
    stop_loss: float
    take_profit: float
    trailing_stop: float | None = None
    engine: str = ""
    entry_time: str = ""
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "long":
            return self.amount * (current_price - self.entry_price)
        return self.amount * (self.entry_price - current_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == "long":
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price


class OrderManager:
    """Manages open positions and order lifecycle."""

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}
        self._trade_count: int = 0
        self._load_positions()

    @property
    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    @property
    def position_count(self) -> int:
        return len(self._positions)

    def get_positions_as_dicts(self) -> list[dict[str, Any]]:
        """Get positions as dicts for risk manager."""
        return [asdict(p) for p in self._positions.values()]

    def add_position(self, position: Position) -> None:
        """Track a new position."""
        self._positions[position.id] = position
        self._save_positions()
        logger.info(
            "Position opened: %s %s %s @ %.4f (size=$%.2f, lev=%dx)",
            position.id, position.side, position.symbol,
            position.entry_price, position.position_size_usd, position.leverage,
        )

    def close_position(self, position_id: str, exit_price: float, exit_reason: str) -> dict | None:
        """Close a position and record to trade journal."""
        pos = self._positions.pop(position_id, None)
        if pos is None:
            logger.warning("Position %s not found", position_id)
            return None

        pnl = pos.unrealized_pnl(exit_price)
        pnl_pct = pos.unrealized_pnl_pct(exit_price)
        fees = pos.position_size_usd * 0.0006 * 2  # entry + exit fees

        trade = {
            "trade_id": position_id,
            "engine": pos.engine,
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "entry_time": pos.entry_time,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "position_size_usd": pos.position_size_usd,
            "leverage": pos.leverage,
            "pnl": pnl - fees,
            "pnl_pct": pnl_pct,
            "fees": fees,
            "exit_reason": exit_reason,
        }

        # Append to trade journal
        self._append_trade_journal(trade)
        self._save_positions()

        logger.info(
            "Position closed: %s %s %s @ %.4f → %.4f PnL=$%.2f (%.2f%%) reason=%s",
            position_id, pos.side, pos.symbol,
            pos.entry_price, exit_price, pnl - fees, pnl_pct * 100, exit_reason,
        )

        return trade

    def update_trailing_stop(self, position_id: str, new_stop: float) -> None:
        """Update trailing stop for a position."""
        pos = self._positions.get(position_id)
        if pos is None:
            return

        if pos.side == "long" and new_stop > (pos.trailing_stop or 0):
            pos.trailing_stop = new_stop
            self._save_positions()
        elif pos.side == "short" and (pos.trailing_stop is None or new_stop < pos.trailing_stop):
            pos.trailing_stop = new_stop
            self._save_positions()

    def check_stops(self, current_prices: dict[str, float]) -> list[tuple[str, float, str]]:
        """Check if any positions have hit their stops.

        Returns list of (position_id, trigger_price, reason).
        """
        triggers = []
        for pid, pos in list(self._positions.items()):
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            effective_stop = pos.trailing_stop or pos.stop_loss

            if pos.side == "long":
                if price <= effective_stop:
                    triggers.append((pid, price, "stop_loss"))
                elif price >= pos.take_profit:
                    triggers.append((pid, price, "take_profit"))
            else:  # short
                if price >= effective_stop:
                    triggers.append((pid, price, "stop_loss"))
                elif price <= pos.take_profit:
                    triggers.append((pid, price, "take_profit"))

        return triggers

    def generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._trade_count += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"HFI-{ts}-{self._trade_count:04d}"

    def _save_positions(self) -> None:
        POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {pid: asdict(pos) for pid, pos in self._positions.items()}
        POSITIONS_FILE.write_text(json.dumps(data, indent=2))

    def _load_positions(self) -> None:
        if POSITIONS_FILE.exists():
            try:
                data = json.loads(POSITIONS_FILE.read_text())
                for pid, pdata in data.items():
                    self._positions[pid] = Position(**pdata)
                logger.info("Loaded %d open positions", len(self._positions))
            except Exception as e:
                logger.warning("Failed to load positions: %s", e)

    def _append_trade_journal(self, trade: dict) -> None:
        TRADE_JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_JOURNAL_FILE, "a") as f:
            f.write(json.dumps(trade) + "\n")
