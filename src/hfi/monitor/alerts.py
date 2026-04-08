"""Telegram alerts — trade notifications and daily summaries."""

from __future__ import annotations

import logging

from telegram import Bot

from hfi.core.config import TelegramConfig

logger = logging.getLogger(__name__)


class TelegramAlerts:
    """Send trading alerts via Telegram bot."""

    def __init__(self, config: TelegramConfig) -> None:
        self._config = config
        self._bot: Bot | None = None

        if config.enabled and config.bot_token and config.chat_id:
            self._bot = Bot(token=config.bot_token)
            logger.info("Telegram alerts initialized")

    async def send(self, message: str) -> None:
        """Send a message to Telegram."""
        if not self._bot or not self._config.chat_id:
            return

        try:
            await self._bot.send_message(
                chat_id=self._config.chat_id,
                text=f"[HFI] {message}",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)

    async def send_trade_open(
        self, symbol: str, side: str, price: float,
        size: float, leverage: int, sl: float, tp: float,
        engine: str, reason: str,
    ) -> None:
        """Send trade open notification."""
        arrow = "\u2191" if side == "long" else "\u2193"
        msg = (
            f"<b>{arrow} Trade Opened</b>\n"
            f"<b>{side.upper()}</b> {symbol}\n"
            f"Price: ${price:.4f}\n"
            f"Size: ${size:.2f} ({leverage}x)\n"
            f"SL: ${sl:.4f} | TP: ${tp:.4f}\n"
            f"Engine: {engine}\n"
            f"<i>{reason}</i>"
        )
        await self.send(msg)

    async def send_trade_close(
        self, symbol: str, side: str, entry: float,
        exit_price: float, pnl: float, pnl_pct: float, reason: str,
    ) -> None:
        """Send trade close notification."""
        emoji = "\u2705" if pnl > 0 else "\u274c"
        msg = (
            f"<b>{emoji} Trade Closed</b>\n"
            f"{side.upper()} {symbol}\n"
            f"Entry: ${entry:.4f} → Exit: ${exit_price:.4f}\n"
            f"PnL: <b>${pnl:+.2f}</b> ({pnl_pct:+.2%})\n"
            f"Reason: {reason}"
        )
        await self.send(msg)

    async def send_daily_summary(
        self, balance: float, daily_pnl: float,
        trades: int, wins: int, drawdown: float,
    ) -> None:
        """Send daily summary."""
        win_rate = wins / trades * 100 if trades > 0 else 0
        msg = (
            f"<b>Daily Summary</b>\n"
            f"Balance: ${balance:.2f}\n"
            f"Daily PnL: ${daily_pnl:+.2f}\n"
            f"Trades: {trades} (WR: {win_rate:.0f}%)\n"
            f"Drawdown: {drawdown:.1%}"
        )
        await self.send(msg)

    async def send_milestone(self, milestone: float, balance: float) -> None:
        """Send milestone reached notification."""
        msg = (
            f"<b>\U0001f389 MILESTONE REACHED!</b>\n"
            f"Balance: ${balance:.2f}\n"
            f"Target: ${milestone:.0f}\n"
            f"Keep going!"
        )
        await self.send(msg)

    async def send_circuit_breaker(self, reason: str) -> None:
        """Send circuit breaker alert."""
        msg = (
            f"<b>\u26a0\ufe0f CIRCUIT BREAKER</b>\n"
            f"{reason}\n"
            f"Trading HALTED. Manual reset required."
        )
        await self.send(msg)
