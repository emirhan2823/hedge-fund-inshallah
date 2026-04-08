"""Constants and enums for HFI."""

from __future__ import annotations

from enum import StrEnum


class Regime(StrEnum):
    TRENDING_LOW_VOL = "TRENDING_LOW_VOL"      # Q1: trend follow paradise
    TRENDING_HIGH_VOL = "TRENDING_HIGH_VOL"    # Q2: trend follow with caution
    RANGING_LOW_VOL = "RANGING_LOW_VOL"        # Q3: mean reversion territory
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"      # Q4: danger zone, sit out


class Bias(StrEnum):
    LONG = "long"
    SHORT = "short"


class Engine(StrEnum):
    TREND_FOLLOWER = "TREND_FOLLOWER"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM_SCALPER = "MOMENTUM_SCALPER"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class PositionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


# Leverage milestones: account_balance -> max_leverage
DEFAULT_LEVERAGE_MILESTONES: dict[int, int] = {
    0: 3,
    500: 5,
    1000: 7,
    5000: 10,
}

# Default trading pairs (Bybit perpetual futures)
DEFAULT_PAIRS: list[str] = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "AVAX/USDT:USDT",
    "DOGE/USDT:USDT",
    "LINK/USDT:USDT",
    "SUI/USDT:USDT",
    "WIF/USDT:USDT",
]

# Timeframes
TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# Minimum account balance to enable momentum scalper
SCALPER_MIN_BALANCE = 500.0
