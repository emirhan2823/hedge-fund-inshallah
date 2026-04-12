"""HFI Config — Pydantic config models + YAML loader.

Pattern from Argus: E:/argus/argus-core/src/argus/core/config.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ExchangeConfig(BaseModel):
    """Exchange connection settings."""

    id: str = "bybit"
    type: str = "future"
    pairs: list[str] = Field(default_factory=lambda: [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
        "AVAX/USDT:USDT", "DOGE/USDT:USDT", "LINK/USDT:USDT",
        "SUI/USDT:USDT", "WIF/USDT:USDT",
    ])
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False


class RiskConfig(BaseModel):
    """Risk management parameters."""

    base_risk_pct: float = 0.02       # 2% per trade
    min_risk_pct: float = 0.005       # 0.5% minimum
    max_risk_pct: float = 0.03        # 3% maximum
    max_position_size_pct: float = 0.30  # 30% of account (higher for small accounts)
    max_concurrent: int = 3
    daily_loss_limit: float = 0.05    # -5% daily
    weekly_loss_limit: float = 0.10   # -10% weekly
    max_drawdown: float = 0.15        # -15% halt
    max_same_direction: int = 2       # max positions same direction

    # Circuit breaker
    consecutive_loss_reduce: int = 3   # reduce size at 3 losses
    consecutive_loss_min: int = 5      # minimum size at 5 losses
    consecutive_loss_halt: int = 7     # halt at 7 losses

    # Drawdown response multipliers
    drawdown_levels: list[float] = Field(default=[0.05, 0.10, 0.15])
    drawdown_multipliers: list[float] = Field(default=[0.50, 0.25, 0.0])


class LeverageConfig(BaseModel):
    """Kademeli leverage milestones."""

    milestones: dict[int, int] = Field(default={0: 3, 500: 5, 1000: 7, 5000: 10})

    def get_leverage(self, balance: float) -> int:
        """Get leverage for current balance."""
        leverage = 1
        for threshold, lev in sorted(self.milestones.items()):
            if balance >= threshold:
                leverage = lev
        return leverage


class TrendFollowerConfig(BaseModel):
    """Trend Follower engine parameters."""

    enabled: bool = True
    timeframe: str = "1h"
    confirmation_timeframe: str = "4h"
    ema_fast: int = 8
    ema_mid: int = 21
    ema_slow: int = 55
    adx_threshold: float = 25.0
    atr_stop_mult: float = 2.0
    rr_ratio: float = 3.0     # risk:reward
    active_regimes: list[str] = Field(default=[
        "TRENDING_LOW_VOL", "TRENDING_HIGH_VOL",
    ])


class MeanReversionConfig(BaseModel):
    """Mean Reversion engine parameters."""

    enabled: bool = True
    timeframe: str = "15m"
    confirmation_timeframe: str = "1h"
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_exit: float = 50.0
    bb_period: int = 20
    bb_std: float = 2.0
    zscore_entry: float = -2.0
    hurst_threshold: float = 0.5
    active_regimes: list[str] = Field(default=["RANGING_LOW_VOL"])


class MomentumScalperConfig(BaseModel):
    """Momentum Scalper engine parameters."""

    enabled: bool = True
    timeframe: str = "5m"
    confirmation_timeframe: str = "15m"
    volume_mult: float = 2.0
    rsi_threshold: float = 60.0
    breakout_lookback: int = 20
    atr_stop_mult: float = 1.5
    rr_ratio: float = 1.5
    min_account_balance: float = 500.0
    active_regimes: list[str] = Field(default=[
        "TRENDING_LOW_VOL", "TRENDING_HIGH_VOL", "RANGING_LOW_VOL",
    ])


class SnowballConfig(BaseModel):
    """Snowball / auto-compound settings."""

    monthly_injection: float = 100.0
    auto_compound: bool = True
    milestones: list[float] = Field(default=[100, 500, 1000, 5000, 10000])


class TelegramConfig(BaseModel):
    """Telegram notification settings."""

    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


class BacktestConfig(BaseModel):
    """Backtesting parameters."""

    fee_rate: float = 0.0006          # 0.06% Bybit taker
    slippage_rate: float = 0.0005     # 0.05%
    train_days: int = 90
    test_days: int = 14
    min_trades_per_fold: int = 30
    initial_capital: float = 100.0


class HFIConfig(BaseModel):
    """Root configuration."""

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    leverage: LeverageConfig = Field(default_factory=LeverageConfig)
    trend_follower: TrendFollowerConfig = Field(default_factory=TrendFollowerConfig)
    mean_reversion: MeanReversionConfig = Field(default_factory=MeanReversionConfig)
    momentum_scalper: MomentumScalperConfig = Field(default_factory=MomentumScalperConfig)
    snowball: SnowballConfig = Field(default_factory=SnowballConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    engine_pairs: dict[str, list[str]] = Field(default_factory=lambda: {
        "TREND_FOLLOWER": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
        "MEAN_REVERSION": ["ETH/USDT:USDT", "SOL/USDT:USDT"],
        "MOMENTUM_SCALPER": [],
    })


def load_config(config_dir: str | Path = "config") -> HFIConfig:
    """Load config from YAML files and environment variables."""
    load_dotenv()
    config_dir = Path(config_dir)

    merged: dict[str, Any] = {}

    # Load main config
    main_file = config_dir / "default.yaml"
    if main_file.exists():
        with open(main_file) as f:
            data = yaml.safe_load(f) or {}
            merged.update(data)

    # Load risk config
    risk_file = config_dir / "risk.yaml"
    if risk_file.exists():
        with open(risk_file) as f:
            data = yaml.safe_load(f) or {}
            if "risk" not in merged:
                merged["risk"] = {}
            merged["risk"].update(data)

    # Load engine configs
    engines_dir = config_dir / "engines"
    if engines_dir.exists():
        for engine_file in engines_dir.glob("*.yaml"):
            with open(engine_file) as f:
                data = yaml.safe_load(f) or {}
                key = engine_file.stem  # e.g. "trend_follower"
                if key in merged:
                    merged[key].update(data)
                else:
                    merged[key] = data

    # Override with env vars
    if "exchange" not in merged:
        merged["exchange"] = {}
    merged["exchange"]["api_key"] = os.getenv("BYBIT_API_KEY", "")
    merged["exchange"]["api_secret"] = os.getenv("BYBIT_API_SECRET", "")

    if "telegram" not in merged:
        merged["telegram"] = {}
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if bot_token:
        merged["telegram"]["enabled"] = True
        merged["telegram"]["bot_token"] = bot_token
        merged["telegram"]["chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "")

    return HFIConfig(**merged)
