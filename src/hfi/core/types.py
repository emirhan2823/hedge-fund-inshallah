"""HFI Core — Data models.

All models: pydantic v2, frozen=True, strict=True, extra='forbid'.
NaN rejection on all non-Optional float fields via model_validator.
Pattern from Argus: E:/argus/argus-core/src/argus/core/types.py
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HFIModel(BaseModel):
    """Base model for all HFI data contracts."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    @model_validator(mode="after")
    def _reject_nan(self) -> "HFIModel":
        for field_name, field_info in self.__class__.model_fields.items():
            val = getattr(self, field_name)
            if isinstance(val, float) and math.isnan(val):
                if field_info.default is not None:
                    raise ValueError(f"NaN not allowed for field '{field_name}'")
        return self


class FeatureVector(HFIModel):
    """Feature snapshot for a single symbol at a single timestamp.

    Simplified from Argus (15 features vs 30+).
    """

    timestamp: datetime
    symbol: str

    # --- Volatility (3) ---
    atr_14: float
    atr_14_pct: float          # ATR as % of close price
    bb_width: float            # Bollinger Band width

    # --- Trend (4) ---
    adx_14: float
    ema_8: float
    ema_21: float
    ema_55: float

    # --- Momentum (4) ---
    rsi_14: float
    macd_hist: float           # MACD histogram
    roc_10: float              # Rate of change 10
    bb_pct_b: float            # %B (0-1, position within bands)

    # --- Volume (2) ---
    volume_ratio: float        # current vol / SMA(20) vol
    volume_sma_20: float

    # --- Derived (2) ---
    zscore_close_20: float     # (close - SMA20) / std20
    hurst_exponent: float = 0.5

    # --- ATR percentile for regime ---
    atr_pctl: Optional[float] = None  # ATR percentile rank [0.0, 1.0]

    # --- Microstructure (6) --- live-only, None in backtests
    funding_rate: Optional[float] = None          # current funding rate (positive = longs pay)
    open_interest_change_pct: Optional[float] = None  # OI change % vs previous
    long_short_ratio: Optional[float] = None      # account L/S ratio
    orderbook_imbalance: Optional[float] = None   # [-1,1] bid-ask volume imbalance
    buy_sell_ratio: Optional[float] = None         # [0,1] recent trades buy fraction
    large_trade_pct: Optional[float] = None        # [0,1] whale trade fraction


class RegimeState(HFIModel):
    """Current market regime determination."""

    regime: str                # Regime enum value
    confidence: float = Field(ge=0.0, le=1.0)
    direction: Optional[int] = None  # +1 / -1 / None
    atr_percentile: float = Field(ge=0.0, le=1.0)
    adx_value: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EngineSignal(HFIModel):
    """Signal output from an engine."""

    engine: str                # Engine enum value
    symbol: str
    bias: str                  # "long" | "short"
    confidence: float = Field(ge=0.0, le=1.0)
    stop_distance: float = Field(gt=0.0)  # as fraction of entry price
    take_profit_distance: float = Field(gt=0.0)  # as fraction of entry price
    expected_return: float
    atr: float
    reason: str = ""           # human-readable signal reason


class SizingResult(HFIModel):
    """Position sizing output."""

    position_size_usd: float = Field(ge=0.0)
    risk_pct: float = Field(ge=0.0)
    leverage: int = Field(ge=1)
    stop_loss_price: float = Field(gt=0.0)
    take_profit_price: float = Field(gt=0.0)
    entry_price: float = Field(gt=0.0)


class TradeRecord(HFIModel):
    """Completed trade record for journaling."""

    trade_id: str
    engine: str
    symbol: str
    bias: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    position_size_usd: float
    leverage: int
    pnl_usd: float
    pnl_pct: float
    fees_usd: float
    exit_reason: str           # "stop_loss" | "take_profit" | "trailing_stop" | "signal_exit"


class PortfolioState(HFIModel):
    """Current portfolio snapshot."""

    balance_usd: float
    equity_usd: float          # balance + unrealized PnL
    unrealized_pnl: float
    open_positions: int
    daily_pnl: float
    daily_pnl_pct: float
    max_equity: float          # high-water mark
    drawdown_pct: float        # current drawdown from HWM
    total_trades: int
    winning_trades: int
    consecutive_losses: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
