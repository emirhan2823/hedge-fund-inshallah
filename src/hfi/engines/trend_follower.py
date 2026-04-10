"""Trend Follower Engine — EMA crossover + MACD + ADX confirmation.

Timeframe: 1h primary, 4h confirmation.
Active in: TRENDING_LOW_VOL, TRENDING_HIGH_VOL regimes.
Skill ref: strategy-framework.md, exit-strategies.md
"""

from __future__ import annotations

import logging
from typing import Optional

from hfi.core.constants import Bias, Engine
from hfi.core.config import TrendFollowerConfig
from hfi.core.types import EngineSignal, FeatureVector, RegimeState

logger = logging.getLogger(__name__)


class TrendFollower:
    """Trend following engine using EMA crossover system.

    Entry conditions (ALL must be true):
    - EMA(8) > EMA(21) > EMA(55) for long (reversed for short)
    - ADX > threshold (strong trend)
    - MACD histogram > 0 for long (< 0 for short)

    Exit:
    - ATR trailing stop (2x ATR)
    - Take profit at 3:1 risk/reward
    """

    def __init__(self, config: TrendFollowerConfig | None = None) -> None:
        self._config = config or TrendFollowerConfig()

    @property
    def name(self) -> str:
        return Engine.TREND_FOLLOWER

    @property
    def active_regimes(self) -> list[str]:
        return self._config.active_regimes

    def generate_signal(
        self,
        *,
        regime: RegimeState,
        features: FeatureVector,
        close_price: float,
    ) -> Optional[EngineSignal]:
        if not self._config.enabled:
            return None

        if regime.regime not in self.active_regimes:
            return None

        c = self._config

        # Volume gate: skip dead volume (no participation = fake trend)
        if features.volume_ratio < 0.8:
            return None

        # Check for LONG setup
        long_ema = features.ema_8 > features.ema_21 > features.ema_55
        long_adx = features.adx_14 > c.adx_threshold
        long_macd = features.macd_hist > 0
        long_rsi_ok = features.rsi_14 < 60  # don't long into overbought

        # Check for SHORT setup
        short_ema = features.ema_8 < features.ema_21 < features.ema_55
        short_adx = features.adx_14 > c.adx_threshold
        short_macd = features.macd_hist < 0
        short_rsi_ok = features.rsi_14 < 35  # only short when momentum is strong

        bias: str | None = None
        if long_ema and long_adx and long_macd and long_rsi_ok:
            bias = Bias.LONG
        elif short_ema and short_adx and short_macd and short_rsi_ok:
            bias = Bias.SHORT
        else:
            return None

        # Calculate stop and take profit distances
        atr = features.atr_14
        stop_distance = (atr * c.atr_stop_mult) / close_price
        tp_distance = stop_distance * c.rr_ratio

        # Confidence from ADX strength and regime confidence
        adx_conf = min(1.0, (features.adx_14 - c.adx_threshold) / 30.0)
        confidence = (adx_conf * 0.6 + regime.confidence * 0.4)

        # Expected return (simplified)
        expected_return = tp_distance * confidence - stop_distance * (1 - confidence)

        reason = (
            f"EMA alignment ({features.ema_8:.1f}>{features.ema_21:.1f}>{features.ema_55:.1f})"
            if bias == Bias.LONG else
            f"EMA alignment ({features.ema_8:.1f}<{features.ema_21:.1f}<{features.ema_55:.1f})"
        )
        reason += f" | ADX={features.adx_14:.1f} | MACD_hist={features.macd_hist:.4f}"

        # Microstructure: OI confirmation (if available)
        if features.open_interest_change_pct is not None:
            if features.open_interest_change_pct > 0.05:
                confidence += 0.05  # OI rising = trend participation growing
            elif features.open_interest_change_pct < -0.05:
                confidence -= 0.10  # OI declining = trend losing steam
                if confidence < 0.3:
                    return None  # Skip weak trend with declining OI
            confidence = min(1.0, max(0.0, confidence))

        return EngineSignal(
            engine=self.name,
            symbol=features.symbol,
            bias=bias,
            confidence=confidence,
            stop_distance=stop_distance,
            take_profit_distance=tp_distance,
            expected_return=expected_return,
            atr=atr,
            reason=reason,
        )
