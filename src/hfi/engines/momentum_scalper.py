"""Momentum Scalper Engine — Volume spike + RSI momentum + breakout.

Timeframe: 5m primary, 15m confirmation.
Active in: Q1, Q2, Q3 (not Q4).
Requires account balance >= $500 (fee drag too high for small accounts).
Skill ref: strategy-framework.md, exit-strategies.md
"""

from __future__ import annotations

import logging
from typing import Optional

from hfi.core.constants import Bias, Engine
from hfi.core.config import MomentumScalperConfig
from hfi.core.types import EngineSignal, FeatureVector, RegimeState

logger = logging.getLogger(__name__)


class MomentumScalper:
    """Aggressive short-term momentum engine.

    Entry conditions (ALL for LONG):
    - Volume > 2x SMA(20) (volume spike)
    - RSI > 60 (momentum confirmation)
    - Price breaks above recent high (breakout)
    - ROC > 0 (positive momentum)

    Entry conditions (ALL for SHORT):
    - Volume > 2x SMA(20)
    - RSI < 40
    - Price breaks below recent low
    - ROC < 0

    Exit:
    - Quick take profit (1.5:1 R:R)
    - Tight stop (1.5x ATR)
    """

    def __init__(self, config: MomentumScalperConfig | None = None) -> None:
        self._config = config or MomentumScalperConfig()

    @property
    def name(self) -> str:
        return Engine.MOMENTUM_SCALPER

    @property
    def active_regimes(self) -> list[str]:
        return self._config.active_regimes

    def generate_signal(
        self,
        *,
        regime: RegimeState,
        features: FeatureVector,
        close_price: float,
        account_balance: float = 0.0,
    ) -> Optional[EngineSignal]:
        if not self._config.enabled:
            return None

        if regime.regime not in self.active_regimes:
            return None

        # Account balance check
        if account_balance < self._config.min_account_balance:
            return None

        c = self._config

        # Volume spike check
        has_volume_spike = features.volume_ratio > c.volume_mult

        if not has_volume_spike:
            return None

        # LONG setup
        long_rsi = features.rsi_14 > c.rsi_threshold
        long_momentum = features.roc_10 > 0
        long_bb = features.bb_pct_b > 0.8  # near or above upper band (breakout)

        # SHORT setup
        short_rsi = features.rsi_14 < (100 - c.rsi_threshold)  # RSI < 40
        short_momentum = features.roc_10 < 0
        short_bb = features.bb_pct_b < 0.2  # near or below lower band (breakdown)

        bias: str | None = None
        if has_volume_spike and long_rsi and long_momentum and long_bb:
            bias = Bias.LONG
        elif has_volume_spike and short_rsi and short_momentum and short_bb:
            bias = Bias.SHORT
        else:
            return None

        # Tight stop and quick TP
        atr = features.atr_14
        stop_distance = (atr * c.atr_stop_mult) / close_price
        tp_distance = stop_distance * c.rr_ratio

        # Confidence from volume spike magnitude and regime
        vol_conf = min(1.0, (features.volume_ratio - c.volume_mult) / 3.0)
        confidence = (vol_conf * 0.5 + regime.confidence * 0.3 + 0.2)

        expected_return = tp_distance * confidence - stop_distance * (1 - confidence)

        reason = (
            f"Volume spike={features.volume_ratio:.1f}x | RSI={features.rsi_14:.1f} | "
            f"ROC={features.roc_10:.2f}% | BB%B={features.bb_pct_b:.2f}"
        )

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
