"""Directional Bias Filter — prevents counter-trend entries.

Core rule: DON'T long in bear market, DON'T short in bull market
(unless signal quality is exceptional).

Adapted from Argus: E:/argus/argus-core/src/argus/filters/directional_bias.py
"""

from __future__ import annotations

import logging

from hfi.core.types import EngineSignal, FeatureVector, RegimeState
from hfi.filters.chain import FilterResult

logger = logging.getLogger(__name__)

_MR_ENGINES = {"MEAN_REVERSION"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class DirectionalBiasFilter:
    """Penalize/reject counter-trend entries.

    Trend direction determined by EMA alignment:
    - Bullish: EMA8 > EMA21 > EMA55
    - Bearish: EMA8 < EMA21 < EMA55

    Counter-trend rules:
    - Trend engines: hard reject if ADX > 25 + counter-trend
    - MR engines: hard reject if ADX > 30 + counter-trend (softer)
    - All: penalty if mild counter-trend
    """

    name = "directional_bias"

    def __init__(
        self,
        *,
        with_trend_bonus: float = 0.05,
        counter_trend_penalty: float = 0.08,
        hard_reject_adx: float = 25.0,
        mr_hard_reject_adx: float = 30.0,
    ) -> None:
        self.with_trend_bonus = with_trend_bonus
        self.counter_trend_penalty = counter_trend_penalty
        self.hard_reject_adx = hard_reject_adx
        self.mr_hard_reject_adx = mr_hard_reject_adx

    def evaluate(
        self,
        signal: EngineSignal,
        features: FeatureVector,
        regime: RegimeState,
    ) -> FilterResult:
        # Determine trend direction from EMA alignment
        bullish = features.ema_8 > features.ema_21 > features.ema_55
        bearish = features.ema_8 < features.ema_21 < features.ema_55

        with_trend = (signal.bias == "long" and bullish) or (signal.bias == "short" and bearish)
        counter_trend = (signal.bias == "long" and bearish) or (signal.bias == "short" and bullish)

        is_mr = signal.engine in _MR_ENGINES
        reject_adx = self.mr_hard_reject_adx if is_mr else self.hard_reject_adx

        if counter_trend and features.adx_14 >= reject_adx:
            # HARD REJECT: strong trend + counter-trend = disaster
            return FilterResult(
                passed=False,
                confidence=signal.confidence,
                reason=(
                    f"REJECTED: {signal.bias} against {'bearish' if bearish else 'bullish'} trend "
                    f"(ADX={features.adx_14:.1f}>={reject_adx})"
                ),
            )

        if counter_trend:
            # Mild counter-trend: penalty
            adj = -self.counter_trend_penalty
            return FilterResult(
                passed=True,
                confidence=_clamp(signal.confidence + adj, 0.0, 1.0),
                reason=f"counter_trend_penalty={adj:.2f} ADX={features.adx_14:.1f}",
            )

        if with_trend:
            # With-trend bonus
            adj = self.with_trend_bonus
            return FilterResult(
                passed=True,
                confidence=_clamp(signal.confidence + adj, 0.0, 1.0),
                reason=f"with_trend_bonus=+{adj:.2f}",
            )

        # Neutral (no clear trend)
        return FilterResult(
            passed=True,
            confidence=signal.confidence,
            reason="no_clear_trend_neutral",
        )
