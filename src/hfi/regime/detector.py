"""4-Quadrant regime detector.

Classifies market into: trending/ranging x low-vol/high-vol.
Skill ref: regime-detection.md
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from hfi.core.constants import Regime
from hfi.core.types import FeatureVector, RegimeState

logger = logging.getLogger(__name__)

# ADX thresholds
ADX_TRENDING = 25.0    # Above = trending
ADX_RANGING = 20.0     # Below = ranging (between 20-25 = transitional, default ranging)

# ATR percentile thresholds
ATR_HIGH_VOL = 0.65    # Above 65th percentile = high volatility
ATR_LOW_VOL = 0.35     # Below 35th percentile = low volatility


def detect_regime(features: FeatureVector) -> RegimeState:
    """Classify current market regime from feature vector.

    4-Quadrant model:
        Q1: TRENDING_LOW_VOL   — Trend following paradise, tight stops work
        Q2: TRENDING_HIGH_VOL  — Trend following with caution, wider stops
        Q3: RANGING_LOW_VOL    — Mean reversion territory
        Q4: RANGING_HIGH_VOL   — Danger zone, choppy, sit out
    """
    adx = features.adx_14
    atr_pctl = features.atr_pctl if features.atr_pctl is not None else 0.5

    # Determine trend axis
    is_trending = adx >= ADX_TRENDING

    # Determine volatility axis
    is_high_vol = atr_pctl >= ATR_HIGH_VOL

    # Classify
    if is_trending and not is_high_vol:
        regime = Regime.TRENDING_LOW_VOL
    elif is_trending and is_high_vol:
        regime = Regime.TRENDING_HIGH_VOL
    elif not is_trending and not is_high_vol:
        regime = Regime.RANGING_LOW_VOL
    else:
        regime = Regime.RANGING_HIGH_VOL

    # Confidence based on how clear the signals are
    adx_clarity = abs(adx - 22.5) / 22.5  # distance from midpoint
    vol_clarity = abs(atr_pctl - 0.5) / 0.5
    confidence = min(1.0, (adx_clarity + vol_clarity) / 2)

    # Direction from EMA alignment
    direction = None
    if is_trending:
        if features.ema_8 > features.ema_21 > features.ema_55:
            direction = 1  # bullish trend
        elif features.ema_8 < features.ema_21 < features.ema_55:
            direction = -1  # bearish trend

    return RegimeState(
        regime=regime,
        confidence=confidence,
        direction=direction,
        atr_percentile=atr_pctl,
        adx_value=adx,
        timestamp=features.timestamp,
    )
