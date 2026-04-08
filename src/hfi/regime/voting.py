"""6-Layer Voting Regime Classifier.

Adapted from Argus: E:/argus/argus-core/src/argus/regime/voting_classifier.py

Layers (weighted votes):
  1. ADX strength          (weight 2.0x) — primary trend indicator
  2. EMA spread            (weight 1.0x) — trend magnitude
  3. Bollinger Band width  (weight 1.0x) — volatility gauge
  4. Price momentum (ROC)  (weight 1.5x) — replaces HTF ADX (no multi-tf yet)
  5. Persistence           (weight 1.0x) — regime stability
  6. Volume ratio          (weight 1.0x) — activity confirmation

CRISIS is a hard override (not voted).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

from hfi.core.constants import Regime
from hfi.core.types import FeatureVector, RegimeState

logger = logging.getLogger(__name__)

_PERSISTENCE_WINDOW = 10

# Vote = (TRENDING, RANGING, VOLATILE)
_Vote = tuple[float, float, float]


@dataclass
class VotingThresholds:
    """Thresholds for each voting layer."""

    # Layer 1: ADX
    adx_trending: float = 28.0
    adx_ranging: float = 18.0

    # Layer 2: EMA spread = abs(ema8 - ema55) / ema55
    ema_spread_trending: float = 0.015
    ema_spread_ranging: float = 0.004

    # Layer 3: BB width
    bb_width_volatile: float = 0.05
    bb_width_ranging: float = 0.02

    # Layer 4: ROC(10) magnitude
    roc_trending: float = 1.5   # >1.5% = strong directional move
    roc_ranging: float = 0.3    # <0.3% = flat

    # Layer 6: Volume ratio
    vol_ratio_volatile: float = 1.5
    vol_ratio_ranging: float = 0.7

    # Crisis detection
    crisis_roc_threshold: float = -5.0  # -5% ROC = crash
    crisis_atr_pct: float = 0.04        # ATR > 4% of price = extreme

    # Volatile combo
    volatile_bb_atr_combo: float = 0.035  # BB width + ATR pct threshold

    # Layer weights
    weight_adx: float = 2.0
    weight_ema_spread: float = 1.0
    weight_bb_width: float = 1.0
    weight_roc: float = 1.5
    weight_persistence: float = 1.0
    weight_volume: float = 1.0


class VotingRegimeClassifier:
    """6-layer weighted voting regime classifier.

    Each layer votes TRENDING/RANGING/VOLATILE with score [0,1].
    Weighted majority determines final regime.
    CRISIS is a hard override checked first.
    """

    def __init__(self, thresholds: VotingThresholds | None = None) -> None:
        self.t = thresholds or VotingThresholds()
        self._history: deque[str] = deque(maxlen=_PERSISTENCE_WINDOW)
        self._candles_in_regime: int = 0
        self._current_regime: str = ""

    def classify(self, features: FeatureVector) -> RegimeState:
        """Classify regime via 6-layer voting. Returns RegimeState."""
        # Hard override: CRISIS
        if self._is_crisis(features):
            regime = "CRISIS"
            confidence = 0.95
        else:
            # Collect weighted votes
            votes: list[tuple[_Vote, float]] = [
                (self._vote_adx(features), self.t.weight_adx),
                (self._vote_ema_spread(features), self.t.weight_ema_spread),
                (self._vote_bb_width(features), self.t.weight_bb_width),
                (self._vote_roc(features), self.t.weight_roc),
                (self._vote_persistence(), self.t.weight_persistence),
                (self._vote_volume(features), self.t.weight_volume),
            ]

            # Weighted sum
            t_score = r_score = v_score = 0.0
            total_weight = 0.0
            for (t, r, v), weight in votes:
                t_score += t * weight
                r_score += r * weight
                v_score += v * weight
                total_weight += weight

            if total_weight > 0:
                t_score /= total_weight
                r_score /= total_weight
                v_score /= total_weight

            # Winner
            scores = {"TRENDING": t_score, "RANGING": r_score, "VOLATILE": v_score}
            regime = max(scores, key=scores.get)
            confidence = scores[regime]

        # Map to 4-quadrant (regime + volatility level)
        atr_pctl = features.atr_pctl if features.atr_pctl is not None else 0.5
        is_high_vol = atr_pctl >= 0.6

        if regime == "CRISIS":
            quadrant = Regime.RANGING_HIGH_VOL  # crisis = sit out
        elif regime == "VOLATILE":
            quadrant = Regime.RANGING_HIGH_VOL if features.adx_14 < 25 else Regime.TRENDING_HIGH_VOL
        elif regime == "TRENDING":
            quadrant = Regime.TRENDING_HIGH_VOL if is_high_vol else Regime.TRENDING_LOW_VOL
        else:  # RANGING
            quadrant = Regime.RANGING_HIGH_VOL if is_high_vol else Regime.RANGING_LOW_VOL

        # Track persistence
        if quadrant == self._current_regime:
            self._candles_in_regime += 1
        else:
            self._candles_in_regime = 1
            self._current_regime = quadrant

        self._history.append(regime)

        # Direction from EMA alignment
        direction = None
        if features.ema_8 > features.ema_21 > features.ema_55:
            direction = 1
        elif features.ema_8 < features.ema_21 < features.ema_55:
            direction = -1

        return RegimeState(
            regime=quadrant,
            confidence=min(confidence, 1.0),
            direction=direction,
            atr_percentile=atr_pctl,
            adx_value=features.adx_14,
            timestamp=features.timestamp,
        )

    @property
    def candles_in_regime(self) -> int:
        return self._candles_in_regime

    def reset(self) -> None:
        self._history.clear()
        self._candles_in_regime = 0
        self._current_regime = ""

    # ── Layer voters ──────────────────────────────────────

    def _vote_adx(self, f: FeatureVector) -> _Vote:
        if f.adx_14 >= self.t.adx_trending:
            strength = min((f.adx_14 - self.t.adx_trending) / 30.0 + 0.6, 1.0)
            return (strength, 0.1, 0.0)
        if f.adx_14 <= self.t.adx_ranging:
            strength = min((self.t.adx_ranging - f.adx_14) / 18.0 + 0.6, 1.0)
            return (0.1, strength, 0.0)
        # Transitional zone
        frac = (f.adx_14 - self.t.adx_ranging) / max(
            self.t.adx_trending - self.t.adx_ranging, 1.0
        )
        return (0.3 + frac * 0.3, 0.3 + (1 - frac) * 0.3, 0.0)

    def _vote_ema_spread(self, f: FeatureVector) -> _Vote:
        spread = abs(f.ema_8 - f.ema_55) / f.ema_55 if f.ema_55 > 0 else 0
        if spread >= self.t.ema_spread_trending:
            return (0.8, 0.1, 0.1)
        if spread <= self.t.ema_spread_ranging:
            return (0.1, 0.8, 0.1)
        frac = (spread - self.t.ema_spread_ranging) / max(
            self.t.ema_spread_trending - self.t.ema_spread_ranging, 1e-6
        )
        return (0.2 + frac * 0.5, 0.5 - frac * 0.3, 0.1)

    def _vote_bb_width(self, f: FeatureVector) -> _Vote:
        if f.bb_width >= self.t.bb_width_volatile:
            return (0.1, 0.0, 0.9)
        if f.bb_width <= self.t.bb_width_ranging:
            return (0.1, 0.8, 0.1)
        frac = (f.bb_width - self.t.bb_width_ranging) / max(
            self.t.bb_width_volatile - self.t.bb_width_ranging, 1e-6
        )
        return (0.3, 0.4 - frac * 0.3, 0.1 + frac * 0.5)

    def _vote_roc(self, f: FeatureVector) -> _Vote:
        roc = abs(f.roc_10)
        if roc >= self.t.roc_trending:
            return (0.8, 0.05, 0.15)
        if roc <= self.t.roc_ranging:
            return (0.1, 0.8, 0.1)
        frac = (roc - self.t.roc_ranging) / max(
            self.t.roc_trending - self.t.roc_ranging, 1e-6
        )
        return (0.2 + frac * 0.5, 0.5 - frac * 0.3, 0.1)

    def _vote_persistence(self) -> _Vote:
        if not self._history:
            return (0.33, 0.34, 0.33)
        total = len(self._history)
        t_count = sum(1 for r in self._history if r == "TRENDING")
        r_count = sum(1 for r in self._history if r == "RANGING")
        v_count = total - t_count - r_count
        return (t_count / total, r_count / total, v_count / total)

    def _vote_volume(self, f: FeatureVector) -> _Vote:
        if f.volume_ratio >= self.t.vol_ratio_volatile:
            return (0.2, 0.0, 0.8)
        if f.volume_ratio <= self.t.vol_ratio_ranging:
            return (0.1, 0.7, 0.2)
        return (0.4, 0.4, 0.2)

    # ── Crisis detection ──────────────────────────────────

    def _is_crisis(self, f: FeatureVector) -> bool:
        if f.roc_10 <= self.t.crisis_roc_threshold:
            return True
        if f.atr_14_pct >= self.t.crisis_atr_pct:
            return True
        return False
