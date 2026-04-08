"""Signal Quality Filter — multi-factor signal validation.

Adapted from Argus: E:/argus/argus-core/src/argus/filters/signal_quality.py

Factors:
  1. Indicator alignment (do RSI/MACD/ROC agree with signal direction?)
  2. Volume confirmation (is there volume behind the move?)
  3. Regime stability (has regime been consistent?)
  4. Trend-bias alignment (does macro trend support entry?)
  5. Toxic pattern detection (known bad setups)

Quality score < MIN_QUALITY → reject signal.
"""

from __future__ import annotations

import logging

from hfi.core.types import EngineSignal, FeatureVector, RegimeState
from hfi.filters.chain import FilterResult
from hfi.regime.voting import VotingRegimeClassifier

logger = logging.getLogger(__name__)

MIN_QUALITY_SCORE = 0.45

_MR_ENGINES = {"MEAN_REVERSION"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SignalQualityFilter:
    """5-factor signal quality assessment."""

    name = "signal_quality"

    def __init__(
        self,
        *,
        min_quality: float = MIN_QUALITY_SCORE,
        regime_classifier: VotingRegimeClassifier | None = None,
    ) -> None:
        self.min_quality = min_quality
        self._regime_classifier = regime_classifier

    def evaluate(
        self,
        signal: EngineSignal,
        features: FeatureVector,
        regime: RegimeState,
    ) -> FilterResult:
        factors: list[float] = []
        conf_adj = 0.0
        details: dict = {}

        # F1: Indicator alignment (do indicators agree with direction?)
        alignment = self._indicator_alignment(signal.bias, features)
        factors.append(alignment)
        details["indicator_alignment"] = round(alignment, 3)
        if alignment < 0.40:
            conf_adj -= 0.08
        elif alignment > 0.70:
            conf_adj += 0.04

        # F2: Volume confirmation
        vol_q = self._volume_quality(features)
        factors.append(vol_q)
        details["volume_quality"] = round(vol_q, 3)
        if vol_q < 0.30:
            conf_adj -= 0.05
        elif vol_q > 0.70:
            conf_adj += 0.03

        # F3: Regime stability (how long have we been in this regime?)
        regime_q = self._regime_stability(regime)
        factors.append(regime_q)
        details["regime_stability"] = round(regime_q, 3)
        if regime_q < 0.30:
            conf_adj -= 0.06

        # F4: Trend alignment (macro trend supports entry?)
        trend_q = self._trend_alignment(signal.bias, signal.engine, features)
        factors.append(trend_q)
        details["trend_alignment"] = round(trend_q, 3)
        if trend_q > 0.70:
            conf_adj += 0.05
        elif trend_q < 0.30:
            conf_adj -= 0.07

        # F5: Toxic pattern detection
        toxic = self._toxic_penalty(signal.bias, features)
        if toxic < 0:
            conf_adj += toxic
            factors.append(0.15)
            details["toxic_penalty"] = round(toxic, 3)
        else:
            factors.append(0.70)

        quality_score = sum(factors) / max(len(factors), 1)
        adjusted_conf = _clamp(signal.confidence + conf_adj, 0.0, 1.0)
        details["quality_score"] = round(quality_score, 3)
        details["conf_adjustment"] = round(conf_adj, 3)

        passed = quality_score >= self.min_quality
        if passed:
            reason = f"quality_pass score={quality_score:.3f} adj={conf_adj:+.3f}"
        else:
            reason = f"REJECTED: quality={quality_score:.3f}<{self.min_quality} factors={details}"

        return FilterResult(
            passed=passed,
            confidence=adjusted_conf,
            reason=reason,
            details=details,
        )

    @staticmethod
    def _indicator_alignment(bias: str, f: FeatureVector) -> float:
        """Count how many indicators agree with signal direction."""
        if bias == "long":
            votes = [
                f.rsi_14 > 40,              # not oversold
                f.macd_hist > 0,             # MACD bullish
                f.roc_10 > 0,               # positive momentum
                f.bb_pct_b > 0.3,           # not at bottom of bands
                f.ema_8 > f.ema_21,         # short-term bullish
            ]
        else:
            votes = [
                f.rsi_14 < 60,              # not overbought
                f.macd_hist < 0,             # MACD bearish
                f.roc_10 < 0,               # negative momentum
                f.bb_pct_b < 0.7,           # not at top of bands
                f.ema_8 < f.ema_21,         # short-term bearish
            ]
        return sum(1 for v in votes if v) / len(votes)

    @staticmethod
    def _volume_quality(f: FeatureVector) -> float:
        """Score volume confirmation."""
        if f.volume_ratio > 2.0:
            return 0.90  # strong volume = good
        elif f.volume_ratio > 1.3:
            return 0.70
        elif f.volume_ratio > 0.8:
            return 0.50
        elif f.volume_ratio > 0.5:
            return 0.30
        return 0.10  # dead volume = bad

    @staticmethod
    def _regime_stability(regime: RegimeState) -> float:
        """Score regime consistency."""
        # Use confidence as proxy for stability
        return regime.confidence

    @staticmethod
    def _trend_alignment(bias: str, engine: str, f: FeatureVector) -> float:
        """Score macro trend alignment. MR engines get neutral score."""
        if engine in _MR_ENGINES:
            return 0.50  # MR is inherently counter-trend

        score = 0.50
        if bias == "long":
            score += 0.20 if f.ema_21 > f.ema_55 else -0.15
            score += 0.15 if f.roc_10 > 0 else -0.10
        else:
            score += 0.20 if f.ema_21 < f.ema_55 else -0.15
            score += 0.15 if f.roc_10 < 0 else -0.10

        if f.adx_14 > 30:
            score *= 1.15

        return _clamp(score, 0.0, 1.0)

    @staticmethod
    def _toxic_penalty(bias: str, f: FeatureVector) -> float:
        """Detect known toxic patterns from backtesting."""
        penalty = 0.0

        # P1: Long in extreme overbought
        if bias == "long" and f.rsi_14 > 80:
            penalty -= 0.12

        # P2: Short in extreme oversold
        if bias == "short" and f.rsi_14 < 20:
            penalty -= 0.12

        # P3: Entry during dead volume
        if f.volume_ratio < 0.5:
            penalty -= 0.10

        # P4: Counter-trend in strong ADX
        if bias == "long" and f.ema_8 < f.ema_55 and f.adx_14 > 35:
            penalty -= 0.15

        if bias == "short" and f.ema_8 > f.ema_55 and f.adx_14 > 35:
            penalty -= 0.15

        return penalty
