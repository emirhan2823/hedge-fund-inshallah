"""Mean Reversion Engine — Bollinger Bands + RSI + Z-score.

Timeframe: 15m primary, 1h confirmation.
Active in: RANGING_LOW_VOL regime only.
Skill ref: mean-reversion.md, exit-strategies.md
"""

from __future__ import annotations

import logging
from typing import Optional

from hfi.core.constants import Bias, Engine
from hfi.core.config import MeanReversionConfig
from hfi.core.types import EngineSignal, FeatureVector, RegimeState

logger = logging.getLogger(__name__)


class MeanReversion:
    """Mean reversion engine using BB + RSI + Z-score.

    Pre-check: Hurst exponent < 0.5 (mean-reverting price action).

    Entry conditions (ALL must be true for LONG):
    - RSI < 30 (oversold)
    - Price below lower Bollinger Band (bb_pct_b < 0)
    - Z-score < -2 (statistically extreme)
    - Hurst < 0.5 (confirms mean-reverting)

    Entry conditions (ALL for SHORT):
    - RSI > 70 (overbought)
    - Price above upper BB (bb_pct_b > 1)
    - Z-score > +2
    - Hurst < 0.5

    Exit:
    - RSI crosses 50 (mean)
    - Price crosses middle BB
    """

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self._config = config or MeanReversionConfig()

    @property
    def name(self) -> str:
        return Engine.MEAN_REVERSION

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

        # Pre-check: Hurst must indicate mean reversion
        if features.hurst_exponent >= c.hurst_threshold:
            return None

        # Score-based entry: at least 2 of 3 extremity signals
        # LONG setup: oversold
        long_rsi = features.rsi_14 < c.rsi_oversold
        long_bb = features.bb_pct_b < 0.15
        long_zscore = features.zscore_close_20 < c.zscore_entry
        long_score = int(long_rsi) + int(long_bb) + int(long_zscore)

        # SHORT setup: overbought
        short_rsi = features.rsi_14 > c.rsi_overbought
        short_bb = features.bb_pct_b > 0.85
        short_zscore = features.zscore_close_20 > abs(c.zscore_entry)
        short_score = int(short_rsi) + int(short_bb) + int(short_zscore)

        bias: str | None = None
        if long_score >= 2:
            bias = Bias.LONG
        elif short_score >= 2:
            bias = Bias.SHORT
        else:
            return None

        # Stop distance: beyond the extreme (wider than trend following)
        atr = features.atr_14
        stop_distance = (atr * 2.5) / close_price

        # Take profit: reversion to mean (middle BB area)
        # Distance from current price to middle BB approximated by zscore
        tp_distance = abs(features.zscore_close_20) * (features.bb_width * close_price / 2) / close_price
        tp_distance = max(tp_distance, stop_distance * 1.5)  # minimum 1.5:1 R:R

        # Confidence from how extreme the deviation is
        zscore_extremity = min(1.0, abs(features.zscore_close_20) / 3.0)
        hurst_conf = 1.0 - features.hurst_exponent  # lower hurst = higher confidence
        confidence = (zscore_extremity * 0.5 + hurst_conf * 0.3 + regime.confidence * 0.2)

        expected_return = tp_distance * confidence - stop_distance * (1 - confidence)

        reason = (
            f"RSI={features.rsi_14:.1f} | BB%B={features.bb_pct_b:.2f} | "
            f"Z={features.zscore_close_20:.2f} | Hurst={features.hurst_exponent:.2f}"
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
