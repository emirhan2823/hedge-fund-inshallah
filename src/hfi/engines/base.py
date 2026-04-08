"""Engine protocol — all engines must satisfy this interface.

Pattern from Argus: E:/argus/argus-core/src/argus/engines/base.py
"""

from __future__ import annotations

from typing import Optional, Protocol

from hfi.core.types import EngineSignal, FeatureVector, RegimeState


class AbstractEngine(Protocol):
    """Protocol every engine implementation must satisfy."""

    @property
    def name(self) -> str:
        """Engine identifier."""
        ...

    @property
    def active_regimes(self) -> list[str]:
        """List of regime names where this engine is active."""
        ...

    def generate_signal(
        self,
        *,
        regime: RegimeState,
        features: FeatureVector,
        close_price: float,
    ) -> Optional[EngineSignal]:
        """Generate a trading signal from current market state.

        Returns None if no signal (no entry conditions met).
        """
        ...
