"""Filter Chain — composable signal validation pipeline.

Adapted from Argus: E:/argus/argus-core/src/argus/filters/chain.py

Each filter evaluates a signal and either:
  - Rejects it (passed=False) with reason
  - Adjusts confidence and passes forward

Chain short-circuits on first rejection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from hfi.core.types import EngineSignal, FeatureVector, RegimeState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterResult:
    """Output of a single filter stage."""

    passed: bool
    confidence: float
    reason: str = ""
    details: dict = field(default_factory=dict)


class FilterStage(Protocol):
    """Protocol for a single filter in the chain."""

    name: str

    def evaluate(
        self,
        signal: EngineSignal,
        features: FeatureVector,
        regime: RegimeState,
    ) -> FilterResult: ...


@dataclass(frozen=True)
class ChainResult:
    """Output of the full filter chain."""

    passed: bool
    final_confidence: float
    original_confidence: float
    stages_passed: int
    stages_total: int
    reasons: list[str] = field(default_factory=list)


class FilterChain:
    """Sequential filter chain — short-circuits on rejection."""

    def __init__(self, stages: list[FilterStage]) -> None:
        self.stages = stages

    def evaluate(
        self,
        signal: EngineSignal,
        features: FeatureVector,
        regime: RegimeState,
    ) -> ChainResult:
        original_confidence = signal.confidence
        current_confidence = signal.confidence
        reasons: list[str] = []
        stages_passed = 0

        for stage in self.stages:
            result = stage.evaluate(signal, features, regime)
            reasons.append(result.reason)

            if not result.passed:
                logger.debug("Filter REJECTED by %s: %s", stage.name, result.reason)
                return ChainResult(
                    passed=False,
                    final_confidence=result.confidence,
                    original_confidence=original_confidence,
                    stages_passed=stages_passed,
                    stages_total=len(self.stages),
                    reasons=reasons,
                )

            current_confidence = result.confidence
            stages_passed += 1

        return ChainResult(
            passed=True,
            final_confidence=current_confidence,
            original_confidence=original_confidence,
            stages_passed=stages_passed,
            stages_total=len(self.stages),
            reasons=reasons,
        )
