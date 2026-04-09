"""Pipeline runner — Data → Features → Regime → Engines → Filters → Risk → Sizing → Decision.

Orchestrates the full trading pipeline with multi-layer validation.
Pattern from Argus: E:/argus/argus-core/src/argus/pipeline/runner.py

Pipeline stages:
  1. Regime detection (6-layer voting)
  2. Engine signal generation
  3. Filter chain validation (directional bias + signal quality)
  4. Risk check (portfolio limits)
  5. Position sizing
  6. Decision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from hfi.core.config import HFIConfig
from hfi.core.types import EngineSignal, FeatureVector, PortfolioState, RegimeState, SizingResult
from hfi.features.microstructure import MicrostructureCollector
from hfi.features.builder import merge_microstructure
from hfi.engines.base import AbstractEngine
from hfi.engines.mean_reversion import MeanReversion
from hfi.engines.momentum_scalper import MomentumScalper
from hfi.engines.trend_follower import TrendFollower
from hfi.filters.chain import FilterChain
from hfi.filters.directional_bias import DirectionalBiasFilter
from hfi.filters.signal_quality import SignalQualityFilter
from hfi.regime.voting import VotingRegimeClassifier
from hfi.risk.manager import RiskManager
from hfi.risk.sizing import SizingInput, compute_size

logger = logging.getLogger(__name__)


@dataclass
class PipelineDecision:
    """Final pipeline output."""

    action: str          # "enter_long" | "enter_short" | "skip"
    signal: EngineSignal | None = None
    sizing: SizingResult | None = None
    regime: RegimeState | None = None
    skip_reason: str = ""
    filter_reasons: list[str] | None = None


class Pipeline:
    """Main trading pipeline orchestrator."""

    def __init__(self, config: HFIConfig, risk_manager: RiskManager, micro_collector: MicrostructureCollector | None = None) -> None:
        self._config = config
        self._risk_manager = risk_manager
        self._micro_collector = micro_collector

        # Initialize 6-layer voting regime classifier
        self._regime_classifier = VotingRegimeClassifier()

        # Initialize engines
        self._engines: list[AbstractEngine] = []
        if config.trend_follower.enabled:
            self._engines.append(TrendFollower(config.trend_follower))
        if config.mean_reversion.enabled:
            self._engines.append(MeanReversion(config.mean_reversion))
        if config.momentum_scalper.enabled:
            self._engines.append(MomentumScalper(config.momentum_scalper))

        # Initialize filter chain
        self._filter_chain = FilterChain([
            DirectionalBiasFilter(),     # Stage 1: block counter-trend
            SignalQualityFilter(         # Stage 2: multi-factor quality
                regime_classifier=self._regime_classifier,
            ),
        ])

        logger.info(
            "Pipeline initialized: %d engines, %d filters",
            len(self._engines), len(self._filter_chain.stages),
        )

    def run(
        self,
        features: FeatureVector,
        close_price: float,
        portfolio: PortfolioState,
        open_positions: list[dict[str, Any]],
    ) -> PipelineDecision:
        """Run full pipeline for a single tick.

        1. Detect regime
        2. Query each active engine for signals
        3. Pick best signal (highest expected return)
        4. Check risk rules
        5. Compute position size
        6. Return decision
        """
        # 1. Detect regime (6-layer voting)
        regime = self._regime_classifier.classify(features)
        logger.debug(
            "Regime: %s (conf=%.2f, dir=%s, ATR_pctl=%.2f, ADX=%.1f, candles=%d)",
            regime.regime, regime.confidence, regime.direction,
            regime.atr_percentile, regime.adx_value,
            self._regime_classifier.candles_in_regime,
        )

        # 2. Collect signals from all active engines
        signals: list[EngineSignal] = []
        for engine in self._engines:
            if regime.regime not in engine.active_regimes:
                continue

            # Special handling for momentum scalper (needs balance)
            if isinstance(engine, MomentumScalper):
                signal = engine.generate_signal(
                    regime=regime,
                    features=features,
                    close_price=close_price,
                    account_balance=portfolio.balance_usd,
                )
            else:
                signal = engine.generate_signal(
                    regime=regime,
                    features=features,
                    close_price=close_price,
                )

            if signal is not None:
                signals.append(signal)
                logger.info(
                    "Signal from %s: %s %s (conf=%.2f, stop=%.4f, tp=%.4f)",
                    signal.engine, signal.bias, signal.symbol,
                    signal.confidence, signal.stop_distance, signal.take_profit_distance,
                )

        if not signals:
            return PipelineDecision(
                action="skip", regime=regime,
                skip_reason=f"No signals in {regime.regime}",
            )

        # 3. Pick best signal (highest expected return * confidence)
        best = max(signals, key=lambda s: s.expected_return * s.confidence)

        # 4. Filter chain validation (directional bias + signal quality)
        chain_result = self._filter_chain.evaluate(best, features, regime)
        if not chain_result.passed:
            return PipelineDecision(
                action="skip", signal=best, regime=regime,
                skip_reason=f"Filter rejected: {chain_result.reasons}",
                filter_reasons=chain_result.reasons,
            )

        logger.info(
            "Filters PASSED (%d/%d): conf %.2f → %.2f",
            chain_result.stages_passed, chain_result.stages_total,
            chain_result.original_confidence, chain_result.final_confidence,
        )

        # 5. Risk check
        can_trade, reason = self._risk_manager.check_can_trade(
            best, portfolio, open_positions,
        )
        if not can_trade:
            return PipelineDecision(
                action="skip", signal=best, regime=regime,
                skip_reason=f"Risk blocked: {reason}",
            )

        # 6. Compute position size
        sizing_input = SizingInput(
            balance=portfolio.balance_usd,
            signal=best,
            regime=regime,
            close_price=close_price,
            risk_config=self._config.risk,
            leverage_config=self._config.leverage,
            consecutive_losses=portfolio.consecutive_losses,
            current_drawdown_pct=portfolio.drawdown_pct,
        )
        sizing = compute_size(sizing_input)

        if sizing is None:
            return PipelineDecision(
                action="skip", signal=best, regime=regime,
                skip_reason="Sizing returned None (halted)",
            )

        # 7. Return decision
        action = f"enter_{best.bias}"
        logger.info(
            "DECISION: %s %s $%.2f (leverage=%dx, SL=%.4f, TP=%.4f) by %s",
            action, best.symbol, sizing.position_size_usd,
            sizing.leverage, sizing.stop_loss_price, sizing.take_profit_price,
            best.engine,
        )

        return PipelineDecision(
            action=action,
            signal=best,
            sizing=sizing,
            regime=regime,
        )
