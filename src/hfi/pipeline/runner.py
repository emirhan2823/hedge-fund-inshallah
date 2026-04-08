"""Pipeline runner — Data → Features → Regime → Engines → Risk → Sizing → Decision.

Orchestrates the full trading pipeline for a single symbol + timeframe tick.
Pattern from Argus: E:/argus/argus-core/src/argus/pipeline/runner.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from hfi.core.config import HFIConfig
from hfi.core.types import EngineSignal, FeatureVector, PortfolioState, RegimeState, SizingResult
from hfi.engines.base import AbstractEngine
from hfi.engines.mean_reversion import MeanReversion
from hfi.engines.momentum_scalper import MomentumScalper
from hfi.engines.trend_follower import TrendFollower
from hfi.regime.detector import detect_regime
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


class Pipeline:
    """Main trading pipeline orchestrator."""

    def __init__(self, config: HFIConfig, risk_manager: RiskManager) -> None:
        self._config = config
        self._risk_manager = risk_manager

        # Initialize engines
        self._engines: list[AbstractEngine] = []
        if config.trend_follower.enabled:
            self._engines.append(TrendFollower(config.trend_follower))
        if config.mean_reversion.enabled:
            self._engines.append(MeanReversion(config.mean_reversion))
        if config.momentum_scalper.enabled:
            self._engines.append(MomentumScalper(config.momentum_scalper))

        logger.info("Pipeline initialized with %d engines", len(self._engines))

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
        # 1. Detect regime
        regime = detect_regime(features)
        logger.debug(
            "Regime: %s (conf=%.2f, dir=%s, ATR_pctl=%.2f, ADX=%.1f)",
            regime.regime, regime.confidence, regime.direction,
            regime.atr_percentile, regime.adx_value,
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

        # 4. Risk check
        can_trade, reason = self._risk_manager.check_can_trade(
            best, portfolio, open_positions,
        )
        if not can_trade:
            return PipelineDecision(
                action="skip", signal=best, regime=regime,
                skip_reason=f"Risk blocked: {reason}",
            )

        # 5. Compute position size
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

        # 6. Return decision
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
