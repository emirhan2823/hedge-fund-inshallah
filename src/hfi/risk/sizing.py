"""Position sizing — multiplicative risk model.

Pattern from Argus: E:/argus/argus-core/src/argus/risk/sizing.py
Skill ref: position-sizing.md, kelly-criterion.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hfi.core.config import LeverageConfig, RiskConfig
from hfi.core.types import EngineSignal, RegimeState, SizingResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizingInput:
    """All inputs needed for position sizing calculation."""

    balance: float
    signal: EngineSignal
    regime: RegimeState
    close_price: float
    risk_config: RiskConfig
    leverage_config: LeverageConfig
    consecutive_losses: int = 0
    current_drawdown_pct: float = 0.0


def compute_size(inp: SizingInput) -> SizingResult | None:
    """Compute position size using multiplicative risk model.

    Formula: risk_pct = base * regime_mult * drawdown_mult * circuit_mult
    Position size = (balance * risk_pct) / stop_distance * leverage
    """
    rc = inp.risk_config
    signal = inp.signal

    # 1. Base risk
    base_risk = rc.base_risk_pct

    # 2. Regime multiplier (higher confidence = closer to 1.0)
    regime_mult = 0.5 + (inp.regime.confidence * 0.5)  # range: 0.5 - 1.0

    # 3. Drawdown multiplier (from risk config levels)
    dd_mult = _drawdown_multiplier(inp.current_drawdown_pct, rc)

    # 4. Circuit breaker multiplier
    circuit_mult = _circuit_multiplier(inp.consecutive_losses, rc)

    # 5. Combined risk percentage
    risk_pct = base_risk * regime_mult * dd_mult * circuit_mult
    risk_pct = max(rc.min_risk_pct, min(rc.max_risk_pct, risk_pct))

    # If halted (dd_mult or circuit_mult = 0), return None
    if risk_pct <= 0 or dd_mult == 0 or circuit_mult == 0:
        logger.warning("Position sizing HALTED: dd_mult=%.2f circuit_mult=%.2f", dd_mult, circuit_mult)
        return None

    # 6. Get leverage for current balance
    leverage = inp.leverage_config.get_leverage(inp.balance)

    # 7. Calculate position size in USD
    risk_amount_usd = inp.balance * risk_pct
    stop_distance = signal.stop_distance  # as fraction of entry

    if stop_distance <= 0:
        logger.error("Invalid stop_distance: %.6f", stop_distance)
        return None

    # Position size: how much USD notional to control
    # risk_amount = position_size * stop_distance (without leverage)
    # With leverage: position_size = risk_amount / stop_distance
    position_size_usd = risk_amount_usd / stop_distance

    # Apply leverage to actual capital needed
    # position_size_usd is notional; capital needed = position_size_usd / leverage
    # But we cap notional at balance * leverage * max_position_size_pct
    max_notional = inp.balance * leverage * rc.max_position_size_pct
    position_size_usd = min(position_size_usd, max_notional)

    # Calculate stop and take profit prices
    if signal.bias == "long":
        stop_loss_price = inp.close_price * (1 - stop_distance)
        take_profit_price = inp.close_price * (1 + signal.take_profit_distance)
    else:
        stop_loss_price = inp.close_price * (1 + stop_distance)
        take_profit_price = inp.close_price * (1 - signal.take_profit_distance)

    logger.info(
        "Sizing: balance=$%.2f risk=%.2f%% (base=%.2f regime=%.2f dd=%.2f circuit=%.2f) "
        "leverage=%dx size=$%.2f SL=%.4f TP=%.4f",
        inp.balance, risk_pct * 100, base_risk, regime_mult, dd_mult, circuit_mult,
        leverage, position_size_usd, stop_loss_price, take_profit_price,
    )

    return SizingResult(
        position_size_usd=position_size_usd,
        risk_pct=risk_pct,
        leverage=leverage,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        entry_price=inp.close_price,
    )


def _drawdown_multiplier(drawdown_pct: float, rc: RiskConfig) -> float:
    """Get sizing multiplier based on current drawdown.

    Drawdown response framework from risk-management.md:
    0-5%: 1.0 (normal)
    5-10%: 0.50 (reduce)
    10-15%: 0.25 (minimum)
    >15%: 0.0 (HALT)
    """
    if drawdown_pct <= 0:
        return 1.0

    for level, mult in zip(rc.drawdown_levels, rc.drawdown_multipliers):
        if drawdown_pct >= level:
            if mult == 0.0:
                return 0.0
            continue
        return 1.0 if level == rc.drawdown_levels[0] else rc.drawdown_multipliers[
            rc.drawdown_levels.index(level) - 1
        ]

    # Beyond all levels
    return rc.drawdown_multipliers[-1]


def _circuit_multiplier(consecutive_losses: int, rc: RiskConfig) -> float:
    """Get sizing multiplier from circuit breaker state.

    3 losses: 50%
    5 losses: 25% (minimum)
    7 losses: 0% (halt)
    """
    if consecutive_losses >= rc.consecutive_loss_halt:
        return 0.0
    elif consecutive_losses >= rc.consecutive_loss_min:
        return 0.25
    elif consecutive_losses >= rc.consecutive_loss_reduce:
        return 0.50
    return 1.0
