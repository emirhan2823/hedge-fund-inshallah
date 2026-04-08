"""Tests for risk management."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hfi.core.config import LeverageConfig, RiskConfig
from hfi.core.types import EngineSignal, PortfolioState, RegimeState
from hfi.risk.sizing import SizingInput, compute_size
from hfi.risk.circuit_breaker import CircuitBreaker


@pytest.fixture
def cb(tmp_path):
    """Create a CircuitBreaker with isolated state file."""
    return CircuitBreaker(state_file=tmp_path / "cb_state.json")


class TestLeverage:
    def test_milestones(self):
        lc = LeverageConfig()
        assert lc.get_leverage(0) == 3
        assert lc.get_leverage(100) == 3
        assert lc.get_leverage(499) == 3
        assert lc.get_leverage(500) == 5
        assert lc.get_leverage(999) == 5
        assert lc.get_leverage(1000) == 7
        assert lc.get_leverage(5000) == 10
        assert lc.get_leverage(50000) == 10


class TestSizing:
    def _make_signal(self, **overrides) -> EngineSignal:
        defaults = dict(
            engine="TREND_FOLLOWER",
            symbol="BTC/USDT:USDT",
            bias="long",
            confidence=0.7,
            stop_distance=0.02,
            take_profit_distance=0.06,
            expected_return=0.03,
            atr=500.0,
        )
        defaults.update(overrides)
        return EngineSignal(**defaults)

    def _make_regime(self) -> RegimeState:
        return RegimeState(
            regime="TRENDING_LOW_VOL",
            confidence=0.7,
            direction=1,
            atr_percentile=0.5,
            adx_value=30.0,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    def test_basic_sizing(self):
        inp = SizingInput(
            balance=100.0,
            signal=self._make_signal(),
            regime=self._make_regime(),
            close_price=42000.0,
            risk_config=RiskConfig(),
            leverage_config=LeverageConfig(),
        )
        result = compute_size(inp)

        assert result is not None
        assert result.position_size_usd > 0
        assert result.leverage == 3  # $100 balance -> 3x
        assert result.stop_loss_price < 42000  # long trade, SL below entry
        assert result.take_profit_price > 42000

    def test_halted_at_max_drawdown(self):
        inp = SizingInput(
            balance=100.0,
            signal=self._make_signal(),
            regime=self._make_regime(),
            close_price=42000.0,
            risk_config=RiskConfig(),
            leverage_config=LeverageConfig(),
            current_drawdown_pct=0.16,  # beyond 15% max
        )
        result = compute_size(inp)

        assert result is None

    def test_reduced_at_consecutive_losses(self):
        normal = SizingInput(
            balance=100.0,
            signal=self._make_signal(),
            regime=self._make_regime(),
            close_price=42000.0,
            risk_config=RiskConfig(),
            leverage_config=LeverageConfig(),
            consecutive_losses=0,
        )
        reduced = SizingInput(
            balance=100.0,
            signal=self._make_signal(),
            regime=self._make_regime(),
            close_price=42000.0,
            risk_config=RiskConfig(),
            leverage_config=LeverageConfig(),
            consecutive_losses=4,  # 3+ -> 50% reduction
        )

        r_normal = compute_size(normal)
        r_reduced = compute_size(reduced)

        assert r_normal is not None and r_reduced is not None
        assert r_reduced.position_size_usd < r_normal.position_size_usd


class TestCircuitBreaker:
    def test_initial_state(self, cb):
        assert cb.consecutive_losses == 0
        assert cb.size_multiplier == 1.0
        assert not cb.is_halted

    def test_reduce_after_3_losses(self, cb):
        for _ in range(3):
            cb.record_trade(-10.0)
        assert cb.consecutive_losses == 3
        assert cb.size_multiplier == 0.50

    def test_minimum_after_5_losses(self, cb):
        for _ in range(5):
            cb.record_trade(-10.0)
        assert cb.size_multiplier == 0.25

    def test_halt_after_7_losses(self, cb):
        for _ in range(7):
            cb.record_trade(-10.0)
        assert cb.is_halted
        assert cb.size_multiplier == 0.0

    def test_reset_on_win(self, cb):
        cb.record_trade(-10.0)
        cb.record_trade(-10.0)
        cb.record_trade(20.0)  # win resets streak
        assert cb.consecutive_losses == 0
        assert cb.size_multiplier == 1.0

    def test_manual_reset(self, cb):
        for _ in range(7):
            cb.record_trade(-10.0)
        assert cb.is_halted
        cb.reset()
        assert not cb.is_halted
        assert cb.consecutive_losses == 0
