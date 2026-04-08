"""Tests for the trading pipeline."""

from datetime import datetime, timezone

import pytest

from hfi.core.config import HFIConfig
from hfi.core.constants import Regime
from hfi.core.types import FeatureVector, PortfolioState
from hfi.pipeline.runner import Pipeline
from hfi.risk.manager import RiskManager


def _make_features(**overrides) -> FeatureVector:
    defaults = dict(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        symbol="BTC/USDT:USDT",
        atr_14=500.0,
        atr_14_pct=0.012,
        bb_width=0.05,
        adx_14=30.0,
        ema_8=42000.0,
        ema_21=41500.0,
        ema_55=40000.0,
        rsi_14=55.0,
        macd_hist=50.0,
        roc_10=2.0,
        bb_pct_b=0.6,
        volume_ratio=1.2,
        volume_sma_20=1000.0,
        zscore_close_20=0.5,
        hurst_exponent=0.55,
        atr_pctl=0.3,  # low vol
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


def _make_portfolio(**overrides) -> PortfolioState:
    defaults = dict(
        balance_usd=100.0,
        equity_usd=100.0,
        unrealized_pnl=0.0,
        open_positions=0,
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        max_equity=100.0,
        drawdown_pct=0.0,
        total_trades=0,
        winning_trades=0,
        consecutive_losses=0,
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


class TestPipeline:
    def test_enter_long_trending(self):
        config = HFIConfig()
        rm = RiskManager(config.risk)
        pipeline = Pipeline(config, rm)

        features = _make_features(
            adx_14=30, ema_8=42000, ema_21=41500, ema_55=40000,
            macd_hist=50, atr_pctl=0.3,
        )
        portfolio = _make_portfolio()

        decision = pipeline.run(
            features=features,
            close_price=42000.0,
            portfolio=portfolio,
            open_positions=[],
        )

        assert decision.action == "enter_long"
        assert decision.signal is not None
        assert decision.sizing is not None
        assert decision.sizing.position_size_usd > 0

    def test_skip_when_halted(self):
        config = HFIConfig()
        rm = RiskManager(config.risk)
        pipeline = Pipeline(config, rm)

        features = _make_features(adx_14=30, macd_hist=50, atr_pctl=0.3)
        portfolio = _make_portfolio(drawdown_pct=0.20)  # beyond max

        decision = pipeline.run(
            features=features,
            close_price=42000.0,
            portfolio=portfolio,
            open_positions=[],
        )

        assert decision.action == "skip"

    def test_skip_max_positions(self):
        config = HFIConfig()
        rm = RiskManager(config.risk)
        pipeline = Pipeline(config, rm)

        features = _make_features(adx_14=30, macd_hist=50, atr_pctl=0.3)
        portfolio = _make_portfolio(open_positions=3)  # at max

        decision = pipeline.run(
            features=features,
            close_price=42000.0,
            portfolio=portfolio,
            open_positions=[{"side": "long"}, {"side": "long"}, {"side": "short"}],
        )

        assert decision.action == "skip"

    def test_skip_no_signal_ranging(self):
        config = HFIConfig()
        rm = RiskManager(config.risk)
        pipeline = Pipeline(config, rm)

        # Ranging high-vol regime (Q4 - no engines active)
        features = _make_features(
            adx_14=15, atr_pctl=0.8,
            ema_8=41000, ema_21=41500, ema_55=41000,  # no clear alignment
            rsi_14=50, macd_hist=0,
            hurst_exponent=0.55,
        )
        portfolio = _make_portfolio()

        decision = pipeline.run(
            features=features,
            close_price=41000.0,
            portfolio=portfolio,
            open_positions=[],
        )

        assert decision.action == "skip"
