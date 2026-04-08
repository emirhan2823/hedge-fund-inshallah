"""Tests for trading engines."""

from datetime import datetime, timezone

import pytest

from hfi.core.constants import Bias, Engine, Regime
from hfi.core.types import EngineSignal, FeatureVector, RegimeState
from hfi.engines.trend_follower import TrendFollower
from hfi.engines.mean_reversion import MeanReversion
from hfi.engines.momentum_scalper import MomentumScalper


def _make_features(**overrides) -> FeatureVector:
    """Create a FeatureVector with sensible defaults."""
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
        atr_pctl=0.5,
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


def _make_regime(regime: str = Regime.TRENDING_LOW_VOL, **overrides) -> RegimeState:
    defaults = dict(
        regime=regime,
        confidence=0.7,
        direction=1,
        atr_percentile=0.5,
        adx_value=30.0,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return RegimeState(**defaults)


class TestTrendFollower:
    def test_long_signal(self):
        engine = TrendFollower()
        features = _make_features(
            ema_8=42000, ema_21=41500, ema_55=40000,
            adx_14=30, macd_hist=50,
        )
        regime = _make_regime(Regime.TRENDING_LOW_VOL)
        signal = engine.generate_signal(regime=regime, features=features, close_price=42000)

        assert signal is not None
        assert signal.bias == Bias.LONG
        assert signal.engine == Engine.TREND_FOLLOWER
        assert signal.stop_distance > 0
        assert signal.take_profit_distance > signal.stop_distance

    def test_short_signal(self):
        engine = TrendFollower()
        features = _make_features(
            ema_8=39000, ema_21=40000, ema_55=41000,
            adx_14=30, macd_hist=-50,
        )
        regime = _make_regime(Regime.TRENDING_LOW_VOL)
        signal = engine.generate_signal(regime=regime, features=features, close_price=39000)

        assert signal is not None
        assert signal.bias == Bias.SHORT

    def test_no_signal_weak_trend(self):
        engine = TrendFollower()
        features = _make_features(adx_14=15)  # weak trend
        regime = _make_regime(Regime.TRENDING_LOW_VOL)
        signal = engine.generate_signal(regime=regime, features=features, close_price=42000)

        assert signal is None

    def test_no_signal_wrong_regime(self):
        engine = TrendFollower()
        features = _make_features()
        regime = _make_regime(Regime.RANGING_LOW_VOL)
        signal = engine.generate_signal(regime=regime, features=features, close_price=42000)

        assert signal is None


class TestMeanReversion:
    def test_long_signal_oversold(self):
        engine = MeanReversion()
        features = _make_features(
            rsi_14=25, bb_pct_b=-0.1, zscore_close_20=-2.5,
            hurst_exponent=0.35,
        )
        regime = _make_regime(Regime.RANGING_LOW_VOL)
        signal = engine.generate_signal(regime=regime, features=features, close_price=42000)

        assert signal is not None
        assert signal.bias == Bias.LONG

    def test_no_signal_trending_hurst(self):
        engine = MeanReversion()
        features = _make_features(
            rsi_14=25, bb_pct_b=-0.1, zscore_close_20=-2.5,
            hurst_exponent=0.65,  # trending, not mean-reverting
        )
        regime = _make_regime(Regime.RANGING_LOW_VOL)
        signal = engine.generate_signal(regime=regime, features=features, close_price=42000)

        assert signal is None

    def test_no_signal_wrong_regime(self):
        engine = MeanReversion()
        features = _make_features(rsi_14=25, bb_pct_b=-0.1, zscore_close_20=-2.5, hurst_exponent=0.35)
        regime = _make_regime(Regime.TRENDING_LOW_VOL)
        signal = engine.generate_signal(regime=regime, features=features, close_price=42000)

        assert signal is None


class TestMomentumScalper:
    def test_long_signal(self):
        engine = MomentumScalper()
        features = _make_features(
            volume_ratio=3.0, rsi_14=65, roc_10=3.0, bb_pct_b=0.9,
        )
        regime = _make_regime(Regime.TRENDING_LOW_VOL)
        signal = engine.generate_signal(
            regime=regime, features=features, close_price=42000,
            account_balance=1000,
        )

        assert signal is not None
        assert signal.bias == Bias.LONG

    def test_blocked_low_balance(self):
        engine = MomentumScalper()
        features = _make_features(volume_ratio=3.0, rsi_14=65, roc_10=3.0, bb_pct_b=0.9)
        regime = _make_regime(Regime.TRENDING_LOW_VOL)
        signal = engine.generate_signal(
            regime=regime, features=features, close_price=42000,
            account_balance=100,  # below $500 threshold
        )

        assert signal is None

    def test_no_volume_spike(self):
        engine = MomentumScalper()
        features = _make_features(volume_ratio=1.2)  # no spike
        regime = _make_regime(Regime.TRENDING_LOW_VOL)
        signal = engine.generate_signal(
            regime=regime, features=features, close_price=42000,
            account_balance=1000,
        )

        assert signal is None
