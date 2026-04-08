"""Backtest runner — vectorbt-based strategy backtesting.

Runs strategies on historical OHLCV data with realistic fees and slippage.
Skill ref: vectorbt.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from hfi.core.config import BacktestConfig, HFIConfig
from hfi.features.builder import build_features_df
from hfi.regime.detector import detect_regime
from hfi.core.types import FeatureVector, RegimeState

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Summary of backtest results."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_bars_held: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    config: HFIConfig,
    engine_name: str = "TREND_FOLLOWER",
) -> BacktestResult:
    """Run backtest on OHLCV DataFrame.

    Uses simple vectorized approach:
    1. Compute features
    2. Generate entry/exit signals
    3. Simulate trades with fees + slippage
    """
    bc = config.backtest

    # Build features
    df = build_features_df(df.copy())
    df = df.dropna()

    if len(df) < 100:
        logger.warning("Not enough data for backtest: %d rows", len(df))
        return BacktestResult()

    # Generate signals based on engine
    entries, exits = _generate_signals(df, engine_name, config)

    if entries.sum() == 0:
        logger.warning("No entry signals generated for %s", engine_name)
        return BacktestResult()

    # Simulate trades
    result = _simulate_trades(
        df, entries, exits,
        initial_capital=bc.initial_capital,
        fee_rate=bc.fee_rate,
        slippage_rate=bc.slippage_rate,
        leverage=config.leverage.get_leverage(bc.initial_capital),
    )

    return result


def _generate_signals(
    df: pd.DataFrame,
    engine_name: str,
    config: HFIConfig,
) -> tuple[pd.Series, pd.Series]:
    """Generate boolean entry/exit signals."""
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)

    if engine_name == "TREND_FOLLOWER":
        tc = config.trend_follower
        # LONG: EMA alignment + ADX + MACD
        long_entry = (
            (df["ema_8"] > df["ema_21"]) &
            (df["ema_21"] > df["ema_55"]) &
            (df["adx_14"] > tc.adx_threshold) &
            (df["macd_hist"] > 0)
        )
        # SHORT: reversed EMA alignment
        short_entry = (
            (df["ema_8"] < df["ema_21"]) &
            (df["ema_21"] < df["ema_55"]) &
            (df["adx_14"] > tc.adx_threshold) &
            (df["macd_hist"] < 0)
        )
        # Enter on crossover (long or short)
        entries = (
            (long_entry & ~long_entry.shift(1, fill_value=False)) |
            (short_entry & ~short_entry.shift(1, fill_value=False))
        )
        # Track direction for simulation
        df["_signal_dir"] = 0
        df.loc[long_entry, "_signal_dir"] = 1
        df.loc[short_entry, "_signal_dir"] = -1

        # Exit: EMA cross back
        exits = (
            (df["ema_8"] < df["ema_21"]) & (df["_signal_dir"].shift(1, fill_value=0) == 1)
        ) | (
            (df["ema_8"] > df["ema_21"]) & (df["_signal_dir"].shift(1, fill_value=0) == -1)
        )

    elif engine_name == "MEAN_REVERSION":
        mc = config.mean_reversion
        # Relaxed conditions: 2 of 3 extremity signals sufficient
        rsi_oversold = df["rsi_14"] < 35  # relaxed from 30
        bb_low = df["bb_pct_b"] < 0.15    # relaxed from 0.0
        zscore_low = df["zscore_close_20"] < -1.5  # relaxed from -2.0

        rsi_overbought = df["rsi_14"] > 65  # relaxed from 70
        bb_high = df["bb_pct_b"] > 0.85
        zscore_high = df["zscore_close_20"] > 1.5

        # LONG: at least 2 of 3 oversold signals
        long_score = rsi_oversold.astype(int) + bb_low.astype(int) + zscore_low.astype(int)
        long_entry = long_score >= 2

        # SHORT: at least 2 of 3 overbought signals
        short_score = rsi_overbought.astype(int) + bb_high.astype(int) + zscore_high.astype(int)
        short_entry = short_score >= 2

        entries = (
            (long_entry & ~long_entry.shift(1, fill_value=False)) |
            (short_entry & ~short_entry.shift(1, fill_value=False))
        )
        df["_signal_dir"] = 0
        df.loc[long_entry, "_signal_dir"] = 1
        df.loc[short_entry, "_signal_dir"] = -1

        # Exit: RSI crosses 50 (mean reversion target)
        exits = (
            ((df["rsi_14"] > 50) & (df["rsi_14"].shift(1) <= 50)) |
            ((df["rsi_14"] < 50) & (df["rsi_14"].shift(1) >= 50))
        )

    elif engine_name == "MOMENTUM_SCALPER":
        sc = config.momentum_scalper
        # Relaxed: volume spike + (RSI OR breakout)
        vol_spike = df["volume_ratio"] > 1.8  # relaxed from 2.0

        # LONG: volume spike + momentum up
        long_entry = (
            vol_spike &
            (df["rsi_14"] > 55) &  # relaxed from 60
            (df["roc_10"] > 0.5)   # require meaningful momentum
        )
        # SHORT: volume spike + momentum down
        short_entry = (
            vol_spike &
            (df["rsi_14"] < 45) &
            (df["roc_10"] < -0.5)
        )

        entries = (
            (long_entry & ~long_entry.shift(1, fill_value=False)) |
            (short_entry & ~short_entry.shift(1, fill_value=False))
        )
        df["_signal_dir"] = 0
        df.loc[long_entry, "_signal_dir"] = 1
        df.loc[short_entry, "_signal_dir"] = -1

        # Exit: quick — RSI mean reversion or momentum flip
        exits = (
            ((df["rsi_14"] < 50) & (df["_signal_dir"].shift(1, fill_value=0) == 1)) |
            ((df["rsi_14"] > 50) & (df["_signal_dir"].shift(1, fill_value=0) == -1))
        )

    return entries, exits


def _simulate_trades(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    initial_capital: float = 100.0,
    fee_rate: float = 0.0006,
    slippage_rate: float = 0.0005,
    leverage: int = 3,
) -> BacktestResult:
    """Simulate trades (long + short) and compute performance metrics."""
    capital = initial_capital
    equity_curve = [capital]
    trades: list[dict] = []
    in_position = False
    entry_price = 0.0
    entry_idx = None
    position_size = 0.0
    trade_dir = 1  # 1 = long, -1 = short
    max_equity = capital

    # Get signal direction if available
    has_dir = "_signal_dir" in df.columns

    for i in range(1, len(df)):
        if not in_position and entries.iloc[i]:
            # Determine direction
            trade_dir = int(df["_signal_dir"].iloc[i]) if has_dir else 1
            if trade_dir == 0:
                trade_dir = 1  # default long

            # Enter trade (slippage against us)
            slip = slippage_rate if trade_dir == 1 else -slippage_rate
            entry_price = df["close"].iloc[i] * (1 + slip)

            risk_pct = 0.02
            atr_val = df["atr_14"].iloc[i] if "atr_14" in df.columns else entry_price * 0.02
            stop_dist = atr_val * 2 / entry_price
            position_size = min(
                capital * risk_pct / max(stop_dist, 0.001),
                capital * leverage * 0.30,
            )
            # Calculate stop loss price
            if trade_dir == 1:
                stop_price = entry_price * (1 - stop_dist)
                tp_price = entry_price * (1 + stop_dist * 3)  # 3:1 R:R
            else:
                stop_price = entry_price * (1 + stop_dist)
                tp_price = entry_price * (1 - stop_dist * 3)

            fee = position_size * fee_rate
            capital -= fee
            in_position = True
            entry_idx = i

        elif in_position:
            current_close = df["close"].iloc[i]

            # Check stop loss and take profit
            hit_stop = (trade_dir == 1 and current_close <= stop_price) or \
                       (trade_dir == -1 and current_close >= stop_price)
            hit_tp = (trade_dir == 1 and current_close >= tp_price) or \
                     (trade_dir == -1 and current_close <= tp_price)
            hit_exit_signal = exits.iloc[i]
            is_last = i == len(df) - 1

            if not (hit_stop or hit_tp or hit_exit_signal or is_last):
                equity_curve.append(capital)
                continue

            # Exit trade (slippage against us)
            if hit_stop:
                exit_price = stop_price  # stopped out at stop price
            elif hit_tp:
                exit_price = tp_price  # TP hit
            else:
                slip = -slippage_rate if trade_dir == 1 else slippage_rate
                exit_price = df["close"].iloc[i] * (1 + slip)

            # PnL depends on direction
            if trade_dir == 1:  # long
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - exit_price) / entry_price

            pnl = position_size * pnl_pct
            fee = position_size * fee_rate
            capital += pnl - fee

            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": "long" if trade_dir == 1 else "short",
                "pnl": pnl - fee * 2,
                "pnl_pct": pnl_pct,
                "bars_held": i - entry_idx,
                "entry_time": str(df.index[entry_idx]),
                "exit_time": str(df.index[i]),
            })

            in_position = False
            entry_price = 0.0
            position_size = 0.0

        max_equity = max(max_equity, capital)
        equity_curve.append(capital)

    # Compute metrics
    if not trades:
        return BacktestResult(equity_curve=equity_curve)

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (capital - initial_capital) / initial_capital

    # Sharpe (annualized, assuming 1h candles)
    if len(pnls) > 1:
        returns = np.array(pnls) / initial_capital
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 24)
        downside = returns[returns < 0]
        sortino = np.mean(returns) / (np.std(downside) + 1e-10) * np.sqrt(365 * 24) if len(downside) > 0 else sharpe
    else:
        sharpe = sortino = 0.0

    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = float(np.max(dd))

    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=max_dd,
        win_rate=len(wins) / len(trades) if trades else 0.0,
        profit_factor=profit_factor,
        total_trades=len(trades),
        avg_trade_pnl=float(np.mean(pnls)),
        avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
        best_trade=max(pnls) if pnls else 0.0,
        worst_trade=min(pnls) if pnls else 0.0,
        avg_bars_held=float(np.mean([t["bars_held"] for t in trades])),
        equity_curve=equity_curve,
        trades=trades,
    )
