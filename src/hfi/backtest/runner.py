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
        # Entry: EMA alignment + ADX + MACD
        long_entry = (
            (df["ema_8"] > df["ema_21"]) &
            (df["ema_21"] > df["ema_55"]) &
            (df["adx_14"] > tc.adx_threshold) &
            (df["macd_hist"] > 0)
        )
        # Only enter on crossover (not while already aligned)
        entries = long_entry & ~long_entry.shift(1).fillna(False)

        # Exit: EMA cross back or ATR trailing (simplified)
        exits = (
            (df["ema_8"] < df["ema_21"]) |
            (df["macd_hist"] < 0)
        )

    elif engine_name == "MEAN_REVERSION":
        mc = config.mean_reversion
        # Entry: RSI oversold + below BB + zscore extreme
        long_entry = (
            (df["rsi_14"] < mc.rsi_oversold) &
            (df["bb_pct_b"] < 0.0) &
            (df["zscore_close_20"] < mc.zscore_entry)
        )
        entries = long_entry & ~long_entry.shift(1).fillna(False)

        # Exit: RSI crosses 50 or price crosses middle BB
        exits = (
            (df["rsi_14"] > mc.rsi_exit) |
            (df["close"] > df["bb_mid"])
        )

    elif engine_name == "MOMENTUM_SCALPER":
        sc = config.momentum_scalper
        # Entry: Volume spike + RSI momentum + breakout
        long_entry = (
            (df["volume_ratio"] > sc.volume_mult) &
            (df["rsi_14"] > sc.rsi_threshold) &
            (df["roc_10"] > 0) &
            (df["bb_pct_b"] > 0.8)
        )
        entries = long_entry & ~long_entry.shift(1).fillna(False)

        # Exit: quick TP or RSI weakness
        exits = (df["rsi_14"] < 50) | (df["roc_10"] < 0)

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
    """Simulate trades and compute performance metrics."""
    capital = initial_capital
    equity_curve = [capital]
    trades: list[dict] = []
    in_position = False
    entry_price = 0.0
    entry_idx = None
    position_size = 0.0
    max_equity = capital

    for i in range(1, len(df)):
        if not in_position and entries.iloc[i]:
            # Enter trade
            entry_price = df["close"].iloc[i] * (1 + slippage_rate)  # slippage on entry
            risk_pct = 0.02  # 2% risk
            stop_dist = df["atr_14"].iloc[i] * 2 / entry_price
            position_size = min(
                capital * risk_pct / max(stop_dist, 0.001),
                capital * leverage * 0.30,  # max 30% position
            )
            fee = position_size * fee_rate
            capital -= fee
            in_position = True
            entry_idx = i

        elif in_position and (exits.iloc[i] or i == len(df) - 1):
            # Exit trade
            exit_price = df["close"].iloc[i] * (1 - slippage_rate)  # slippage on exit
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl = position_size * pnl_pct
            fee = position_size * fee_rate
            capital += pnl - fee

            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl - fee * 2,  # both entry and exit fees
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
