"""Walk-forward validation + overfit detection.

Rolling window validation with Deflated Sharpe Ratio.
Skill ref: walk-forward-validation.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hfi.backtest.runner import BacktestResult, run_backtest
from hfi.core.config import HFIConfig

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Walk-forward validation results."""

    oos_sharpe: float = 0.0          # Out-of-sample Sharpe
    oos_total_return: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_total_trades: int = 0
    num_folds: int = 0
    fold_results: list[BacktestResult] = field(default_factory=list)
    deflated_sharpe: float = 0.0     # Overfit-adjusted Sharpe
    is_overfit: bool = False


def walk_forward_validate(
    df: pd.DataFrame,
    config: HFIConfig,
    engine_name: str = "TREND_FOLLOWER",
    train_days: int = 90,
    test_days: int = 14,
    step_days: int = 14,
) -> WalkForwardResult:
    """Run rolling walk-forward validation.

    1. Split data into train/test folds
    2. Train (optimize) on train set
    3. Test on out-of-sample test set
    4. Aggregate OOS metrics
    5. Check for overfitting via Deflated Sharpe
    """
    # Determine timeframe from data frequency
    if len(df) < 2:
        return WalkForwardResult()

    freq = pd.infer_freq(df.index)
    bars_per_day = _bars_per_day(df)

    train_bars = int(train_days * bars_per_day)
    test_bars = int(test_days * bars_per_day)
    step_bars = int(step_days * bars_per_day)

    if len(df) < train_bars + test_bars:
        logger.warning(
            "Not enough data for walk-forward: %d bars (need %d+%d)",
            len(df), train_bars, test_bars,
        )
        return WalkForwardResult()

    # Generate folds
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    start = 0
    while start + train_bars + test_bars <= len(df):
        train_df = df.iloc[start:start + train_bars]
        test_df = df.iloc[start + train_bars:start + train_bars + test_bars]
        folds.append((train_df, test_df))
        start += step_bars

    if not folds:
        return WalkForwardResult()

    logger.info("Walk-forward: %d folds (train=%d, test=%d bars)", len(folds), train_bars, test_bars)

    # Run OOS backtests
    oos_results: list[BacktestResult] = []
    for i, (train_df, test_df) in enumerate(folds):
        # For now, use fixed config (no optimization per fold)
        # Future: optimize params on train_df, test on test_df
        result = run_backtest(test_df, config, engine_name)
        oos_results.append(result)

        logger.info(
            "Fold %d/%d: trades=%d, return=%.2f%%, sharpe=%.2f, MDD=%.2f%%",
            i + 1, len(folds), result.total_trades,
            result.total_return * 100, result.sharpe_ratio, result.max_drawdown * 100,
        )

    # Aggregate OOS metrics
    total_trades = sum(r.total_trades for r in oos_results)
    if total_trades == 0:
        return WalkForwardResult(num_folds=len(folds), fold_results=oos_results)

    # Weighted average metrics
    valid_results = [r for r in oos_results if r.total_trades > 0]
    if not valid_results:
        return WalkForwardResult(num_folds=len(folds), fold_results=oos_results)

    avg_sharpe = np.mean([r.sharpe_ratio for r in valid_results])
    avg_return = np.mean([r.total_return for r in valid_results])
    avg_mdd = np.max([r.max_drawdown for r in valid_results])
    avg_wr = np.mean([r.win_rate for r in valid_results])
    avg_pf = np.mean([r.profit_factor for r in valid_results])

    # Deflated Sharpe Ratio (simplified)
    dsr = _deflated_sharpe(
        sharpe_observed=avg_sharpe,
        num_trials=len(folds),
        num_trades=total_trades,
        skew=0.0,  # simplified: assume normal
        kurtosis=3.0,
    )

    is_overfit = dsr < 0.95

    result = WalkForwardResult(
        oos_sharpe=float(avg_sharpe),
        oos_total_return=float(avg_return),
        oos_max_drawdown=float(avg_mdd),
        oos_win_rate=float(avg_wr),
        oos_profit_factor=float(avg_pf),
        oos_total_trades=total_trades,
        num_folds=len(folds),
        fold_results=oos_results,
        deflated_sharpe=float(dsr),
        is_overfit=is_overfit,
    )

    logger.info(
        "Walk-forward summary: OOS Sharpe=%.2f, DSR=%.2f, overfit=%s, trades=%d",
        avg_sharpe, dsr, is_overfit, total_trades,
    )

    return result


def _bars_per_day(df: pd.DataFrame) -> float:
    """Estimate bars per day from data."""
    if len(df) < 2:
        return 24.0  # assume hourly

    total_seconds = (df.index[-1] - df.index[0]).total_seconds()
    total_bars = len(df)
    seconds_per_bar = total_seconds / total_bars
    return 86400 / seconds_per_bar


def _deflated_sharpe(
    sharpe_observed: float,
    num_trials: int,
    num_trades: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Deflated Sharpe Ratio (simplified Bailey & Lopez de Prado).

    Adjusts for multiple testing, non-normality, and sample size.
    DSR < 0.95 suggests overfitting.
    """
    from scipy import stats

    if num_trades < 10 or num_trials < 2:
        return 0.0

    # Expected maximum Sharpe under null (multiple testing adjustment)
    e_max_sharpe = stats.norm.ppf(1 - 1 / num_trials) if num_trials > 1 else 0.0

    # Standard error of Sharpe
    se_sharpe = np.sqrt(
        (1 - skew * sharpe_observed + (kurtosis - 1) / 4 * sharpe_observed**2)
        / (num_trades - 1)
    )

    if se_sharpe <= 0:
        return 0.0

    # DSR = P(SR* < SR_observed)
    test_stat = (sharpe_observed - e_max_sharpe) / se_sharpe
    dsr = float(stats.norm.cdf(test_stat))

    return dsr
