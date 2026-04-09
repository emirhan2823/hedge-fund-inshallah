"""Robustness validator — anti-overfit testing suite.

Tests strategy across random market periods with Monte Carlo analysis.
Ensures strategy doesn't just work on one specific time window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from hfi.backtest.market_conditions import MarketConditionLabeler
from hfi.backtest.random_sampler import RandomPeriodSampler, SampledPeriod
from hfi.backtest.runner import BacktestResult, run_backtest
from hfi.core.config import HFIConfig

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Results of robustness testing."""
    # Overall
    pass_rate: float = 0.0
    avg_return: float = 0.0
    std_return: float = 0.0
    avg_sharpe: float = 0.0
    avg_max_dd: float = 0.0
    avg_win_rate: float = 0.0
    avg_profit_factor: float = 0.0
    total_trades: int = 0

    # Condition breakdown
    condition_breakdown: dict[str, dict] = field(default_factory=dict)

    # Overfit detection
    is_robust: bool = False
    is_overfit: bool = False
    oos_degradation_pct: float = 0.0
    mc_dd_95th: float = 0.0
    mc_dd_actual: float = 0.0

    # Per-iteration details
    iterations: list[dict] = field(default_factory=list)

    # Summary
    verdict: str = ""


class RobustnessValidator:
    """Anti-overfit testing orchestrator."""

    def __init__(
        self,
        labeler: MarketConditionLabeler | None = None,
        sampler: RandomPeriodSampler | None = None,
    ) -> None:
        self._labeler = labeler or MarketConditionLabeler()
        self._sampler = sampler or RandomPeriodSampler()

    def run_robustness_test(
        self,
        hourly_data: dict[str, pd.DataFrame],
        daily_data: pd.DataFrame,
        config: HFIConfig,
        engine_name: str,
        symbol: str,
        n_iterations: int = 20,
        n_samples_per_iter: int = 5,
        seed: int | None = None,
    ) -> RobustnessResult:
        """Run full robustness test suite.

        Args:
            hourly_data: Dict of {symbol: hourly_ohlcv_df} for backtesting
            daily_data: Daily OHLCV for market condition labeling
            config: HFI config
            engine_name: Engine to test
            symbol: Primary symbol
            n_iterations: Number of random sampling iterations
            n_samples_per_iter: Periods per iteration
            seed: Random seed for reproducibility
        """
        # Step 1: Label market conditions
        labeled = self._labeler.label_conditions(daily_data)
        periods = self._labeler.get_periods(labeled)

        if not periods:
            logger.error("No valid periods found for robustness testing")
            return RobustnessResult(verdict="FAIL: No valid periods in data")

        # Step 2: Run iterations
        all_results: list[dict] = []
        all_trades_pnls: list[float] = []

        for iteration in range(n_iterations):
            iter_seed = (seed + iteration) if seed is not None else None
            samples = self._sampler.sample_periods(
                periods, n_samples=n_samples_per_iter,
                min_days=21, max_days=90, seed=iter_seed,
            )

            iter_results: list[BacktestResult] = []
            iter_conditions: list[str] = []

            for sample in samples:
                # Extract hourly data for this period
                df = hourly_data.get(symbol)
                if df is None or df.empty:
                    continue

                period_df = df[sample.start:sample.end]
                if len(period_df) < 100:
                    continue

                result = run_backtest(period_df.copy(), config, engine_name, symbol=symbol)
                if result.total_trades > 0:
                    iter_results.append(result)
                    iter_conditions.append(sample.condition)
                    all_trades_pnls.extend(t.get("pnl_usd", t.get("pnl", 0)) for t in result.trades)

            # Iteration summary
            if iter_results:
                iter_return = np.mean([r.total_return for r in iter_results])
                iter_sharpe = np.mean([r.sharpe_ratio for r in iter_results])
                iter_mdd = np.max([r.max_drawdown for r in iter_results])
                iter_trades = sum(r.total_trades for r in iter_results)
                iter_wr = np.mean([r.win_rate for r in iter_results])
                iter_pf = np.mean([r.profit_factor for r in iter_results])
            else:
                iter_return = iter_sharpe = iter_mdd = iter_wr = iter_pf = 0.0
                iter_trades = 0

            all_results.append({
                "iteration": iteration,
                "n_periods": len(iter_results),
                "conditions": iter_conditions,
                "total_return": iter_return,
                "sharpe": iter_sharpe,
                "max_dd": iter_mdd,
                "total_trades": iter_trades,
                "win_rate": iter_wr,
                "profit_factor": iter_pf,
                "passed": iter_sharpe > 0 and iter_pf > 1.0 and iter_trades >= 5,
            })

        # Step 3: Aggregate
        if not all_results:
            return RobustnessResult(verdict="FAIL: No results generated")

        passed = [r for r in all_results if r["passed"]]
        pass_rate = len(passed) / len(all_results)

        active = [r for r in all_results if r["total_trades"] > 0]
        avg_return = np.mean([r["total_return"] for r in active]) if active else 0.0
        std_return = np.std([r["total_return"] for r in active]) if active else 0.0
        avg_sharpe = np.mean([r["sharpe"] for r in active]) if active else 0.0
        avg_mdd = np.mean([r["max_dd"] for r in active]) if active else 0.0
        avg_wr = np.mean([r["win_rate"] for r in active]) if active else 0.0
        avg_pf = np.mean([r["profit_factor"] for r in active]) if active else 0.0

        # Condition breakdown
        cond_breakdown: dict[str, dict] = {}
        for cond in ["BULL", "BEAR", "RANGING", "CRASH"]:
            cond_iters = []
            for r in all_results:
                for i, c in enumerate(r.get("conditions", [])):
                    if c == cond:
                        cond_iters.append(r)
                        break
            if cond_iters:
                cond_breakdown[cond] = {
                    "n_samples": len(cond_iters),
                    "avg_return": float(np.mean([r["total_return"] for r in cond_iters])),
                    "avg_sharpe": float(np.mean([r["sharpe"] for r in cond_iters])),
                    "pass_rate": sum(1 for r in cond_iters if r["passed"]) / len(cond_iters),
                }

        # Step 4: Monte Carlo trade shuffle
        mc_dd_95th, mc_dd_actual = self._monte_carlo_dd(all_trades_pnls)

        # Step 5: IS/OOS degradation (simplified)
        oos_deg = self._estimate_oos_degradation(all_results)

        # Step 6: Verdict
        is_robust = pass_rate >= 0.60 and std_return < abs(avg_return) * 3
        is_overfit = oos_deg > 0.50 or (mc_dd_actual > mc_dd_95th and mc_dd_95th > 0)

        if is_robust and not is_overfit:
            verdict = f"ROBUST: {pass_rate:.0%} pass rate, consistent across conditions"
        elif is_robust and is_overfit:
            verdict = f"CAUTION: {pass_rate:.0%} pass rate but overfit signals detected"
        elif not is_robust and not is_overfit:
            verdict = f"WEAK: {pass_rate:.0%} pass rate, strategy needs improvement"
        else:
            verdict = f"OVERFIT: {pass_rate:.0%} pass rate with overfit confirmed"

        return RobustnessResult(
            pass_rate=float(pass_rate),
            avg_return=float(avg_return),
            std_return=float(std_return),
            avg_sharpe=float(avg_sharpe),
            avg_max_dd=float(avg_mdd),
            avg_win_rate=float(avg_wr),
            avg_profit_factor=float(avg_pf),
            total_trades=sum(r["total_trades"] for r in all_results),
            condition_breakdown=cond_breakdown,
            is_robust=is_robust,
            is_overfit=is_overfit,
            oos_degradation_pct=float(oos_deg),
            mc_dd_95th=float(mc_dd_95th),
            mc_dd_actual=float(mc_dd_actual),
            iterations=all_results,
            verdict=verdict,
        )

    @staticmethod
    def _monte_carlo_dd(pnls: list[float], n_shuffles: int = 1000) -> tuple[float, float]:
        """Monte Carlo max drawdown test.

        Shuffles trade PnLs randomly and computes max DD distribution.
        If actual DD > 95th percentile of shuffled DDs → path-dependent (fragile).
        """
        if len(pnls) < 5:
            return 0.0, 0.0

        arr = np.array(pnls)

        # Actual max drawdown from sequential PnLs
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        actual_dd = float(np.max(peak - cum))

        # Shuffled max drawdowns
        rng = np.random.RandomState(42)
        shuffled_dds = []
        for _ in range(n_shuffles):
            rng.shuffle(arr)
            cum = np.cumsum(arr)
            peak = np.maximum.accumulate(cum)
            shuffled_dds.append(float(np.max(peak - cum)))

        dd_95th = float(np.percentile(shuffled_dds, 95))

        return dd_95th, actual_dd

    @staticmethod
    def _estimate_oos_degradation(results: list[dict]) -> float:
        """Estimate in-sample vs out-of-sample degradation.

        Uses first half vs second half of iterations as proxy.
        """
        if len(results) < 4:
            return 0.0

        mid = len(results) // 2
        first_half = [r for r in results[:mid] if r["total_trades"] > 0]
        second_half = [r for r in results[mid:] if r["total_trades"] > 0]

        if not first_half or not second_half:
            return 0.0

        is_return = np.mean([r["total_return"] for r in first_half])
        oos_return = np.mean([r["total_return"] for r in second_half])

        if is_return <= 0:
            return 0.0

        degradation = (is_return - oos_return) / abs(is_return) if is_return != 0 else 0.0
        return max(0.0, degradation)
