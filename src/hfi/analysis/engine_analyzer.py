"""Engine-specific performance analyzer.

Analyzes each engine's trades with metrics tailored to its strategy type.
Generates actionable improvement recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EngineReport:
    """Analysis report for a single engine."""
    engine: str
    symbol: str
    total_trades: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    parameter_suggestions: dict[str, Any] = field(default_factory=dict)
    direction_stats: dict[str, dict] = field(default_factory=dict)
    exit_reason_stats: dict[str, dict] = field(default_factory=dict)


class EngineAnalyzer:
    """Dispatches to engine-specific analyzers."""

    def analyze(
        self,
        trades: list[dict],
        engine_name: str,
        symbol: str,
        ohlcv_df: pd.DataFrame | None = None,
    ) -> EngineReport:
        """Analyze trades for a specific engine.

        Args:
            trades: List of enriched trade dicts (from BacktestResult.trades)
            engine_name: Which engine generated these trades
            symbol: Trading pair
            ohlcv_df: Original OHLCV data for context (optional)
        """
        if not trades:
            return EngineReport(engine=engine_name, symbol=symbol)

        # Common analysis
        report = EngineReport(
            engine=engine_name,
            symbol=symbol,
            total_trades=len(trades),
        )

        # Common metrics
        self._common_metrics(trades, report)

        # Engine-specific
        if engine_name == "TREND_FOLLOWER":
            self._analyze_trend_follower(trades, report, ohlcv_df)
        elif engine_name == "MEAN_REVERSION":
            self._analyze_mean_reversion(trades, report, ohlcv_df)
        elif engine_name == "MOMENTUM_SCALPER":
            self._analyze_momentum_scalper(trades, report, ohlcv_df)

        return report

    def _common_metrics(self, trades: list[dict], report: EngineReport) -> None:
        """Compute metrics common to all engines."""
        pnls = [t.get("pnl_usd", t.get("pnl", 0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        report.metrics["total_pnl"] = sum(pnls)
        report.metrics["win_rate"] = len(wins) / len(trades) if trades else 0
        report.metrics["avg_win"] = np.mean(wins) if wins else 0
        report.metrics["avg_loss"] = np.mean(losses) if losses else 0
        report.metrics["payoff_ratio"] = abs(np.mean(wins) / np.mean(losses)) if losses and wins else 0
        report.metrics["expectancy"] = np.mean(pnls) if pnls else 0

        # Direction breakdown
        longs = [t for t in trades if t.get("direction") == "long"]
        shorts = [t for t in trades if t.get("direction") == "short"]

        for label, subset in [("long", longs), ("short", shorts)]:
            if subset:
                s_pnls = [t.get("pnl_usd", t.get("pnl", 0)) for t in subset]
                s_wins = sum(1 for p in s_pnls if p > 0)
                report.direction_stats[label] = {
                    "count": len(subset),
                    "win_rate": s_wins / len(subset),
                    "avg_pnl": np.mean(s_pnls),
                    "total_pnl": sum(s_pnls),
                }

        # Exit reason breakdown
        for t in trades:
            reason = t.get("exit_reason", "unknown")
            if reason not in report.exit_reason_stats:
                report.exit_reason_stats[reason] = {"count": 0, "wins": 0, "total_pnl": 0}
            report.exit_reason_stats[reason]["count"] += 1
            pnl = t.get("pnl_usd", t.get("pnl", 0))
            report.exit_reason_stats[reason]["total_pnl"] += pnl
            if pnl > 0:
                report.exit_reason_stats[reason]["wins"] += 1

    def _analyze_trend_follower(
        self, trades: list[dict], report: EngineReport, ohlcv_df: pd.DataFrame | None,
    ) -> None:
        """Trend Follower specific analysis."""
        bars_held = [t.get("bars_held", 0) for t in trades]
        pnls = [t.get("pnl_usd", t.get("pnl", 0)) for t in trades]

        # 1. Holding duration analysis
        report.metrics["median_bars_held"] = float(np.median(bars_held)) if bars_held else 0
        report.metrics["pct_held_under_10"] = sum(1 for b in bars_held if b < 10) / len(bars_held) if bars_held else 0
        report.metrics["pct_held_over_100"] = sum(1 for b in bars_held if b > 100) / len(bars_held) if bars_held else 0

        # 2. Stop loss hit rate
        sl_trades = [t for t in trades if t.get("exit_reason") == "stop_loss"]
        report.metrics["stop_loss_hit_rate"] = len(sl_trades) / len(trades) if trades else 0

        # 3. Take profit hit rate
        tp_trades = [t for t in trades if t.get("exit_reason") == "take_profit"]
        report.metrics["take_profit_hit_rate"] = len(tp_trades) / len(trades) if trades else 0

        # 4. R:R achieved vs target (3:1)
        for t in trades:
            entry = t.get("entry_price", 0)
            sl = t.get("stop_loss", 0)
            if entry > 0 and sl > 0 and entry != sl:
                risk = abs(entry - sl)
                reward = abs(t.get("pnl_usd", 0))  # crude but directional
                t["_rr_actual"] = reward / risk if risk > 0 else 0

        rr_actuals = [t.get("_rr_actual", 0) for t in trades if t.get("pnl_usd", 0) > 0]
        report.metrics["avg_rr_achieved"] = float(np.mean(rr_actuals)) if rr_actuals else 0

        # 5. Direction accuracy
        if "long" in report.direction_stats:
            report.metrics["long_win_rate"] = report.direction_stats["long"]["win_rate"]
        if "short" in report.direction_stats:
            report.metrics["short_win_rate"] = report.direction_stats["short"]["win_rate"]

        # 6. ADX at entry analysis
        adx_values = [t.get("adx_at_entry", 0) for t in trades if t.get("adx_at_entry")]
        if adx_values:
            report.metrics["avg_adx_at_entry"] = float(np.mean(adx_values))
            # Win rate by ADX bucket
            for lo, hi, label in [(25, 35, "25-35"), (35, 50, "35-50"), (50, 100, "50+")]:
                bucket = [t for t in trades if lo <= t.get("adx_at_entry", 0) < hi]
                if bucket:
                    wins = sum(1 for t in bucket if t.get("pnl_usd", 0) > 0)
                    report.metrics[f"wr_adx_{label}"] = wins / len(bucket)

        # Recommendations
        if report.metrics.get("stop_loss_hit_rate", 0) > 0.60:
            report.recommendations.append(
                f"Stop loss hit rate too high ({report.metrics['stop_loss_hit_rate']:.0%}). "
                f"Consider widening ATR multiplier from 2.0 to 2.5."
            )
            report.parameter_suggestions["atr_stop_mult"] = 2.5

        if report.metrics.get("pct_held_under_10", 0) > 0.40:
            report.recommendations.append(
                f"{report.metrics['pct_held_under_10']:.0%} of trades held < 10 bars. "
                f"Exit signals may be too aggressive. Consider using EMA(21) cross instead of EMA(8)."
            )

        if report.metrics.get("long_win_rate", 0.5) < 0.35:
            report.recommendations.append(
                f"Long win rate is only {report.metrics.get('long_win_rate', 0):.0%}. "
                f"Consider adding stronger long confirmation or restricting longs in bearish EMAs."
            )

        if report.metrics.get("short_win_rate", 0.5) < 0.35:
            report.recommendations.append(
                f"Short win rate is only {report.metrics.get('short_win_rate', 0):.0%}. "
                f"Consider adding stronger short confirmation."
            )

        if report.metrics.get("avg_rr_achieved", 0) < 2.0:
            report.recommendations.append(
                f"Avg R:R achieved is {report.metrics.get('avg_rr_achieved', 0):.1f}:1 vs target 3:1. "
                f"Exits may be too early. Consider trailing stop instead of fixed TP."
            )

    def _analyze_mean_reversion(
        self, trades: list[dict], report: EngineReport, ohlcv_df: pd.DataFrame | None,
    ) -> None:
        """Mean Reversion specific analysis."""
        bars_held = [t.get("bars_held", 0) for t in trades]
        pnls = [t.get("pnl_usd", t.get("pnl", 0)) for t in trades]

        # 1. Reversion speed (bars held for winning trades)
        winning_bars = [t.get("bars_held", 0) for t in trades if t.get("pnl_usd", t.get("pnl", 0)) > 0]
        report.metrics["median_win_bars"] = float(np.median(winning_bars)) if winning_bars else 0
        report.metrics["pct_fast_wins"] = sum(1 for b in winning_bars if b < 20) / len(winning_bars) if winning_bars else 0

        # 2. Z-score bucket analysis
        for lo, hi, label in [(-3.0, -2.0, "-3to-2"), (-2.0, -1.5, "-2to-1.5"), (-1.5, -1.0, "-1.5to-1"),
                               (1.0, 1.5, "1to1.5"), (1.5, 2.0, "1.5to2"), (2.0, 3.0, "2to3")]:
            bucket = [t for t in trades if lo <= t.get("zscore_at_entry", 0) < hi]
            if bucket:
                wins = sum(1 for t in bucket if t.get("pnl_usd", t.get("pnl", 0)) > 0)
                avg_pnl = np.mean([t.get("pnl_usd", t.get("pnl", 0)) for t in bucket])
                report.metrics[f"wr_zscore_{label}"] = wins / len(bucket)
                report.metrics[f"avg_pnl_zscore_{label}"] = float(avg_pnl)

        # 3. Hurst accuracy
        low_hurst = [t for t in trades if t.get("hurst_at_entry", 0.5) < 0.45]
        mid_hurst = [t for t in trades if 0.45 <= t.get("hurst_at_entry", 0.5) < 0.50]
        if low_hurst:
            report.metrics["wr_hurst_low"] = sum(1 for t in low_hurst if t.get("pnl_usd", t.get("pnl", 0)) > 0) / len(low_hurst)
        if mid_hurst:
            report.metrics["wr_hurst_mid"] = sum(1 for t in mid_hurst if t.get("pnl_usd", t.get("pnl", 0)) > 0) / len(mid_hurst)

        # 4. Stop loss hit rate
        sl_trades = [t for t in trades if t.get("exit_reason") == "stop_loss"]
        report.metrics["stop_loss_hit_rate"] = len(sl_trades) / len(trades) if trades else 0

        # 5. RSI at entry analysis
        rsi_values = [t.get("rsi_at_entry", 50) for t in trades]
        report.metrics["avg_rsi_at_entry"] = float(np.mean(rsi_values)) if rsi_values else 50

        # Recommendations
        best_z_wr = 0
        best_z_label = ""
        for key, val in report.metrics.items():
            if key.startswith("wr_zscore_") and val > best_z_wr:
                best_z_wr = val
                best_z_label = key.replace("wr_zscore_", "")

        if best_z_wr > 0.65 and best_z_label:
            report.recommendations.append(
                f"Z-score bucket {best_z_label} has {best_z_wr:.0%} win rate. "
                f"Consider tightening entry to more extreme z-scores."
            )

        if report.metrics.get("stop_loss_hit_rate", 0) > 0.50:
            report.recommendations.append(
                f"Stop loss hit rate {report.metrics['stop_loss_hit_rate']:.0%}. "
                f"Widen stops (ATR mult 2.5 -> 3.0) since MR trades need room to breathe."
            )
            report.parameter_suggestions["atr_stop_mult"] = 3.0

        hurst_low_wr = report.metrics.get("wr_hurst_low", 0)
        hurst_mid_wr = report.metrics.get("wr_hurst_mid", 0)
        if hurst_low_wr > 0 and hurst_mid_wr > 0 and abs(hurst_low_wr - hurst_mid_wr) < 0.05:
            report.recommendations.append(
                f"Hurst filter shows no edge (low={hurst_low_wr:.0%} vs mid={hurst_mid_wr:.0%}). "
                f"Consider removing or loosening hurst threshold."
            )

        if report.metrics.get("median_win_bars", 0) > 50:
            report.recommendations.append(
                f"Winning trades take {report.metrics['median_win_bars']:.0f} bars median. "
                f"Reversion is slow — consider exiting earlier (RSI 45 instead of 50)."
            )

    def _analyze_momentum_scalper(
        self, trades: list[dict], report: EngineReport, ohlcv_df: pd.DataFrame | None,
    ) -> None:
        """Momentum Scalper specific analysis."""
        bars_held = [t.get("bars_held", 0) for t in trades]
        pnls = [t.get("pnl_usd", t.get("pnl", 0)) for t in trades]

        # 1. Time in trade (should be short for scalper)
        report.metrics["median_bars_held"] = float(np.median(bars_held)) if bars_held else 0
        report.metrics["pct_held_over_20"] = sum(1 for b in bars_held if b > 20) / len(bars_held) if bars_held else 0

        # 2. False breakout rate (exit by stop loss within first few bars)
        fast_sl = [t for t in trades if t.get("exit_reason") == "stop_loss" and t.get("bars_held", 0) <= 3]
        report.metrics["false_breakout_rate"] = len(fast_sl) / len(trades) if trades else 0

        # 3. Volume spike quality
        vol_values = [t.get("volume_ratio_at_entry", 1) for t in trades]
        report.metrics["avg_volume_at_entry"] = float(np.mean(vol_values)) if vol_values else 1

        # Win rate by volume bucket
        for lo, hi, label in [(1.5, 2.0, "1.5-2x"), (2.0, 3.0, "2-3x"), (3.0, 100, "3x+")]:
            bucket = [t for t in trades if lo <= t.get("volume_ratio_at_entry", 1) < hi]
            if bucket:
                wins = sum(1 for t in bucket if t.get("pnl_usd", t.get("pnl", 0)) > 0)
                report.metrics[f"wr_vol_{label}"] = wins / len(bucket)

        # 4. Fee impact
        fees = [t.get("fees_usd", 0) for t in trades]
        gross_pnls = [abs(t.get("pnl_usd", t.get("pnl", 0))) for t in trades if t.get("pnl_usd", t.get("pnl", 0)) > 0]
        if fees and gross_pnls:
            report.metrics["avg_fee_pct_of_profit"] = float(np.mean(fees)) / float(np.mean(gross_pnls)) if np.mean(gross_pnls) > 0 else 1.0

        # 5. Stop loss hit rate
        sl_trades = [t for t in trades if t.get("exit_reason") == "stop_loss"]
        report.metrics["stop_loss_hit_rate"] = len(sl_trades) / len(trades) if trades else 0

        # Recommendations
        if report.metrics.get("false_breakout_rate", 0) > 0.30:
            report.recommendations.append(
                f"False breakout rate is {report.metrics['false_breakout_rate']:.0%}. "
                f"Volume spikes are traps — require stronger confirmation (RSI > 65 or ROC > 1.0)."
            )
            report.parameter_suggestions["rsi_threshold"] = 65

        if report.metrics.get("median_bars_held", 0) > 20:
            report.recommendations.append(
                f"Median hold time {report.metrics['median_bars_held']:.0f} bars — too long for a scalper. "
                f"Tighten exit to RSI crossing 55 instead of 50."
            )

        if report.metrics.get("avg_fee_pct_of_profit", 0) > 0.30:
            report.recommendations.append(
                f"Fees eat {report.metrics['avg_fee_pct_of_profit']:.0%} of average profit. "
                f"Edge too small — increase min volume threshold or R:R ratio."
            )
            report.parameter_suggestions["volume_mult"] = 2.5
            report.parameter_suggestions["rr_ratio"] = 2.0

        if report.metrics.get("stop_loss_hit_rate", 0) > 0.60:
            report.recommendations.append(
                f"Stop loss hit rate {report.metrics['stop_loss_hit_rate']:.0%}. "
                f"Stops too tight for scalper — try ATR mult 2.0 instead of 1.5."
            )
            report.parameter_suggestions["atr_stop_mult"] = 2.0
