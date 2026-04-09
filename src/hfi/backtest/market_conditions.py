"""Market condition labeler — auto-classify historical periods.

Labels daily OHLCV data as BULL, BEAR, RANGING, or CRASH.
Used for sampling diverse market conditions in robustness testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConditionPeriod:
    """A contiguous period of one market condition."""
    condition: str
    start: datetime
    end: datetime
    duration_days: int


class MarketConditionLabeler:
    """Labels daily price data with market conditions."""

    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        crash_dd_threshold: float = 0.20,
        crash_window: int = 30,
        min_period_days: int = 14,
        smooth_days: int = 5,
    ) -> None:
        self._sma_fast = sma_fast
        self._sma_slow = sma_slow
        self._crash_dd = crash_dd_threshold
        self._crash_window = crash_window
        self._min_period_days = min_period_days
        self._smooth_days = smooth_days

    def label_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'condition' column to daily OHLCV DataFrame.

        Args:
            df: Daily OHLCV with DatetimeIndex, must have 'close' column.

        Returns:
            Same DataFrame with 'condition' column added.
        """
        if len(df) < self._sma_slow + 10:
            logger.warning("Not enough data for condition labeling: %d rows", len(df))
            df["condition"] = "RANGING"
            return df

        df = df.copy()

        # Compute indicators
        df["sma_fast"] = df["close"].rolling(self._sma_fast).mean()
        df["sma_slow"] = df["close"].rolling(self._sma_slow).mean()

        # Rolling max drawdown over crash_window days
        rolling_max = df["close"].rolling(self._crash_window).max()
        df["rolling_dd"] = (rolling_max - df["close"]) / rolling_max

        # Raw classification
        conditions = []
        for i in range(len(df)):
            if pd.isna(df["sma_fast"].iloc[i]) or pd.isna(df["sma_slow"].iloc[i]):
                conditions.append("RANGING")
                continue

            close = df["close"].iloc[i]
            sma_f = df["sma_fast"].iloc[i]
            sma_s = df["sma_slow"].iloc[i]
            dd = df["rolling_dd"].iloc[i]

            if dd >= self._crash_dd and close < sma_f:
                conditions.append("CRASH")
            elif sma_f > sma_s and close > sma_f:
                conditions.append("BULL")
            elif sma_f < sma_s and close < sma_f:
                conditions.append("BEAR")
            else:
                conditions.append("RANGING")

        df["condition"] = conditions

        # Smooth: require N consecutive days to confirm a label change
        df["condition"] = self._smooth_labels(df["condition"], self._smooth_days)

        # Cleanup temp columns
        df.drop(columns=["sma_fast", "sma_slow", "rolling_dd"], inplace=True, errors="ignore")

        # Log distribution
        dist = df["condition"].value_counts()
        logger.info("Condition distribution: %s", dict(dist))

        return df

    def get_periods(self, df: pd.DataFrame) -> list[ConditionPeriod]:
        """Extract contiguous periods from labeled DataFrame.

        Filters out periods shorter than min_period_days.
        """
        if "condition" not in df.columns:
            df = self.label_conditions(df)

        periods: list[ConditionPeriod] = []
        current_condition = None
        period_start = None

        for i, (ts, row) in enumerate(df.iterrows()):
            cond = row["condition"]
            if cond != current_condition:
                # Close previous period
                if current_condition is not None and period_start is not None:
                    end = df.index[i - 1]
                    duration = (end - period_start).days + 1
                    if duration >= self._min_period_days:
                        periods.append(ConditionPeriod(
                            condition=current_condition,
                            start=period_start.to_pydatetime(),
                            end=end.to_pydatetime(),
                            duration_days=duration,
                        ))
                current_condition = cond
                period_start = ts

        # Close last period
        if current_condition is not None and period_start is not None:
            end = df.index[-1]
            duration = (end - period_start).days + 1
            if duration >= self._min_period_days:
                periods.append(ConditionPeriod(
                    condition=current_condition,
                    start=period_start.to_pydatetime(),
                    end=end.to_pydatetime(),
                    duration_days=duration,
                ))

        logger.info(
            "Found %d periods: %s",
            len(periods),
            {c: sum(1 for p in periods if p.condition == c)
             for c in ["BULL", "BEAR", "RANGING", "CRASH"]},
        )
        return periods

    @staticmethod
    def _smooth_labels(labels: pd.Series, window: int) -> pd.Series:
        """Smooth labels: require `window` consecutive same labels to confirm change."""
        smoothed = labels.copy()
        current = labels.iloc[0]
        streak = 1

        for i in range(1, len(labels)):
            if labels.iloc[i] == labels.iloc[i - 1]:
                streak += 1
            else:
                streak = 1

            if streak >= window:
                current = labels.iloc[i]

            smoothed.iloc[i] = current

        return smoothed
