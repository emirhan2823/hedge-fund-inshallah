"""Random period sampler — diverse market condition sampling.

Selects random periods from different market conditions
to prevent period-specific overfitting.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from hfi.backtest.market_conditions import ConditionPeriod

logger = logging.getLogger(__name__)


@dataclass
class SampledPeriod:
    """A randomly sampled period for testing."""
    condition: str
    start: datetime
    end: datetime
    duration_days: int
    source_period_idx: int


class RandomPeriodSampler:
    """Samples random periods from different market conditions."""

    def sample_periods(
        self,
        periods: list[ConditionPeriod],
        n_samples: int = 5,
        min_days: int = 30,
        max_days: int = 120,
        seed: int | None = None,
    ) -> list[SampledPeriod]:
        """Sample random periods ensuring condition diversity.

        Guarantees at least 1 sample from each available condition.
        """
        rng = random.Random(seed)

        if not periods:
            logger.warning("No periods available for sampling")
            return []

        # Group by condition
        by_condition: dict[str, list[tuple[int, ConditionPeriod]]] = {}
        for idx, p in enumerate(periods):
            by_condition.setdefault(p.condition, []).append((idx, p))

        samples: list[SampledPeriod] = []
        conditions = list(by_condition.keys())

        # Phase 1: one from each condition (diversity guarantee)
        for cond in conditions:
            pool = by_condition[cond]
            s = self._sample_one(pool, rng, min_days, max_days)
            if s:
                samples.append(s)

        # Phase 2: fill remaining slots randomly
        remaining = n_samples - len(samples)
        all_pool = [(idx, p) for idx, p in enumerate(periods)]

        for _ in range(remaining):
            s = self._sample_one(all_pool, rng, min_days, max_days)
            if s:
                samples.append(s)

        rng.shuffle(samples)

        logger.info(
            "Sampled %d periods: %s",
            len(samples),
            {c: sum(1 for s in samples if s.condition == c) for c in conditions},
        )
        return samples

    @staticmethod
    def _sample_one(
        pool: list[tuple[int, ConditionPeriod]],
        rng: random.Random,
        min_days: int,
        max_days: int,
    ) -> SampledPeriod | None:
        """Sample one random sub-period from a pool of periods."""
        # Filter to periods long enough
        valid = [(idx, p) for idx, p in pool if p.duration_days >= min_days]
        if not valid:
            valid = pool  # fallback to any period

        if not valid:
            return None

        idx, period = rng.choice(valid)

        # Random duration
        max_dur = min(max_days, period.duration_days)
        min_dur = min(min_days, max_dur)
        duration = rng.randint(min_dur, max_dur)

        # Random start within period
        max_start_offset = period.duration_days - duration
        if max_start_offset > 0:
            start_offset = rng.randint(0, max_start_offset)
        else:
            start_offset = 0

        start = period.start + timedelta(days=start_offset)
        end = start + timedelta(days=duration)

        return SampledPeriod(
            condition=period.condition,
            start=start,
            end=end,
            duration_days=duration,
            source_period_idx=idx,
        )
