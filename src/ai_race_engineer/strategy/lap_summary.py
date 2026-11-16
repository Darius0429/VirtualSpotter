from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ai_race_engineer.strategy.corner_card import CardLimits, SegmentCard
from ai_race_engineer.strategy.lap_segments import LapSamples


@dataclass(frozen=True)
class LapShares:
    """How one lap's time was spent on the pedals.
    full_throttle  float  share of the lap time at full throttle, 0-1
    braking        float  share of the lap time on the brake, 0-1
    coasting       float  share of the lap time off both pedals, 0-1
    """

    full_throttle: float
    braking: float
    coasting: float


def lap_shares(samples: LapSamples, limits: CardLimits) -> LapShares:
    """Shares of one lap's time at full throttle, on the brake and coasting.
    In:  samples  LapSamples
         limits   CardLimits
    Out:          LapShares
    """
    step_s = np.gradient(samples.time_s)
    braking = samples.brake > limits.brake_on
    return LapShares(
        full_throttle=float(step_s[samples.throttle >= limits.full_throttle].sum() / step_s.sum()),
        braking=float(step_s[braking].sum() / step_s.sum()),
        coasting=float(step_s[~braking & (samples.throttle <= limits.throttle_on)].sum() / step_s.sum()),
    )


def theoretical_best_s(laps: Sequence[Sequence[SegmentCard]]) -> float:
    """The lap time made of the best pass of every segment.
    In:  laps  Sequence[Sequence[SegmentCard]]  the cards of each lap, segments in the same order
    Out:       float                            s
    """
    return float(sum(min(card.time_s for card in segment) for segment in zip(*laps, strict=True)))
