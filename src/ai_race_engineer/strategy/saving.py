import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ai_race_engineer.strategy.pit_time import PitRates


@dataclass(frozen=True)
class SavingNeed:
    """How much energy must be saved per lap to reach a goal.
    usage_per_lap    float  fraction/lap the driver must get down to
    saving_per_lap   float  fraction of energy per lap to save
    saving_fraction  float  share of the current usage (0.021 = 2.1 %)
    laps             float  laps it must be held
    """

    usage_per_lap: float
    saving_per_lap: float
    saving_fraction: float
    laps: float


@dataclass(frozen=True)
class SavingCheck:
    """Whether a saving is realistic and what it costs.
    need            SavingNeed
    within_limit    bool          saving_fraction <= max_saving_fraction
    within_shown    bool | None   the target usage is not below the lowest usage driven; None without laps
    cost_s_per_lap  float | None  s/lap lost at that saving; None = unknown
    """

    need: SavingNeed
    within_limit: bool
    within_shown: bool | None
    cost_s_per_lap: float | None


def saving_to_reach(amount: float, per_lap: float, laps_needed: float) -> SavingNeed:
    """Saving that makes the energy on board last a number of laps.
    In:  amount       float  fraction on board
         per_lap      float  fraction/lap used now
         laps_needed  float  laps the amount must last
    Out:              SavingNeed
    """
    usage = amount / laps_needed
    return SavingNeed(usage, per_lap - usage, 1 - usage / per_lap, laps_needed)


def saving_for_extra_lap(amount: float, per_lap: float, lap_fraction: float) -> SavingNeed:
    """Saving that lets the car cross the line once more before it has to stop.
    In:  amount        float  fraction on board
         per_lap       float  fraction/lap used now
         lap_fraction  float  fraction of the current lap already driven, 0-1
    Out:               SavingNeed
    """
    return saving_to_reach(amount, per_lap, math.floor(lap_fraction + amount / per_lap) + 1 - lap_fraction)


def saving_to_skip_stop(
    amount: float, per_lap: float, capacity: float, laps_remaining: float, stops: int
) -> SavingNeed | None:
    """Saving that covers the remaining laps with one stop fewer: the highest usage at which the amount on board
    lasts the first stint and a full load lasts each of the later, equally long whole-lap stints.
    In:  amount          float  fraction on board
         per_lap         float  fraction/lap used now
         capacity        float  fraction a full load holds
         laps_remaining  float  laps to the flag
         stops           int    stops needed at the current usage
    Out:                 SavingNeed | None  None when no stop is needed, or when the laps are too few for one stop
                                            fewer with whole-lap stints
    """
    if stops == 0:
        return None
    later = stops - 1
    usages = (
        [amount / laps_remaining]
        if later == 0
        else [
            min(amount / (laps_remaining - later * stint_laps), capacity / stint_laps)
            for stint_laps in range(1, math.ceil(laps_remaining / later))
        ]
    )
    if not usages:
        return None
    usage = max(usages)
    return SavingNeed(usage, per_lap - usage, 1 - usage / per_lap, laps_remaining)


def saving_to_shorten_stop(per_lap: float, laps_remaining: float, shorten_s: float, rates: PitRates) -> SavingNeed:
    """Saving that shortens the filling at a stop: the energy not added is spread over the remaining laps.
    In:  per_lap         float     fraction/lap used now
         laps_remaining  float     laps to the flag
         shorten_s       float     s the stop should be shorter
         rates           PitRates
    Out:                 SavingNeed
    """
    saving = shorten_s * rates.energy_rate_per_s / rates.scale / laps_remaining
    return SavingNeed(per_lap - saving, saving, saving / per_lap, laps_remaining)


def cost_s_per_energy(laps: Sequence[tuple[float, float]], min_laps: int, min_spread: float) -> float | None:
    """Lap time lost per unit of energy saved: the slope of lap time against usage over the driver's clean laps.
    In:  laps        Sequence[tuple[float, float]]  (lap time s, energy fraction used) per clean lap
         min_laps    int                            laps needed for an estimate
         min_spread  float                          share of the mean usage the lowest lap must be below it
    Out:             float | None                   s per fraction of energy; None with too few laps, too little
                                                    spread in the usage, or laps that show no cost (the laps that
                                                    used less were not slower)
    """
    if len(laps) < min_laps:
        return None
    times = np.array([lap_s for lap_s, _ in laps])
    usage = np.array([used for _, used in laps])
    if usage.min() > usage.mean() * (1 - min_spread):
        return None
    cost = -float(np.polyfit(usage, times, 1)[0])
    return cost if cost > 0 else None


def check_saving(
    need: SavingNeed, max_saving_fraction: float, laps: Sequence[tuple[float, float]], min_laps: int, min_spread: float
) -> SavingCheck:
    """Judge a saving against the realistic limit, the lowest usage the driver has shown and its lap-time cost.
    In:  need                 SavingNeed
         max_saving_fraction  float                          share of the usage per lap that is realistic to save
         laps                 Sequence[tuple[float, float]]  (lap time s, energy fraction used) per clean lap
         min_laps             int                            laps needed before a cost is estimated
         min_spread           float                          spread in the usage needed before a cost is estimated
    Out:                      SavingCheck
    """
    cost = cost_s_per_energy(laps, min_laps, min_spread)
    return SavingCheck(
        need=need,
        within_limit=need.saving_fraction <= max_saving_fraction,
        within_shown=need.usage_per_lap >= min(used for _, used in laps) if laps else None,
        cost_s_per_lap=None if cost is None else cost * need.saving_per_lap,
    )
