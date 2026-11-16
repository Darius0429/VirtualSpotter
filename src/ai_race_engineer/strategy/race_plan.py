import math
from dataclasses import dataclass

from ai_race_engineer.strategy.race_finish import CarInput, CarProjection, strategy_resource


@dataclass(frozen=True)
class RacePlan:
    """The player's plan to the flag on the strategy resource (virtual energy, or fuel for cars without it).
    race_laps         int           laps the player's race will have (completed + remaining)
    average_lap_s     float         s/lap to the flag, stops included
    stint_laps        int           whole laps a full load lasts
    stops             int           stops still to make
    pit_lap           int | None    completed-lap count at which to pit; None if no stop is needed
    amount_on_board   float         fraction of energy (litres for a fuel car)
    to_add_total      float         fraction or litres to add in total to reach the flag; negative = surplus
    first_stop_add    float         fraction or litres added at the first stop, at most a full load; 0 without a stop
    fuel_to_add_l     float | None  litres in total through the fuel ratio; None when the ratio is unknown
    min_stops         int           stops the remaining laps need with a full load now
    min_start_amount  float         fraction or litres: the lowest amount now that still needs only min_stops
    """

    race_laps: int
    average_lap_s: float
    stint_laps: int
    stops: int
    pit_lap: int | None
    amount_on_board: float
    to_add_total: float
    first_stop_add: float
    fuel_to_add_l: float | None
    min_stops: int
    min_start_amount: float


def race_plan(car: CarInput, projection: CarProjection) -> RacePlan | None:
    """Plan of one car from its projection.
    In:  car         CarInput
         projection  CarProjection    of the same car
    Out:             RacePlan | None  None when the usage is unknown
    """
    resource = strategy_resource(car)
    if resource is None:
        return None
    _, amount, per_lap, capacity = resource
    laps = projection.laps_remaining
    laps_to_stop = 0.0 if projection.pit_lap is None else projection.pit_lap - car.progress_laps
    return RacePlan(
        race_laps=round(car.progress_laps + laps),
        average_lap_s=projection.finish_time_s / laps if laps > 0 else car.lap_time_s,
        stint_laps=_stint_laps(capacity, per_lap),
        stops=projection.stops_remaining,
        pit_lap=projection.pit_lap,
        amount_on_board=amount,
        to_add_total=laps * per_lap - amount,
        first_stop_add=0.0
        if projection.pit_lap is None
        else min(capacity, (laps - laps_to_stop) * per_lap) - (amount - laps_to_stop * per_lap),
        fuel_to_add_l=projection.fuel_to_add_l,
        min_stops=min_stops(laps, per_lap, capacity),
        min_start_amount=min_start_amount(laps, per_lap, capacity),
    )


def min_stops(laps: float, per_lap: float, capacity: float) -> int:
    """Stops a number of laps needs when starting on a full load, with whole-lap stints.
    In:  laps      float  laps
         per_lap   float  fraction/lap or litres/lap
         capacity  float  fraction or litres
    Out:           int    stops
    """
    return max(0, math.ceil(laps / _stint_laps(capacity, per_lap) - 1e-9) - 1)


def min_start_amount(laps: float, per_lap: float, capacity: float) -> float:
    """Lowest amount on board at the start that still needs only the minimum number of stops: every later stint is a
    full one, the first covers the rest.
    In:  laps      float  laps
         per_lap   float  fraction/lap or litres/lap
         capacity  float  fraction or litres
    Out:           float  fraction or litres
    """
    first_stint_laps = laps - min_stops(laps, per_lap, capacity) * _stint_laps(capacity, per_lap)
    return min(capacity, max(0.0, first_stint_laps * per_lap))


def _stint_laps(capacity: float, per_lap: float) -> int:
    """Laps a full load lasts.
    In:  capacity  float  fraction or litres
         per_lap   float  fraction/lap or litres/lap
    Out:           int    whole laps, at least 1
    """
    return max(1, math.floor(capacity / per_lap))
