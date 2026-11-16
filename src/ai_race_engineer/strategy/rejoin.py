from collections.abc import Sequence
from dataclasses import dataclass

from ai_race_engineer.strategy.pit_time import PitModel, lane_loss_s, service_time_s
from ai_race_engineer.strategy.race_finish import CarInput, strategy_resource


@dataclass(frozen=True)
class Rejoin:
    """Where a car rejoins if it pits now and nobody else pits meanwhile.
    slot_id         int           game slot id
    lane_loss_s     float         s lost driving through the pit lane
    service_s       float         s standing still
    stop_time_s     float         s lost in total
    energy_to_add   float         fraction of virtual energy added at this stop
    fuel_to_add_l   float         litres added at this stop
    place           int           overall position after the stop, 1-based
    class_place     int           position in class after the stop, 1-based
    ahead_slot_id   int | None    nearest car of its class ahead after the stop
    gap_ahead_s     float | None  s to that car
    behind_slot_id  int | None    nearest car of its class behind after the stop
    gap_behind_s    float | None  s to that car
    traffic         int           cars of any class within traffic_window_s on the road at the exit
    """

    slot_id: int
    lane_loss_s: float
    service_s: float
    stop_time_s: float
    energy_to_add: float
    fuel_to_add_l: float
    place: int
    class_place: int
    ahead_slot_id: int | None
    gap_ahead_s: float | None
    behind_slot_id: int | None
    gap_behind_s: float | None
    traffic: int


def stop_amounts(car: CarInput, laps_remaining: float) -> tuple[float, float]:
    """Energy and fuel a car takes at a stop now: what it needs to the flag, capped by its capacity, never negative.
    In:  car             CarInput
         laps_remaining  float     laps to its chequered flag
    Out:                 tuple[float, float]  (energy fraction, fuel litres); fuel follows energy through
                                              fuel_per_energy for an energy car; (0, 0) when the usage is unknown
    """
    resource = strategy_resource(car)
    if resource is None:
        return 0.0, 0.0
    name, amount, per_lap, capacity = resource
    to_add = max(0.0, min(capacity - amount, laps_remaining * per_lap - amount))
    if name == "fuel":
        return 0.0, to_add
    return to_add, to_add * 100 * (car.fuel_per_energy or 0.0)


def predict_rejoin(
    car: CarInput, laps_remaining: float, others: Sequence[CarInput], pit: PitModel, traffic_window_s: float
) -> Rejoin:
    """Predict a car's stop time and where it rejoins if it pits now.
    In:  car               CarInput            the car that pits
         laps_remaining    float               laps to its chequered flag
         others            Sequence[CarInput]  all other cars
         pit               PitModel
         traffic_window_s  float               s around the rejoin point that count as traffic
    Out:                   Rejoin
    """
    energy_to_add, fuel_to_add_l = stop_amounts(car, laps_remaining)
    loss_s = lane_loss_s(pit, car.lap_time_s)
    service_s = service_time_s(energy_to_add, fuel_to_add_l, pit.rates)
    progress = car.progress_laps - (loss_s + service_s) / car.lap_time_s
    rivals = [o for o in others if o.class_name == car.class_name]
    ahead = min((o for o in rivals if o.progress_laps > progress), key=lambda o: o.progress_laps, default=None)
    behind = max((o for o in rivals if o.progress_laps <= progress), key=lambda o: o.progress_laps, default=None)
    return Rejoin(
        slot_id=car.slot_id,
        lane_loss_s=loss_s,
        service_s=service_s,
        stop_time_s=loss_s + service_s,
        energy_to_add=energy_to_add,
        fuel_to_add_l=fuel_to_add_l,
        place=1 + sum(o.progress_laps > progress for o in others),
        class_place=1 + sum(o.progress_laps > progress for o in rivals),
        ahead_slot_id=None if ahead is None else ahead.slot_id,
        gap_ahead_s=None if ahead is None else (ahead.progress_laps - progress) * car.lap_time_s,
        behind_slot_id=None if behind is None else behind.slot_id,
        gap_behind_s=None if behind is None else (progress - behind.progress_laps) * car.lap_time_s,
        traffic=sum(
            abs((o.progress_laps - progress + 0.5) % 1.0 - 0.5) * car.lap_time_s <= traffic_window_s for o in others
        ),
    )
