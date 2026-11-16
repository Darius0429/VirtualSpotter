import math
from collections.abc import Sequence
from dataclasses import dataclass

from ai_race_engineer.strategy.pit_time import PitModel, fill_time_s, lane_loss_s


@dataclass(frozen=True)
class CarInput:
    """What the projection needs to know about one car.
    slot_id          int           game slot id
    class_name       str
    progress_laps    float         laps (completed laps + lap fraction)
    lap_time_s       float         s/lap
    finished         bool          the car has taken the chequered flag
    in_pits          bool
    fuel_l           float | None  litres on board
    fuel_capacity_l  float | None  litres
    virtual_energy   float | None  fraction 0-1 on board
    fuel_l_per_lap   float | None  litres/lap
    energy_per_lap   float | None  fraction/lap
    fuel_per_energy  float | None  litres per 1 % energy
    """

    slot_id: int
    class_name: str
    progress_laps: float
    lap_time_s: float
    finished: bool
    in_pits: bool
    fuel_l: float | None
    fuel_capacity_l: float | None
    virtual_energy: float | None
    fuel_l_per_lap: float | None
    energy_per_lap: float | None
    fuel_per_energy: float | None


@dataclass(frozen=True)
class CarProjection:
    """Projected rest of the race of one car.
    slot_id             int           game slot id
    class_name          str
    laps_remaining      float         laps still to drive to its own chequered flag
    finish_time_s       float         s from now
    flag_progress_laps  float         laps covered when the flag falls (ordering key)
    place               int           projected overall position, 1-based
    class_place         int           projected position in class, 1-based
    resource            str | None    "energy" | "fuel": the budget the strategy is computed on; None = usage unknown
    stops_remaining     int           stops
    laps_on_board       float | None  laps on the strategy resource
    pit_lap             int | None    completed-lap count at which it must pit; None if no stop is needed
    energy_to_add       float | None  fraction to add in total to reach the flag (negative = surplus); None on fuel
    fuel_to_add_l       float | None  litres in total: energy_to_add * 100 * fuel_per_energy, or from fuel for fuel
                                      cars; None when unknown
    fuel_laps_on_board  float | None  laps of fuel on board
    fuel_margin_laps    float | None  fuel laps - energy laps on board; negative = fuel runs out first
    fuel_runs_out_first bool          the fuel margin is below minus the tolerance
    """

    slot_id: int
    class_name: str
    laps_remaining: float
    finish_time_s: float
    flag_progress_laps: float
    place: int
    class_place: int
    resource: str | None
    stops_remaining: int
    laps_on_board: float | None
    pit_lap: int | None
    energy_to_add: float | None
    fuel_to_add_l: float | None
    fuel_laps_on_board: float | None
    fuel_margin_laps: float | None
    fuel_runs_out_first: bool


@dataclass(frozen=True)
class RaceProjection:
    """Projected finish of the race.
    flag_time_s     float                     s from now until the net leader takes the flag
    leader_slot_id  int                       net leader
    cars            dict[int, CarProjection]  slot id -> projection
    """

    flag_time_s: float
    leader_slot_id: int
    cars: dict[int, CarProjection]


def project_race(
    cars: Sequence[CarInput],
    remaining_s: float | None,
    max_laps: int | None,
    pit: PitModel,
    fuel_tolerance_laps: float = 0.0,
) -> RaceProjection:
    """Project every car to its own chequered flag, with the stops its strategy resource requires.
    In:  cars                 Sequence[CarInput]  all classified cars with a lap time
         remaining_s          float | None        s on the session clock (None = no time limit)
         max_laps             int | None          laps (None = no lap limit)
         pit                  PitModel
         fuel_tolerance_laps  float               laps the fuel may be short of the energy before it is a warning
    Out:                      RaceProjection
    """
    flag_s, leader_slot_id = flag_time_s(cars, remaining_s, max_laps, pit)
    flag_progress = {c.slot_id: c.progress_laps + (0.0 if c.finished else laps_in_time(c, flag_s, pit)) for c in cars}
    order = sorted(cars, key=lambda c: flag_progress[c.slot_id], reverse=True)
    projections: dict[int, CarProjection] = {}
    class_counts: dict[str, int] = {}
    for place, car in enumerate(order, start=1):
        class_counts[car.class_name] = class_counts.get(car.class_name, 0) + 1
        projections[car.slot_id] = _car_projection(
            car, flag_progress[car.slot_id], place, class_counts[car.class_name], pit, fuel_tolerance_laps
        )
    return RaceProjection(flag_s, leader_slot_id, projections)


def flag_time_s(
    cars: Sequence[CarInput], remaining_s: float | None, max_laps: int | None, pit: PitModel
) -> tuple[float, int]:
    """When the chequered flag falls and for whom: the net leader over all classes, stops included.
    In:  cars         Sequence[CarInput]
         remaining_s  float | None        s on the session clock (None = no time limit)
         max_laps     int | None          laps (None = no lap limit)
         pit          PitModel
    Out:              tuple[float, int]   (s from now, slot id of the net leader); with both limits the earlier flag
                                          wins; (0, most advanced finished car) once a car has finished
    """
    finished = [c for c in cars if c.finished]
    if finished:
        return 0.0, max(finished, key=lambda c: c.progress_laps).slot_id
    flags: list[tuple[float, int]] = []
    if remaining_s is not None:
        reach = {c.slot_id: c.progress_laps + laps_in_time(c, max(remaining_s, 0.0), pit) for c in cars}
        leader = max(cars, key=lambda c: reach[c.slot_id])
        laps = math.floor(reach[leader.slot_id]) + 1 - leader.progress_laps
        flags.append((time_for_laps_s(leader, laps, pit), leader.slot_id))
    if max_laps is not None:
        flags.append(min((time_for_laps_s(c, max_laps - c.progress_laps, pit), c.slot_id) for c in cars))
    return min(flags)


def time_for_laps_s(car: CarInput, laps: float, pit: PitModel) -> float:
    """Time a car needs to drive a number of laps, with the stops and the filling time this takes.
    In:  car   CarInput
         laps  float     laps
         pit   PitModel
    Out:       float     s
    """
    energy_to_add, fuel_to_add_l = _amounts_to_add(car, laps)
    return (
        laps * car.lap_time_s
        + stops_needed(car, laps) * (lane_loss_s(pit, car.lap_time_s) + pit.rates.scale * pit.rates.base_s)
        + fill_time_s(energy_to_add, fuel_to_add_l, pit.rates)
    )


def laps_in_time(car: CarInput, time_s: float, pit: PitModel) -> float:
    """Laps a car drives in a time, the inverse of time_for_laps_s; while a stop is under way the car stays at the
    lap where that stop is due.
    In:  car     CarInput
         time_s  float     s
         pit     PitModel
    Out:         float     laps
    """
    resource = strategy_resource(car)
    if resource is None:
        return time_s / car.lap_time_s
    _, amount, per_lap, capacity = resource
    fill_s = fill_time_s(*_amounts_to_add(car, (amount + 1.0) / per_lap), pit.rates)
    loss_s = lane_loss_s(pit, car.lap_time_s) + pit.rates.scale * pit.rates.base_s
    stops = 0
    while True:
        driving_s = time_s - stops * loss_s
        laps = min(driving_s / car.lap_time_s, (driving_s + fill_s * amount) / (car.lap_time_s + fill_s * per_lap))
        if stops > 0:
            laps = max(laps, amount / per_lap + (stops - 1) * _stint_laps(capacity, per_lap))
        if stops_needed(car, laps) <= stops:
            return laps
        stops += 1


def stops_needed(car: CarInput, laps: float) -> int:
    """Stops a car needs to drive a number of laps on its strategy resource. A stint is the whole laps a full load
    lasts and at least one lap: a car that cannot do a lap on a full load still stops every lap.
    In:  car   CarInput
         laps  float     laps
    Out:       int       stops; 0 when the usage is unknown
    """
    resource = strategy_resource(car)
    if resource is None:
        return 0
    _, amount, per_lap, capacity = resource
    return max(0, math.ceil((laps - amount / per_lap) / _stint_laps(capacity, per_lap) - 1e-9))


def strategy_resource(car: CarInput) -> tuple[str, float, float, float] | None:
    """The budget a car's strategy is computed on: virtual energy, or fuel for cars without virtual energy.
    In:  car  CarInput
    Out:      tuple[str, float, float, float] | None  ("energy" | "fuel", amount on board, usage per lap, capacity);
                                                      None when the usage is unknown
    """
    if car.virtual_energy is not None and car.energy_per_lap:
        return "energy", car.virtual_energy, car.energy_per_lap, 1.0
    if car.fuel_l is not None and car.fuel_l_per_lap and car.fuel_capacity_l is not None:
        return "fuel", car.fuel_l, car.fuel_l_per_lap, car.fuel_capacity_l
    return None


def fuel_margin_laps(car: CarInput) -> float | None:
    """Laps of fuel on board beyond the laps of energy on board, for a car on an energy strategy.
    In:  car  CarInput
    Out:      float | None  laps (negative = fuel runs out first); None for fuel cars or unknown fuel usage
    """
    resource = strategy_resource(car)
    fuel_laps = _fuel_laps_on_board(car)
    if resource is None or resource[0] != "energy" or fuel_laps is None:
        return None
    return fuel_laps - resource[1] / resource[2]


def _car_projection(
    car: CarInput, flag_progress_laps: float, place: int, class_place: int, pit: PitModel, fuel_tolerance_laps: float
) -> CarProjection:
    """Projection of one car, given where it is when the flag falls.
    In:  car                  CarInput
         flag_progress_laps   float     laps covered when the flag falls
         place                int       projected overall position
         class_place          int       projected position in class
         pit                  PitModel
         fuel_tolerance_laps  float     laps the fuel may be short of the energy before it is a warning
    Out:                      CarProjection
    """
    laps = 0.0 if car.finished else math.ceil(flag_progress_laps - 1e-9) - car.progress_laps
    resource = strategy_resource(car)
    stops = stops_needed(car, laps)
    laps_on_board = None if resource is None else resource[1] / resource[2]
    energy_to_add, fuel_to_add_l = _amounts_to_add(car, laps)
    margin = fuel_margin_laps(car)
    is_energy = resource is not None and resource[0] == "energy"
    return CarProjection(
        slot_id=car.slot_id,
        class_name=car.class_name,
        laps_remaining=laps,
        finish_time_s=time_for_laps_s(car, laps, pit),
        flag_progress_laps=flag_progress_laps,
        place=place,
        class_place=class_place,
        resource=None if resource is None else resource[0],
        stops_remaining=stops,
        laps_on_board=laps_on_board,
        pit_lap=math.floor(car.progress_laps + laps_on_board) if stops > 0 else None,
        energy_to_add=energy_to_add if is_energy else None,
        fuel_to_add_l=None if resource is None or (is_energy and car.fuel_per_energy is None) else fuel_to_add_l,
        fuel_laps_on_board=_fuel_laps_on_board(car),
        fuel_margin_laps=margin,
        fuel_runs_out_first=margin is not None and margin < -fuel_tolerance_laps,
    )


def _stint_laps(capacity: float, per_lap: float) -> int:
    """Laps a full load of the strategy resource lasts.
    In:  capacity  float  fraction or litres
         per_lap   float  fraction/lap or litres/lap
    Out:           int    whole laps, at least 1
    """
    return max(1, math.floor(capacity / per_lap))


def _amounts_to_add(car: CarInput, laps: float) -> tuple[float, float]:
    """Energy and fuel a car must add in total to drive a number of laps.
    In:  car   CarInput
         laps  float     laps
    Out:       tuple[float, float]  (energy fraction, fuel litres), negative = surplus; fuel follows energy through
                                    fuel_per_energy for an energy car (0 when the ratio is unknown); (0, 0) when the
                                    usage is unknown
    """
    resource = strategy_resource(car)
    if resource is None:
        return 0.0, 0.0
    name, amount, per_lap, _ = resource
    to_add = laps * per_lap - amount
    if name == "fuel":
        return 0.0, to_add
    return to_add, to_add * 100 * (car.fuel_per_energy or 0.0)


def _fuel_laps_on_board(car: CarInput) -> float | None:
    """Laps the fuel on board lasts.
    In:  car  CarInput
    Out:      float | None  laps; None when fuel or its usage is unknown
    """
    if car.fuel_l is None or not car.fuel_l_per_lap:
        return None
    return car.fuel_l / car.fuel_l_per_lap
