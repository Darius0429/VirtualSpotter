from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ai_race_engineer.domains.common.analysis.strategy.race_finish import CarInput, RaceProjection
from ai_race_engineer.domains.common.analysis.field.rivals import Rival
from ai_race_engineer.domains.common.data.race_snapshot import CarTyres
from ai_race_engineer.domains.common.data.snapshot import VehicleState


@dataclass(frozen=True)
class GridCar:
    """One car the driver is told about before the start, compared with the player's car.
    slot_id             int
    driver_name         str
    relation            str           "ahead" | "behind" | "rival"
    class_place         int           place in class on the grid
    energy_delta        float | None  fraction of energy on board, car - player
    energy_laps_delta   float | None  laps on board, car - player
    compound_front      str | None
    compound_rear       str | None
    same_compound       bool | None   both axles on the player's compounds; None when unknown
    qualifying_delta_s  float | None  s, qualifying lap car - player (negative = the car was faster); None without
                                      this event's qualifying results
    stops               int           projected stops
    stops_delta         int           projected stops, car - player
    """

    slot_id: int
    driver_name: str
    relation: str
    class_place: int
    energy_delta: float | None
    energy_laps_delta: float | None
    compound_front: str | None
    compound_rear: str | None
    same_compound: bool | None
    qualifying_delta_s: float | None
    stops: int
    stops_delta: int


def grid_briefing(
    player_slot_id: int,
    vehicles: Sequence[VehicleState],
    cars: Sequence[CarInput],
    projection: RaceProjection,
    rivals: Sequence[Rival],
    tyres: Mapping[int, CarTyres],
    qualifying_best_s: Mapping[str, float],
) -> tuple[GridCar, ...]:
    """Compare the player with the cars it starts among: the class neighbours by place and every strategy rival.
    The pace comparison is the qualifying lap of this event only, never a lap of the race.
    In:  player_slot_id     int
         vehicles           Sequence[VehicleState]  all cars
         cars               Sequence[CarInput]      projected cars
         projection         RaceProjection
         rivals             Sequence[Rival]         the player's rivals
         tyres              Mapping[int, CarTyres]  slot id -> tyres, cars with telemetry
         qualifying_best_s  Mapping[str, float]     driver name -> s, best qualifying lap of this event
    Out:                    tuple[GridCar, ...]     the car ahead, the car behind, then the strategy rivals
    """
    names = {v.slot_id: v.driver_name for v in vehicles}
    inputs = {c.slot_id: c for c in cars}
    order = {"ahead": 0, "behind": 1, "rival": 2}
    picked = sorted(
        ((_relation(r), r) for r in rivals if "ahead" in r.groups or "behind" in r.groups or "strategy" in r.groups),
        key=lambda pair: (order[pair[0]], pair[1].class_place),
    )
    mine, my_tyres = projection.cars[player_slot_id], tyres.get(player_slot_id)
    my_lap_s = qualifying_best_s.get(names[player_slot_id])
    grid: list[GridCar] = []
    for relation, rival in picked:
        theirs, their_tyres = projection.cars[rival.slot_id], tyres.get(rival.slot_id)
        their_lap_s = qualifying_best_s.get(rival.driver_name)
        grid.append(
            GridCar(
                slot_id=rival.slot_id,
                driver_name=rival.driver_name,
                relation=relation,
                class_place=rival.class_place,
                energy_delta=_delta(inputs[rival.slot_id].virtual_energy, inputs[player_slot_id].virtual_energy),
                energy_laps_delta=_delta(theirs.laps_on_board, mine.laps_on_board),
                compound_front=None if their_tyres is None else their_tyres.compound_front,
                compound_rear=None if their_tyres is None else their_tyres.compound_rear,
                same_compound=None
                if their_tyres is None or my_tyres is None
                else (their_tyres.compound_front, their_tyres.compound_rear)
                == (my_tyres.compound_front, my_tyres.compound_rear),
                qualifying_delta_s=_delta(their_lap_s, my_lap_s),
                stops=theirs.stops_remaining,
                stops_delta=theirs.stops_remaining - mine.stops_remaining,
            )
        )
    return tuple(grid)


def briefing_text(cars: Sequence[GridCar], max_cars: int, min_energy_delta: float) -> str:
    """The grid comparison as speech.
    In:  cars              Sequence[GridCar]
         max_cars          int                cars named
         min_energy_delta  float              fraction of energy; differences below this are not mentioned
    Out:                   str                one sentence per car; "" without cars
    """
    return " ".join(_sentence(car, min_energy_delta) for car in cars[:max_cars])


def _relation(rival: Rival) -> str:
    """How a rival stands to the player on the grid.
    In:  rival  Rival
    Out:        str  "ahead" | "behind" | "rival"
    """
    if "ahead" in rival.groups:
        return "ahead"
    return "behind" if "behind" in rival.groups else "rival"


def _sentence(car: GridCar, min_energy_delta: float) -> str:
    """One car of the grid comparison as a sentence.
    In:  car               GridCar
         min_energy_delta  float    fraction of energy; differences below this are not mentioned
    Out:                   str
    """
    lead = {"ahead": "Ahead", "behind": "Behind", "rival": "Rival"}[car.relation]
    parts = [f"{lead} {car.driver_name}, P{car.class_place} in class"]
    if car.qualifying_delta_s is not None:
        side = "faster" if car.qualifying_delta_s < 0 else "slower"
        parts.append(f"{abs(car.qualifying_delta_s):.1f} seconds {side} in qualifying")
    if car.same_compound is False:
        parts.append(f"on {car.compound_front} tyres")
    if car.energy_delta is not None and abs(car.energy_delta) >= min_energy_delta:
        parts.append(f"{abs(car.energy_delta) * 100:.0f} percent {'more' if car.energy_delta > 0 else 'less'} energy")
    if car.stops_delta != 0:
        count = abs(car.stops_delta)
        parts.append(f"{count} stop{'s' if count > 1 else ''} {'more' if car.stops_delta > 0 else 'fewer'}")
    return ", ".join(parts) + "."


def _delta(theirs: float | None, mine: float | None) -> float | None:
    """Difference of two values that may be unknown.
    In:  theirs  float | None
         mine    float | None
    Out:         float | None  theirs - mine; None when either is unknown
    """
    return None if theirs is None or mine is None else theirs - mine
