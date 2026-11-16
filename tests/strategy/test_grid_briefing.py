import dataclasses

import pytest

from ai_race_engineer.strategy.grid_briefing import GridCar, briefing_text, grid_briefing
from ai_race_engineer.strategy.race_finish import CarInput, CarProjection, RaceProjection
from ai_race_engineer.strategy.rivals import Rival
from ai_race_engineer.telemetry.race_snapshot import CarTyres
from ai_race_engineer.telemetry.snapshot import VehicleState

QUALIFYING: dict[str, float] = {"driver0": 106.0, "driver1": 105.7, "driver2": 106.4}


def _vehicle(slot_id: int) -> VehicleState:
    """Car on the grid.
    In:  slot_id  int
    Out:          VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(slot_id=slot_id, driver_name=f"driver{slot_id}", class_name="Hyper")
    return VehicleState(**base)


def _car(slot_id: int, virtual_energy: float) -> CarInput:
    """Projection input with an amount of energy on board.
    In:  slot_id         int
         virtual_energy  float  fraction
    Out:                 CarInput
    """
    return CarInput(slot_id, "Hyper", 0.0, 106.0, False, False, 60.0, 100.0, virtual_energy, 3.0, 0.045, 0.66)


def _projected(slot_id: int, stops: int, laps_on_board: float) -> CarProjection:
    """Projection with stops and laps on board.
    In:  slot_id        int
         stops          int
         laps_on_board  float  laps
    Out:                CarProjection
    """
    return CarProjection(
        slot_id, "Hyper", 34.0, 3650.0, 34.0, 1, 1, "energy", stops, laps_on_board, 22, 0.5, 30.0, 22.0, 0.0, False
    )


def _rival(slot_id: int, groups: tuple[str, ...], class_place: int) -> Rival:
    """Rival of the player on the grid.
    In:  slot_id      int
         groups       tuple[str, ...]
         class_place  int
    Out:              Rival
    """
    return Rival(
        slot_id,
        f"driver{slot_id}",
        groups,
        class_place,
        0.0,
        0.0,
        False,
        0,
        106.0,
        None,
        0.0,
        class_place,
        0.0,
        1,
        22.0,
        22,
    )


def _briefing(
    tyres: dict[int, CarTyres] | None = None, qualifying: dict[str, float] | None = None
) -> tuple[GridCar, ...]:
    """Grid comparison of the player (slot 0, full energy, one stop) with the car ahead (slot 1, 96 %), the car
    behind (slot 2, two stops), a strategy rival (slot 3) and a pace-only rival (slot 4).
    In:  tyres       dict[int, CarTyres] | None    default: none known
         qualifying  dict[str, float] | None       default: QUALIFYING
    Out:             tuple[GridCar, ...]
    """
    energy = {0: 1.0, 1: 0.96, 2: 1.0, 3: 1.0, 4: 1.0}
    projection = RaceProjection(
        3600.0,
        1,
        {s: _projected(s, 2 if s == 2 else 1, energy[s] / 0.045) for s in energy},
    )
    rivals = (
        _rival(3, ("strategy",), 9),
        _rival(2, ("behind", "pace"), 5),
        _rival(4, ("pace",), 7),
        _rival(1, ("ahead",), 3),
    )
    return grid_briefing(
        0,
        [_vehicle(s) for s in energy],
        [_car(s, energy[s]) for s in energy],
        projection,
        rivals,
        tyres or {},
        QUALIFYING if qualifying is None else qualifying,
    )


def test_neighbours_and_strategy_rivals_are_picked_in_order() -> None:
    """The car ahead, the car behind, then the strategy rivals; a rival on the same pace only is left out."""
    cars = _briefing()
    assert [(c.slot_id, c.relation, c.class_place) for c in cars] == [
        (1, "ahead", 3),
        (2, "behind", 5),
        (3, "rival", 9),
    ]


def test_energy_laps_and_stops_deltas() -> None:
    """Energy and laps on board are the car minus the player; so are the projected stops."""
    ahead, behind, _ = _briefing()
    assert ahead.energy_delta == pytest.approx(-0.04)
    assert ahead.energy_laps_delta == pytest.approx(-0.04 / 0.045)
    assert (ahead.stops, ahead.stops_delta) == (1, 0)
    assert (behind.energy_delta, behind.stops, behind.stops_delta) == (0.0, 2, 1)


def test_qualifying_delta_present_and_absent() -> None:
    """The pace comparison is the qualifying lap of this event; a car without one, or no file at all, gives none."""
    ahead, behind, rival = _briefing()
    assert ahead.qualifying_delta_s == pytest.approx(-0.3)
    assert behind.qualifying_delta_s == pytest.approx(0.4)
    assert rival.qualifying_delta_s is None
    assert all(c.qualifying_delta_s is None for c in _briefing(qualifying={}))


def test_compounds() -> None:
    """Compounds are compared axle by axle; a car without telemetry has none."""
    tyres = {
        0: CarTyres(0, "Medium", "Medium", (0.9, 0.9, 0.9, 0.9)),
        1: CarTyres(1, "Hard", "Medium", (0.91, 0.91, 0.91, 0.91)),
        2: CarTyres(2, "Medium", "Medium", (1.0, 1.0, 0.96, 0.96)),
    }
    ahead, behind, rival = _briefing(tyres)
    assert (ahead.compound_front, ahead.compound_rear, ahead.same_compound) == ("Hard", "Medium", False)
    assert behind.same_compound is True
    assert (rival.compound_front, rival.same_compound) == (None, None)


def test_text_names_at_most_max_cars() -> None:
    """One sentence per car with only what differs; cars beyond the limit are left out."""
    tyres = {0: CarTyres(0, "Medium", "Medium", (1.0,) * 4), 1: CarTyres(1, "Hard", "Hard", (1.0,) * 4)}
    cars = _briefing(tyres)
    assert briefing_text(cars, 2, 0.005) == (
        "Ahead driver1, P3 in class, 0.3 seconds faster in qualifying, on Hard tyres, 4 percent less energy. "
        "Behind driver2, P5 in class, 0.4 seconds slower in qualifying, 1 stop more."
    )
    assert briefing_text(cars, 3, 0.005).endswith("Rival driver3, P9 in class.")
    assert briefing_text((), 3, 0.005) == ""
    assert "energy" not in briefing_text(cars, 2, 0.05)
