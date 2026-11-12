import dataclasses

import pytest

from ai_race_engineer.strategy.standings import standings
from ai_race_engineer.telemetry.snapshot import VehicleState


def _vehicle(slot_id: int, place: int, class_name: str, best_lap_s: float | None) -> VehicleState:
    """Vehicle with the given scoring values and neutral other values.
    In:  slot_id     int
         place       int           overall position
         class_name  str
         best_lap_s  float | None  s
    Out:             VehicleState
    """
    base = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        slot_id=slot_id,
        place=place,
        class_name=class_name,
        best_lap_s=best_lap_s,
        driver_name=f"driver{slot_id}",
        vehicle_name="",
        position_xz_m=(0.0, 0.0),
    )
    return VehicleState(**base)


def test_single_class_position_equals_place() -> None:
    """In one class the class position is the overall place."""
    table = standings([_vehicle(7, 2, "GT3", 100.0), _vehicle(3, 1, "GT3", 99.0), _vehicle(5, 3, "GT3", 101.0)])
    assert table["position_in_class"].to_dict() == {7: 2, 3: 1, 5: 3}


def test_multiclass_ranking() -> None:
    """Class positions rank within each class."""
    table = standings(
        [
            _vehicle(1, 1, "Hypercar", 90.0),
            _vehicle(2, 2, "GT3", 100.0),
            _vehicle(3, 3, "Hypercar", 91.0),
            _vehicle(4, 4, "GT3", 101.0),
        ]
    )
    assert table["position_in_class"].to_dict() == {1: 1, 2: 1, 3: 2, 4: 2}


def test_class_best_ignores_none() -> None:
    """Cars without a best lap do not affect the class best."""
    table = standings([_vehicle(1, 1, "GT3", None), _vehicle(2, 2, "GT3", 100.5), _vehicle(3, 3, "GT3", 101.0)])
    assert table["class_best_lap_s"].tolist() == pytest.approx([100.5, 100.5, 100.5])


def test_session_best() -> None:
    """Session best is the fastest lap of any class."""
    table = standings([_vehicle(1, 1, "Hypercar", 90.0), _vehicle(2, 2, "GT3", 100.0)])
    assert table["session_best_lap_s"].tolist() == pytest.approx([90.0, 90.0])
    assert table["class_best_lap_s"].to_dict() == {1: 90.0, 2: 100.0}


def test_index_is_slot_id() -> None:
    """Rows are indexed by slot id."""
    table = standings([_vehicle(42, 1, "GT3", 100.0), _vehicle(9, 2, "GT3", None)])
    assert table.index.name == "slot_id"
    assert list(table.index) == [42, 9]
