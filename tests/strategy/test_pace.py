import dataclasses

from ai_race_engineer.strategy.pace import PaceStats, class_references, pace_stats
from ai_race_engineer.telemetry.snapshot import VehicleState


def _vehicle(
    slot_id: int,
    place: int,
    class_name: str = "Hyper",
    last_lap_s: float | None = None,
    best_lap_s: float | None = None,
) -> VehicleState:
    """Vehicle with the given scoring values.
    In:  slot_id     int
         place       int           overall position
         class_name  str
         last_lap_s  float | None  s
         best_lap_s  float | None  s
    Out:             VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(slot_id=slot_id, place=place, class_name=class_name, last_lap_s=last_lap_s, best_lap_s=best_lap_s)
    return VehicleState(**base)


def test_pace_numbers() -> None:
    """Last and best lap come from scoring; mean, median and count from the clean-lap window."""
    stats = pace_stats(_vehicle(0, 1, last_lap_s=108.0, best_lap_s=106.5), (107.0, 107.5, 112.0, 108.0))
    assert stats == PaceStats(108.0, 106.5, 108.625, 107.75, 4)


def test_pace_without_clean_laps() -> None:
    """Without clean laps there is no mean or median; an invalidated last lap has no time."""
    assert pace_stats(_vehicle(0, 1, best_lap_s=106.5), ()) == PaceStats(None, 106.5, None, None, 0)


def test_class_references_in_the_middle_of_the_class() -> None:
    """The class leader and the class neighbours, with another class in between overall."""
    player = _vehicle(0, 4)
    vehicles = [_vehicle(1, 1), _vehicle(9, 2, "GT3"), _vehicle(2, 3), player, _vehicle(8, 5, "GT3"), _vehicle(3, 6)]
    references = class_references(player, vehicles)
    assert {key: car.slot_id for key, car in references.items()} == {"class_leader": 1, "ahead": 2, "behind": 3}


def test_class_references_for_the_class_leader_and_the_last_car() -> None:
    """The class leader has only a car behind; the last car has no car behind; second has the leader ahead."""
    first, second, last = _vehicle(0, 1), _vehicle(1, 2), _vehicle(2, 3)
    vehicles = [first, second, last]
    assert {k: c.slot_id for k, c in class_references(first, vehicles).items()} == {"behind": 1}
    assert {k: c.slot_id for k, c in class_references(second, vehicles).items()} == {
        "class_leader": 0,
        "ahead": 0,
        "behind": 2,
    }
    assert {k: c.slot_id for k, c in class_references(last, vehicles).items()} == {"class_leader": 0, "ahead": 1}
