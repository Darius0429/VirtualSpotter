import dataclasses

import pytest

from ai_race_engineer.strategy.field import CarEstimate
from ai_race_engineer.strategy.history import CarHistory, ClassHistory
from ai_race_engineer.strategy.prior_field import PriorFieldTracker
from ai_race_engineer.strategy.priors import EventPriors, PriorSource
from ai_race_engineer.telemetry.garage import GarageInfo
from ai_race_engineer.telemetry.race_snapshot import CarResources
from ai_race_engineer.telemetry.snapshot import VehicleState

GREEN: int = 5
PRIORS: EventPriors = EventPriors("Track", "Car A", {"driver0": 100.0, "driver1": 102.0}, {})
GARAGE: GarageInfo = GarageInfo(fuel_ratio=0.8, energy_per_lap=0.05)
_OTHER_CLASS: CarEstimate = CarEstimate(9, 130.0, 2.0, 0.03, 0.7)


def _vehicle(slot_id: int, class_name: str = "Hyper", **overrides: object) -> VehicleState:
    """Car before its first lap; slot 0 is the player.
    In:  slot_id     int
         class_name  str
         overrides   object  VehicleState field values
    Out:             VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        slot_id=slot_id,
        is_player=slot_id == 0,
        driver_name=f"driver{slot_id}",
        class_name=class_name,
        best_lap_s=None,
        last_lap_s=None,
    )
    base.update(overrides)
    return VehicleState(**base)


def _tracker(
    priors: EventPriors | None = PRIORS, garage: GarageInfo | None = GARAGE, start_laps: int = 0
) -> PriorFieldTracker:
    """Tracker with a pace factor of 1.01.
    In:  priors      EventPriors | None
         garage      GarageInfo | None
         start_laps  int                 laps of the player before measured values are used (default: at once)
    Out:         PriorFieldTracker
    """
    return PriorFieldTracker(3, 3, 1.01, lambda: priors, lambda: garage, start_laps)


def test_priors_fill_cars_without_laps() -> None:
    """On the grid the player gets the qualifying pace and the garage usage; another car of its class its own
    qualifying pace and the class usage (here the player's garage figure); a car without a qualifying lap the best
    pace of its class; a car of a class nothing is known about has neither."""
    tracker = _tracker()
    estimates = tracker.update((_vehicle(0), _vehicle(1), _vehicle(2), _vehicle(3, "GT3")), (), GREEN)
    assert estimates[0] == CarEstimate(0, pytest.approx(101.0), pytest.approx(4.0), 0.05, 0.8)
    assert estimates[1] == CarEstimate(1, pytest.approx(103.02), None, 0.05, None)
    assert estimates[2].lap_time_s == pytest.approx(101.0)
    assert estimates[3].lap_time_s is None
    assert tracker.sources == {
        0: PriorSource("qualifying", "game"),
        1: PriorSource("qualifying", "class"),
        2: PriorSource("class_best", "class"),
        3: PriorSource("none", "none"),
    }


def test_measured_holds_the_raw_values() -> None:
    """What the session measured is kept apart from the filled estimates."""
    tracker = _tracker()
    tracker.update((_vehicle(0),), (), GREEN)
    assert tracker.measured == {0: CarEstimate(0, None, None, None, None)}


def test_clean_lap_replaces_the_prior() -> None:
    """After a clean lap the pace and the usage are the measured ones."""
    tracker = _tracker()
    tracker.update((_vehicle(0),), (CarResources(0, 50.0, 100.0, 1.0, 0),), GREEN)
    tracker.update((_vehicle(0, lap_start_s=10.0, completed_laps=1),), (CarResources(0, 50.0, 100.0, 1.0, 0),), GREEN)
    lapped = _vehicle(0, lap_start_s=115.0, completed_laps=2, last_lap_s=105.0, best_lap_s=105.0)
    estimates = tracker.update((lapped,), (CarResources(0, 46.5, 100.0, 0.955, 0),), GREEN)
    assert estimates[0].lap_time_s == 105.0
    assert estimates[0].energy_per_lap == pytest.approx(0.045)
    assert estimates[0].fuel_per_energy == pytest.approx(3.5 / 4.5)
    assert tracker.sources[0] == PriorSource("measured", "measured")


def test_best_lap_and_remembered_values() -> None:
    """Without a qualifying file the best lap is used; values remembered from an earlier session come first, for the
    same driver in the same class only; the remembered usage is the best-known figure of the class."""
    remembered = {("driver1", "Hyper"): CarEstimate(7, 104.0, 3.6, 0.046, 0.79), ("driver0", "GT3"): _OTHER_CLASS}
    tracker = _tracker(EventPriors("Track", "Car A", {}, remembered), None)
    estimates = tracker.update((_vehicle(0, best_lap_s=100.0), _vehicle(1, best_lap_s=100.0)), (), GREEN)
    assert estimates[0].lap_time_s == pytest.approx(101.0)
    assert estimates[1] == CarEstimate(1, 104.0, pytest.approx(0.046 * 100 * 0.79), 0.046, 0.79)
    assert estimates[0].energy_per_lap == 0.046
    assert tracker.sources == {0: PriorSource("best_lap", "class"), 1: PriorSource("remembered", "remembered")}


def test_nothing_known_gives_none() -> None:
    """Without priors, garage figures or laps there is no pace and no usage."""
    tracker = _tracker(None, None)
    estimates = tracker.update((_vehicle(0),), (), GREEN)
    assert estimates[0] == CarEstimate(0, None, None, None, None)
    assert tracker.sources[0] == PriorSource("none", "none")


def test_qualifying_file_outranks_the_remembered_pace() -> None:
    """With both a qualifying lap and a remembered pace the qualifying lap is used; the remembered usage stays."""
    remembered = {("driver0", "Hyper"): CarEstimate(0, 104.0, 3.6, 0.046, 0.79)}
    tracker = _tracker(EventPriors("Track", "Car A", {"driver0": 100.0}, remembered))
    estimates = tracker.update((_vehicle(0),), (), GREEN)
    assert estimates[0].lap_time_s == pytest.approx(101.0)
    assert tracker.sources[0] == PriorSource("qualifying", "remembered")


def test_measured_usage_of_the_class_beats_the_garage_figure() -> None:
    """Once a car of the class has a measured usage, cars without their own get the mean of the measured ones."""
    tracker = _tracker(None, None)
    cars = (_vehicle(0), _vehicle(1), _vehicle(2))
    full = tuple(CarResources(s, 50.0, 100.0, 1.0, 0) for s in (0, 1))
    tracker.update(cars, full, GREEN)
    tracker.update(tuple(dataclasses.replace(c, lap_start_s=10.0, completed_laps=1) for c in cars), full, GREEN)
    lapped = tuple(dataclasses.replace(c, lap_start_s=110.0, completed_laps=2, last_lap_s=100.0) for c in cars)
    used = (CarResources(0, 47.0, 100.0, 0.96, 0), CarResources(1, 47.0, 100.0, 0.95, 0))
    estimates = tracker.update(lapped, used, GREEN)
    assert estimates[2].energy_per_lap == pytest.approx(0.045)
    assert tracker.sources[2].energy == "class"


def _lap(tracker: PriorFieldTracker, lap: int, energy: float) -> dict[int, CarEstimate]:
    """Both cars cross the line at the end of a 105 s lap.
    In:  tracker  PriorFieldTracker
         lap      int                completed laps after the crossing
         energy   float              fraction on board at the crossing
    Out:          dict[int, CarEstimate]
    """
    cars = tuple(
        _vehicle(s, lap_start_s=10.0 + 105.0 * lap, completed_laps=lap, last_lap_s=105.0, best_lap_s=105.0)
        for s in (0, 1)
    )
    return tracker.update(cars, tuple(CarResources(s, 50.0, 100.0, energy, 0) for s in (0, 1)), GREEN)


def test_priors_are_held_until_the_strategy_starts() -> None:
    """Through the player's lap 4 every car keeps its pre-race pace and usage although laps are measured in the
    background (not even its best lap of the race is used); when lap 5 is completed the measured values take over
    and hold the laps since the first counted crossing."""
    tracker = _tracker(start_laps=5)
    tracker.update((_vehicle(0), _vehicle(1)), (), GREEN)
    for lap in range(1, 5):
        estimates = _lap(tracker, lap, 1.0 - 0.045 * lap)
        assert tracker.pre_race
        assert (estimates[0].lap_time_s, estimates[0].energy_per_lap) == (pytest.approx(101.0), 0.05)
        assert estimates[1].lap_time_s == pytest.approx(103.02)
        assert tracker.sources == {0: PriorSource("qualifying", "game"), 1: PriorSource("qualifying", "class")}
    assert tracker.measured[0].clean_laps_s == (105.0, 105.0, 105.0)
    assert estimates[0].clean_laps_s == (105.0, 105.0, 105.0)
    estimates = _lap(tracker, 5, 1.0 - 0.045 * 5)
    assert not tracker.pre_race
    assert (estimates[0].lap_time_s, estimates[0].energy_per_lap) == (105.0, pytest.approx(0.045))
    assert tracker.sources == {0: PriorSource("measured", "measured"), 1: PriorSource("measured", "measured")}


def test_history_fills_what_the_event_does_not_give() -> None:
    """Without qualifying file or laps the player gets pace and usage from the history of its car (before the
    garage figure), another car of the class the class history's pace and the player's usage as the class figure;
    the best pace of the class is built from this event only, never from the history."""
    car = CarHistory(101.0, 100.5, 104.0, (None, None, None), 0.046, 0.7, 21, None, None, None, (), 3, 0, ())
    of_class = ClassHistory(pole_lap_s=100.0, race_pace_s=None, energy_per_lap=0.044, fuel_per_energy=None)
    priors = EventPriors("Track", "Car A", {}, {}, car, {"Hyper": of_class})
    tracker = _tracker(priors, GARAGE)
    estimates = tracker.update((_vehicle(0), _vehicle(1, "GT3"), _vehicle(2)), (), GREEN)
    assert (estimates[0].lap_time_s, estimates[0].energy_per_lap, estimates[0].fuel_per_energy) == (104.0, 0.046, 0.7)
    assert estimates[2].lap_time_s == pytest.approx(101.0)
    assert estimates[2].energy_per_lap == 0.046
    assert tracker.sources[0] == PriorSource("history_car", "history_car")
    assert tracker.sources[2] == PriorSource("history_class", "class")
    assert tracker.sources[1] == PriorSource("none", "none")
    alone = _tracker(EventPriors("Track", "Car A", {}, {}, None, {"Hyper": of_class}), None)
    estimates = alone.update((_vehicle(0),), (), GREEN)
    assert (estimates[0].lap_time_s, estimates[0].energy_per_lap) == (pytest.approx(101.0), 0.044)
    assert alone.sources[0] == PriorSource("history_class", "history_class")
