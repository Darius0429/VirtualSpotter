import dataclasses

import pytest

from ai_race_engineer.domains.common.analysis.field.field import CarEstimate, FieldTracker
from ai_race_engineer.domains.common.data.race_snapshot import CarResources
from ai_race_engineer.domains.common.data.snapshot import VehicleState

GREEN: int = 5
FORMATION: int = 3


class _Car:
    """One car fed to a field tracker lap by lap.
    tracker  FieldTracker  3-lap windows
    _laps    int           completed laps
    _time_s  float         s session time the current lap started
    """

    def __init__(self) -> None:
        """Start the car at first sight, in the middle of a lap.
        In:  -
        Out: None
        """
        self.tracker: FieldTracker = FieldTracker(3, 3)
        self._laps: int = 0
        self._time_s: float = 50.0

    def see(
        self,
        fuel_l: float | None = None,
        energy: float | None = None,
        in_pits: bool = False,
        game_phase: int = GREEN,
        best_lap_s: float | None = None,
        last_lap_s: float | None = None,
    ) -> CarEstimate:
        """One update during the current lap.
        In:  fuel_l      float | None  litres on board; None = car without telemetry
             energy      float | None  fraction on board
             in_pits     bool
             game_phase  int           mGamePhase
             best_lap_s  float | None  s
             last_lap_s  float | None  s
        Out:             CarEstimate
        """
        base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
        base.update(
            slot_id=1,
            completed_laps=self._laps,
            lap_start_s=self._time_s,
            last_lap_s=last_lap_s,
            best_lap_s=best_lap_s,
            in_pits=in_pits,
        )
        resources = [] if fuel_l is None else [CarResources(1, fuel_l, 100.0, energy, 0)]
        return self.tracker.update([VehicleState(**base)], resources, game_phase)[1]

    def cross(
        self,
        lap_s: float,
        fuel_l: float | None = None,
        energy: float | None = None,
        game_phase: int = GREEN,
        counts: bool = True,
        in_pits: bool = False,
        valid: bool = True,
    ) -> CarEstimate:
        """Cross the line: a new lap starts, the lap just driven took lap_s.
        In:  lap_s       float         s of the lap just driven
             fuel_l      float | None  litres on board at the line
             energy      float | None  fraction on board at the line
             game_phase  int           mGamePhase at the crossing
             counts      bool          the game counts the lap (False for the crossing at the start of the race)
             in_pits     bool          the line is crossed in the pit lane
             valid       bool          the game gives a time for the lap (False: invalidated lap, time -1)
        Out:             CarEstimate
        """
        self._laps += counts
        self._time_s += lap_s
        return self.see(fuel_l, energy, in_pits, game_phase, best_lap_s=lap_s, last_lap_s=lap_s if valid else None)


def test_no_lap_time_without_laps() -> None:
    """A car without a clean lap or a best lap has no lap time."""
    assert _Car().see().lap_time_s is None


def test_best_lap_fallback() -> None:
    """Before the first clean lap the best lap is used."""
    assert _Car().see(best_lap_s=99.0).lap_time_s == 99.0


def test_pace_and_usage_are_means_of_clean_laps() -> None:
    """Laps started at a crossing under green and driven without the pits are averaged."""
    car = _Car()
    car.see(50.0, 0.9)
    car.cross(60.0, 49.0, 0.89)
    car.cross(100.0, 46.5, 0.86)
    estimate = car.cross(102.0, 43.5, 0.82)
    assert estimate.lap_time_s == 101.0
    assert estimate.fuel_l_per_lap == pytest.approx(2.75)
    assert estimate.energy_per_lap == pytest.approx(0.035)


def test_lap_in_progress_at_first_sight_is_dropped() -> None:
    """A tool started mid-lap does not take the rest of that lap for a lap."""
    car = _Car()
    car.see(50.0, 0.9, best_lap_s=99.0)
    estimate = car.cross(40.0, 49.0, 0.89)
    assert estimate == CarEstimate(1, 40.0, None, None, None)
    assert car.cross(100.0, 46.5, 0.86).fuel_l_per_lap == 2.5


def test_lap_of_the_start_is_not_measured() -> None:
    """At the green the game sets a new lap start time without a line crossing (the lap count stays): neither the
    formation lap nor the lap of the standing start is a clean lap; measuring starts at the first counted crossing."""
    car = _Car()
    car.see(50.0, 0.9, game_phase=FORMATION)
    car.see(48.0, 0.88, game_phase=FORMATION)
    car.cross(120.0, 47.5, 0.875, counts=False)
    start_lap = car.cross(114.0, 45.0, 0.845)
    assert (start_lap.clean_laps_s, start_lap.fuel_l_per_lap, start_lap.energy_per_lap) == ((), None, None)
    estimate = car.cross(100.0, 42.5, 0.815)
    assert estimate.clean_laps_s == (100.0,)
    assert estimate.fuel_l_per_lap == 2.5
    assert estimate.energy_per_lap == pytest.approx(0.03)


def test_crossing_before_the_green_starts_nothing() -> None:
    """A line crossing while the phase is not green does not start a measured lap."""
    car = _Car()
    car.see(50.0, 0.9, game_phase=FORMATION)
    car.cross(120.0, 48.0, 0.88, game_phase=FORMATION)
    assert car.cross(100.0, 45.0, 0.85).fuel_l_per_lap is None
    assert car.cross(100.0, 42.5, 0.82).fuel_l_per_lap == 2.5


def test_in_lap_and_out_lap_are_excluded_from_pace_and_usage() -> None:
    """A lap with the pits, also one with only a 2 % splash, changes neither the pace nor the usage average."""
    car = _Car()
    car.see(50.0, 0.5)
    car.cross(60.0, 50.0, 0.5)
    before = car.cross(100.0, 47.5, 0.47)
    car.see(46.0, 0.45, in_pits=True)
    in_lap = car.cross(125.0, 46.0, 0.45, in_pits=True)
    car.see(46.0, 0.47, in_pits=True)
    out_lap = car.cross(115.0, 44.0, 0.45)
    assert in_lap == out_lap == before
    after = car.cross(104.0, 41.0, 0.41)
    assert after.lap_time_s == 102.0
    assert after.fuel_l_per_lap == 2.75
    assert after.energy_per_lap == pytest.approx(0.035)


def test_amount_that_rose_is_skipped() -> None:
    """A lap on which the fuel rose does not enter the usage."""
    car = _Car()
    car.see(50.0, 0.9)
    car.cross(60.0, 50.0, 0.9)
    assert car.cross(100.0, 47.5, 0.85).fuel_l_per_lap == 2.5
    assert car.cross(100.0, 80.0, 1.0).fuel_l_per_lap == 2.5


def test_fuel_per_energy() -> None:
    """2.41 L and 2.83 % per lap give 0.85 L per 1 % energy."""
    car = _Car()
    car.see(50.0, 0.9)
    car.cross(60.0, 50.0, 0.9)
    estimate = car.cross(100.0, 50.0 - 2.41, 0.9 - 0.0283)
    assert estimate.fuel_l_per_lap == pytest.approx(2.41)
    assert estimate.energy_per_lap == pytest.approx(0.0283)
    assert estimate.fuel_per_energy == pytest.approx(0.85, abs=0.005)


def test_lap_decrease_restarts_the_car() -> None:
    """A lap count going down forgets lap times and usage, and the lap in progress is dropped again."""
    car = _Car()
    car.see(50.0, 0.9)
    car.cross(60.0, 50.0, 0.9)
    car.cross(100.0, 47.5, 0.85)
    restarted = _Car()
    restarted.tracker = car.tracker
    assert restarted.see(60.0, 1.0) == CarEstimate(1, None, None, None, None)
    assert restarted.cross(30.0, 59.0, 0.99).fuel_l_per_lap is None


def test_car_without_resources() -> None:
    """A car without telemetry has a pace but no usage."""
    car = _Car()
    car.see()
    car.cross(60.0)
    assert car.cross(100.0) == CarEstimate(1, 100.0, None, None, None, (100.0,))


def test_lap_without_a_time_is_not_a_lap_time() -> None:
    """An invalidated lap (the game gives -1, read as no time) adds no lap time and leaves the window intact; its
    usage still counts."""
    car = _Car()
    car.see(50.0, 0.9)
    car.cross(60.0, 50.0, 0.9)
    car.cross(100.0, 47.5, 0.87)
    estimate = car.cross(98.0, 45.0, 0.84, valid=False)
    assert estimate.clean_laps_s == (100.0,)
    assert estimate.lap_time_s == 100.0
    assert estimate.fuel_l_per_lap == 2.5
    assert car.cross(104.0, 42.5, 0.81).clean_laps_s == (100.0, 104.0)


def test_pace_window_keeps_the_last_laps() -> None:
    """With an 8-lap window the pace is the mean of the last 8 clean laps."""
    car = _Car()
    car.tracker = FieldTracker(8, 3)
    car.see()
    car.cross(60.0)
    for lap_s in range(100, 110):
        estimate = car.cross(float(lap_s))
    assert estimate.clean_laps_s == tuple(float(lap_s) for lap_s in range(102, 110))
    assert estimate.lap_time_s == 105.5
