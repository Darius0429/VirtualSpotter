import dataclasses

import pytest

from ai_race_engineer.domains.common.analysis.strategy.consumption import ConsumptionEstimate, ConsumptionTracker
from ai_race_engineer.domains.common.data.snapshot import PlayerState, VehicleState


def _player(completed_laps: int, fuel_l: float, virtual_energy: float | None) -> PlayerState:
    """Player state with the given lap count, fuel and energy and neutral other values.
    In:  completed_laps  int
         fuel_l          float         litres
         virtual_energy  float | None  fraction 0-1
    Out:                 PlayerState
    """
    vehicle = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    vehicle.update(completed_laps=completed_laps, driver_name="", vehicle_name="", class_name="", position_xz_m=(0, 0))
    player = dict.fromkeys((f.name for f in dataclasses.fields(PlayerState)), 0)
    player.update(vehicle=VehicleState(**vehicle), fuel_l=fuel_l, virtual_energy=virtual_energy)
    return PlayerState(**player)


def test_fuel_estimate_after_laps() -> None:
    """Measured laps give the average fuel per lap."""
    tracker = ConsumptionTracker(window_laps=3)
    tracker.update(_player(0, 50.0, None))
    tracker.update(_player(1, 47.0, None))
    assert tracker.update(_player(2, 43.0, None)).fuel_l_per_lap == pytest.approx(3.5)


def test_energy_fed_when_present() -> None:
    """Virtual energy is tracked when the game provides it."""
    tracker = ConsumptionTracker(window_laps=3)
    tracker.update(_player(0, 50.0, 0.9))
    assert tracker.update(_player(1, 47.0, 0.85)).energy_per_lap == pytest.approx(0.05)


def test_no_energy_stays_none() -> None:
    """Without virtual energy (rF2) the energy estimate stays None, even with a seed."""
    tracker = ConsumptionTracker(window_laps=3, energy_seed_per_lap=0.05)
    tracker.update(_player(0, 50.0, None))
    assert tracker.update(_player(1, 47.0, None)).energy_per_lap is None


def test_seeds_before_first_lap() -> None:
    """Seeds are used until a lap is measured."""
    tracker = ConsumptionTracker(window_laps=3, fuel_seed_l_per_lap=4.0, energy_seed_per_lap=0.06)
    assert tracker.update(_player(0, 50.0, 0.9)) == ConsumptionEstimate(4.0, 0.06)


def test_reset_keeps_seeds_drops_laps() -> None:
    """Reset forgets measured laps but keeps the seeds."""
    tracker = ConsumptionTracker(window_laps=3, fuel_seed_l_per_lap=4.0)
    tracker.update(_player(0, 50.0, None))
    tracker.update(_player(1, 47.0, None))
    tracker.reset()
    assert tracker.update(_player(1, 47.0, None)).fuel_l_per_lap == 4.0


def test_estimate_equals_update_return() -> None:
    """The stored estimate is the last update's result."""
    tracker = ConsumptionTracker(window_laps=3, fuel_seed_l_per_lap=4.0)
    result = tracker.update(_player(0, 50.0, None))
    assert tracker.estimate is result
