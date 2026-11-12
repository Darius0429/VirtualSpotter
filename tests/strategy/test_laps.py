import dataclasses

import pytest

from ai_race_engineer.strategy.laps import lap_fraction, laps_remaining, leader_laps_remaining, reference_lap_time_s
from ai_race_engineer.telemetry.snapshot import VehicleState

LAP_S = 100.0


def _vehicle(best_lap_s: float | None) -> VehicleState:
    """Vehicle with the given best lap and neutral other values.
    In:  best_lap_s  float | None  s
    Out:             VehicleState
    """
    names = {f.name for f in dataclasses.fields(VehicleState)}
    base = dict.fromkeys(names, 0)
    base.update(driver_name="", vehicle_name="", class_name="", position_xz_m=(0.0, 0.0), best_lap_s=best_lap_s)
    return VehicleState(**base)


def test_lap_fraction() -> None:
    """Distance over track length."""
    assert lap_fraction(1500.0, 6000.0) == pytest.approx(0.25)


def test_timed_player_same_lap_behind_leader() -> None:
    """Leader 10.4 with 250 s left crosses at 13 after the clock ends; player 10.2 then finishes at 13."""
    assert leader_laps_remaining(250.0, 1000, 10.4, LAP_S) == pytest.approx(2.6)
    assert laps_remaining(250.0, 1000, 10.4, LAP_S, 10.2, LAP_S, False) == pytest.approx(2.8)


def test_timed_player_lapped() -> None:
    """Lapped player finishes one lap short of the leader's lap count."""
    assert leader_laps_remaining(240.0, 1000, 10.5, LAP_S) == pytest.approx(2.5)
    assert laps_remaining(240.0, 1000, 10.5, LAP_S, 9.3, LAP_S, False) == pytest.approx(2.7)


def test_player_is_leader() -> None:
    """The leader's own value is returned, also when progress plus laps lands on an integer."""
    assert laps_remaining(250.0, 1000, 10.4, LAP_S, 10.4, LAP_S, True) == pytest.approx(2.6)
    assert laps_remaining(250.0, 1000, 10.5, LAP_S, 10.5, LAP_S, True) == pytest.approx(3.5)


def test_lap_limited() -> None:
    """max_laps caps the race when the clock is not limiting."""
    assert leader_laps_remaining(1e9, 20, 17.25, LAP_S) == pytest.approx(2.75)
    assert laps_remaining(1e9, 20, 17.25, LAP_S, 17.0, LAP_S, False) == pytest.approx(3.0)


@pytest.mark.parametrize("remaining_s", [0.0, -30.0])
def test_clock_expired_finishes_at_next_crossing(remaining_s: float) -> None:
    """With the clock at or past zero the leader finishes at the next line crossing."""
    assert leader_laps_remaining(remaining_s, 1000, 10.4, LAP_S) == pytest.approx(0.6)


def test_crossing_exactly_at_zero_needs_one_more_lap() -> None:
    """A crossing exactly when the clock hits zero does not end the race."""
    assert leader_laps_remaining(300.0, 1000, 10.0, LAP_S) == pytest.approx(4.0)


def test_slower_player_drives_fewer_laps() -> None:
    """A slower player lap time yields fewer laps than at leader pace and than the leader."""
    slow = laps_remaining(250.0, 1000, 10.4, LAP_S, 10.0, 150.0, False)
    at_pace = laps_remaining(250.0, 1000, 10.4, LAP_S, 10.0, LAP_S, False)
    assert slow == pytest.approx(2.0)
    assert slow < at_pace
    assert slow < leader_laps_remaining(250.0, 1000, 10.4, LAP_S)


def test_reference_lap_time_is_best_lap() -> None:
    """The reference lap time is the car's best lap."""
    assert reference_lap_time_s(_vehicle(98.7)) == 98.7


def test_reference_lap_time_none_without_best_lap() -> None:
    """No best lap gives no reference lap time."""
    assert reference_lap_time_s(_vehicle(None)) is None
