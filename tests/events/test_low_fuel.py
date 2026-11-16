import dataclasses
from collections.abc import Callable

import pytest

from ai_race_engineer.events.base import Callout, Priority
from ai_race_engineer.events.low_fuel import LowFuel
from ai_race_engineer.strategy.consumption import ConsumptionTracker
from ai_race_engineer.telemetry.snapshot import Snapshot, VehicleState


def _check(event: LowFuel, tracker: ConsumptionTracker, snapshot: Snapshot) -> Callout | None:
    """Update consumption then check, as the monitor does.
    In:  event     LowFuel
         tracker   ConsumptionTracker  the tracker the event was built with
         snapshot  Snapshot
    Out:           Callout | None
    """
    tracker.update(snapshot.player)
    return event.check(snapshot, 0.0)


def _setup(fuel_seed: float | None = 3.0, energy_seed: float | None = 0.05) -> tuple[LowFuel, ConsumptionTracker]:
    """LowFuel with 2-lap thresholds on a seeded tracker.
    In:  fuel_seed    float | None  litres/lap
         energy_seed  float | None  fraction/lap
    Out:              tuple[LowFuel, ConsumptionTracker]
    """
    tracker = ConsumptionTracker(3, fuel_seed, energy_seed)
    return LowFuel(tracker, 2.0, 2.0), tracker


@pytest.fixture
def leader(make_vehicle: Callable[..., VehicleState]) -> VehicleState:
    """Player as overall leader with a 100 s best lap.
    In:  make_vehicle  Callable[..., VehicleState]
    Out:               VehicleState
    """
    return make_vehicle(slot_id=0, is_player=True, place=1, best_lap_s=100.0)


def test_no_estimate_silent(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """Without a consumption estimate there is no warning."""
    event, tracker = _setup(fuel_seed=None)
    assert _check(event, tracker, make_snapshot(leader, fuel_l=1.0)) is None


def test_above_threshold_silent(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """More laps on board than the threshold is not a warning."""
    event, tracker = _setup()
    assert _check(event, tracker, make_snapshot(leader, fuel_l=7.0)) is None


def test_low_and_short_warns(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """At or below threshold and short of the flag: one callout with laps and litres short."""
    event, tracker = _setup()
    callout = _check(event, tracker, make_snapshot(leader, remaining_s=1000.0, fuel_l=5.0))
    assert callout is not None
    assert callout.kind == "low_fuel"
    assert callout.priority == Priority.STRATEGY
    assert callout.text == "Fuel for 1.7 laps, 28 litres short"
    assert callout.facts == pytest.approx({"laps_on_board": 5.0 / 3.0, "laps_remaining": 11.0, "to_add": 28.0})


def test_low_but_enough_silent(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """Low fuel that still reaches the flag is not a warning."""
    event, tracker = _setup()
    assert _check(event, tracker, make_snapshot(leader, remaining_s=0.0, fuel_l=5.0)) is None


def test_unknown_laps_remaining_warns(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """Without lap times the threshold alone decides."""
    event, tracker = _setup()
    me = make_vehicle(slot_id=0, is_player=True, place=1)
    callout = _check(event, tracker, make_snapshot(me, fuel_l=5.0))
    assert callout is not None
    assert callout.text == "Fuel for 1.7 laps"
    assert callout.facts == pytest.approx({"laps_on_board": 5.0 / 3.0})


def test_latch_and_rearm_after_refuel(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """Warns once per stint; a refuel re-arms the warning."""
    event, tracker = _setup()
    in_pits = dataclasses.replace(leader, in_pits=True)
    assert _check(event, tracker, make_snapshot(leader, fuel_l=5.0)) is not None
    assert _check(event, tracker, make_snapshot(leader, fuel_l=4.9)) is None
    assert _check(event, tracker, make_snapshot(in_pits, fuel_l=60.0)) is None
    assert _check(event, tracker, make_snapshot(leader, fuel_l=60.0)) is None
    assert _check(event, tracker, make_snapshot(leader, fuel_l=5.0)) is not None


def test_energy_warning(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """Low virtual energy has its own callout in percent."""
    event, tracker = _setup()
    callout = _check(event, tracker, make_snapshot(leader, remaining_s=1000.0, fuel_l=50.0, virtual_energy=0.08))
    assert callout is not None
    assert callout.kind == "low_energy"
    assert callout.text == "Energy for 1.6 laps, 47 percent short"


def test_fuel_first_then_energy(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """When both are low, fuel is called first and energy on the next tick."""
    event, tracker = _setup()
    snap = make_snapshot(leader, fuel_l=5.0, virtual_energy=0.08)
    first = _check(event, tracker, snap)
    second = _check(event, tracker, snap)
    assert first is not None and first.kind == "low_fuel"
    assert second is not None and second.kind == "low_energy"


def test_energy_none_ignored(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """Without virtual energy only fuel is checked."""
    event, tracker = _setup()
    assert _check(event, tracker, make_snapshot(leader, fuel_l=50.0, virtual_energy=None)) is None


def test_in_pits_silent(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """No warnings while the player is in the pits."""
    event, tracker = _setup()
    me = make_vehicle(slot_id=0, is_player=True, place=1, best_lap_s=100.0, in_pits=True)
    assert _check(event, tracker, make_snapshot(me, fuel_l=1.0)) is None


def test_leader_based_laps_remaining(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """Laps remaining follow the leader: 2 laps for the player here, while the player's own clock gives only 1."""
    event, tracker = _setup()
    me = make_vehicle(slot_id=0, is_player=True, place=2, best_lap_s=50.0)
    leader = make_vehicle(slot_id=1, place=1, best_lap_s=100.0)
    callout = _check(event, tracker, make_snapshot(me, [leader], remaining_s=10.0, fuel_l=4.5))
    assert callout is not None
    assert callout.facts["laps_remaining"] == pytest.approx(2.0)


def test_reset_rearms(leader: VehicleState, make_snapshot: Callable[..., Snapshot]) -> None:
    """Reset re-arms the warning."""
    event, tracker = _setup()
    assert _check(event, tracker, make_snapshot(leader, fuel_l=5.0)) is not None
    event.reset()
    assert _check(event, tracker, make_snapshot(leader, fuel_l=4.9)) is not None


def test_no_car_in_p1_warns_on_threshold_alone(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """A torn or transitional read with no car in P1 does not crash; laps remaining is treated as unknown."""
    event, tracker = _setup()
    player = make_vehicle(slot_id=0, is_player=True, place=0, best_lap_s=100.0)
    callout = _check(event, tracker, make_snapshot(player, fuel_l=5.0))
    assert callout is not None
    assert "laps_remaining" not in callout.facts
