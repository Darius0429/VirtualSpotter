from collections.abc import Callable

from ai_race_engineer.events.base import Priority
from ai_race_engineer.events.fastest_lap import FastestLap
from ai_race_engineer.telemetry.snapshot import Snapshot, VehicleState


def _seeded(make_snapshot: Callable[..., Snapshot], vehicles: list[VehicleState]) -> FastestLap:
    """FastestLap after a silent seeding check.
    In:  make_snapshot  Callable[..., Snapshot]
         vehicles       list[VehicleState]  player first, then rivals
    Out:                FastestLap
    """
    event = FastestLap()
    assert event.check(make_snapshot(vehicles[0], vehicles[1:]), 0.0) is None
    return event


def test_seed_is_silent(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """Existing times are not announced on the first check."""
    me = make_vehicle(slot_id=0, is_player=True, best_lap_s=100.0)
    assert FastestLap().check(make_snapshot(me, [make_vehicle(slot_id=1, best_lap_s=99.0)]), 0.0) is None


def test_rival_overall_best(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """A rival beating the overall best is announced."""
    me = make_vehicle(slot_id=0, is_player=True, best_lap_s=100.0)
    event = _seeded(make_snapshot, [me, make_vehicle(slot_id=1, best_lap_s=99.0)])
    callout = event.check(make_snapshot(me, [make_vehicle(slot_id=1, driver_name="Ann", best_lap_s=98.5)]), 1.0)
    assert callout is not None
    assert callout.kind == "fastest_lap"
    assert callout.priority == Priority.INFO
    assert callout.text == "Ann set the fastest time of the session with a 1:38.5"
    assert callout.facts == {"driver": "Ann", "class_name": "GT3", "lap_s": 98.5, "is_player": 0}


def test_player_overall_best_no_pb_after(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """The player's overall best gives one 'You set' callout and no personal best afterwards."""
    rival = make_vehicle(slot_id=1, best_lap_s=99.0)
    event = _seeded(make_snapshot, [make_vehicle(slot_id=0, is_player=True, best_lap_s=100.0), rival])
    me = make_vehicle(slot_id=0, is_player=True, best_lap_s=98.0)
    callout = event.check(make_snapshot(me, [rival]), 1.0)
    assert callout is not None
    assert callout.text == "You set the fastest time of the session with a 1:38.0"
    assert callout.facts["is_player"] == 1
    assert event.check(make_snapshot(me, [rival]), 2.0) is None


def test_player_pb_only(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """A player improvement that is not the overall best is a personal best."""
    rival = make_vehicle(slot_id=1, best_lap_s=95.0)
    event = _seeded(make_snapshot, [make_vehicle(slot_id=0, is_player=True, best_lap_s=100.0), rival])
    callout = event.check(make_snapshot(make_vehicle(slot_id=0, is_player=True, best_lap_s=99.0), [rival]), 1.0)
    assert callout is not None
    assert callout.kind == "personal_best"
    assert callout.text == "You set your fastest lap with a 1:39.0"
    assert callout.facts == {"lap_s": 99.0}


def test_first_player_lap_is_pb(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """A first player lap after seeding without a time is a personal best."""
    rival = make_vehicle(slot_id=1, best_lap_s=95.0)
    event = _seeded(make_snapshot, [make_vehicle(slot_id=0, is_player=True), rival])
    callout = event.check(make_snapshot(make_vehicle(slot_id=0, is_player=True, best_lap_s=99.0), [rival]), 1.0)
    assert callout is not None
    assert callout.kind == "personal_best"


def test_rival_overall_and_player_pb_same_tick(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """Overall best is announced first, the player's personal best on the next tick."""
    event = _seeded(
        make_snapshot,
        [make_vehicle(slot_id=0, is_player=True, best_lap_s=100.0), make_vehicle(slot_id=1, best_lap_s=99.0)],
    )
    snap = make_snapshot(
        make_vehicle(slot_id=0, is_player=True, best_lap_s=99.5), [make_vehicle(slot_id=1, best_lap_s=98.0)]
    )
    first = event.check(snap, 1.0)
    second = event.check(snap, 1.05)
    assert first is not None and first.kind == "fastest_lap"
    assert second is not None and second.kind == "personal_best"
    assert event.check(snap, 1.1) is None


def test_other_class_overall_best_announced(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """An overall best from another class is announced too."""
    me = make_vehicle(slot_id=0, is_player=True, best_lap_s=100.0)
    event = _seeded(make_snapshot, [me, make_vehicle(slot_id=1, class_name="Hypercar", best_lap_s=90.0)])
    callout = event.check(make_snapshot(me, [make_vehicle(slot_id=1, class_name="Hypercar", best_lap_s=89.0)]), 1.0)
    assert callout is not None
    assert callout.facts["class_name"] == "Hypercar"


def test_slower_times_silent(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """Improvements that beat neither stored time are not announced."""
    me = make_vehicle(slot_id=0, is_player=True, best_lap_s=100.0)
    event = _seeded(make_snapshot, [me, make_vehicle(slot_id=1, best_lap_s=99.0), make_vehicle(slot_id=2)])
    snap = make_snapshot(me, [make_vehicle(slot_id=1, best_lap_s=99.0), make_vehicle(slot_id=2, best_lap_s=101.0)])
    assert event.check(snap, 1.0) is None


def test_reset_reseeds(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """After reset the next check seeds silently again."""
    me = make_vehicle(slot_id=0, is_player=True, best_lap_s=100.0)
    event = _seeded(make_snapshot, [me, make_vehicle(slot_id=1, best_lap_s=99.0)])
    event.reset()
    assert event.check(make_snapshot(me, [make_vehicle(slot_id=1, best_lap_s=120.0)]), 1.0) is None
    assert event.check(make_snapshot(me, [make_vehicle(slot_id=1, best_lap_s=99.9)]), 2.0) is not None
