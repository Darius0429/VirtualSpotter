from collections.abc import Callable

import pytest

from ai_race_engineer.events.base import Priority
from ai_race_engineer.events.spotter import Spotter
from ai_race_engineer.telemetry.snapshot import Snapshot, VehicleState

type At = Callable[..., Snapshot]


@pytest.fixture
def spotter() -> Spotter:
    """Spotter with the default config values.
    In:  -
    Out: Spotter
    """
    return Spotter(4.8, 2.0, 6.0, 0.5, 0.6, 2.0, 8.0)


@pytest.fixture
def at(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> At:
    """Factory: snapshot with the player at the origin (heading 0) and cars at given (lateral, longitudinal) offsets.
    In:  make_vehicle   Callable[..., VehicleState]
         make_snapshot  Callable[..., Snapshot]
    Out:                At  (*offsets, player_speed_mps=30.0, **car_overrides) -> Snapshot
    """

    def build(*offsets: tuple[float, float], player_speed_mps: float = 30.0, **car_overrides: object) -> Snapshot:
        """Snapshot with cars at the given offsets.
        In:  offsets           tuple[float, float]  m (lateral + right, longitudinal + ahead) per car
             player_speed_mps  float                m/s
             car_overrides     object               VehicleState overrides for every other car
        Out:                   Snapshot
        """
        me = make_vehicle(slot_id=0, is_player=True, speed_mps=player_speed_mps)
        cars = [make_vehicle(slot_id=i + 1, position_xz_m=xz, **car_overrides) for i, xz in enumerate(offsets)]
        return make_snapshot(me, cars)

    return build


def _text(spotter: Spotter, snapshot: Snapshot, now_s: float) -> str | None:
    """Spotter callout text, or None.
    In:  spotter   Spotter
         snapshot  Snapshot
         now_s     float     s
    Out:           str | None
    """
    callout = spotter.check(snapshot, now_s)
    return None if callout is None else callout.text


def test_car_left(spotter: Spotter, at: At) -> None:
    """A car on the left is called at once."""
    callout = spotter.check(at((-3.0, 0.0)), 0.0)
    assert callout is not None
    assert (callout.kind, callout.text, callout.priority) == ("spotter", "car left", Priority.SPOTTER)
    assert callout.facts == {"left": 1, "right": 0}


def test_car_right(spotter: Spotter, at: At) -> None:
    """A car on the right is called at once."""
    assert _text(spotter, at((3.0, 0.0)), 0.0) == "car right"


def test_three_wide(spotter: Spotter, at: At) -> None:
    """Cars on both sides are three wide."""
    assert _text(spotter, at((-3.0, 0.0), (3.0, 1.0)), 0.0) == "three wide"


def test_ahead_beyond_length_ignored(spotter: Spotter, at: At) -> None:
    """A car further ahead than length plus margin is not beside."""
    assert _text(spotter, at((-3.0, 5.4)), 0.0) is None


def test_beyond_lateral_range_ignored(spotter: Spotter, at: At) -> None:
    """A car beyond the lateral range is not beside."""
    assert _text(spotter, at((6.5, 0.0)), 0.0) is None


def test_within_half_width_ignored(spotter: Spotter, at: At) -> None:
    """A car with lateral offset below half the width is not beside."""
    assert _text(spotter, at((0.5, 3.0)), 0.0) is None


def test_pits_and_garage_ignored(spotter: Spotter, at: At) -> None:
    """Cars in the pits or garage are ignored."""
    assert _text(spotter, at((-3.0, 0.0), in_pits=True), 0.0) is None
    assert _text(spotter, at((-3.0, 0.0), in_garage=True), 0.1) is None


def test_clear_left_after_hold(spotter: Spotter, at: At) -> None:
    """'clear left' only once the side was empty for the hold time (and suppress time passed)."""
    spotter.check(at((-3.0, 0.0)), 0.0)
    spotter.check(at((-3.0, 0.0)), 2.0)
    assert _text(spotter, at(), 2.3) is None
    assert _text(spotter, at(), 2.6) == "clear left"


def test_return_within_hold_silent(spotter: Spotter, at: At) -> None:
    """A car that comes back within the hold time causes no callout."""
    spotter.check(at((-3.0, 0.0)), 0.0)
    spotter.check(at((-3.0, 0.0)), 2.0)
    assert _text(spotter, at(), 2.3) is None
    assert _text(spotter, at((-3.0, 0.0)), 2.4) is None
    assert _text(spotter, at(), 2.9) is None


def test_clear_delayed_by_repeat_suppress(spotter: Spotter, at: At) -> None:
    """'clear' waits until repeat_suppress_s after the call even when the hold has passed."""
    spotter.check(at((-3.0, 0.0)), 0.0)
    assert _text(spotter, at(), 1.0) is None
    assert _text(spotter, at(), 1.9) is None
    assert _text(spotter, at(), 2.0) == "clear left"


def test_three_wide_right_leaves(spotter: Spotter, at: At) -> None:
    """From three wide, the right car leaving gives 'clear right' after the hold."""
    spotter.check(at((-3.0, 0.0), (3.0, 0.0)), 0.0)
    spotter.check(at((-3.0, 0.0), (3.0, 0.0)), 2.0)
    assert _text(spotter, at((-3.0, 0.0)), 2.3) is None
    assert _text(spotter, at((-3.0, 0.0)), 2.6) == "clear right"


def test_three_wide_both_leave(spotter: Spotter, at: At) -> None:
    """From three wide, both cars leaving gives 'clear'."""
    spotter.check(at((-3.0, 0.0), (3.0, 0.0)), 0.0)
    assert _text(spotter, at(), 3.0) == "clear"


def test_left_to_right_directly(spotter: Spotter, at: At) -> None:
    """Left car gone and right car arriving in the same tick gives 'car right'."""
    spotter.check(at((-3.0, 0.0)), 0.0)
    callout = spotter.check(at((3.0, 0.0)), 3.0)
    assert callout is not None
    assert callout.text == "car right"
    assert callout.facts == {"left": 0, "right": 1}


def test_held_left_then_lone_right_is_car_right(spotter: Spotter, at: At) -> None:
    """Right car arriving while the empty left side is still held gives 'car right', not 'three wide'."""
    for t in (0.0, 1.0, 2.0, 3.0):
        spotter.check(at((-3.0, 0.0)), t)
    assert _text(spotter, at(), 3.1) is None
    assert _text(spotter, at((3.0, 0.0)), 3.3) == "car right"
    assert _text(spotter, at((3.0, 0.0)), 3.8) is None


def test_held_left_car_returns_after_lone_right_is_three_wide(spotter: Spotter, at: At) -> None:
    """If the left car is really still there and reappears, 'three wide' is called at once."""
    for t in (0.0, 1.0, 2.0, 3.0):
        spotter.check(at((-3.0, 0.0)), t)
    spotter.check(at(), 3.1)
    spotter.check(at((3.0, 0.0)), 3.3)
    assert _text(spotter, at((-3.0, 0.0), (3.0, 0.0)), 3.4) == "three wide"


def test_slow_player_silent_reset(spotter: Spotter, at: At) -> None:
    """Below min speed the spotter resets silently and does not call 'clear'."""
    spotter.check(at((-3.0, 0.0)), 0.0)
    assert _text(spotter, at((-3.0, 0.0), player_speed_mps=5.0), 1.0) is None
    assert _text(spotter, at(), 5.0) is None
    assert _text(spotter, at((-3.0, 0.0)), 6.0) == "car left"


def test_reset_clean_state(spotter: Spotter, at: At) -> None:
    """After reset an occupied side is called again."""
    spotter.check(at((-3.0, 0.0)), 0.0)
    spotter.reset()
    assert _text(spotter, at((-3.0, 0.0)), 0.1) == "car left"


def test_many_cars(spotter: Spotter, at: At) -> None:
    """Many cars in one tick: only those beside count."""
    cars = [(-3.0, 1.0), (-4.0, -2.0), (0.0, 20.0), (10.0, 0.0), (0.0, -8.0), (1.5, -30.0)]
    assert _text(spotter, at(*cars), 0.0) == "car left"
