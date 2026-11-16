import dataclasses

import pytest

from ai_race_engineer.strategy.straight import on_straight
from ai_race_engineer.strategy.straight_live import LiveStraightDetector
from ai_race_engineer.telemetry.snapshot import PlayerState, SessionState, SessionType, Snapshot, VehicleState


def _snapshot(elapsed_s: float, steering: float = 0.0, lateral_mps2: float = 0.0, speed_mps: float = 60.0) -> Snapshot:
    """Snapshot with the player's steering, lateral acceleration and speed at a session time.
    In:  elapsed_s     float  s
         steering      float  -1 to 1
         lateral_mps2  float  m/s^2
         speed_mps     float  m/s
    Out:               Snapshot
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(is_player=True, speed_mps=speed_mps)
    vehicle = VehicleState(**base)
    session = SessionState(SessionType.RACE, elapsed_s, 3600.0, 1000, "Track", 1000.0, 20.0, 20.0)
    player = PlayerState(
        vehicle, 50.0, None, None, 0.0, 0.0, 0.0, steering, 0.0, 0, lateral_mps2, 0.0, 0.0, "", "", (0.0, 0.0, 0.0, 0.0)
    )
    return Snapshot(session, player, (vehicle,))


def _detector() -> LiveStraightDetector:
    """Detector with the config thresholds: 0.08 steering, 3 m/s^2, 30 m/s, 1 s hold.
    In:  -
    Out: LiveStraightDetector
    """
    return LiveStraightDetector(max_steering=0.08, max_lateral_mps2=3.0, min_speed_mps=30.0, hold_s=1.0)


@pytest.mark.parametrize(
    ("steering", "lateral_mps2", "speed_mps", "expected"),
    [
        (0.08, 3.0, 30.0, True),
        (-0.08, -3.0, 80.0, True),
        (0.09, 0.0, 60.0, False),
        (-0.09, 0.0, 60.0, False),
        (0.0, 3.1, 60.0, False),
        (0.0, -3.1, 60.0, False),
        (0.0, 0.0, 29.9, False),
    ],
)
def test_on_straight_thresholds(steering: float, lateral_mps2: float, speed_mps: float, expected: bool) -> None:
    """Each threshold is inclusive and applies to both sides."""
    assert on_straight(steering, lateral_mps2, speed_mps, 0.08, 3.0, 30.0) is expected


def test_straight_only_after_the_hold_time() -> None:
    """Driving straight counts once it has lasted hold_s."""
    detector = _detector()
    detector.update(_snapshot(10.0))
    assert not detector.is_straight()
    detector.update(_snapshot(10.8))
    assert not detector.is_straight()
    detector.update(_snapshot(11.0))
    assert detector.is_straight()


def test_corner_resets_the_hold() -> None:
    """A corner ends the straight and the hold time starts again."""
    detector = _detector()
    detector.update(_snapshot(10.0))
    detector.update(_snapshot(11.0))
    detector.update(_snapshot(11.2, steering=0.3))
    assert not detector.is_straight()
    detector.update(_snapshot(11.4))
    detector.update(_snapshot(12.2))
    assert not detector.is_straight()
    detector.update(_snapshot(12.4))
    assert detector.is_straight()


def test_slow_car_is_not_on_a_straight() -> None:
    """Below the minimum speed the detector stays false."""
    detector = _detector()
    detector.update(_snapshot(10.0, speed_mps=10.0))
    detector.update(_snapshot(12.0, speed_mps=10.0))
    assert not detector.is_straight()
