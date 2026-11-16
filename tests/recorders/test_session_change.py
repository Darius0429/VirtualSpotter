import dataclasses

from ai_race_engineer.recorders.session_change import is_new_session
from ai_race_engineer.telemetry.frame import SessionInfo
from ai_race_engineer.telemetry.snapshot import SessionType

_INFO = SessionInfo(
    game="lmu",
    track_name="Sebring",
    track_length_m=6000.0,
    vehicle_name="Car #1",
    class_name="GT3",
    session_type=SessionType.PRACTICE,
    session_number=1,
    max_laps=100,
)


def test_first_frame() -> None:
    """The first frame starts a session."""
    assert is_new_session(None, _INFO, 0.0, 10.0)


def test_continuing_session() -> None:
    """Same session with the clock moving forward continues."""
    assert not is_new_session(_INFO, _INFO, 10.0, 10.02)


def test_session_number_or_type_change() -> None:
    """A new session number or session type starts a session."""
    assert is_new_session(_INFO, dataclasses.replace(_INFO, session_number=2), 10.0, 10.02)
    assert is_new_session(_INFO, dataclasses.replace(_INFO, session_type=SessionType.RACE), 10.0, 10.02)


def test_track_or_vehicle_change() -> None:
    """A new track or vehicle starts a session."""
    assert is_new_session(_INFO, dataclasses.replace(_INFO, track_name="Spa"), 10.0, 10.02)
    assert is_new_session(_INFO, dataclasses.replace(_INFO, vehicle_name="Car #2"), 10.0, 10.02)


def test_clock_backwards() -> None:
    """The clock going backwards (restart) starts a session."""
    assert is_new_session(_INFO, _INFO, 10.0, 1.0)
