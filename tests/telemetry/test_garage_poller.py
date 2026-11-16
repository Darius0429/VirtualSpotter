import dataclasses

import pytest

from ai_race_engineer.telemetry.garage import GarageInfo
from ai_race_engineer.telemetry.garage_poller import GaragePoller
from ai_race_engineer.telemetry.snapshot import PlayerState, SessionState, SessionType, Snapshot, VehicleState

FIGURES: GarageInfo = GarageInfo(0.69, 0.044)


def _snapshot(
    elapsed_s: float,
    session_type: SessionType = SessionType.QUALIFY,
    track_name: str = "Track",
    vehicle_name: str = "car 1",
) -> Snapshot:
    """Snapshot of the player on track.
    In:  elapsed_s     float  s
         session_type  SessionType
         track_name    str
         vehicle_name  str
    Out:               Snapshot
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    vehicle = VehicleState(**{**base, "is_player": True, "vehicle_name": vehicle_name})
    player = PlayerState(vehicle, 50.0, 0.5, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, "", "", (0.0,) * 4)
    session = SessionState(session_type, elapsed_s, 3600.0, 1000, track_name, 5000.0, 20.0, 20.0)
    return Snapshot(session, player, (vehicle,))


def _poller() -> GaragePoller:
    """Poller that holds figures read in a qualifying session.
    In:  -
    Out: GaragePoller
    """
    poller = GaragePoller(lambda: None, 5.0)
    poller.update(_snapshot(100.0))
    poller._latest = FIGURES
    return poller


def test_figures_are_kept_within_a_session() -> None:
    """The same session and car with the clock moving on keeps the figures and asks for no read."""
    poller = _poller()
    poller.update(_snapshot(101.0))
    assert (poller.latest(), poller.fuel_ratio(), poller._due) == (FIGURES, 0.69, False)


@pytest.mark.parametrize(
    "new",
    [
        _snapshot(200.0, session_type=SessionType.RACE),
        _snapshot(200.0, track_name="Other"),
        _snapshot(200.0, vehicle_name="car 2"),
        _snapshot(5.0),
    ],
)
def test_new_session_or_car_drops_the_figures_and_asks_for_one_read(new: Snapshot) -> None:
    """Another session type, track or car, or a restarted clock, forgets the figures and owes one read."""
    poller = _poller()
    poller.update(new)
    assert (poller.latest(), poller.fuel_ratio(), poller._due) == (None, None, True)


def test_read_in_flight_during_a_change_is_discarded() -> None:
    """Figures read while the session changes belong to the old session: they are dropped and the read stays owed."""
    poller = _poller()
    poller._read = lambda: poller.update(_snapshot(5.0)) or GarageInfo(0.8, 0.05)
    poller._poll()
    assert (poller.latest(), poller._due) == (None, True)
    poller._read = lambda: GarageInfo(0.8, 0.05)
    poller._poll()
    assert poller.latest() == GarageInfo(0.8, 0.05)
