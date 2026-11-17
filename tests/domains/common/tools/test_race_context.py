from types import SimpleNamespace

from ai_race_engineer.domains.common.analysis.session.track_limits import TrackLimitStatus
from ai_race_engineer.domains.common.tools.race_context import RaceContext
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment
from ai_race_engineer.domains.common.data.snapshot import SessionType


def _context(
    in_pits: bool = False,
    lap_type: str | None = None,
    limits: TrackLimitStatus | None = None,
    session_type: SessionType = SessionType.RACE,
) -> RaceContext:
    """Context for the player (slot 0) P2 in class on lap 3 at 300 m, with Ann ahead and a GT3 car between them
    overall.
    In:  in_pits       bool
         lap_type      str | None               provisional type of the lap in progress
         limits        TrackLimitStatus | None  the player's track limits and penalties
         session_type  SessionType
    Out:               RaceContext
    """

    def vehicle(slot_id: int, name: str, class_name: str, place: int) -> SimpleNamespace:
        return SimpleNamespace(
            slot_id=slot_id,
            driver_name=name,
            class_name=class_name,
            place=place,
            completed_laps=2,
            lap_distance_m=300.0,
            in_pits=in_pits,
        )

    me = vehicle(0, "Balazs Szakter", "Hyper", 3)
    snapshot = SimpleNamespace(
        session=SimpleNamespace(session_type=session_type),
        player=SimpleNamespace(vehicle=me, virtual_energy=0.614),
        vehicles=[vehicle(1, "Ann Gamble", "Hyper", 1), vehicle(9, "Gus Powell", "GT3", 2), me],
    )
    projection = SimpleNamespace(cars={0: SimpleNamespace(laps_remaining=3.44)})
    tracker = SimpleNamespace(state=SimpleNamespace(player_slot_id=0, before_green=False, projection=projection))
    segments = SimpleNamespace(latest=snapshot, segments=(Segment("turn 7", True, 250.0, 400.0, 320.0),))
    return RaceContext(tracker, segments, lambda: lap_type, lambda: limits)


def test_context_says_session_lap_laps_left_place_segment_and_energy() -> None:
    """One line with where the driver is now; in the pits it says so instead of the lap."""
    assert _context()() == "Now: race, on lap 3, 3.4 laps to go, P2 in class, at turn 7, 61 percent energy."
    assert _context(in_pits=True)().startswith("Now: race, in the pits, ")


def test_context_without_a_session_and_driver_names() -> None:
    """Without a snapshot there is no session; the names are the family names of the drivers."""
    empty = RaceContext(
        SimpleNamespace(state=None), SimpleNamespace(latest=None, segments=()), lambda: None, lambda: None
    )
    assert (empty(), empty.driver_names()) == ("Now: no session running.", "")
    assert _context().driver_names() == "Drivers: Gamble, Powell, Szakter."


def test_context_names_the_type_of_the_lap_in_progress() -> None:
    """A known lap type follows the lap number in brackets; in the pits there is none."""
    assert _context(lap_type="invalid lap")().startswith("Now: race, on lap 3 (invalid lap), 3.4 laps to go, ")
    assert _context(in_pits=True, lap_type="out lap")().startswith("Now: race, in the pits, ")


def test_context_gives_the_track_limits_in_a_race_only() -> None:
    """Points and penalties above zero end the line in a race; zero counts and other sessions add nothing."""
    limits = TrackLimitStatus(1.25, 5.0, 1, ())
    end = "61 percent energy, track limits 1.25 of 5 points, 1 penalty outstanding."
    assert _context(limits=limits)().endswith(end)
    assert _context(limits=TrackLimitStatus(0.5, 5.0, 0, ()))().endswith("energy, track limits 0.5 of 5 points.")
    assert _context(limits=TrackLimitStatus(0.0, 5.0, 2, ()))().endswith("energy, 2 penalties outstanding.")
    assert _context(limits=TrackLimitStatus(0.0, 5.0, 0, ()))().endswith("61 percent energy.")
    assert _context(limits=limits, session_type=SessionType.QUALIFY)().endswith("61 percent energy.")
