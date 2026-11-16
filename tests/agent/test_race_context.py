from types import SimpleNamespace

from ai_race_engineer.agent.race_context import RaceContext
from ai_race_engineer.strategy.track_map import Segment
from ai_race_engineer.telemetry.snapshot import SessionType


def _context(in_pits: bool = False) -> RaceContext:
    """Context for the player (slot 0) P2 in class on lap 3 at 300 m, with Ann ahead and a GT3 car between them
    overall.
    In:  in_pits  bool
    Out:          RaceContext
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
        session=SimpleNamespace(session_type=SessionType.RACE),
        player=SimpleNamespace(vehicle=me, virtual_energy=0.614),
        vehicles=[vehicle(1, "Ann Gamble", "Hyper", 1), vehicle(9, "Gus Powell", "GT3", 2), me],
    )
    projection = SimpleNamespace(cars={0: SimpleNamespace(laps_remaining=3.44)})
    tracker = SimpleNamespace(state=SimpleNamespace(player_slot_id=0, before_green=False, projection=projection))
    segments = SimpleNamespace(latest=snapshot, segments=(Segment("turn 7", True, 250.0, 400.0, 320.0),))
    return RaceContext(tracker, segments)


def test_context_says_session_lap_laps_left_place_segment_and_energy() -> None:
    """One line with where the driver is now; in the pits it says so instead of the lap."""
    assert _context()() == "Now: race, on lap 3, 3.4 laps to go, P2 in class, at turn 7, 61 percent energy."
    assert _context(in_pits=True)().startswith("Now: race, in the pits, ")


def test_context_without_a_session_and_driver_names() -> None:
    """Without a snapshot there is no session; the names are the family names of the drivers."""
    empty = RaceContext(SimpleNamespace(state=None), SimpleNamespace(latest=None, segments=()))
    assert (empty(), empty.driver_names()) == ("Now: no session running.", "")
    assert _context().driver_names() == "Drivers: Gamble, Powell, Szakter."
