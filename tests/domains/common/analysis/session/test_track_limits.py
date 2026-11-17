from types import SimpleNamespace

from ai_race_engineer.domains.common.analysis.corners.track_map import Segment
from ai_race_engineer.domains.common.analysis.session.track_limits import PENALTY, TRACK_LIMITS, TrackLimitTracker
from ai_race_engineer.domains.common.data.race_snapshot import PenaltyState
from ai_race_engineer.domains.common.data.snapshot import SessionType

SEGMENTS: tuple[Segment, ...] = (
    Segment("straight 1", False, 0.0, 300.0, None),
    Segment("turn 2", True, 300.0, 600.0, 450.0),
    Segment("straight 3", False, 600.0, 1000.0, None),
)


def _snapshot(
    time_s: float,
    steps: int = 0,
    penalties: int = 0,
    lap_distance_m: float = 100.0,
    lap_time_counts: bool | None = True,
    has_penalties: bool = True,
    session_type: SessionType = SessionType.RACE,
    in_pits: bool = False,
    completed_laps: int = 2,
) -> SimpleNamespace:
    """Snapshot stand-in: the player on lap 3 with 4 steps per point and 20 steps per penalty.
    In:  time_s           float        s session time
         steps            int          track-limit steps held
         penalties        int          penalties outstanding
         lap_distance_m   float        m from the start/finish line
         lap_time_counts  bool | None  the game is timing the lap
         has_penalties    bool         False = the source has no penalty state
         session_type     SessionType
         in_pits          bool         the car is in the pit lane
         completed_laps   int          laps the player has completed
    Out:                  SimpleNamespace
    """
    return SimpleNamespace(
        session=SimpleNamespace(session_type=session_type, track_name="Track", elapsed_s=time_s),
        player=SimpleNamespace(
            vehicle=SimpleNamespace(
                completed_laps=completed_laps,
                lap_distance_m=lap_distance_m,
                lap_time_counts=lap_time_counts,
                in_pits=in_pits,
            )
        ),
        penalties=PenaltyState(steps, 4, 20, penalties) if has_penalties else None,
    )


def _tracker(recorded: list[tuple[str, str, dict[str, object]]]) -> TrackLimitTracker:
    """Tracker on the three-segment map that keeps what it records.
    In:  recorded  list[tuple[str, str, dict[str, object]]]  (kind, text, fields) of every timeline entry added
    Out:           TrackLimitTracker
    """
    return TrackLimitTracker(
        lambda kind, text, fields: recorded.append((kind, text, dict(fields))), lambda: SEGMENTS, 20.0
    )


def test_step_gives_one_event_with_place_and_lap_and_one_entry() -> None:
    """A step counted with the lap timed all along is placed where the car is; with 4 steps per point and 20 per
    penalty one step is 0.25 of 5 points."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0))
    tracker.update(_snapshot(20.0, steps=1, lap_distance_m=500.0))
    tracker.update(_snapshot(21.0, steps=1, lap_distance_m=540.0))
    (event,) = tracker.status.events
    assert (event.time_s, event.lap, event.steps, event.lap_distance_m, event.place) == (
        20.0,
        3,
        1,
        500.0,
        "exit of turn 2",
    )
    assert (tracker.status.points, tracker.status.penalty_points, tracker.status.penalties) == (0.25, 5.0, 0)
    ((kind, text, fields),) = recorded
    assert (kind, text) == (TRACK_LIMITS, "Track limits at exit of turn 2: 0.25 of 5 points")
    assert fields == {
        "steps": 1,
        "points": 0.25,
        "penalty_points": 5.0,
        "place": "exit of turn 2",
        "lap_distance_m": 500.0,
    }


def test_first_snapshot_only_notes_the_values() -> None:
    """Steps and penalties already held when the session is first seen give no entry, but are in the status."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0, steps=5, penalties=1))
    assert recorded == []
    assert (tracker.status.points, tracker.status.penalties, tracker.status.events) == (1.25, 1, ())


def test_source_without_penalties_gives_no_status() -> None:
    """A snapshot without a penalty state gives no status and no entries."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0, has_penalties=False))
    tracker.update(_snapshot(20.0, has_penalties=False))
    assert (tracker.status, recorded) == (None, [])


def test_change_of_the_penalty_count_is_recorded() -> None:
    """A penalty that comes, a second one and their being served each give a "penalty" entry."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    for time_s, penalties in ((10.0, 0), (20.0, 1), (30.0, 1), (40.0, 2), (50.0, 0)):
        tracker.update(_snapshot(time_s, penalties=penalties))
    assert [(kind, text) for kind, text, _ in recorded] == [
        (PENALTY, "1 penalty outstanding"),
        (PENALTY, "2 penalties outstanding"),
        (PENALTY, "No penalty outstanding"),
    ]
    assert tracker.status.penalties == 0


def test_new_session_clears_the_events() -> None:
    """The clock going back or another session type drops the offences and notes the values again."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0))
    tracker.update(_snapshot(20.0, steps=1))
    tracker.update(_snapshot(5.0, steps=2))
    assert (tracker.status.events, len(recorded)) == ((), 1)
    tracker.update(_snapshot(6.0, steps=3))
    assert len(tracker.status.events) == 1
    tracker.update(_snapshot(7.0, steps=4, session_type=SessionType.QUALIFY))
    assert (tracker.status.events, len(recorded)) == ((), 2)


def test_steps_going_down_are_followed_without_an_entry() -> None:
    """A count that falls (jump to the garage) gives nothing; the next step counts from there."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0, steps=3))
    tracker.update(_snapshot(20.0, steps=0))
    assert (recorded, tracker.status.points, tracker.status.events) == ([], 0.0, ())
    tracker.update(_snapshot(30.0, steps=1))
    assert [event.steps for event in tracker.status.events] == [1]


def test_place_is_where_the_game_stopped_timing_the_lap() -> None:
    """The step is counted seconds after the car left the track: the offence is placed where the lap stopped being
    timed, not where the step was counted."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0, lap_distance_m=400.0))
    tracker.update(_snapshot(11.0, lap_distance_m=455.0, lap_time_counts=False))
    tracker.update(_snapshot(15.0, lap_distance_m=680.0, lap_time_counts=False))
    tracker.update(_snapshot(15.5, steps=1, lap_distance_m=700.0, lap_time_counts=False))
    (event,) = tracker.status.events
    assert (event.lap_distance_m, event.place) == (455.0, "apex of turn 2")
    assert recorded[0][1] == "Track limits at apex of turn 2: 0.25 of 5 points"
    tracker.update(_snapshot(40.0, steps=2, lap_distance_m=800.0, lap_time_counts=False))
    assert tracker.status.events[1].place == "straight 3"


def test_drop_that_recovers_is_forgotten() -> None:
    """When the game times the lap again the place it stopped at is dropped; a later step is placed where it is
    counted."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0, lap_distance_m=400.0))
    tracker.update(_snapshot(11.0, lap_distance_m=455.0, lap_time_counts=False))
    tracker.update(_snapshot(12.6, lap_distance_m=560.0))
    tracker.update(_snapshot(30.0, steps=1, lap_distance_m=700.0))
    assert tracker.status.events[0].place == "straight 3"


def test_without_a_map_the_entry_has_no_place() -> None:
    """Until the track map is known the offence is recorded without a place."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = TrackLimitTracker(
        lambda kind, text, fields: recorded.append((kind, text, dict(fields))), lambda: (), 20.0
    )
    tracker.update(_snapshot(10.0))
    tracker.update(_snapshot(20.0, steps=1))
    assert recorded[0][1] == "Track limits: 0.25 of 5 points"


def test_untimed_lap_from_the_pits_does_not_give_the_place() -> None:
    """The game stops timing at the pit entry and through the out lap: an offence on the out lap is placed where the
    step was counted."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0))
    tracker.update(_snapshot(11.0, lap_distance_m=900.0, lap_time_counts=False, in_pits=True))
    tracker.update(_snapshot(30.0, lap_distance_m=200.0, lap_time_counts=False, completed_laps=3))
    tracker.update(_snapshot(40.0, steps=1, lap_distance_m=500.0, lap_time_counts=False, completed_laps=3))
    (event,) = tracker.status.events
    assert (event.lap, event.lap_distance_m, event.place) == (4, 500.0, "exit of turn 2")


def test_place_is_kept_over_the_line_for_one_lap_only() -> None:
    """An offence in the last corner is counted after the line and keeps its place and lap; a place more than a lap
    old is not used."""
    recorded: list[tuple[str, str, dict[str, object]]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0))
    tracker.update(_snapshot(11.0, lap_distance_m=320.0, lap_time_counts=False))
    tracker.update(_snapshot(12.0, steps=1, lap_distance_m=50.0, lap_time_counts=False, completed_laps=3))
    tracker.update(_snapshot(13.0, steps=1, lap_distance_m=100.0, lap_time_counts=True, completed_laps=3))
    tracker.update(_snapshot(14.0, steps=1, lap_distance_m=320.0, lap_time_counts=False, completed_laps=3))
    tracker.update(_snapshot(90.0, steps=2, lap_distance_m=700.0, lap_time_counts=False, completed_laps=5))
    first, second = tracker.status.events
    assert (first.lap, first.place) == (3, "entry of turn 2")
    assert (second.lap, second.place) == (6, "straight 3")
