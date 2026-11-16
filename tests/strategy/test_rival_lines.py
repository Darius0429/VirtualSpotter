import pytest

from ai_race_engineer.strategy.rival_lines import RivalLineTracker, segment_line
from ai_race_engineer.strategy.track_map import Segment
from tests.oval import LENGTH_M, TURN_1_M, oval_lap, with_rival

TURN_1: Segment = Segment("turn 1", True, 100.0, 650.0, 400.0)
STRAIGHT: Segment = Segment("straight 1", False, 0.0, 100.0, None)


def _drive(tracker: RivalLineTracker, *laps: tuple[int, float | None, bool]) -> None:
    """Feed laps of the oval with a rival in slot 1.
    In:  tracker  RivalLineTracker
         laps     tuple[int, float | None, bool]  completed laps, the rival's last lap time (s) and in pits per lap
    Out:          None
    """
    for completed_laps, last_lap_s, in_pits in laps:
        for snapshot in with_rival(oval_lap(completed_laps, in_pits=in_pits), "Rival One", last_lap_s):
            tracker.update(snapshot)


def test_fastest_full_lap_of_another_car_is_kept() -> None:
    """The lap the tracker joins in is skipped; of the full laps the fastest stays, with the car's line."""
    tracker = RivalLineTracker()
    _drive(tracker, (0, None, False), (1, 41.0, False))
    assert tracker.best == {}
    _drive(tracker, (2, 40.0, False), (3, 39.0, False), (4, 39.5, False))
    lap = tracker.best[1]
    assert (lap.driver_name, lap.lap_time_s) == ("Rival One", 39.0)
    assert tracker.last[1].lap_time_s == 39.5
    assert lap.line.distance_m[-1] == pytest.approx(LENGTH_M)
    assert lap.line.time_s[-1] == pytest.approx(39.0)
    assert 0 not in tracker.best


def test_pit_laps_and_untimed_laps_are_left_out_and_a_new_session_starts_empty() -> None:
    """A lap through the pits or without a time is not kept; a lap count that goes down starts a new
    session: the car's fastest lap is kept, its last lap is not."""
    tracker = RivalLineTracker()
    _drive(tracker, (0, None, False), (1, 41.0, True), (2, 60.0, False), (3, None, False))
    assert tracker.best == {}
    _drive(tracker, (4, 40.0, False))
    assert tracker.best[1].lap_time_s == 40.0
    _drive(tracker, (0, None, True))
    assert (tracker.best[1].lap_time_s, tracker.last) == (40.0, {})


def test_segment_figures_from_line_and_speed() -> None:
    """A corner gives the time, the speeds, where the car starts slowing and where it is slowest; a straight has no
    apex time."""
    tracker = RivalLineTracker()
    _drive(tracker, (0, None, False), (1, 41.0, False), (2, 38.2, False))
    corner = segment_line(tracker.best[1].line, TURN_1, search_m=150.0)
    assert corner.slowing_point_m == pytest.approx(TURN_1_M[0] - 100.0, abs=4.0)
    assert corner.min_speed_m == pytest.approx(TURN_1_M[0], abs=4.0)
    assert (corner.min_kph, corner.entry_kph) == (pytest.approx(108.0), pytest.approx(216.0))
    assert corner.time_to_apex_s == pytest.approx(8.0, abs=0.3)
    assert segment_line(tracker.best[1].line, STRAIGHT, search_m=150.0).time_to_apex_s is None
