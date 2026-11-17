import pytest

from ai_race_engineer.domains.common.analysis.corners.live_segments import LiveSegmentTracker
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment
from tests.oval import ARC_M, LENGTH_M, oval_lap, with_rival

SEGMENTS: tuple[Segment, ...] = (
    Segment("straight 1", False, 0.0, 250.0, None),
    Segment("turn 1", True, 250.0, 250.0 + ARC_M, 400.0),
    Segment("straight 2", False, 250.0 + ARC_M, LENGTH_M, None),
)


def _drive(tracker: LiveSegmentTracker, laps: int, in_pits: bool = False) -> None:
    """Feed laps of the oval with a rival in slot 1 driving the same lap.
    In:  tracker  LiveSegmentTracker
         laps     int                 laps driven
         in_pits  bool                the cars are in the pits
    Out:          None
    """
    time_s = 0.0
    for lap in range(laps):
        for snapshot in with_rival(oval_lap(lap, in_pits=in_pits), "Rival One", None):
            snapshot.session.__dict__["elapsed_s"] = time_s
            tracker.update(snapshot)
            time_s += 0.05


def test_every_car_is_timed_through_a_segment_the_moment_it_leaves_it() -> None:
    """The segment a car is in when first seen is not timed; from then on each segment gives a pass with its time,
    minimum and exit speed and the car's lap, for the player and for the rival alike."""
    tracker = LiveSegmentTracker(lambda: SEGMENTS)
    _drive(tracker, 2)
    mine, theirs = tracker.last[0], tracker.last[1]
    assert set(mine) == {"straight 1", "turn 1", "straight 2"}
    assert mine["turn 1"].time_s == pytest.approx(ARC_M / 30.0, abs=0.06)
    assert mine["turn 1"].min_kph == pytest.approx(108.0)
    assert (mine["turn 1"].lap, mine["straight 1"].lap, mine["straight 2"].lap) == (2, 2, 1)
    assert theirs["turn 1"].time_s == pytest.approx(mine["turn 1"].time_s)
    assert tracker.best[0]["turn 1"].time_s <= mine["turn 1"].time_s


def test_nothing_is_timed_without_a_map_or_in_the_pits() -> None:
    """Without a track map, and for cars in the pits, there are no passes."""
    no_map = LiveSegmentTracker(lambda: ())
    _drive(no_map, 2)
    in_pits = LiveSegmentTracker(lambda: SEGMENTS)
    _drive(in_pits, 2, in_pits=True)
    assert (no_map.last, in_pits.last) == ({}, {})
