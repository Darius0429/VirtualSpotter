import dataclasses

from ai_race_engineer.domains.common.analysis.corners.segment_tracker import SegmentTracker
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot
from tests.oval import LENGTH_M, oval_lap
from tests.domains.common.analysis.corners.test_corner_card import LIMITS

STORED: tuple[Segment, ...] = (Segment("pit straight", False, 0.0, 1000.0, None),)


class _Store:
    """Stand-in for the track map store.
    stored  dict[str, tuple[Segment, ...]]  track name -> map
    best    dict[tuple[str, str], tuple[float, object]]  (track, car) -> stored best lap
    saves   int                             maps saved
    """

    def __init__(self, stored: dict[str, tuple[Segment, ...]]) -> None:
        """Hold the stored maps.
        In:  stored  dict[str, tuple[Segment, ...]]
        Out:         None
        """
        self.stored: dict[str, tuple[Segment, ...]] = stored
        self.best: dict[tuple[str, str], tuple[float, object]] = {}
        self.saves: int = 0

    def load(self, track_name: str) -> tuple[Segment, ...]:
        """Stored map of a track.
        In:  track_name  str
        Out:             tuple[Segment, ...]  empty when there is none
        """
        return self.stored.get(track_name, ())

    def save(self, track_name: str, segments: tuple[Segment, ...]) -> None:
        """Keep the map of a track.
        In:  track_name  str
             segments    tuple[Segment, ...]
        Out:             None
        """
        self.stored[track_name] = segments
        self.saves += 1


def _tracker(store: _Store, keep_laps: int = 30) -> SegmentTracker:
    """Tracker with the config values.
    In:  store      _Store
         keep_laps  int
    Out:            SegmentTracker
    """
    return SegmentTracker(
        store.load,
        store.save,
        lambda track, car: store.best.get((track, car)),
        lambda track, car, lap_time_s, samples: store.best.update({(track, car): (lap_time_s, samples)}),
        keep_laps=keep_laps,
        limits=LIMITS,
        step_m=5.0,
        smooth_m=55.0,
        max_radius_m=400.0,
        min_turn_deg=18.0,
        entry_pad_m=100.0,
        exit_pad_m=50.0,
        merge_gap_m=50.0,
    )


def _drive(tracker: SegmentTracker, *laps: tuple[int, float | None, bool]) -> None:
    """Feed laps of the oval.
    In:  tracker  SegmentTracker
         laps     tuple[int, float | None, bool]  completed laps, last lap time (s) and in pits during each lap
    Out:          None
    """
    for completed_laps, last_lap_s, in_pits in laps:
        for snapshot in oval_lap(completed_laps, last_lap_s, in_pits):
            tracker.update(snapshot)


def test_first_full_lap_gives_the_map_and_every_full_lap_is_analysed() -> None:
    """The lap the tracker joins in is skipped; the next lap builds and saves the map once and is analysed, as is
    the one after."""
    store = _Store({})
    tracker = _tracker(store)
    _drive(tracker, (0, None, False), (1, 41.0, False))
    assert (tracker.laps, tracker.segments) == ((), ())
    _drive(tracker, (2, 40.0, False), (3, 39.5, False))
    assert [(lap.lap, lap.lap_time_s) for lap in tracker.laps] == [(2, 40.0), (3, 39.5)]
    assert [s.name for s in tracker.segments] == ["straight 1", "turn 1", "straight 2", "turn 2", "straight 3"]
    assert [c.name for c in tracker.laps[0].cards] == [s.name for s in tracker.segments]
    assert tracker.laps[0].samples.distance_m[-1] > 1600.0
    assert 0.0 < tracker.laps[0].shares.braking < 0.2
    assert store.saves == 1
    assert tracker.latest is not None


def test_laps_through_the_pits_and_untimed_laps_are_not_analysed() -> None:
    """An in-lap or out-lap and a lap the game gave no time are left out."""
    tracker = _tracker(_Store({}))
    _drive(tracker, (0, None, False), (1, 41.0, True), (2, 60.0, False), (3, None, False), (4, 40.0, False))
    assert [lap.lap for lap in tracker.laps] == [4]


def test_stored_map_is_used_and_not_saved_again() -> None:
    """A map on disk is taken as it is."""
    store = _Store({"Oval": STORED})
    tracker = _tracker(store)
    _drive(tracker, (0, None, False), (1, 41.0, False), (2, 40.0, False))
    assert tracker.segments == STORED
    assert [c.name for c in tracker.laps[0].cards] == ["pit straight"]
    assert store.saves == 0


def test_only_the_newest_laps_are_kept_and_a_new_session_starts_empty() -> None:
    """The lap list is capped; a lap count that goes down drops the laps of the earlier session."""
    tracker = _tracker(_Store({}), keep_laps=2)
    _drive(tracker, (0, None, False), (1, 41.0, False), (2, 40.0, False), (3, 39.0, False), (4, 38.0, False))
    assert [lap.lap for lap in tracker.laps] == [3, 4]
    _drive(tracker, (0, None, True))
    assert tracker.laps == ()


def test_best_lap_is_stored_and_is_the_reference_of_a_later_session() -> None:
    """The fastest lap is kept as the reference and saved per track and car; a new session, and a tracker started
    later with the same store, still have it though their own laps start empty."""
    store = _Store({})
    tracker = _tracker(store)
    _drive(tracker, (0, None, False), (1, 41.0, False), (2, 40.0, False), (3, 39.0, False), (4, 39.5, False))
    assert (tracker.reference.lap, tracker.reference.lap_time_s) == (0, 39.0)
    assert store.best[("Oval", "Hyper Team Car")][0] == 39.0
    _drive(tracker, (0, None, True))
    assert tracker.laps == ()
    assert tracker.reference.lap_time_s == 39.0
    later = _tracker(store)
    _drive(later, (0, None, False))
    assert (later.laps, later.reference.lap_time_s) == ((), 39.0)
    assert [c.name for c in later.reference.cards] == [s.name for s in later.segments]


def test_segments_the_car_was_not_polled_in_are_skipped_at_a_start_from_the_grid() -> None:
    """A car standing on the grid 100 m behind the line, whose lap distance then moves to the end of the lap, gets
    a card for the segment it starts in and none for the segments before it."""
    quarters = tuple(Segment(name, False, 400.0 * i, 400.0 * (i + 1), None) for i, name in enumerate("abcd"))
    tracker = _tracker(_Store({"Oval": quarters}))
    grid_m = LENGTH_M - 100.0
    driven = [snapshot for snapshot in oval_lap(0) if snapshot.player.vehicle.lap_distance_m >= grid_m]
    standing = [_placed(driven[0], -100.0, 0.01 * i, i == 0) for i in range(4)]
    for snapshot in (*standing, _placed(driven[0], grid_m, 0.03, False), *driven[1:]):
        tracker.update(snapshot)
    assert (tracker.live_first, [card.name for card in tracker.live]) == (3, ["d"])


def _placed(snapshot: RaceSnapshot, lap_distance_m: float, moved_m: float, in_pits: bool) -> RaceSnapshot:
    """A snapshot with the player's car standing where it is, at another lap distance.
    In:  snapshot        RaceSnapshot
         lap_distance_m  float         m from the line as the game counts it; negative behind it
         moved_m         float         m the car has crept along world x
         in_pits         bool
    Out:                 RaceSnapshot
    """
    x_m, z_m = snapshot.player.vehicle.position_xz_m
    vehicle = dataclasses.replace(
        snapshot.player.vehicle, lap_distance_m=lap_distance_m, position_xz_m=(x_m + moved_m, z_m), in_pits=in_pits
    )
    return dataclasses.replace(snapshot, player=dataclasses.replace(snapshot.player, vehicle=vehicle))
