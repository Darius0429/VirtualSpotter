import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

from ai_race_engineer.domains.common.store.segment_recorder import SegmentRecorder
from ai_race_engineer.domains.common.store.segment_store import SegmentStore
from ai_race_engineer.domains.common.analysis.corners.lap_types import LapTypeTracker
from ai_race_engineer.domains.common.analysis.corners.live_segments import LiveSegmentTracker
from ai_race_engineer.domains.common.analysis.corners.rival_lines import RivalLineTracker
from ai_race_engineer.domains.common.analysis.corners.segment_records import LAP, SECTOR, SEGMENT, Record
from ai_race_engineer.domains.common.analysis.corners.segment_tracker import SegmentTracker
from tests.oval import POLL_S, oval_lap, with_rival
from tests.domains.common.analysis.corners.test_corner_card import LIMITS
from tests.domains.common.analysis.corners.test_corner_compare import FINDINGS

SECTOR_2_M: float = 600.0
SECTOR_3_M: float = 1200.0
NAMES: list[str] = ["straight 1", "turn 1", "straight 2", "turn 2", "straight 3"]


@dataclass
class Run:
    """The trackers, the recorder and the store on the oval.
    tracker   SegmentTracker
    rivals    RivalLineTracker
    live      LiveSegmentTracker
    store     SegmentStore
    lap_types LapTypeTracker
    recorder  SegmentRecorder
    time_s    float                    s session time
    """

    tracker: SegmentTracker
    rivals: RivalLineTracker
    live: LiveSegmentTracker
    lap_types: LapTypeTracker
    store: SegmentStore
    recorder: SegmentRecorder
    time_s: float = 0.0

    def drive(
        self,
        completed_laps: int,
        last_lap_s: float | None,
        brake_zone_m: float,
        share: float = 1.0,
        pits_m: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Drive one lap of the oval, or the first part of it, with Rival One on the same lap 3 m further in x and
        0.5 s faster; sector 2 starts at 600 m and sector 3 at 1200 m.
        In:  completed_laps  int           laps completed during this lap
             last_lap_s      float | None  s the game shows as the last lap time during this lap
             brake_zone_m    float         m braked before each turn
             share           float         part of the lap driven, 0-1
             pits_m          tuple[float, float]  m lap distance from and to which both cars are in the pits
        Out:                 None
        """
        lap = oval_lap(completed_laps, last_lap_s, brake_zone_m=brake_zone_m)
        for snapshot in lap[: int(len(lap) * share)]:
            distance_m = snapshot.player.vehicle.lap_distance_m
            sector = (distance_m >= SECTOR_2_M) + (distance_m >= SECTOR_3_M)
            pitted = pits_m[0] <= distance_m < pits_m[1]
            vehicle = dataclasses.replace(snapshot.player.vehicle, sector_index=int(sector), in_pits=pitted)
            timed = dataclasses.replace(
                snapshot,
                player=dataclasses.replace(snapshot.player, vehicle=vehicle),
                session=dataclasses.replace(snapshot.session, elapsed_s=self.time_s),
            )
            rival_s = None if last_lap_s is None else last_lap_s - 0.5
            both = with_rival([timed], "Rival One", rival_s, shift_x_m=3.0)[0]
            for block in (self.tracker, self.rivals, self.live, self.lap_types, self.recorder):
                block.update(both)
            self.time_s += POLL_S


def run(folder: Path) -> Run:
    """A run after five laps of the oval: lap 1 gives nothing, lap 2 (40.0 s, braked 20 m earlier) gives the track
    map, lap 3 (39.5 s) and lap 4 (40.5 s) are recorded as they are driven, and lap 5 is driven but not finished.
    In:  folder  Path  where the store's file goes
    Out:         Run
    """
    rivals = RivalLineTracker()
    tracker = SegmentTracker(
        lambda track: (),
        lambda track, segments: None,
        lambda track, car: None,
        lambda track, car, lap_time_s, samples: None,
        keep_laps=30,
        limits=LIMITS,
        step_m=5.0,
        smooth_m=55.0,
        max_radius_m=400.0,
        min_turn_deg=18.0,
        entry_pad_m=100.0,
        exit_pad_m=50.0,
        merge_gap_m=50.0,
    )
    live = LiveSegmentTracker(lambda: tracker.segments)
    store = SegmentStore(folder / "run.sqlite")
    lap_types = LapTypeTracker()
    load = lambda track, class_name: None  # noqa: E731
    recorder = SegmentRecorder(tracker, rivals, live, lap_types, store, load, LIMITS, FINDINGS, 60.0)
    driven = Run(tracker, rivals, live, lap_types, store, recorder)
    for completed_laps, last_lap_s, brake_zone_m in (
        (0, None, 100.0),
        (1, 41.0, 120.0),
        (2, 40.0, 100.0),
        (3, 39.5, 100.0),
        (4, 40.5, 100.0),
    ):
        driven.drive(completed_laps, last_lap_s, brake_zone_m)
    return driven


def _mine(records: list[Record], kind: str, lap: int) -> list[Record]:
    """The player's records of one kind on one lap.
    In:  records  list[Record]
         kind     str
         lap      int
    Out:          list[Record]
    """
    return [r for r in records if r.me and r.kind == kind and r.lap == lap]


def test_segments_sectors_and_laps_are_recorded_in_the_order_they_finish(tmp_path: Path) -> None:
    """Every segment of a lap is in the list once, a sector follows its last segment before the lap is over, and
    the lap closes it."""
    records = run(tmp_path).store.select("1 = 1")
    assert [r.lap for r in records if r.me and r.kind == LAP] == [2, 3, 4]
    for lap in (2, 3, 4):
        assert [r.name for r in _mine(records, SEGMENT, lap)] == NAMES
        assert [r.name for r in _mine(records, SECTOR, lap)] == ["sector 1", "sector 2", "sector 3"]
    lap_3 = [r for r in records if r.me and r.lap == 3]
    names = [r.name for r in lap_3]
    assert names.index("turn 1") < names.index("sector 1") < names.index("straight 2")
    assert names[-1] == LAP
    assert [r.figures["sector"] for r in _mine(records, SEGMENT, 3)] == [1, 1, 2, 2, 3]
    assert _mine(records, SECTOR, 3)[0].figures["segments"] == ["straight 1", "turn 1"]
    assert _mine(records, SECTOR, 3)[0].time_s == pytest.approx(15.86, abs=0.1)


def test_differences_are_stored_against_the_references_of_the_time(tmp_path: Path) -> None:
    """A lap is compared with the laps that existed when it started: lap 2 with nothing, lap 3 with lap 2 and with
    the rival's lap, lap 4 with lap 3; a lap is never compared with itself."""
    records = run(tmp_path).store.select("1 = 1")
    assert all(r.vs == {} for r in records if r.me and r.lap == 2)
    lap_3 = _mine(records, LAP, 3)[0]
    assert (lap_3.vs["car"]["reference"], lap_3.vs["car"]["lost_s"]) == ("lap 2", pytest.approx(-0.5))
    assert (lap_3.vs["class"]["reference"], lap_3.vs["class"]["lost_s"]) == ("Rival One", pytest.approx(0.0))
    assert lap_3.vs["track"]["reference"] == "lap 2"
    lap_4 = _mine(records, LAP, 4)[0]
    assert (lap_4.vs["car"]["reference"], lap_4.vs["car"]["lost_s"]) == ("lap 3", pytest.approx(1.0))
    assert (lap_4.vs["class"]["reference"], lap_4.vs["class"]["reference_lap_s"]) == ("Rival One", 39.0)
    segments = _mine(records, SEGMENT, 3)
    assert sum(r.vs["car"]["lost_s"] for r in segments) == pytest.approx(sum(r.time_s for r in segments) - 40.0)
    turn_1 = segments[1]
    assert turn_1.vs["car"]["findings"][0].endswith(" m later")
    assert turn_1.vs["class"]["deltas"] == {}
    assert abs(turn_1.vs["class"]["line_wider_m"]) == pytest.approx(3.0, abs=0.5)


def test_segment_keeps_its_telemetry_with_the_difference_per_sample(tmp_path: Path) -> None:
    """A segment of the player carries its telemetry with a margin, and per sample the time lost since the start
    of the segment and the speed difference to each reference; the reference's line stays in memory."""
    store = run(tmp_path).store
    turn_1 = store.telemetry(_mine(store.select("1 = 1"), SEGMENT, 4)[1])
    samples, trace = turn_1.samples, turn_1.traces["car"]
    assert samples.distance_m[0] == pytest.approx(150.0 - 60.0, abs=4.0)
    assert len(trace["time_s"]) == len(trace["speed_kph"]) == len(samples.distance_m)
    assert trace["time_s"][-1] - trace["time_s"][0] == pytest.approx(turn_1.vs["car"]["lost_s"], abs=0.1)
    assert turn_1.lines["car"].distance_m[0] == pytest.approx(samples.distance_m[0], abs=4.0)
    assert set(turn_1.traces) == {"class", "car", "track"}
    assert set(turn_1.lines) == {"car", "track"}


def test_other_cars_have_their_segment_times_and_their_laps(tmp_path: Path) -> None:
    """Another car's segments are in the list as it leaves them and its full laps as it finishes them."""
    records = run(tmp_path).store.select("1 = 1")
    laps = [r for r in records if not r.me and r.kind == LAP]
    assert [(r.driver, r.lap, r.time_s) for r in laps] == [
        ("Rival One", 2, 39.5),
        ("Rival One", 3, 39.0),
        ("Rival One", 4, 40.0),
    ]
    passes = [r for r in records if not r.me and r.kind == SEGMENT and r.lap == 4]
    assert {r.name for r in passes} == set(NAMES)
    assert sum(r.time_s for r in passes) == pytest.approx(sum(r.time_s for r in _mine(records, SEGMENT, 4)), abs=0.2)
    assert all(r.samples is None and r.vs == {} for r in passes)


def test_lap_in_progress_is_recorded_segment_by_segment(tmp_path: Path) -> None:
    """On a lap that is not finished the segments already left are in the list with their differences."""
    driven = run(tmp_path)
    driven.drive(5, 40.5, 120.0, share=0.5)
    now = _mine(driven.store.select("1 = 1"), SEGMENT, 6)
    assert [r.name for r in now] == ["straight 1", "turn 1"]
    assert now[1].vs["car"]["findings"][0].endswith(" m earlier")
    assert [r.name for r in _mine(driven.store.select("1 = 1"), SECTOR, 6)] == ["sector 1"]


def test_new_session_is_counted_and_the_file_can_be_read_after_the_run(tmp_path: Path) -> None:
    """A session clock that goes back starts session 2; a store opened on the file of a finished run has the same
    records and carries on counting its sessions."""
    driven = run(tmp_path)
    assert driven.store.session == 1
    driven.time_s = 0.0
    driven.drive(0, None, 100.0, share=0.1)
    assert driven.store.session == 2
    kept = driven.store.select("me = 1")
    driven.store.close()
    with SegmentStore(tmp_path / "run.sqlite") as later:
        assert later.session == 2
        assert later.select("me = 1") == kept
        turn_1 = later.telemetry(_mine(kept, SEGMENT, 4)[1])
    assert turn_1.samples.speed_mps.size == turn_1.traces["car"]["time_s"].size


def test_in_lap_and_out_lap_are_recorded_as_such_with_the_segments_driven(tmp_path: Path) -> None:
    """A lap that ends in the pits is an in-lap with the segments before the pit entry; the lap from the pit exit
    to the line is an out-lap with the segments after the exit, but for the one
    that ends at the line; neither is a timed lap, for the player or for the
    other car, and the next lap from the line is a timed lap again."""
    driven = run(tmp_path)
    driven.drive(5, 40.5, 100.0, pits_m=(1000.0, 2000.0))
    driven.drive(6, None, 100.0, pits_m=(0.0, 200.0))
    driven.drive(7, None, 100.0)
    driven.drive(8, 40.2, 100.0, share=0.1)
    records = driven.store.select("1 = 1")
    laps = [(r.lap, r.name) for r in records if r.me and r.kind == LAP]
    assert laps == [(2, "lap"), (3, "lap"), (4, "lap"), (5, "lap"), (6, "in lap"), (7, "out lap"), (8, "lap")]
    assert [r.name for r in _mine(records, SEGMENT, 6)] == ["straight 1", "turn 1", "straight 2"]
    assert [r.name for r in _mine(records, SEGMENT, 7)] == ["straight 2", "turn 2"]
    assert _mine(records, SECTOR, 7) == []
    theirs = [(r.lap, r.name) for r in records if not r.me and r.kind == LAP and r.lap >= 6]
    assert theirs == [(6, "in lap"), (7, "out lap"), (8, "lap")]
