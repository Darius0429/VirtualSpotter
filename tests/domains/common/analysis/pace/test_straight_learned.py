import dataclasses

import numpy as np

from ai_race_engineer.domains.common.analysis.pace.straight import StraightDetector
from ai_race_engineer.domains.common.analysis.pace.straight_learned import (
    LearnedStraightDetector,
    StraightSegment,
    learn_straights,
    straight_left_m,
)
from ai_race_engineer.domains.common.analysis.strategy.track_info import TrackInfo
from ai_race_engineer.domains.common.data.snapshot import PlayerState, SessionState, SessionType, Snapshot, VehicleState

TRACK_M: float = 1000.0
SEGMENTS: tuple[StraightSegment, ...] = (StraightSegment(100.0, 400.0, 300.0), StraightSegment(900.0, 60.0, 160.0))


class _Fallback(StraightDetector):
    """Fallback with a fixed answer that counts its updates.
    answer   bool  what is_straight returns
    updates  int   snapshots fed
    """

    def __init__(self, answer: bool) -> None:
        """Set the answer.
        In:  answer  bool
        Out:         None
        """
        self.answer: bool = answer
        self.updates: int = 0

    def update(self, snapshot: Snapshot) -> None:
        """Count the snapshot.
        In:  snapshot  Snapshot
        Out:           None
        """
        self.updates += 1

    def is_straight(self) -> bool:
        """The fixed answer.
        In:  -
        Out: bool
        """
        return self.answer


def _is_straight_at(lap_distance_m: float) -> bool:
    """Whether a position is on a straight of the synthetic track: 100-400 m, 600-700 m, 900 m across the line to 60 m.
    In:  lap_distance_m  float  m
    Out:                 bool
    """
    return 100 <= lap_distance_m < 400 or 600 <= lap_distance_m < 700 or lap_distance_m >= 900 or lap_distance_m < 60


def _snapshot(
    completed_laps: int,
    lap_distance_m: float,
    in_pits: bool = False,
    elapsed_s: float = 100.0,
    track_name: str = "Track",
) -> Snapshot:
    """Snapshot of the player at 50 m/s on the synthetic track, steering only off the straights.
    In:  completed_laps  int    laps
         lap_distance_m  float  m
         in_pits         bool
         elapsed_s       float  s
         track_name      str
    Out:                 Snapshot
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        is_player=True, speed_mps=50.0, completed_laps=completed_laps, lap_distance_m=lap_distance_m, in_pits=in_pits
    )
    vehicle = VehicleState(**base)
    steering = 0.0 if _is_straight_at(lap_distance_m) else 0.5
    session = SessionState(SessionType.RACE, elapsed_s, 3600.0, 1000, track_name, TRACK_M, 20.0, 20.0)
    player = PlayerState(vehicle, 50.0, None, None, 0.0, 0.0, 0.0, steering, 0.0, 0, 0.0, 0.0, 0.0, "", "", (0.0,) * 4)
    return Snapshot(session, player, (vehicle,))


def _detector(fallback: StraightDetector, tracks: dict[str, TrackInfo], learn_laps: int = 1) -> LearnedStraightDetector:
    """Detector with 20 m bins, 150 m minimum length and 4 s of straight required.
    In:  fallback    StraightDetector
         tracks      dict[str, TrackInfo]
         learn_laps  int
    Out:             LearnedStraightDetector
    """
    return LearnedStraightDetector(fallback, tracks, 0.08, 3.0, 30.0, 1.0, learn_laps, 20.0, 0.8, 150.0, 4.0)


def _drive_lap(detector: LearnedStraightDetector, completed_laps: int, in_pits: bool = False) -> None:
    """Feed one lap in 5 m steps.
    In:  detector        LearnedStraightDetector
         completed_laps  int   laps completed before this lap
         in_pits         bool  the whole lap counts as in the pits
    Out:                 None
    """
    for lap_distance_m in np.arange(0.0, TRACK_M, 5.0):
        detector.update(_snapshot(completed_laps, float(lap_distance_m), in_pits))


def test_learn_straights_from_synthetic_laps() -> None:
    """Two straights are found, the longest first; the 100 m one is dropped; the one across the line is merged."""
    distance = np.tile(np.arange(0.0, TRACK_M, 5.0), 3)
    straight = np.array([_is_straight_at(d) for d in distance])
    assert learn_straights(distance, straight, TRACK_M, 20.0, 0.8, 150.0) == SEGMENTS


def test_learn_straights_needs_the_share_of_straight_samples() -> None:
    """A bin with too few straight samples is not straight."""
    distance = np.tile(np.arange(0.0, TRACK_M, 5.0), 4)
    straight = np.array([_is_straight_at(d) for d in distance])
    straight[: len(straight) // 4] = False
    assert learn_straights(distance, straight, TRACK_M, 20.0, 0.8, 150.0) == ()
    assert learn_straights(distance, straight, TRACK_M, 20.0, 0.75, 150.0) == SEGMENTS


def test_learn_straights_whole_track() -> None:
    """A track that is straight everywhere is one segment."""
    distance = np.arange(0.0, TRACK_M, 5.0)
    whole = (StraightSegment(0.0, TRACK_M, TRACK_M),)
    assert learn_straights(distance, np.ones(len(distance), dtype=np.bool_), TRACK_M, 20.0, 0.8, 150.0) == whole


def test_straight_left_inside_outside_and_across_the_line() -> None:
    """Metres to the end of the segment the position is in, also when the segment crosses the line."""
    assert straight_left_m(SEGMENTS, 250.0, TRACK_M) == 150.0
    assert straight_left_m(SEGMENTS, 500.0, TRACK_M) == 0.0
    assert straight_left_m(SEGMENTS, 400.0, TRACK_M) == 0.0
    assert straight_left_m(SEGMENTS, 950.0, TRACK_M) == 110.0
    assert straight_left_m(SEGMENTS, 30.0, TRACK_M) == 30.0


def test_fallback_answers_before_the_laps_are_learned() -> None:
    """Until learn_laps clean laps are driven the fallback's answer is passed on, and it sees every snapshot."""
    fallback = _Fallback(True)
    detector = _detector(fallback, {}, learn_laps=2)
    _drive_lap(detector, 0)
    detector.update(_snapshot(1, 500.0))
    assert detector.is_straight()
    assert fallback.updates == 201
    fallback.answer = False
    assert not detector.is_straight()


def test_learned_straights_take_over() -> None:
    """After learn_laps the detector answers from the learned straights, whatever the fallback says."""
    fallback = _Fallback(True)
    detector = _detector(fallback, {})
    _drive_lap(detector, 0)
    detector.update(_snapshot(1, 0.0))
    detector.update(_snapshot(1, 500.0))
    assert not detector.is_straight()
    fallback.answer = False
    detector.update(_snapshot(1, 110.0))
    assert detector.is_straight()


def test_enough_straight_left_early_but_not_near_the_end() -> None:
    """At 50 m/s and 4 s required, 290 m of straight left is enough and 100 m is not."""
    detector = _detector(_Fallback(False), {})
    _drive_lap(detector, 0)
    detector.update(_snapshot(1, 0.0))
    detector.update(_snapshot(1, 110.0))
    assert detector.is_straight()
    detector.update(_snapshot(1, 300.0))
    assert not detector.is_straight()


def test_pit_lap_is_not_learned() -> None:
    """A lap with the pits does not count towards learn_laps."""
    detector = _detector(_Fallback(True), {})
    _drive_lap(detector, 0, in_pits=True)
    detector.update(_snapshot(1, 0.0))
    detector.update(_snapshot(1, 500.0))
    assert detector.is_straight()


def test_laps_without_a_straight_keep_the_fallback() -> None:
    """A lap that shows no straight (formation lap) leaves the fallback in charge; the next lap is learned."""
    detector = _detector(_Fallback(True), {})
    for lap_distance_m in np.arange(0.0, TRACK_M, 5.0):
        detector.update(_snapshot(0, 450.0 + float(lap_distance_m) / 20.0))
    detector.update(_snapshot(1, 500.0))
    assert detector.is_straight()
    _drive_lap(detector, 1)
    detector.update(_snapshot(2, 0.0))
    detector.update(_snapshot(2, 500.0))
    assert not detector.is_straight()


def test_track_straights_are_used_from_the_first_tick() -> None:
    """Straights given for the track need no learning."""
    tracks = {"Track": TrackInfo(TRACK_M, None, ((100.0, 400.0), (900.0, 60.0)))}
    detector = _detector(_Fallback(False), tracks, learn_laps=3)
    detector.update(_snapshot(0, 150.0))
    assert detector.is_straight()
    detector.update(_snapshot(0, 950.0, elapsed_s=101.0))
    assert not detector.is_straight()
    detector.update(_snapshot(0, 500.0, elapsed_s=102.0))
    assert not detector.is_straight()


def test_track_change_restarts_learning() -> None:
    """On another track the learned straights are dropped and the fallback answers again."""
    detector = _detector(_Fallback(True), {})
    _drive_lap(detector, 0)
    detector.update(_snapshot(1, 0.0))
    detector.update(_snapshot(1, 500.0))
    assert not detector.is_straight()
    detector.update(_snapshot(0, 500.0, track_name="Other"))
    assert detector.is_straight()
