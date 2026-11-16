from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.strategy.straight import StraightDetector, on_straight
from ai_race_engineer.strategy.track_info import TrackInfo
from ai_race_engineer.telemetry.snapshot import SessionState, Snapshot


@dataclass(frozen=True)
class StraightSegment:
    """One straight of a track.
    start_m   float  m lap distance where it starts
    end_m     float  m lap distance where it ends; end_m < start_m when it crosses the start/finish line
    length_m  float  m
    """

    start_m: float
    end_m: float
    length_m: float


def learn_straights(
    lap_distance_m: NDArray[np.float64],
    straight: NDArray[np.bool_],
    track_length_m: float,
    bin_m: float,
    min_fraction: float,
    min_length_m: float,
) -> tuple[StraightSegment, ...]:
    """Find the straights of a track from samples of driven laps.
    In:  lap_distance_m  NDArray[np.float64]  m lap distance of each sample
         straight        NDArray[np.bool_]    whether the car drove straight at each sample
         track_length_m  float                m
         bin_m           float                m lap-distance bin
         min_fraction    float                share of a bin's samples that must be straight
         min_length_m    float                m; shorter straights are dropped
    Out:                 tuple[StraightSegment, ...]  longest first
    """
    bins = int(np.ceil(track_length_m / bin_m))
    index = np.clip((lap_distance_m // bin_m).astype(int), 0, bins - 1)
    samples = np.bincount(index, minlength=bins)
    hits = np.bincount(index, weights=straight, minlength=bins)
    is_straight = (samples > 0) & (hits >= min_fraction * samples)
    if is_straight.all():
        return (StraightSegment(0.0, track_length_m, track_length_m),)
    edges = np.flatnonzero(is_straight != np.roll(is_straight, 1))
    starts = [int(e) for e in edges if is_straight[e]]
    ends = [int(e) for e in edges if not is_straight[e]]
    segments = [
        _segment(start, next((e for e in ends if e > start), ends[0]), bin_m, track_length_m) for start in starts
    ]
    return tuple(sorted((s for s in segments if s.length_m >= min_length_m), key=lambda s: s.length_m, reverse=True))


def straight_left_m(segments: Sequence[StraightSegment], lap_distance_m: float, track_length_m: float) -> float:
    """Metres of straight left ahead of a position.
    In:  segments        Sequence[StraightSegment]
         lap_distance_m  float  m position
         track_length_m  float  m
    Out:                 float  m to the end of the segment containing the position; 0 outside every segment
    """
    for segment in segments:
        if segment.start_m <= lap_distance_m < segment.end_m or (
            segment.end_m < segment.start_m and lap_distance_m < segment.end_m
        ):
            return segment.end_m - lap_distance_m
        if segment.end_m < segment.start_m <= lap_distance_m:
            return track_length_m - lap_distance_m + segment.end_m
    return 0.0


class LearnedStraightDetector(StraightDetector):
    """Straight detection from the track's known straights, so it knows how much straight is left. The straights
    come from the track table, else they are learned from the last clean laps; until then the fallback answers. Laps
    that show no straight (e.g. a formation lap) teach nothing: learning goes on with the next lap.
    _fallback          StraightDetector               answers until the straights are known
    _tracks            Mapping[str, TrackInfo]        track name -> characteristics
    _max_steering      float                          |steering| limit, fraction of lock (labels samples)
    _max_lateral_mps2  float                          m/s^2 (labels samples)
    _min_speed_mps     float                          m/s (labels samples)
    _learn_laps        int                            clean laps to learn from
    _bin_m             float                          m lap-distance bin
    _min_fraction      float                          share of a bin's samples that must be straight
    _min_length_m      float                          m; shorter straights are dropped
    _min_remaining_s   float                          s of straight that must be left
    _session           SessionState | None            session of the previous update
    _segments          tuple[StraightSegment, ...] | None  None until given or learned
    _lap               list[tuple[float, bool]]       (m lap distance, straight) samples of the current lap
    _laps              deque[list[tuple[float, bool]]]  samples of the last learn_laps finished clean laps
    _completed_laps    int                            player's completed laps at the previous update
    _pitted            bool                           the player was in the pits during the current lap
    _straight          bool                           latest own answer (read from other threads)
    """

    def __init__(
        self,
        fallback: StraightDetector,
        tracks: Mapping[str, TrackInfo],
        max_steering: float,
        max_lateral_mps2: float,
        min_speed_mps: float,
        hold_s: float,
        learn_laps: int,
        bin_m: float,
        min_fraction: float,
        min_length_m: float,
        min_remaining_s: float,
    ) -> None:
        """Set the fallback, the track table and the thresholds.
        In:  fallback          StraightDetector         answers until the straights are known
             tracks            Mapping[str, TrackInfo]  track name -> characteristics
             max_steering      float                    |steering| limit, fraction of lock
             max_lateral_mps2  float                    |lateral acceleration| limit, m/s^2
             min_speed_mps     float                    m/s
             hold_s            float                    s (accepted for the shared config block, unused)
             learn_laps        int                      clean laps before the learned straights take over
             bin_m             float                    m lap-distance bin
             min_fraction      float                    share of a bin's samples that must be straight
             min_length_m      float                    m; shorter straights are dropped
             min_remaining_s   float                    s of straight that must be left to start speaking
        Out:                   None
        """
        self._fallback: StraightDetector = fallback
        self._tracks: Mapping[str, TrackInfo] = tracks
        self._max_steering: float = max_steering
        self._max_lateral_mps2: float = max_lateral_mps2
        self._min_speed_mps: float = min_speed_mps
        self._learn_laps: int = learn_laps
        self._bin_m: float = bin_m
        self._min_fraction: float = min_fraction
        self._min_length_m: float = min_length_m
        self._min_remaining_s: float = min_remaining_s
        self._session: SessionState | None = None
        self._segments: tuple[StraightSegment, ...] | None = None
        self._lap: list[tuple[float, bool]] = []
        self._laps: deque[list[tuple[float, bool]]] = deque(maxlen=learn_laps)
        self._completed_laps: int = 0
        self._pitted: bool = False
        self._straight: bool = False

    def update(self, snapshot: Snapshot) -> None:
        """Feed one snapshot: learn while the straights are unknown, then judge the straight left ahead.
        In:  snapshot  Snapshot
        Out:           None
        """
        self._fallback.update(snapshot)
        session, player = snapshot.session, snapshot.player
        if _is_new_session(self._session, session):
            self._restart(session, player.vehicle.completed_laps)
        self._session = session
        if self._segments is None:
            self._learn(snapshot)
            return
        left_m = straight_left_m(self._segments, player.vehicle.lap_distance_m, session.track_length_m)
        self._straight = left_m > 0 and left_m >= self._min_remaining_s * player.vehicle.speed_mps

    def is_straight(self) -> bool:
        """Whether enough straight is left ahead to speak; the fallback's answer until the straights are known.
        In:  -
        Out: bool
        """
        return self._fallback.is_straight() if self._segments is None else self._straight

    def segments(self) -> tuple[tuple[float, float], ...]:
        """The straights in use, given by the track table or learned.
        In:  -
        Out: tuple[tuple[float, float], ...]  (start m, end m) lap distance per straight, longest first; empty until
                                              known
        """
        return tuple((s.start_m, s.end_m) for s in self._segments or ())

    def _restart(self, session: SessionState, completed_laps: int) -> None:
        """Start over for a new track or session: take the track's given straights, else learn again.
        In:  session         SessionState
             completed_laps  int           player's completed laps
        Out:                 None
        """
        track = self._tracks.get(session.track_name)
        self._segments = None
        if track is not None and track.straights:
            self._segments = tuple(
                StraightSegment(start_m, end_m, (end_m - start_m) % track.length_m)
                for start_m, end_m in track.straights
            )
        self._lap = []
        self._laps.clear()
        self._completed_laps, self._pitted, self._straight = completed_laps, False, False

    def _learn(self, snapshot: Snapshot) -> None:
        """Collect one sample; at the end of a lap without the pits keep its samples, and learn once learn_laps such
        laps are there (again after the next lap when they show no straight).
        In:  snapshot  Snapshot
        Out:           None
        """
        player = snapshot.player
        if player.vehicle.completed_laps != self._completed_laps:
            if not self._pitted:
                self._laps.append(self._lap)
            self._lap, self._pitted, self._completed_laps = [], False, player.vehicle.completed_laps
            if len(self._laps) == self._learn_laps:
                samples = [sample for lap in self._laps for sample in lap]
                self._segments = (
                    learn_straights(
                        np.array([d for d, _ in samples], dtype=np.float64),
                        np.array([s for _, s in samples], dtype=np.bool_),
                        snapshot.session.track_length_m,
                        self._bin_m,
                        self._min_fraction,
                        self._min_length_m,
                    )
                    or None
                )
            if self._segments is not None:
                return
        self._pitted = self._pitted or player.vehicle.in_pits
        self._lap.append(
            (
                player.vehicle.lap_distance_m,
                on_straight(
                    player.steering,
                    player.accel_lateral_mps2,
                    player.vehicle.speed_mps,
                    self._max_steering,
                    self._max_lateral_mps2,
                    self._min_speed_mps,
                ),
            )
        )


def _segment(start_bin: int, end_bin: int, bin_m: float, track_length_m: float) -> StraightSegment:
    """Segment from a run of straight bins.
    In:  start_bin       int    first straight bin
         end_bin         int    first bin after the run; at or below start_bin when the run crosses the line
         bin_m           float  m
         track_length_m  float  m
    Out:                 StraightSegment
    """
    start_m = start_bin * bin_m
    end_m = end_bin * bin_m if end_bin > 0 else track_length_m
    return StraightSegment(start_m, end_m, end_m - start_m if end_m > start_m else track_length_m - start_m + end_m)


def _is_new_session(prev: SessionState | None, cur: SessionState) -> bool:
    """Whether the current session differs from the previous one.
    In:  prev  SessionState | None  None before the first update
         cur   SessionState
    Out:       bool  True on the first update, a session type or track change, or the session clock going backwards
    """
    return (
        prev is None
        or cur.session_type != prev.session_type
        or cur.track_name != prev.track_name
        or cur.elapsed_s < prev.elapsed_s
    )
