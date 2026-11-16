from collections.abc import Callable
from dataclasses import dataclass

from ai_race_engineer.strategy.corner_card import CardLimits, SegmentCard, lap_cards
from ai_race_engineer.strategy.lap_segments import LapSamples, lap_samples
from ai_race_engineer.strategy.lap_summary import LapShares, lap_shares
from ai_race_engineer.strategy.track_map import Segment, track_map
from ai_race_engineer.telemetry.race_snapshot import RaceSnapshot
from ai_race_engineer.telemetry.snapshot import PlayerState


@dataclass(frozen=True)
class AnalysedLap:
    """One lap of the player split into the track's segments.
    lap         int                      lap number as the driver counts it
    lap_time_s  float                    s
    samples     LapSamples               the telemetry of the lap
    cards       tuple[SegmentCard, ...]  one per segment, in driving order
    shares      LapShares                how the lap time was spent on the pedals
    """

    lap: int
    lap_time_s: float
    samples: LapSamples
    cards: tuple[SegmentCard, ...]
    shares: LapShares


class SegmentTracker:
    """Collects the player's telemetry of the running lap and, at the start/finish line, splits the finished lap
    into the track's corners and straights. The track map comes from the store, else from the first full lap driven
    outside the pits. Out-laps, in-laps and laps the game did not time are not analysed.
    laps          tuple[AnalysedLap, ...]  analysed laps of the session, oldest first (replaced, never mutated)
    segments      tuple[Segment, ...]      the track map; empty until known (replaced, never mutated)
    latest        RaceSnapshot | None      the latest snapshot (read from other threads)
    _load         Callable[[str], tuple[Segment, ...]]        stored map of a track; empty when there is none
    _save         Callable[[str, tuple[Segment, ...]], None]  stores the map of a track
    _map_args     dict[str, float]         track_map arguments: step_m, smooth_m, max_radius_m, min_turn_deg,
                                           entry_pad_m, exit_pad_m, merge_gap_m
    _limits       CardLimits               thresholds of the corner events
    _keep_laps    int                      analysed laps kept
    _track        str                      track of the map
    _lap          int | None               completed laps at the last snapshot; None before the first
    _full         bool                     the running lap was seen from the line and stayed out of the pits
    _buffer       list[PlayerState]        states of the running lap
    """

    def __init__(
        self,
        load: Callable[[str], tuple[Segment, ...]],
        save: Callable[[str, tuple[Segment, ...]], None],
        keep_laps: int,
        limits: CardLimits,
        **map_args: float,
    ) -> None:
        """Store the map store, the thresholds and the map settings.
        In:  load         Callable[[str], tuple[Segment, ...]]        track name -> stored map; empty when none
             save         Callable[[str, tuple[Segment, ...]], None]  stores the map of a track
             keep_laps    int         analysed laps kept
             limits       CardLimits  thresholds of the corner events
             map_args     float  track_map arguments: step_m, smooth_m, max_radius_m, min_turn_deg, entry_pad_m,
                                 exit_pad_m, merge_gap_m
        Out:              None
        """
        self.laps: tuple[AnalysedLap, ...] = ()
        self.segments: tuple[Segment, ...] = ()
        self.latest: RaceSnapshot | None = None
        self._load: Callable[[str], tuple[Segment, ...]] = load
        self._save: Callable[[str, tuple[Segment, ...]], None] = save
        self._map_args: dict[str, float] = map_args
        self._limits: CardLimits = limits
        self._keep_laps: int = keep_laps
        self._track: str = ""
        self._lap: int | None = None
        self._full: bool = False
        self._buffer: list[PlayerState] = []

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        vehicle = snapshot.player.vehicle
        if snapshot.session.track_name != self._track:
            self._track = snapshot.session.track_name
            self.segments = self._load(self._track)
            self.laps, self._lap = (), None
        if vehicle.completed_laps != self._lap:
            if self._lap is not None and vehicle.completed_laps < self._lap:
                self.laps = ()
            elif self._full and vehicle.completed_laps == self._lap + 1 and vehicle.last_lap_s is not None:
                self._finish(snapshot)
            self._full = self._lap is not None
            self._lap = vehicle.completed_laps
            self._buffer = []
        if vehicle.in_pits:
            self._full = False
        self._buffer.append(snapshot.player)
        self.latest = snapshot

    def _finish(self, snapshot: RaceSnapshot) -> None:
        """Analyse the lap in the buffer; the first one also gives the track map when none is stored.
        In:  snapshot  RaceSnapshot  first snapshot after the line: carries the lap count and the lap time
        Out:           None
        """
        vehicle = snapshot.player.vehicle
        samples = lap_samples(self._buffer, snapshot.session.track_length_m, vehicle.last_lap_s)
        if not self.segments:
            self.segments = track_map(
                samples.distance_m, samples.x_m, samples.z_m, snapshot.session.track_length_m, **self._map_args
            )
            self._save(self._track, self.segments)
        lap = AnalysedLap(
            vehicle.completed_laps,
            vehicle.last_lap_s,
            samples,
            lap_cards(samples, self.segments, self._limits),
            lap_shares(samples, self._limits),
        )
        self.laps = (*self.laps, lap)[-self._keep_laps :]
