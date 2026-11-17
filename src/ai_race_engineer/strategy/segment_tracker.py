import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

from ai_race_engineer.strategy.corner_card import CardLimits, SegmentCard, lap_cards, segment_card
from ai_race_engineer.strategy.lap_segments import LapSamples, lap_samples
from ai_race_engineer.strategy.lap_summary import LapShares, lap_shares
from ai_race_engineer.strategy.track_map import Segment, track_map
from ai_race_engineer.telemetry.race_snapshot import RaceSnapshot
from ai_race_engineer.telemetry.snapshot import PlayerState


@dataclass(frozen=True)
class AnalysedLap:
    """One lap of the player split into the track's segments.
    lap         int                      lap number as the driver counts it; 0 = the stored best lap
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
    outside the pits. Out-laps, in-laps and laps the game did not time are not analysed as laps, but their segments
    are in live as they are driven: on an out-lap from the pit exit, on an in-lap up to the pit entry.
    laps          tuple[AnalysedLap, ...]  analysed laps of the session, oldest first (replaced, never mutated)
    reference     AnalysedLap | None       the driver's best lap of this car on this track, from this or an earlier
                                           session (lap number 0); it outlives a change of session
    live          tuple[SegmentCard, ...]  the segments already driven on the lap in progress, in driving order; a
                                           corner is added once the car is past its exit (replaced, never mutated)
    live_first    int                      position in the track map of the first segment in live: 0 on a lap
                                           from the line, the first segment after the pit exit on an out-lap
    live_samples  LapSamples | None        the telemetry of the lap in progress when the last card was made
    _floor_m      float                    m lap distance before which the braking of the next corner is not sought
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
    _carding      bool                     segments of the running lap are being added to live
    _pitted       bool                     the car was in the pits or the garage at the last snapshot
    _buffer       list[PlayerState]        states of the running lap; on an out-lap from where the car left the pits
    """

    def __init__(
        self,
        load: Callable[[str], tuple[Segment, ...]],
        save: Callable[[str, tuple[Segment, ...]], None],
        load_best: Callable[[str, str], tuple[float, LapSamples] | None],
        save_best: Callable[[str, str, float, LapSamples], None],
        keep_laps: int,
        limits: CardLimits,
        **map_args: float,
    ) -> None:
        """Store the map store, the thresholds and the map settings.
        In:  load         Callable[[str], tuple[Segment, ...]]        track name -> stored map; empty when none
             save         Callable[[str, tuple[Segment, ...]], None]  stores the map of a track
             load_best    Callable[[str, str], tuple[float, LapSamples] | None]  track, car -> stored best lap
             save_best    Callable[[str, str, float, LapSamples], None]  stores a best lap of a car on a track
             keep_laps    int         analysed laps kept
             limits       CardLimits  thresholds of the corner events
             map_args     float  track_map arguments: step_m, smooth_m, max_radius_m, min_turn_deg, entry_pad_m,
                                 exit_pad_m, merge_gap_m
        Out:              None
        """
        self.laps: tuple[AnalysedLap, ...] = ()
        self.reference: AnalysedLap | None = None
        self.live: tuple[SegmentCard, ...] = ()
        self.live_first: int = 0
        self.live_samples: LapSamples | None = None
        self._floor_m: float = 0.0
        self.segments: tuple[Segment, ...] = ()
        self.latest: RaceSnapshot | None = None
        self._load_best: Callable[[str, str], tuple[float, LapSamples] | None] = load_best
        self._save_best: Callable[[str, str, float, LapSamples], None] = save_best
        self._car: str = ""
        self._load: Callable[[str], tuple[Segment, ...]] = load
        self._save: Callable[[str, tuple[Segment, ...]], None] = save
        self._map_args: dict[str, float] = map_args
        self._limits: CardLimits = limits
        self._keep_laps: int = keep_laps
        self._track: str = ""
        self._lap: int | None = None
        self._full: bool = False
        self._carding: bool = False
        self._pitted: bool = False
        self._buffer: list[PlayerState] = []

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        vehicle = snapshot.player.vehicle
        car = f"{vehicle.class_name} {vehicle.vehicle_name.split('#')[0].strip()}"
        if snapshot.session.track_name != self._track or car != self._car:
            self._track, self._car = snapshot.session.track_name, car
            self.segments = self._load(self._track)
            self.laps, self._lap = (), None
            stored = self._load_best(self._track, car) if self.segments else None
            self.reference = None if stored is None else self._analysed(0, *stored)
        if vehicle.completed_laps != self._lap:
            if self._lap is not None and vehicle.completed_laps < self._lap:
                self.laps = ()
            elif self._full and vehicle.completed_laps == self._lap + 1 and vehicle.last_lap_s is not None:
                self._finish(snapshot)
            self._full = self._carding = self._lap is not None
            self._lap = vehicle.completed_laps
            self._buffer = []
            self.live, self.live_first, self.live_samples, self._floor_m = (), 0, None, 0.0
        pitted = vehicle.in_pits or vehicle.in_garage
        if pitted:
            self._full = self._carding = False
        elif self._pitted:
            self._carding = True
            self._buffer = []
            first = sum(segment.start_m < vehicle.lap_distance_m for segment in self.segments)
            self.live, self.live_first, self.live_samples, self._floor_m = (), first, None, 0.0
        self._pitted = pitted
        self._buffer.append(snapshot.player)
        self._add_live()
        self.latest = snapshot

    def _add_live(self) -> None:
        """Add the card of the next segment of the lap in progress once the car has left it; a corner counts as
        left when the car is past the stretch after it where the exit is measured.
        In:  -
        Out: None
        """
        index = self.live_first + len(self.live)
        if not self._carding or index >= len(self.segments):
            return
        segment = self.segments[index]
        left_m = segment.end_m + (self._limits.after_exit_m if segment.corner else 0.0)
        if self._buffer[-1].vehicle.lap_distance_m < left_m:
            return
        samples = lap_samples(self._buffer, None)
        search_from_m = max(self._floor_m, segment.start_m - self._limits.search_m)
        self.live = (*self.live, segment_card(samples, segment, search_from_m, self._limits))
        self.live_samples = samples
        self._floor_m = segment.apex_m if segment.corner else self._floor_m

    def _finish(self, snapshot: RaceSnapshot) -> None:
        """Analyse the lap in the buffer; the first one also gives the track map when none is stored.
        In:  snapshot  RaceSnapshot  first snapshot after the line: carries the lap count and the lap time
        Out:           None
        """
        vehicle = snapshot.player.vehicle
        samples = lap_samples(self._buffer, vehicle.last_lap_s)
        if not self.segments:
            self.segments = track_map(
                samples.distance_m, samples.x_m, samples.z_m, snapshot.session.track_length_m, **self._map_args
            )
            self._save(self._track, self.segments)
        lap = self._analysed(vehicle.completed_laps, vehicle.last_lap_s, samples)
        self.laps = (*self.laps, lap)[-self._keep_laps :]
        if self.reference is None or lap.lap_time_s < self.reference.lap_time_s:
            self.reference = dataclasses.replace(lap, lap=0)
            self._save_best(self._track, self._car, lap.lap_time_s, samples)

    def _analysed(self, lap: int, lap_time_s: float, samples: LapSamples) -> AnalysedLap:
        """One lap split into the track's segments.
        In:  lap         int         lap number; 0 for the stored best lap
             lap_time_s  float       s
             samples     LapSamples
        Out:             AnalysedLap
        """
        return AnalysedLap(
            lap, lap_time_s, samples, lap_cards(samples, self.segments, self._limits), lap_shares(samples, self._limits)
        )
