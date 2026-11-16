from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ai_race_engineer.strategy.track_map import Segment
from ai_race_engineer.telemetry.race_snapshot import RaceSnapshot
from ai_race_engineer.telemetry.snapshot import VehicleState

MPS_TO_KPH: float = 3.6


@dataclass(frozen=True)
class SegmentPass:
    """One car's drive through one segment, known the moment the car leaves it.
    slot_id      int    game slot id
    driver_name  str
    class_name   str
    segment      str    name of the segment
    lap          int    the car's lap on which it drove the segment, as its driver counts it
    time_s       float  s in the segment
    min_kph      float  km/h
    exit_kph     float  km/h at the end of the segment
    finished_s   float  s session time when the car left the segment
    """

    slot_id: int
    driver_name: str
    class_name: str
    segment: str
    lap: int
    time_s: float
    min_kph: float
    exit_kph: float
    finished_s: float


@dataclass
class _Run:
    """A car's drive through the segment it is in.
    index       int           position of the segment in the track map
    entered_s   float | None  s session time it entered; None when it was not seen entering
    min_mps     float         m/s lowest speed so far
    distance_m  float         m lap distance at the last scoring tick
    time_s      float         s session time of that tick
    """

    index: int
    entered_s: float | None
    min_mps: float
    distance_m: float
    time_s: float


class LiveSegmentTracker:
    """Times every car through every corner and straight as it happens, so the driver can be compared with any car
    in the corner both have just driven, without waiting for the end of the lap. The moment a car crosses the end
    of a segment is worked out between two position updates of the game. Cars in the pits are not timed.
    last        dict[int, dict[str, SegmentPass]]  slot id -> segment name -> the car's latest pass (replaced, never
                                                   mutated)
    best        dict[int, dict[str, SegmentPass]]  slot id -> segment name -> the car's fastest pass of the session
    _segments   Callable[[], Sequence[Segment]]    the track map, empty until known
    _runs       dict[int, _Run]                    slot id -> the segment the car is in
    _time_s     float                              s session time of the last snapshot
    """

    def __init__(self, segments: Callable[[], Sequence[Segment]]) -> None:
        """Store where the track map comes from.
        In:  segments  Callable[[], Sequence[Segment]]  the track map, empty until known
        Out:           None
        """
        self.last: dict[int, dict[str, SegmentPass]] = {}
        self.best: dict[int, dict[str, SegmentPass]] = {}
        self._segments: Callable[[], Sequence[Segment]] = segments
        self._runs: dict[int, _Run] = {}
        self._time_s: float = -1.0

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer): keep the pass of every car that just left a segment. A session clock
        that goes back starts a new session.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        segments = self._segments()
        now_s = snapshot.session.elapsed_s
        if now_s < self._time_s:
            self.last, self.best, self._runs = {}, {}, {}
        self._time_s = now_s
        if not segments:
            return
        for vehicle in snapshot.vehicles:
            if vehicle.in_pits or vehicle.in_garage:
                self._runs.pop(vehicle.slot_id, None)
            else:
                self._update_car(vehicle, segments, now_s, snapshot.session.track_length_m)

    def _update_car(self, vehicle: VehicleState, segments: Sequence[Segment], now_s: float, length_m: float) -> None:
        """Feed one car's state.
        In:  vehicle   VehicleState
             segments  Sequence[Segment]  the track map
             now_s     float              s session time
             length_m  float              m track length
        Out:           None
        """
        index = next((i for i, s in enumerate(segments) if s.start_m <= vehicle.lap_distance_m < s.end_m), None)
        run = self._runs.get(vehicle.slot_id)
        if index is None:
            return
        if run is None:
            self._runs[vehicle.slot_id] = _Run(index, None, vehicle.speed_mps, vehicle.lap_distance_m, now_s)
            return
        run.min_mps = min(run.min_mps, vehicle.speed_mps)
        if vehicle.lap_distance_m == run.distance_m:
            return
        if index != run.index:
            driven_m = (vehicle.lap_distance_m - run.distance_m) % length_m
            to_end_m = (segments[run.index].end_m - run.distance_m) % length_m
            crossed_s = run.time_s + (now_s - run.time_s) * to_end_m / driven_m
            if run.entered_s is not None and index == (run.index + 1) % len(segments):
                self._keep(vehicle, segments[run.index], run, crossed_s)
            entered_s = crossed_s if index == (run.index + 1) % len(segments) else None
            self._runs[vehicle.slot_id] = _Run(index, entered_s, vehicle.speed_mps, vehicle.lap_distance_m, now_s)
            return
        run.distance_m, run.time_s = vehicle.lap_distance_m, now_s

    def _keep(self, vehicle: VehicleState, segment: Segment, run: _Run, crossed_s: float) -> None:
        """Keep a finished pass as the car's latest, and as its best when it is the fastest.
        In:  vehicle    VehicleState  the car, already in the next segment
             segment    Segment       the segment it left
             run        _Run          its drive through that segment
             crossed_s  float         s session time it crossed the end of the segment
        Out:            None
        """
        wrapped = vehicle.lap_distance_m < segment.start_m
        done = SegmentPass(
            slot_id=vehicle.slot_id,
            driver_name=vehicle.driver_name,
            class_name=vehicle.class_name,
            segment=segment.name,
            lap=vehicle.completed_laps + (0 if wrapped else 1),
            time_s=crossed_s - run.entered_s,
            min_kph=run.min_mps * MPS_TO_KPH,
            exit_kph=vehicle.speed_mps * MPS_TO_KPH,
            finished_s=crossed_s,
        )
        self.last = {**self.last, vehicle.slot_id: {**self.last.get(vehicle.slot_id, {}), segment.name: done}}
        fastest = self.best.get(vehicle.slot_id, {}).get(segment.name)
        if fastest is None or done.time_s < fastest.time_s:
            self.best = {**self.best, vehicle.slot_id: {**self.best.get(vehicle.slot_id, {}), segment.name: done}}
