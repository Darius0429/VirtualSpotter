from dataclasses import dataclass

import numpy as np

from ai_race_engineer.domains.common.analysis.corners.lap_segments import LineSamples, line_samples
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot
from ai_race_engineer.domains.common.data.snapshot import VehicleState

MPS_TO_KPH: float = 3.6


@dataclass(frozen=True)
class RivalLap:
    """One full lap seen of another car.
    driver_name  str
    class_name   str
    lap_time_s   float        s
    line         LineSamples  the line and the speed of the lap
    """

    driver_name: str
    class_name: str
    lap_time_s: float
    line: LineSamples


@dataclass(frozen=True)
class SegmentLine:
    """What a car's line and speed show in one segment; the game gives no pedals of other cars.
    time_s           float         s in the segment
    time_to_apex_s   float | None  s from the start of the segment to the apex; None for a straight
    entry_kph        float         km/h at the start of the segment
    min_kph          float         km/h
    exit_kph         float         km/h at the end of the segment
    top_kph          float         km/h
    slowing_point_m  float         m lap distance where the car was last at its highest speed before the minimum
                                   speed: where it starts slowing for the corner
    min_speed_m      float         m lap distance of the minimum speed
    """

    time_s: float
    time_to_apex_s: float | None
    entry_kph: float
    min_kph: float
    exit_kph: float
    top_kph: float
    slowing_point_m: float
    min_speed_m: float


def segment_line(line: LineSamples, segment: Segment, search_m: float) -> SegmentLine:
    """Figures of one segment from a car's line and speed.
    In:  line      LineSamples
         segment   Segment
         search_m  float        m before the segment in which the slowing point is looked for
    Out:           SegmentLine
    """
    edges_m = (segment.start_m, segment.end_m)
    first, last = (int(i) for i in np.searchsorted(line.distance_m, edges_m))
    last = max(last, first + 1)
    start_s, end_s = np.interp(edges_m, line.distance_m, line.time_s)
    entry_mps, exit_mps = np.interp(edges_m, line.distance_m, line.speed_mps)
    slowest = first + int(np.argmin(line.speed_mps[first:last]))
    search = int(np.searchsorted(line.distance_m, segment.start_m - search_m))
    fastest = slowest - int(np.argmax(line.speed_mps[search : slowest + 1][::-1]))
    apex_s = None if segment.apex_m is None else float(np.interp(segment.apex_m, line.distance_m, line.time_s))
    return SegmentLine(
        time_s=float(end_s - start_s),
        time_to_apex_s=None if apex_s is None else apex_s - float(start_s),
        entry_kph=float(entry_mps) * MPS_TO_KPH,
        min_kph=float(line.speed_mps[slowest]) * MPS_TO_KPH,
        exit_kph=float(exit_mps) * MPS_TO_KPH,
        top_kph=float(line.speed_mps[first:last].max()) * MPS_TO_KPH,
        slowing_point_m=float(line.distance_m[fastest]),
        min_speed_m=float(line.distance_m[slowest]),
    )


class RivalLineTracker:
    """Collects the position and speed of every other car and keeps the line of each car's fastest and of its last
    full lap; the fastest lap is kept when a new session starts on the same track. Laps through the pits, laps
    joined halfway and laps the game did not time are left out.
    best           dict[int, RivalLap]  slot id -> fastest lap seen (replaced, never mutated)
    last           dict[int, RivalLap]  slot id -> last full lap seen (replaced, never mutated)
    player_class   str                  class of the player's car
    _track         str                  track of the lines
    _laps          dict[int, int]       slot id -> completed laps at the last snapshot
    _full          dict[int, bool]      slot id -> the running lap was seen from the line and stayed out of the pits
    _points        dict[int, list[tuple[float, float, float, float]]]  slot id -> (x m, z m, speed m/s, lap distance
                                                                m) of the running lap
    """

    def __init__(self) -> None:
        """Start without lines.
        In:  -
        Out: None
        """
        self.best: dict[int, RivalLap] = {}
        self.last: dict[int, RivalLap] = {}
        self.player_class: str = ""
        self._track: str = ""
        self._laps: dict[int, int] = {}
        self._full: dict[int, bool] = {}
        self._points: dict[int, list[tuple[float, float, float, float]]] = {}

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        if snapshot.session.track_name != self._track:
            self._track = snapshot.session.track_name
            self.best, self.last, self._laps, self._full, self._points = {}, {}, {}, {}, {}
        self.player_class = snapshot.player.vehicle.class_name
        for vehicle in snapshot.vehicles:
            if not vehicle.is_player:
                self._update_car(vehicle)

    def _update_car(self, vehicle: VehicleState) -> None:
        """Feed one car's state.
        In:  vehicle  VehicleState
        Out:          None
        """
        slot = vehicle.slot_id
        laps = self._laps.get(slot)
        if vehicle.completed_laps != laps:
            if laps is not None and vehicle.completed_laps < laps:
                self.last = {key: lap for key, lap in self.last.items() if key != slot}
            elif self._full.get(slot) and vehicle.completed_laps == laps + 1 and vehicle.last_lap_s is not None:
                self._finish(vehicle)
            self._full[slot] = laps is not None
            self._laps[slot] = vehicle.completed_laps
            self._points[slot] = []
        if vehicle.in_pits:
            self._full[slot] = False
        point = (*vehicle.position_xz_m, vehicle.speed_mps, vehicle.lap_distance_m)
        if not self._points[slot] or self._points[slot][-1][:2] != point[:2]:
            self._points[slot].append(point)

    def _finish(self, vehicle: VehicleState) -> None:
        """Keep the lap in the buffer as the car's last lap, and as its best when it is the fastest so far.
        In:  vehicle  VehicleState  first state after the line: carries the lap time
        Out:          None
        """
        x_m, z_m, speed_mps, lap_distance_m = np.array(self._points[vehicle.slot_id], dtype=np.float64).T
        line = line_samples(x_m, z_m, speed_mps, lap_distance_m, vehicle.last_lap_s)
        lap = RivalLap(vehicle.driver_name, vehicle.class_name, vehicle.last_lap_s, line)
        self.last = {**self.last, vehicle.slot_id: lap}
        known = self.best.get(vehicle.slot_id)
        if known is None or lap.lap_time_s < known.lap_time_s:
            self.best = {**self.best, vehicle.slot_id: lap}
