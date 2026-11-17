import dataclasses
import itertools
from collections.abc import Callable

import numpy as np

from ai_race_engineer.domains.common.store.segment_store import SegmentStore
from ai_race_engineer.domains.common.analysis.corners.corner_card import CardLimits, SegmentCard, lap_cards
from ai_race_engineer.domains.common.analysis.corners.corner_compare import FindingLimits
from ai_race_engineer.domains.common.analysis.corners.lap_segments import LapSamples
from ai_race_engineer.domains.common.analysis.corners.lap_summary import theoretical_best_s
from ai_race_engineer.domains.common.analysis.corners.lap_types import LapTypeTracker
from ai_race_engineer.domains.common.analysis.corners.live_segments import LiveSegmentTracker, SegmentPass
from ai_race_engineer.domains.common.analysis.corners.references import Reference, fastest, own_reference, rival_reference
from ai_race_engineer.domains.common.analysis.corners.rival_lines import RivalLap, RivalLineTracker
from ai_race_engineer.domains.common.analysis.pace.sector_delta import Splits, splits
from ai_race_engineer.domains.common.analysis.corners.segment_records import (
    LAP,
    SECTOR,
    car_record,
    lap_record,
    pass_record,
    sector_record,
    segment_record,
)
from ai_race_engineer.domains.common.analysis.corners.segment_tracker import AnalysedLap, SegmentTracker
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot

SECTORS: int = 3


class SegmentRecorder:
    """Fills the segment store as things happen: a segment of the player when the car has left it, a sector when its
    last segment is done, a lap at the line, each with its differences to the best lap of the class, the driver's
    best lap in this car and the driver's best lap on this track in this class, as those laps were at the start of
    the lap. Every other car's segments, sectors and full laps are added with their times; every full lap of every car
    also carries its line. Out-laps, in-laps, laps the game did not time and laps cut short by a jump to the garage
    are added for every car as laps of that kind, the player's with the segments driven on them.
    _tracker     SegmentTracker      the player's cards of the lap in progress and the analysed laps
    _rivals      RivalLineTracker    the best and the last full lap of every other car
    _live        LiveSegmentTracker  every car's latest pass of every segment
    _lap_types   LapTypeTracker      what the laps that have just ended were
    _store       SegmentStore
    _load_best   Callable[[str, str], tuple[float, LapSamples] | None]  track, class -> stored best lap in the class
    _limits      CardLimits          thresholds of the corner events
    _findings    FindingLimits       differences from which something is worth saying
    _margin_m    float               m of telemetry kept before and after a segment
    _key         tuple[str, object, str]  track, session type and class of the running session
    _time_s      float               s session time of the last snapshot
    _stored      tuple[float, LapSamples] | None  the stored best lap on this track in this class
    _track_best  Reference | None    that lap as a reference, once the track map is known
    _references  dict[str, Reference]  "class" | "car" | "track" -> the lap the running lap is compared with
    _lap         int | None          the player's completed laps when the references were last chosen
    _count       int                 segments of the running lap already recorded
    _sectors     int                 sectors of the running lap already recorded
    _last        AnalysedLap | None  the last finished lap recorded
    _sector      int | None          the player's sector at the last snapshot
    _bounds      dict[int, float]    1, 2 -> m lap distance where sector 2 and sector 3 start: where the car was seen
                                     to change sector, then where the game's sector times of a finished lap put it
    _passes      dict[int, dict[str, SegmentPass]]  the other cars' passes already recorded
    _laps        dict[int, RivalLap]  the other cars' last laps already recorded
    _splits      dict[int, tuple[float | None, ...]]  slot id -> s the game showed at the last snapshot for sector
                                     1 and 2 of the car's lap in progress
    """

    def __init__(
        self,
        tracker: SegmentTracker,
        rivals: RivalLineTracker,
        live: LiveSegmentTracker,
        lap_types: LapTypeTracker,
        store: SegmentStore,
        load_best: Callable[[str, str], tuple[float, LapSamples] | None],
        limits: CardLimits,
        findings: FindingLimits,
        margin_m: float,
    ) -> None:
        """Store the sources, the store and the limits.
        In:  tracker    SegmentTracker
             rivals     RivalLineTracker
             live       LiveSegmentTracker
             lap_types  LapTypeTracker
             store      SegmentStore
             load_best  Callable[[str, str], tuple[float, LapSamples] | None]  track, class -> stored best lap
             limits     CardLimits
             findings   FindingLimits
             margin_m   float  m of telemetry kept before and after a segment
        Out:            None
        """
        self._tracker: SegmentTracker = tracker
        self._rivals: RivalLineTracker = rivals
        self._live: LiveSegmentTracker = live
        self._lap_types: LapTypeTracker = lap_types
        self._store: SegmentStore = store
        self._load_best: Callable[[str, str], tuple[float, LapSamples] | None] = load_best
        self._limits: CardLimits = limits
        self._findings: FindingLimits = findings
        self._margin_m: float = margin_m
        self._key: tuple[str, object, str] = ("", None, "")
        self._time_s: float = 0.0
        self._stored: tuple[float, LapSamples] | None = None
        self._track_best: Reference | None = None
        self._references: dict[str, Reference] = {}
        self._lap: int | None = None
        self._count: int = 0
        self._sectors: int = 0
        self._last: AnalysedLap | None = None
        self._sector: int | None = None
        self._bounds: dict[int, float] = {}
        self._passes: dict[int, dict[str, SegmentPass]] = {}
        self._laps: dict[int, RivalLap] = {}
        self._splits: dict[int, tuple[float | None, ...]] = {}

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer, runs after the three trackers): record what was finished since the
        last one. Another track, class or kind of session, or a session clock that goes back, starts a new session.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        vehicle, session = snapshot.player.vehicle, snapshot.session
        key = (session.track_name, session.session_type, vehicle.class_name)
        if key != self._key or session.elapsed_s < self._time_s:
            if (key[0], key[2]) != (self._key[0], self._key[2]):
                self._stored, self._track_best, self._bounds = self._load_best(key[0], key[2]), None, {}
            self._key = key
            self._count, self._sectors, self._lap, self._splits = 0, 0, None, {}
            self._store.start_session()
        self._time_s = session.elapsed_s
        on_track = not (vehicle.in_pits or vehicle.in_garage)
        if on_track and self._sector is not None and vehicle.sector_index == self._sector + 1:
            self._bounds.setdefault(vehicle.sector_index, vehicle.lap_distance_m)
        self._sector = vehicle.sector_index
        self._finished_lap(snapshot)
        self._driven_segments(snapshot)
        self._other_cars(snapshot)
        for end in self._lap_types.ended:
            if end.name != LAP:
                done = car_record(self._store.session, end.vehicle, LAP, end.lap, end.name, end.time_s, self._time_s)
                self._store.append(done)
        if vehicle.completed_laps != self._lap:
            self._lap = vehicle.completed_laps
            self._references = self._current_references()

    def _finished_lap(self, snapshot: RaceSnapshot) -> None:
        """Record the lap the tracker has just analysed: the segments not recorded while it was driven, the sectors
        still open and the lap. A lap in progress that was given up starts again from nothing.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        laps = self._tracker.laps
        if laps and laps[-1] is not self._last:
            lap = self._last = laps[-1]
            for index in range(self._count, len(lap.cards)):
                self._segment(snapshot, lap.lap, lap.cards[index], lap.samples, index)
            self._count = len(lap.cards)
            sectors = next((s for s in snapshot.sectors if s.slot_id == snapshot.player.vehicle.slot_id), None)
            times = splits(sectors and sectors.last_sector1_s, sectors and sectors.last_sector2_cum_s, lap.lap_time_s)
            self._close_sectors(snapshot, lap.lap, lap.samples, float(lap.samples.distance_m[-1]), times)
            done = lap_record(
                self._store.session,
                snapshot.player.vehicle,
                lap,
                snapshot.session.elapsed_s,
                theoretical_best_s([entry.cards for entry in laps]),
                self._references,
            )
            self._store.append(dataclasses.replace(done, lines={LAP: lap.samples}))
            self._count, self._sectors = 0, 0
            if None not in times:
                for sector, split_s in enumerate(itertools.accumulate(times[:-1]), start=1):
                    self._bounds[sector] = float(np.interp(split_s, lap.samples.time_s, lap.samples.distance_m))
        elif len(self._tracker.live) < self._count:
            self._count, self._sectors = 0, 0

    def _driven_segments(self, snapshot: RaceSnapshot) -> None:
        """Record the segments of the lap in progress the car has left since the last snapshot, then the sectors
        they complete.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        live, samples = self._tracker.live, self._tracker.live_samples
        if len(live) <= self._count:
            return
        lap = snapshot.player.vehicle.completed_laps + 1
        first = self._tracker.live_first
        for index in range(self._count, len(live)):
            self._segment(snapshot, lap, live[index], samples, first + index)
        self._count = len(live)
        if first:
            return
        sectors = next((s for s in snapshot.sectors if s.slot_id == snapshot.player.vehicle.slot_id), None)
        self._close_sectors(
            snapshot, lap, samples, None, splits(sectors and sectors.sector1_s, sectors and sectors.sector2_cum_s, None)
        )

    def _segment(self, snapshot: RaceSnapshot, lap: int, card: SegmentCard, samples: LapSamples, index: int) -> None:
        """Record one segment of the player.
        In:  snapshot  RaceSnapshot
             lap       int          lap number as the driver counts it
             card      SegmentCard
             samples   LapSamples   telemetry of the lap up to at least the end of the segment
             index     int          position of the segment in the track map
        Out:           None
        """
        segment = self._tracker.segments[index]
        self._store.append(
            segment_record(
                self._store.session,
                snapshot.player.vehicle,
                lap,
                snapshot.session.elapsed_s,
                card,
                samples,
                segment,
                1 + sum(bound_m <= _middle_m(segment) for bound_m in self._bounds.values()),
                self._references,
                self._findings,
                self._limits.search_m,
                self._margin_m,
            )
        )

    def _close_sectors(
        self,
        snapshot: RaceSnapshot,
        lap: int,
        samples: LapSamples,
        lap_end_m: float | None,
        times: Splits,
    ) -> None:
        """Record every sector of the running lap whose end the telemetry has reached and whose segments are all
        recorded.
        In:  snapshot   RaceSnapshot
             lap        int           lap number as the driver counts it
             samples    LapSamples    telemetry of the lap so far
             lap_end_m  float | None  m lap distance of the line when the lap is finished; None while it is driven
             times      Splits        s the game timed sector 1, 2 and 3 of this lap; None where it has no time
        Out:            None
        """
        segments = self._tracker.segments
        while self._sectors < SECTORS:
            start_m = self._bounds.get(self._sectors) if self._sectors else 0.0
            end_m = self._bounds.get(self._sectors + 1) if self._sectors < SECTORS - 1 else lap_end_m
            if start_m is None or end_m is None or samples.distance_m[-1] < end_m:
                return
            inside = [index for index, s in enumerate(segments) if start_m <= _middle_m(s) < end_m]
            if inside and inside[-1] >= self._count:
                return
            self._sectors += 1
            self._store.append(
                sector_record(
                    self._store.session,
                    snapshot.player.vehicle,
                    lap,
                    snapshot.session.elapsed_s,
                    self._sectors,
                    times[self._sectors - 1],
                    start_m,
                    end_m,
                    samples,
                    [segments[index].name for index in inside],
                    self._references,
                )
            )

    def _other_cars(self, snapshot: RaceSnapshot) -> None:
        """Record the segments the other cars have left, the sectors they have finished and the full laps they have
        finished since the last snapshot.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        session, now_s = self._store.session, snapshot.session.elapsed_s
        cars = {vehicle.slot_id: vehicle for vehicle in snapshot.vehicles if not vehicle.is_player}
        passes, laps = self._live.last, self._rivals.last
        if passes is not self._passes:
            for slot_id, done in passes.items():
                seen = self._passes.get(slot_id, {})
                if slot_id not in cars or done is seen:
                    continue
                for name, last in done.items():
                    if seen.get(name) is not last:
                        self._store.append(pass_record(session, last))
            self._passes = passes
        for entry in snapshot.sectors:
            now = splits(entry.sector1_s, entry.sector2_cum_s, None)[:2]
            seen = self._splits.get(entry.slot_id, (None, None))
            car = cars.get(entry.slot_id)
            for sector, (time_s, known_s) in enumerate(zip(now, seen, strict=True), start=1):
                if car is not None and time_s is not None and known_s is None:
                    name = f"{SECTOR} {sector}"
                    self._store.append(car_record(session, car, SECTOR, car.completed_laps + 1, name, time_s, now_s))
            self._splits[entry.slot_id] = now
        if laps is not self._laps:
            sectors = {entry.slot_id: entry for entry in snapshot.sectors}
            for slot_id, lap in laps.items():
                if self._laps.get(slot_id) is lap or slot_id not in cars:
                    continue
                car, entry = cars[slot_id], sectors.get(slot_id)
                last_s = splits(entry and entry.last_sector1_s, entry and entry.last_sector2_cum_s, lap.lap_time_s)[2]
                if last_s is not None:
                    self._store.append(
                        car_record(session, car, SECTOR, car.completed_laps, f"{SECTOR} {SECTORS}", last_s, now_s)
                    )
                done = car_record(session, car, LAP, car.completed_laps, LAP, lap.lap_time_s, now_s)
                self._store.append(dataclasses.replace(done, lines={LAP: lap.line}))
            self._laps = laps

    def _current_references(self) -> dict[str, Reference]:
        """The laps a lap that starts now is compared with.
        In:  -
        Out: dict[str, Reference]  "class": the fastest lap of the class in this run, the driver's included; "car":
                                   the driver's best lap in this car on this track; "track": the driver's best lap
                                   on this track in any car of the class; a key is missing while there is no such lap
        """
        tracker = self._tracker
        if self._track_best is None and self._stored is not None and tracker.segments:
            lap_time_s, samples = self._stored
            cards = lap_cards(samples, tracker.segments, self._limits)
            self._track_best = Reference("stored track best", lap_time_s, samples, cards)
        car = None if tracker.reference is None else own_reference(tracker.reference, tracker.laps)
        mine = [own_reference(lap, tracker.laps) for lap in tracker.laps]
        theirs = [rival_reference(lap) for lap in self._rivals.best.values() if lap.class_name == self._key[2]]
        found = {"class": fastest([*mine, *theirs]), "car": car, "track": fastest([car, self._track_best])}
        return {key: reference for key, reference in found.items() if reference is not None}


def _middle_m(segment: Segment) -> float:
    """The lap distance that says which sector a segment belongs to: the apex of a corner, the middle of a straight.
    In:  segment  Segment
    Out:          float  m
    """
    return segment.apex_m if segment.corner else (segment.start_m + segment.end_m) / 2
