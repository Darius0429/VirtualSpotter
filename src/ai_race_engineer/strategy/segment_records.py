import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.strategy.corner_card import SegmentCard
from ai_race_engineer.strategy.corner_compare import FindingLimits, compare, line_wider_m
from ai_race_engineer.strategy.lap_segments import LapSamples, LineSamples
from ai_race_engineer.strategy.live_segments import SegmentPass
from ai_race_engineer.strategy.references import Reference
from ai_race_engineer.strategy.rival_lines import SegmentLine, segment_line
from ai_race_engineer.strategy.segment_tracker import AnalysedLap
from ai_race_engineer.strategy.track_map import Segment
from ai_race_engineer.telemetry.snapshot import VehicleState

MPS_TO_KPH: float = 3.6
SEGMENT: str = "segment"
SECTOR: str = "sector"
LAP: str = "lap"


@dataclass(frozen=True)
class Record:
    """One finished segment, sector or lap of one car, with its differences to the reference laps as they were when
    it was finished. Every difference is this car minus the reference.
    kind        str    "segment" | "sector" | "lap"
    session     int    number of the session within the run, from 1
    slot_id     int    game slot id
    driver      str
    class_name  str
    me          bool   the player's car
    lap         int    lap number as the car's driver counts it
    name        str    "turn 3" | "straight 2" | "sector 2" | "lap" (a timed lap) | "out lap" | "in lap" |
                       "invalid lap" | "aborted lap" | "retired" | "first lap"
    time_s      float  s
    finished_s  float  s session time it was finished
    figures     dict[str, object]             speeds, the corner card, sector times: what the kind has
    vs          dict[str, dict[str, object]]  "class" | "car" | "track" -> {reference, reference_lap_s, lost_s, ...};
                                              empty for other cars
    samples     LapSamples | None             the player's telemetry of a segment with a margin on both sides
    traces      dict[str, dict[str, NDArray[np.float64]]]  reference -> {time_s, speed_kph}: per sample the time lost
                                              since the start of the segment and the speed difference
    lines       dict[str, LineSamples]        for a segment of the player: reference -> its line around the segment,
                                              for a reference lap of the player; for a full lap of any car: "lap" ->
                                              the line of the lap
    """

    kind: str
    session: int
    slot_id: int
    driver: str
    class_name: str
    me: bool
    lap: int
    name: str
    time_s: float
    finished_s: float
    figures: dict[str, object]
    vs: dict[str, dict[str, object]] = field(default_factory=dict)
    samples: LapSamples | None = None
    traces: dict[str, dict[str, NDArray[np.float64]]] = field(default_factory=dict)
    lines: dict[str, LineSamples] = field(default_factory=dict)


def segment_record(
    session: int,
    vehicle: VehicleState,
    lap: int,
    finished_s: float,
    card: SegmentCard,
    samples: LapSamples,
    segment: Segment,
    sector: int,
    references: Mapping[str, Reference],
    findings: FindingLimits,
    search_m: float,
    margin_m: float,
) -> Record:
    """Record of one segment the player has just driven.
    In:  session     int
         vehicle     VehicleState               the player's car
         lap         int                        lap number as the driver counts it
         finished_s  float                      s session time
         card        SegmentCard                what the driver did in the segment
         samples     LapSamples                 telemetry of the lap up to at least the end of the segment
         segment     Segment
         sector      int                        1, 2 or 3
         references  Mapping[str, Reference]    "class" | "car" | "track" -> the lap to compare with
         findings    FindingLimits              differences from which something is worth saying
         search_m    float                      m before a corner in which a car's slowing point is looked for
         margin_m    float                      m of telemetry kept before and after the segment
    Out:             Record
    """
    low_m, high_m = segment.start_m - margin_m, segment.end_m + margin_m
    own = (samples.distance_m >= low_m) & (samples.distance_m <= high_m)
    piece = LapSamples(**{f.name: getattr(samples, f.name)[own] for f in dataclasses.fields(LapSamples)})
    figures = {key: value for key, value in dataclasses.asdict(card).items() if key not in ("name", "time_s", "energy")}
    return Record(
        kind=SEGMENT,
        session=session,
        slot_id=vehicle.slot_id,
        driver=vehicle.driver_name,
        class_name=vehicle.class_name,
        me=True,
        lap=lap,
        name=card.name,
        time_s=card.time_s,
        finished_s=finished_s,
        figures={"sector": sector, **figures, "energy_percent": None if card.energy is None else card.energy * 100},
        vs={key: _against(card, samples, segment, ref, findings, search_m) for key, ref in references.items()},
        samples=piece,
        traces={key: _trace(piece, ref.line, segment.start_m) for key, ref in references.items()},
        lines={key: _cut(ref.line, low_m, high_m) for key, ref in references.items() if ref.cards is not None},
    )


def sector_record(
    session: int,
    vehicle: VehicleState,
    lap: int,
    finished_s: float,
    sector: int,
    time_s: float | None,
    start_m: float,
    end_m: float,
    samples: LapSamples,
    segments: Sequence[str],
    references: Mapping[str, Reference],
) -> Record:
    """Record of one sector the player has just driven.
    In:  session     int
         vehicle     VehicleState             the player's car
         lap         int
         finished_s  float                    s session time
         sector      int                      1, 2 or 3
         time_s      float | None             s the game timed the sector; None = taken from the telemetry
         start_m     float                    m lap distance where the sector starts
         end_m       float                    m lap distance where it ends
         samples     LapSamples               telemetry of the lap up to at least the end of the sector
         segments    Sequence[str]            names of the segments of the sector, in driving order
         references  Mapping[str, Reference]
    Out:             Record
    """
    time_s = _between(samples, start_m, end_m) if time_s is None else time_s
    return Record(
        kind=SECTOR,
        session=session,
        slot_id=vehicle.slot_id,
        driver=vehicle.driver_name,
        class_name=vehicle.class_name,
        me=True,
        lap=lap,
        name=f"sector {sector}",
        time_s=time_s,
        finished_s=finished_s,
        figures={"segments": list(segments)},
        vs={key: _lost(ref, time_s - _between(ref.line, start_m, end_m)) for key, ref in references.items()},
    )


def lap_record(
    session: int,
    vehicle: VehicleState,
    lap: AnalysedLap,
    finished_s: float,
    theoretical_best_s: float,
    references: Mapping[str, Reference],
) -> Record:
    """Record of one lap the player has just finished.
    In:  session             int
         vehicle             VehicleState             the player's car
         lap                 AnalysedLap
         finished_s          float                    s session time
         theoretical_best_s  float                    s the best pass of every segment added up
         references          Mapping[str, Reference]
    Out:                     Record
    """
    energy = [card.energy for card in lap.cards]
    return Record(
        kind=LAP,
        session=session,
        slot_id=vehicle.slot_id,
        driver=vehicle.driver_name,
        class_name=vehicle.class_name,
        me=True,
        lap=lap.lap,
        name=LAP,
        time_s=lap.lap_time_s,
        finished_s=finished_s,
        figures={
            "theoretical_best_s": theoretical_best_s,
            "full_throttle_percent": lap.shares.full_throttle * 100,
            "braking_percent": lap.shares.braking * 100,
            "coasting_percent": lap.shares.coasting * 100,
            "fuel_l": sum(card.fuel_l for card in lap.cards),
            "energy_percent": None if None in energy else sum(energy) * 100,
        },
        vs={key: _lost(ref, lap.lap_time_s - ref.lap_time_s) for key, ref in references.items()},
    )


def pass_record(session: int, done: SegmentPass) -> Record:
    """Record of one segment another car has just driven.
    In:  session  int
         done     SegmentPass
    Out:          Record
    """
    return Record(
        kind=SEGMENT,
        session=session,
        slot_id=done.slot_id,
        driver=done.driver_name,
        class_name=done.class_name,
        me=False,
        lap=done.lap,
        name=done.segment,
        time_s=done.time_s,
        finished_s=done.finished_s,
        figures={"min_kph": done.min_kph, "exit_kph": done.exit_kph},
    )


def car_record(
    session: int, vehicle: VehicleState, kind: str, lap: int, name: str, time_s: float, finished_s: float
) -> Record:
    """Record of a sector or a lap with nothing but its time: of another car, or a lap of the player that was
    not a timed lap.
    In:  session     int
         vehicle     VehicleState  the car
         kind        str           "sector" | "lap"
         lap         int           lap number as the car's driver counts it
         name        str           "sector 2" | a kind of lap: "out lap", "in lap", "invalid lap", "aborted lap",
                                   "retired", "first lap"
         time_s      float         s
         finished_s  float         s session time
    Out:             Record
    """
    return Record(
        kind=kind,
        session=session,
        slot_id=vehicle.slot_id,
        driver=vehicle.driver_name,
        class_name=vehicle.class_name,
        me=vehicle.is_player,
        lap=lap,
        name=name,
        time_s=time_s,
        finished_s=finished_s,
        figures={},
    )


def _against(
    card: SegmentCard,
    samples: LapSamples,
    segment: Segment,
    reference: Reference,
    limits: FindingLimits,
    search_m: float,
) -> dict[str, object]:
    """One segment of the player against the same segment of a reference lap.
    In:  card       SegmentCard
         samples    LapSamples     telemetry of the player's lap
         segment    Segment
         reference  Reference
         limits     FindingLimits
         search_m   float          m before a corner in which a car's slowing point is looked for
    Out:            dict[str, object]  {reference, reference_lap_s, lost_s, lost_to_apex_s, lost_from_apex_s, min_kph,
                                       exit_kph, line_wider_m, findings, deltas}
    """
    theirs = segment_line(reference.line, segment, search_m)
    wider_m = line_wider_m(samples, reference.line, segment.apex_m) if segment.corner else None
    lost_s = card.time_s - theirs.time_s
    to_apex_s = None if card.time_to_apex_s is None else card.time_to_apex_s - theirs.time_to_apex_s
    if reference.cards is None:
        found = _line_findings(segment_line(samples, segment, search_m), theirs, segment.corner, wider_m, limits)
        deltas: dict[str, float] = {}
    else:
        delta = compare(card, next(c for c in reference.cards if c.name == card.name), wider_m, limits)
        found, deltas = delta.findings, delta.deltas
    return {
        **_lost(reference, lost_s),
        "lost_to_apex_s": to_apex_s,
        "lost_from_apex_s": None if to_apex_s is None else lost_s - to_apex_s,
        "min_kph": card.min_kph - theirs.min_kph,
        "exit_kph": card.exit_kph - theirs.exit_kph,
        "line_wider_m": wider_m,
        "findings": list(found),
        "deltas": deltas,
    }


def _line_findings(
    mine: SegmentLine, theirs: SegmentLine, corner: bool, wider_m: float | None, limits: FindingLimits
) -> tuple[str, ...]:
    """The differences worth saying against a car whose pedals are not known, in driving order.
    In:  mine     SegmentLine
         theirs   SegmentLine
         corner   bool           False = straight
         wider_m  float | None   m wider at the apex
         limits   FindingLimits
    Out:          tuple[str, ...]
    """
    found: list[str] = []
    slowing_m = mine.slowing_point_m - theirs.slowing_point_m
    if corner and abs(slowing_m) >= limits.distance_m:
        found.append(f"started slowing {abs(slowing_m):.0f} m {'earlier' if slowing_m < 0 else 'later'}")
    if wider_m is not None and abs(wider_m) >= limits.distance_m / 2:
        found.append(f"{abs(wider_m):.1f} m {'wider' if wider_m > 0 else 'tighter'} at the apex")
    for label, kph in (("minimum", mine.min_kph - theirs.min_kph), ("exit", mine.exit_kph - theirs.exit_kph)):
        if abs(kph) >= limits.speed_kph:
            found.append(f"{label} speed {abs(kph):.0f} kph {'lower' if kph < 0 else 'higher'}")
    return tuple(found)


def _lost(reference: Reference, lost_s: float) -> dict[str, object]:
    """The time lost to a reference lap, with which lap it was.
    In:  reference  Reference
         lost_s     float      s
    Out:            dict[str, object]  {reference, reference_lap_s, lost_s}
    """
    return {"reference": reference.name, "reference_lap_s": reference.lap_time_s, "lost_s": lost_s}


def _trace(piece: LapSamples, line: LineSamples, start_m: float) -> dict[str, NDArray[np.float64]]:
    """Per sample of the player's telemetry, the time lost to a reference lap since the start of the segment and the
    speed difference at the same lap distance.
    In:  piece    LapSamples   the player's telemetry around the segment
         line     LineSamples  the reference lap
         start_m  float        m lap distance where the segment starts
    Out:          dict[str, NDArray[np.float64]]  {time_s, speed_kph}
    """
    own_s = piece.time_s - np.interp(start_m, piece.distance_m, piece.time_s)
    their_s = np.interp(piece.distance_m, line.distance_m, line.time_s) - np.interp(
        start_m, line.distance_m, line.time_s
    )
    their_mps = np.interp(piece.distance_m, line.distance_m, line.speed_mps)
    return {"time_s": own_s - their_s, "speed_kph": (piece.speed_mps - their_mps) * MPS_TO_KPH}


def _cut(line: LineSamples, low_m: float, high_m: float) -> LineSamples:
    """The part of a line between two lap distances, with one sample more on each side so that it can be read at
    both ends.
    In:  line    LineSamples
         low_m   float        m
         high_m  float        m
    Out:         LineSamples
    """
    first, last = (int(i) for i in np.searchsorted(line.distance_m, (low_m, high_m)))
    inside = slice(max(first - 1, 0), last + 1)
    return LineSamples(
        line.distance_m[inside], line.time_s[inside], line.x_m[inside], line.z_m[inside], line.speed_mps[inside]
    )


def _between(line: LineSamples, start_m: float, end_m: float) -> float:
    """Time a lap took between two lap distances.
    In:  line     LineSamples
         start_m  float        m
         end_m    float        m
    Out:          float        s
    """
    start_s, end_s = np.interp((start_m, end_m), line.distance_m, line.time_s)
    return float(end_s - start_s)
