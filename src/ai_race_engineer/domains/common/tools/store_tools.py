import dataclasses
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ai_race_engineer.domains.common.store.corner_plot import CornerPlotter
from ai_race_engineer.domains.common.store.segment_store import SegmentStore
from ai_race_engineer.domains.common.analysis.corners.corner_compare import line_wider_m
from ai_race_engineer.domains.common.analysis.corners.lap_segments import LineSamples
from ai_race_engineer.domains.common.analysis.corners.rival_lines import segment_line
from ai_race_engineer.domains.common.analysis.corners.segment_records import LAP, MPS_TO_KPH, SECTOR, SEGMENT, Record
from ai_race_engineer.domains.common.analysis.corners.segment_tracker import SegmentTracker
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment
from ai_race_engineer.domains.common.data.snapshot import WheelState

UNAVAILABLE: dict[str, object] = {"available": False}
WHEELS: tuple[str, ...] = ("front_left", "front_right", "rear_left", "rear_right")
LEFT_WHEELS: tuple[int, ...] = (0, 2)
REFERENCES: tuple[str, ...] = ("class", "car", "track")
ME: str = "me"
FASTEST: str = "fastest"
NOW: str = "now"


@dataclass(frozen=True)
class _Side:
    """What the store holds of one car on the lap asked for.
    driver  str            "me" for the player
    me      bool
    lap     int | None     lap number; None for the latest passes of another car, which can span two laps
    record  Record | None  the lap's record; None while the lap is in progress
    parts   list[Record]   its segments and sectors, oldest first
    """

    driver: str
    me: bool
    lap: int | None
    record: Record | None
    parts: list[Record]


class StoreTools:
    """Agent tools on the segment store: one query for every time and difference of any car, lap, sector or segment,
    the plot of a corner, the line of the driver against another car in a corner, and the state of the car.
    _store     SegmentStore      everything timed in this run
    _tracker   SegmentTracker    the track map and the latest snapshot
    _plotter   CornerPlotter     draws the corner plots
    _search_m  float             m before a corner in which a car's slowing point is looked for
    """

    def __init__(
        self,
        store: SegmentStore,
        tracker: SegmentTracker,
        plotter: CornerPlotter,
        search_m: float,
    ) -> None:
        """Store the sources and the plotter.
        In:  store     SegmentStore
             tracker   SegmentTracker
             plotter   CornerPlotter
             search_m  float  m before a corner in which a car's slowing point is looked for
        Out:           None
        """
        self._store: SegmentStore = store
        self._tracker: SegmentTracker = tracker
        self._plotter: CornerPlotter = plotter
        self._search_m: float = search_m

    def lap_data(self, car: str, lap: str, part: str, against: str) -> dict[str, object]:
        """Times and time lost of any car on any lap, sector, corner or straight, from the list that is filled the
        moment a segment, a sector or a lap is finished. Use it for every question about where time is lost or
        won, what happened in a corner, this lap, last lap, a sector, and how the driver compares with another car.
        The driver against a reference lap (against "class", "car" or "track"): the differences were worked out
        when the segment was driven, against the reference lap as it was at the start of that lap, and "reference"
        names that lap. Without a part: the lap, its sectors and every segment with the time lost and what the
        driver did differently in plain words (brake point, a locked wheel, the line, minimum speed, throttle). With
        a sector: that sector and its segments. With a corner or straight: everything the driver did there (entry:
        brake point, braking, trail braking, a locking wheel; mid: minimum speed, cornering load, coasting; exit:
        throttle pick-up, full throttle, wheelspin), the differences to the reference in "vs" (negative
        brake_point_m = braked earlier, positive line_wider_m = wider at the apex) and the time lost to the other
        two references. Against another car, or for another car: both cars' times per sector and segment and the
        difference in time, minimum and exit speed; the game gives no pedals of other cars. Every difference is
        the car asked about minus what it is compared with: positive lost_s = slower. Segments carry their
        real-life turn numbers or names; straights are "straight N" from the start/finish line. A corner made of
        two turns ("turn 9 and 10") is found by either number. "best" and "last" are timed laps; lap_type says
        what a numbered lap was: "lap", "first lap" (of a race, from the grid), "out lap", "in lap", "invalid lap"
        (not timed by the game), "aborted lap" (jump to the garage) or "retired".
        In:  car      str  whose lap: empty = the driver; else a driver's name or part of it
             lap      str  "now" = the lap in progress (another car: its latest pass of each segment), "last" = the
                           last finished lap, "best" = the fastest lap, or a lap number as the driver counts it
             part     str  empty = the whole lap; "sector 2"; or a segment, e.g. "turn 10", "Arnage", "straight 2"
             against  str  for the driver: "class" = the fastest lap of the class, "car" = the driver's best lap in
                           this car, "track" = the driver's best lap on this track in the class (each as it was at
                           the start of that lap), a driver's name, or "fastest" = the car with the fastest lap of
                           the class in this session; with a driver or "fastest" the same lap word is used for
                           both cars: best against best, last against last, lap 10 against lap 10; for another
                           car: "me"
        Out:          dict[str, object]  {car, lap, in_progress, against, reference, time_s, lost_s, sectors:
                                         [{name, time_s, lost_s}], segments: [{name, time_s, lost_s, findings}]},
                                         most lost first, on a lap in progress latest first; {"available": False}
                                         when nothing of that car, lap or part is recorded
        """
        mine = self._side(car, lap)
        key = against.strip().lower() or "car"
        if mine is None:
            return UNAVAILABLE
        if mine.me and key in REFERENCES:
            return _rounded(_against_reference(mine, part.strip().lower(), key))
        named = self._class_fastest() if key == FASTEST else ME if key in (ME, *REFERENCES) else against
        other = None if named is None else self._side(named, lap)
        if other is None or other.driver == mine.driver:
            return UNAVAILABLE
        return _rounded(self._against_car(mine, other, part.strip().lower(), lap.strip().lower()))

    def corner_plot(self, corner: str, lap: str, against: str) -> Path | dict[str, object]:
        """Draw one corner of one lap of the driver against another lap, and look at the picture: on the left the
        driven line, on the right the speed over the lap distance, both coloured red where the driver braked, grey
        where the car coasted and green on the throttle. The lap it is compared with is the dashed black line, the
        orange line marks the apex and the purple line is the time lost to that lap as it builds up through the
        corner. It works for a corner of the lap in progress as soon as the car has left it. Use it after lap_data
        shows time lost in a corner, to see where on the line and on the pedals it went. The picture is saved for
        the driver.
        In:  corner   str  name of the corner, e.g. "turn 3" or "Arnage"
             lap      str  the driver's lap: "now", "last", "best" or a lap number as the driver counts it
             against  str  "class", "car" or "track" as in lap_data; a driver's name or "fastest" for that car's
                           lap of the same kind (best against best, last against last, lap 10 against lap 10; with
                           "now" its last full lap); or another lap of the driver: "my best", "my last", "my lap 7"
        Out:          Path | dict[str, object]  the PNG file; {"available": False} for a straight, when the corner
                                                is not recorded on that lap or there is no such lap to compare with
        """
        side = self._side("", lap)
        index = _index(self._tracker.segments, corner)
        if side is None or index is None or not self._tracker.segments[index].corner:
            return UNAVAILABLE
        segment = self._tracker.segments[index]
        found = next((r for r in side.parts if r.kind == SEGMENT and r.name == segment.name), None)
        key, which = against.strip().lower() or "car", lap.strip().lower()
        if found is None:
            return UNAVAILABLE
        record = self._store.telemetry(found)
        if key in REFERENCES:
            line, name = self._reference_line(record, key), record.vs.get(key, {}).get("reference")
        else:
            own = key.startswith("my ")
            driver = ME if own else self._class_fastest() if key == FASTEST else against
            theirs = key.split()[-1] if own else "last" if which == NOW else which
            other = self._lap_line(driver, theirs)
            line = None if other is None else other[1]
            name = None if other is None else f"{'my' if own else other[0].driver} lap {other[0].lap}"
        if line is None or record.samples is None:
            return UNAVAILABLE
        return self._plotter.save(record.samples, line, segment, record.lap, name)

    def corner_lines(self, driver: str, corner: str, lap: str) -> dict[str, object]:
        """One corner or straight of the driver against another car in detail, from the line and the speed of both
        on a full lap: for both cars the time, the entry, minimum, exit and top speed, where the car starts slowing
        and where it is slowest, and how much wider the driver's line is at the apex. The line of every full lap of
        every car is kept, so any lap can be compared: the fastest lap of each, the last full lap of each, or the
        same lap number of both. Every difference is the driver minus the other car. Use it after lap_data shows a
        corner where another car is faster, to see what that car does differently.
        In:  driver  str  driver of the other car, or part of the name; empty = the fastest in the driver's class
             corner  str  name of the segment, e.g. "turn 3"
             lap     str  "best" = the fastest lap of each, "last" = the last full lap of each, or a lap number for
                          that lap of both cars
        Out:         dict[str, object]  {laps_compared, driver, their_lap, their_lap_time_s, my_lap, my_lap_time_s,
                                        name, me: {...}, them: {...}, difference: {...}, line_wider_m};
                                        {"available": False} when either car has no such full lap or the track has
                                        no such segment
        """
        mine = self._lap_line(ME, lap)
        theirs = self._lap_line(driver or self._class_fastest(), lap)
        index = _index(self._tracker.segments, corner)
        if mine is None or theirs is None or index is None:
            return UNAVAILABLE
        segment = self._tracker.segments[index]
        me = dataclasses.asdict(segment_line(mine[1], segment, self._search_m))
        them = dataclasses.asdict(segment_line(theirs[1], segment, self._search_m))
        return _rounded(
            {
                "laps_compared": lap.strip().lower(),
                "driver": theirs[0].driver,
                "their_lap": theirs[0].lap,
                "their_lap_time_s": theirs[0].time_s,
                "my_lap": mine[0].lap,
                "my_lap_time_s": mine[0].time_s,
                "name": segment.name,
                "me": me,
                "them": them,
                "difference": {key: me[key] - them[key] for key in me if me[key] is not None},
                "line_wider_m": None if segment.apex_m is None else line_wider_m(mine[1], theirs[1], segment.apex_m),
            }
        )

    def car_health(self) -> dict[str, object]:
        """The state of the driver's car right now: fuel and energy on board, tyre compounds, and per wheel the
        tread temperature on the inner, middle and outer part, the carcass temperature, the pressure, the brake
        temperature and how much of the tyre is left; also the track temperature.
        In:  -
        Out: dict[str, object]  {fuel_l, energy_percent, compound_front, compound_rear, track_temp_c, wheels:
                                {wheel: {tread_inner_c, tread_middle_c, tread_outer_c, carcass_c, pressure_kpa,
                                brake_c, tyre_left_percent}}}; {"available": False} without telemetry
        """
        snapshot = self._tracker.latest
        if snapshot is None:
            return UNAVAILABLE
        player = snapshot.player
        tyres = next((t for t in snapshot.tyres if t.slot_id == player.vehicle.slot_id), None)
        return _rounded(
            {
                "fuel_l": player.fuel_l,
                "energy_percent": None if player.virtual_energy is None else player.virtual_energy * 100,
                "compound_front": player.tyre_compound_front,
                "compound_rear": player.tyre_compound_rear,
                "track_temp_c": snapshot.session.track_temp_c,
                "wheels": {
                    name: {**_wheel(wheel, index), "tyre_left_percent": tyres and tyres.wear[index] * 100}
                    for index, (name, wheel) in enumerate(zip(WHEELS, player.wheels, strict=False))
                },
            },
            digits=1,
        )

    def tools(self) -> list[Callable[..., object]]:
        """The functions an agent may call.
        In:  -
        Out: list[Callable[..., object]]  lap_data, corner_plot, corner_lines, car_health
        """
        return [self.lap_data, self.corner_plot, self.corner_lines, self.car_health]

    def _side(self, driver: str, lap: str) -> _Side | None:
        """What the store holds of one car on one lap.
        In:  driver  str  empty or "me" = the player; else a driver's name or part of it, any case
             lap     str  "now" | "last" | "best" | a lap number of the running session
        Out:         _Side | None  None when no such car or nothing of that lap is recorded
        """
        wanted, which = driver.strip().lower(), lap.strip().lower()
        own = wanted in ("", ME)
        newest = self._store.select("id IN (SELECT MAX(id) FROM times GROUP BY me, driver)")
        cars = sorted((r for r in newest if r.me == own), key=lambda r: r.driver)
        match = [r for r in cars if own or wanted in r.driver.lower()]
        if not match:
            return None
        car, of_car = "me = ? AND driver = ?", (match[0].me, match[0].driver)
        name, session, latest = ME if own else match[0].driver, self._store.session, self._tracker.latest
        if which == NOW and not own:
            newest = f"SELECT MAX(id) FROM times WHERE {car} AND session = ? AND kind != ? GROUP BY name"
            parts = self._store.select(f"id IN ({newest})", (*of_car, session, LAP))
            return _Side(name, False, None, None, parts) if parts else None
        if which == "best":
            laps = self._fastest(car, of_car)
        elif which == "last":
            newest = f"SELECT MAX(id) FROM times WHERE {car} AND session = ? AND name = ?"
            laps = self._store.select(f"id = ({newest})", (*of_car, session, LAP))
        else:
            laps = []
        if laps:
            session, number = laps[0].session, laps[0].lap
        elif which.isdigit():
            number = int(which)
        else:
            number = latest.player.vehicle.completed_laps + 1 if which == NOW and latest is not None else 0
        rows = self._store.select(f"{car} AND session = ? AND lap = ?", (*of_car, session, number))
        record = next((r for r in rows if r.kind == LAP), None)
        return _Side(name, own, number, record, [r for r in rows if r.kind != LAP]) if rows else None

    def _against_car(self, mine: _Side, other: _Side, part: str, which: str) -> dict[str, object]:
        """One car against another from their recorded times.
        In:  mine   _Side  the car asked about
             other  _Side  the car it is compared with
             part   str    "" | "sector N" | a segment name, lower case
             which  str    the lap asked for, lower case
        Out:        dict[str, object]  {car, against, laps_compared, lap, their_lap, part, time_s, their_time_s,
                                       lost_s, sectors: [{name, time_s, their_time_s, lost_s}], segments: [{name,
                                       time_s, their_time_s, lost_s, min_kph, exit_kph}]}; {"available": False}
                                       when the two share no recorded segment of that part
        """
        player = mine.parts if mine.me else other.parts
        sector_of = {r.name: r.figures.get("sector") for r in player if r.kind == SEGMENT}
        theirs = {r.name: r for r in other.parts}
        both = [r for r in mine.parts if r.name in theirs]
        rows = [
            {
                "name": r.name,
                "time_s": r.time_s,
                "their_time_s": theirs[r.name].time_s,
                "lost_s": r.time_s - theirs[r.name].time_s,
                "min_kph": r.figures["min_kph"] - theirs[r.name].figures["min_kph"],
                "exit_kph": r.figures["exit_kph"] - theirs[r.name].figures["exit_kph"],
            }
            for r in both
            if r.kind == SEGMENT and _in_part(r.name.lower(), sector_of.get(r.name), part)
        ]
        if not rows:
            return UNAVAILABLE
        whole = mine.record is not None and other.record is not None and not part
        return {
            "car": mine.driver,
            "against": other.driver,
            "laps_compared": which,
            "lap": mine.lap,
            "their_lap": other.lap,
            "part": part or LAP,
            "time_s": mine.record.time_s if whole else None,
            "their_time_s": other.record.time_s if whole else None,
            "lost_s": mine.record.time_s - other.record.time_s if whole else sum(row["lost_s"] for row in rows),
            "sectors": [
                {
                    "name": r.name,
                    "time_s": r.time_s,
                    "their_time_s": theirs[r.name].time_s,
                    "lost_s": r.time_s - theirs[r.name].time_s,
                }
                for r in both
                if r.kind == SECTOR and part in ("", r.name)
            ],
            "segments": rows[::-1] if which == NOW else sorted(rows, key=lambda row: row["lost_s"], reverse=True),
        }

    def _reference_line(self, record: Record, key: str) -> LineSamples | None:
        """The line of the reference lap a segment of the player was compared with: the stored piece for a lap of
        the player, the line of the lap for another car's lap, else the player's own line with the speed and the
        time of the reference.
        In:  record  Record  a segment of the player with its telemetry
             key     str     "class" | "car" | "track"
        Out:         LineSamples | None  None when the segment had no such reference
        """
        vs, samples = record.vs.get(key), record.samples
        if vs is None or samples is None:
            return None
        if key in record.lines:
            return record.lines[key]
        where = "me = 0 AND kind = ? AND driver = ? AND time_s = ?"
        laps = self._store.select(where, (LAP, vs["reference"], vs["reference_lap_s"]))
        line = self._store.telemetry(laps[-1]).lines.get(LAP) if laps else None
        if line is not None:
            return line
        trace = record.traces[key]
        their_mps = samples.speed_mps - trace["speed_kph"] / MPS_TO_KPH
        return LineSamples(samples.distance_m, samples.time_s - trace["time_s"], samples.x_m, samples.z_m, their_mps)

    def _lap_line(self, driver: str | None, lap: str) -> tuple[Record, LineSamples] | None:
        """A full lap of a car with its line.
        In:  driver  str | None  "me" = the player; a driver's name or part of it; None = no such car
             lap     str         "best" | "last" | a lap number
        Out:         tuple[Record, LineSamples] | None  the lap's record and its line; None when there is none
        """
        side = None if driver is None else self._side(driver, lap)
        if side is None or side.record is None:
            return None
        line = self._store.telemetry(side.record).lines.get(LAP)
        return None if line is None else (side.record, line)

    def _class_fastest(self) -> str | None:
        """The driver of the fastest timed lap of another car in the player's class.
        In:  -
        Out: str | None  None when no such lap is recorded
        """
        mine = self._store.select("id = (SELECT MAX(id) FROM times WHERE me = 1)")
        laps = self._fastest("me = 0 AND class_name = ?", (mine[0].class_name,)) if mine else []
        return laps[0].driver if laps else None

    def _fastest(self, cars: str, values: tuple[object, ...]) -> list[Record]:
        """The fastest timed lap of some cars: of the running session, or of the whole run while the session has
        none.
        In:  cars    str                 SQL condition that picks the cars
             values  tuple[object, ...]  its values
        Out:         list[Record]        the lap's record; empty when there is none
        """
        for scope in ((f"{cars} AND session = ?", (*values, self._store.session)), (cars, values)):
            fastest = f"SELECT MIN(time_s) FROM times WHERE {scope[0]} AND name = ?"
            laps = self._store.select(f"{scope[0]} AND name = ? AND time_s = ({fastest})", (*scope[1], LAP) * 2)
            if laps:
                return laps
        return []


def _against_reference(side: _Side, part: str, key: str) -> dict[str, object]:
    """The player's lap, one sector or one segment against a reference lap, from the stored differences.
    In:  side  _Side  the player's lap
         part  str    "" | "sector N" | a segment name, lower case
         key   str    "class" | "car" | "track"
    Out:       dict[str, object]  see lap_data; {"available": False} when the part is not recorded
    """
    segments = [r for r in side.parts if r.kind == SEGMENT and _in_part(r.name.lower(), r.figures["sector"], part)]
    kind = "in progress" if side.record is None else side.record.name
    head = {"car": ME, "lap": side.lap, "lap_type": kind, "in_progress": side.record is None, "against": key}
    if not segments:
        return UNAVAILABLE
    if part and not part.startswith(SECTOR):
        record = segments[0]
        others = {name: _short(vs) for name, vs in record.vs.items() if name != key}
        return {
            **head,
            "name": record.name,
            "time_s": record.time_s,
            **record.figures,
            "vs": record.vs.get(key),
            "other_references": others,
        }
    rows = [{"name": r.name, "time_s": r.time_s, **_brief(r.vs.get(key, {}))} for r in segments]
    sectors = [r for r in side.parts if r.kind == SECTOR and (not part or r.name == part)]
    whole = next(iter(sectors), None) if part else side.record
    known = (segments[0] if whole is None else whole).vs.get(key, {})
    return {
        **head,
        "reference": known.get("reference"),
        "reference_lap_s": known.get("reference_lap_s"),
        "name": part or LAP,
        "time_s": whole and whole.time_s,
        "lost_s": whole.vs.get(key, {}).get("lost_s") if whole else sum(row["lost_s"] or 0.0 for row in rows),
        **({} if part or side.record is None else side.record.figures),
        "sectors": [{"name": r.name, "time_s": r.time_s, "lost_s": r.vs.get(key, {}).get("lost_s")} for r in sectors],
        "segments": rows[::-1] if side.record is None else sorted(rows, key=lambda row: -(row["lost_s"] or 0.0)),
    }


def _in_part(name: str, sector: object, part: str) -> bool:
    """Whether a segment belongs to the part asked for.
    In:  name    str     name of the segment, lower case
         sector  object  1, 2, 3, or None when not known
         part    str     "" | "sector N" | a segment name, lower case
    Out:         bool
    """
    return not part or _named(name, part) or part == f"{SECTOR} {sector}"


def _brief(vs: dict[str, object]) -> dict[str, object]:
    """What a list of segments says of one segment's difference to a reference.
    In:  vs  dict[str, object]  the stored difference; empty when there was no reference
    Out:     dict[str, object]  {lost_s, findings} and for a corner {lost_to_apex_s, lost_from_apex_s}
    """
    splits = {key: vs[key] for key in ("lost_to_apex_s", "lost_from_apex_s") if vs.get(key) is not None}
    return {"lost_s": vs.get("lost_s"), **splits, "findings": vs.get("findings", [])}


def _short(vs: dict[str, object]) -> dict[str, object]:
    """The time lost to a reference and which lap it was.
    In:  vs  dict[str, object]  the stored difference
    Out:     dict[str, object]  {reference, reference_lap_s, lost_s}
    """
    return {key: vs[key] for key in ("reference", "reference_lap_s", "lost_s")}


def _index(segments: tuple[Segment, ...], name: str) -> int | None:
    """Position of a segment in the track map, whatever the case of its name.
    In:  segments  tuple[Segment, ...]
         name      str
    Out:           int | None  None when the track has no such segment
    """
    return next((i for i, segment in enumerate(segments) if _named(segment.name.lower(), name.strip().lower())), None)


def _named(name: str, wanted: str) -> bool:
    """Whether a segment is the one asked for: by its name, or by one of the turns it is made of, so that "turn 10"
    finds "turn 9 and 10".
    In:  name    str  name of the segment, lower case
         wanted  str  what was asked for, lower case
    Out:         bool
    """
    turn = re.fullmatch(r"turn (\d+)", wanted)
    return name == wanted or (turn is not None and name.startswith("turn") and turn[1] in re.findall(r"\d+", name))


def _wheel(wheel: WheelState, index: int) -> dict[str, object]:
    """One wheel as tool output, with the tread temperatures named from the car's centre line outwards.
    In:  wheel  WheelState
         index  int         0 FL, 1 FR, 2 RL, 3 RR
    Out:        dict[str, object]
    """
    left, middle, right = wheel.surface_temp_c
    inner, outer = (right, left) if index in LEFT_WHEELS else (left, right)
    return {
        "tread_inner_c": inner,
        "tread_middle_c": middle,
        "tread_outer_c": outer,
        "carcass_c": wheel.carcass_temp_c,
        "pressure_kpa": wheel.pressure_kpa,
        "brake_c": wheel.brake_temp_c,
    }


def _rounded(value: object, digits: int = 2) -> object:
    """Tool output with every number rounded, so the model reads short figures.
    In:  value   object  mappings, lists, numbers, text or None
         digits  int     decimals kept
    Out:         object  the same structure
    """
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(item, digits) for item in value]
    return value
