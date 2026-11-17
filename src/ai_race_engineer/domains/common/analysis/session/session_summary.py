import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace

from ai_race_engineer.domains.common.analysis.corners.lap_types import IN_LAP, INVALID_LAP, LAP
from ai_race_engineer.domains.common.analysis.session.session_log import Note, PlayerLap, SessionRecord
from ai_race_engineer.domains.common.analysis.session.track_limits import TrackLimitStatus
from ai_race_engineer.domains.common.data.results import TRACK_FIELDS, ResultDriver, SessionResults

type Sectors = tuple[float | None, float | None, float | None]


@dataclass(frozen=True)
class ClassEntry:
    """One car of the player's class in the order of a session.
    class_place  int
    driver       str
    best_lap_s   float | None  s
    laps         int
    pitstops     int
    """

    class_place: int
    driver: str
    best_lap_s: float | None
    laps: int
    pitstops: int


@dataclass(frozen=True)
class PitStopNote:
    """One pit stop of the player.
    lap           int           lap at the end of which the car pitted
    stationary_s  float | None  s standing still
    energy_added  float | None  fraction of virtual energy
    fuel_added_l  float | None  litres
    """

    lap: int
    stationary_s: float | None
    energy_added: float | None
    fuel_added_l: float | None


@dataclass(frozen=True)
class PlayerSummary:
    """The player's session in numbers.
    class_place           int
    grid_class_place      int | None    place in class at the start
    positions_gained      int | None    grid place - place
    laps                  int           completed laps
    best_lap_s            float | None  s
    theoretical_best_s    float | None  s, sum of the best sectors
    clean_pace_s          float | None  s/lap, median of the clean laps
    clean_laps            int
    invalid_laps          int | None    laps the game flagged for track limits and did not time; None from a
                                        results file
    best_sectors_s        Sectors       s (S1, S2, S3)
    energy_per_lap        float | None  fraction/lap, mean over the clean laps
    fuel_per_energy       float | None  litres per 1 % energy over the clean laps
    stint_laps            int | None    whole laps a full load of energy lasts
    pit_stops             tuple[PitStopNote, ...]
    incidents             int | None    contacts the game reported; None until the results file exists
    penalties             tuple[str, ...]  the game's penalty messages
    track_limit_warnings  int | None    None until the results file exists
    lap_list              tuple[PlayerLap, ...]
    limits                TrackLimitStatus | None  track-limit points, the penalty limit, penalties outstanding and
                                        each offence; None from a results file or when the source has none
    """

    class_place: int
    grid_class_place: int | None
    positions_gained: int | None
    laps: int
    best_lap_s: float | None
    theoretical_best_s: float | None
    clean_pace_s: float | None
    clean_laps: int
    invalid_laps: int | None
    best_sectors_s: Sectors
    energy_per_lap: float | None
    fuel_per_energy: float | None
    stint_laps: int | None
    pit_stops: tuple[PitStopNote, ...]
    incidents: int | None
    penalties: tuple[str, ...]
    track_limit_warnings: int | None
    lap_list: tuple[PlayerLap, ...]
    limits: TrackLimitStatus | None = None


@dataclass(frozen=True)
class SessionSummary:
    """Summary of one session, stored as one file.
    session_id    str
    track         str
    kind          str    "practice" | "qualify" | "race" | ...
    car           str    car model
    car_class     str
    started_unix  float  s unix time
    time_s        float  s session time the summary covers
    complete      bool   the session is over
    class_order   tuple[ClassEntry, ...]  the top of the class plus the player and its neighbours
    yellow_flags  int | None              times a sector flag turned yellow; None from a results file or when they
                                          are not counted
    player        PlayerSummary
    notes         tuple[Note, ...]
    """

    session_id: str
    track: str
    kind: str
    car: str
    car_class: str
    started_unix: float
    time_s: float
    complete: bool
    class_order: tuple[ClassEntry, ...]
    yellow_flags: int | None
    player: PlayerSummary
    notes: tuple[Note, ...]


def summarise(record: SessionRecord, car: str, class_top: int) -> SessionSummary:
    """Summary of a session from the engineer's own record (it has energy and fuel).
    In:  record     SessionRecord
         car        str            car model
         class_top  int            cars of the class listed from the top (the player and its neighbours are added)
    Out:            SessionSummary
    """
    in_class = sorted((v for v in record.vehicles if v.class_name == record.class_name), key=lambda v: v.place)
    player = next(v for v in in_class if v.driver_name == record.player_name)
    place = in_class.index(player) + 1
    entries = [
        ClassEntry(i, v.driver_name, v.best_lap_s, v.completed_laps, v.pitstops) for i, v in enumerate(in_class, 1)
    ]
    clean = [lap for lap in record.laps if lap.green and not lap.pitted]
    with_energy = [lap for lap in clean if lap.energy_used is not None]
    energy = statistics.fmean(lap.energy_used for lap in with_energy) if with_energy else None
    with_fuel = [lap for lap in with_energy if lap.fuel_used_l is not None]
    sectors = _best_sectors(record.laps)
    stops = [s for s in record.pit_samples if s.is_player]
    stop_notes = [n for n in record.notes if n.auto and n.text.startswith("Pit stop")]
    return SessionSummary(
        session_id=record.session_id,
        track=record.track,
        kind=record.kind,
        car=car,
        car_class=record.class_name,
        started_unix=record.started_unix,
        time_s=record.time_s,
        complete=record.ended,
        class_order=_shown(entries, place, class_top),
        yellow_flags=record.yellow_flags,
        player=PlayerSummary(
            class_place=place,
            grid_class_place=record.grid_class_place,
            positions_gained=record.grid_class_place - place,
            laps=player.completed_laps,
            best_lap_s=player.best_lap_s,
            theoretical_best_s=None if None in sectors else sum(sectors),
            clean_pace_s=statistics.median(lap.lap_time_s for lap in clean) if clean else None,
            clean_laps=len(clean),
            invalid_laps=sum(lap.lap_type == INVALID_LAP for lap in record.laps),
            best_sectors_s=sectors,
            energy_per_lap=energy,
            fuel_per_energy=statistics.fmean(lap.fuel_used_l for lap in with_fuel)
            / (statistics.fmean(lap.energy_used for lap in with_fuel) * 100)
            if with_fuel and energy
            else None,
            stint_laps=max(1, math.floor(1 / energy)) if energy else None,
            pit_stops=tuple(
                PitStopNote(note.lap, stop.stationary_s, stop.energy_added, stop.fuel_added_l)
                for note, stop in zip(stop_notes, stops[len(stops) - len(stop_notes) :], strict=False)
            ),
            incidents=None,
            penalties=(),
            track_limit_warnings=None,
            lap_list=record.laps,
            limits=record.limits,
        ),
        notes=record.notes,
    )


def summary_from_results(
    results: SessionResults, session_id: str, track_field: str, class_top: int, player_name: str
) -> SessionSummary | None:
    """Summary of a session from its results file alone (no energy, no fuel).
    In:  results      SessionResults
         session_id   str             the file's stem
         track_field  str             results field equal to the game's track name
         class_top    int             cars of the class listed from the top
         player_name  str             the player's driver name
    Out:              SessionSummary | None  None when the driver is not in the file
    """
    player = next((d for d in results.drivers if d.name == player_name), None)
    if player is None:
        return None
    in_class = sorted((d for d in results.drivers if d.car_class == player.car_class), key=lambda d: d.class_place)
    entries = [ClassEntry(d.class_place, d.name, d.best_lap_s, d.laps, d.pitstops) for d in in_class]
    laps = _laps_of_results(player)
    clean = [lap for lap in laps if lap.green and not lap.pitted]
    sectors = _best_sectors(laps)
    summary = SessionSummary(
        session_id=session_id,
        track=getattr(results, TRACK_FIELDS[track_field]),
        kind=results.kind,
        car=player.car_type,
        car_class=player.car_class,
        started_unix=results.started_unix,
        time_s=max((lap.start_s + lap.lap_time_s for lap in laps), default=0.0),
        complete=True,
        class_order=_shown(entries, player.class_place, class_top),
        yellow_flags=None,
        player=PlayerSummary(
            class_place=player.class_place,
            grid_class_place=player.class_grid_place,
            positions_gained=None if player.class_grid_place is None else player.class_grid_place - player.class_place,
            laps=player.laps,
            best_lap_s=player.best_lap_s,
            theoretical_best_s=None if None in sectors else sum(sectors),
            clean_pace_s=statistics.median(lap.lap_time_s for lap in clean) if clean else None,
            clean_laps=len(clean),
            invalid_laps=None,
            best_sectors_s=sectors,
            energy_per_lap=None,
            fuel_per_energy=None,
            stint_laps=None,
            pit_stops=tuple(PitStopNote(lap.num, None, None, None) for lap in player.lap_list if lap.is_pit),
            incidents=None,
            penalties=(),
            track_limit_warnings=None,
            lap_list=laps,
        ),
        notes=(),
    )
    return with_results(summary, results, player_name)


def with_results(summary: SessionSummary, results: SessionResults, player_name: str) -> SessionSummary:
    """Add to a summary what only the results file has: the player's contacts, penalties and track-limit warnings,
    the grid place, and an automatic note for each penalty and contact. A contact is the player's when its message
    names the player as a whole name: at the start or after "vehicle ", followed by the car number in brackets.
    In:  summary      SessionSummary
         results      SessionResults  the results file of the same session
         player_name  str             the player's driver name
    Out:              SessionSummary
    """
    driver = next((d for d in results.drivers if d.name == player_name), None)
    named = re.compile(rf"(?:^|vehicle ){re.escape(player_name)}\(")
    contacts = [e for e in results.events if e.kind == "incident" and named.search(e.text)]
    penalties = [e for e in results.events if e.kind == "penalty" and e.driver == player_name]
    limits = [e for e in results.events if e.kind == "track_limits" and e.driver == player_name]
    known = driver is not None and driver.class_grid_place is not None
    grid = driver.class_grid_place if known else summary.player.grid_class_place
    notes = tuple(
        Note(e.time_s, 0, f"{'Penalty' if e.kind == 'penalty' else 'Contact'}: {e.text}", True)
        for e in sorted(contacts + penalties, key=lambda e: e.time_s)
    )
    return replace(
        summary,
        player=replace(
            summary.player,
            grid_class_place=grid,
            positions_gained=None if grid is None else grid - summary.player.class_place,
            incidents=len(contacts),
            penalties=tuple(e.text for e in penalties),
            track_limit_warnings=len(limits),
        ),
        notes=tuple(sorted(summary.notes + notes, key=lambda n: n.time_s)),
    )


def results_of_session(results: SessionResults, kind: str, session_start_unix: float, margin_s: float) -> bool:
    """Whether a results file belongs to a session the engineer watched.
    In:  results             SessionResults
         kind                str    kind of the watched session
         session_start_unix  float  s unix time the engineer first saw the session
         margin_s            float  s the two start times may differ
    Out:                     bool
    """
    return results.kind == kind and abs(results.started_unix - session_start_unix) <= margin_s


def _shown(entries: Sequence[ClassEntry], place: int, class_top: int) -> tuple[ClassEntry, ...]:
    """The part of the class order worth storing.
    In:  entries    Sequence[ClassEntry]  the whole class, in order
         place      int                   the player's place in class
         class_top  int                   cars listed from the top
    Out:            tuple[ClassEntry, ...]  the top, the player and its two neighbours
    """
    return tuple(e for e in entries if e.class_place <= class_top or abs(e.class_place - place) <= 1)


def _best_sectors(laps: Sequence[PlayerLap]) -> Sectors:
    """Best time of each sector over a set of laps.
    In:  laps  Sequence[PlayerLap]
    Out:       Sectors  s; None for a sector without a time
    """
    columns = ([lap.sector1_s for lap in laps], [lap.sector2_s for lap in laps], [lap.sector3_s for lap in laps])
    best = [min((s for s in column if s is not None), default=None) for column in columns]
    return best[0], best[1], best[2]


def _laps_of_results(driver: ResultDriver) -> tuple[PlayerLap, ...]:
    """The timed laps of a driver in a results file in the form of the engineer's own lap list.
    In:  driver  ResultDriver
    Out:         tuple[PlayerLap, ...]  laps with a time; an in-lap and the lap after it count as pitted, lap 1 as not
                                        green; usage is unknown; lap_type "in lap" for a lap into the pits, else "lap"
    """
    after_pit = {lap.num + 1 for lap in driver.lap_list if lap.is_pit}
    return tuple(
        PlayerLap(
            lap=lap.num,
            start_s=lap.start_s or 0.0,
            lap_time_s=lap.lap_time_s,
            timed=True,
            sector1_s=lap.sector1_s,
            sector2_s=lap.sector2_s,
            sector3_s=lap.sector3_s,
            energy_used=None,
            fuel_used_l=None,
            class_place=driver.class_place,
            compound_front=lap.compound_front,
            compound_rear=lap.compound_rear,
            pitted=lap.is_pit or lap.num in after_pit,
            green=lap.num > 1,
            lap_type=IN_LAP if lap.is_pit else LAP,
        )
        for lap in driver.lap_list
        if lap.lap_time_s is not None
    )
