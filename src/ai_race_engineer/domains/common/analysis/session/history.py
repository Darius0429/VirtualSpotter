import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitLane, PitRates, calibrate, measured_pit_lane
from ai_race_engineer.domains.common.analysis.session.session_log import SessionRecord
from ai_race_engineer.domains.common.analysis.session.session_summary import SessionSummary
from ai_race_engineer.domains.common.data.results import TRACK_FIELDS, ResultDriver, SessionResults

type Sectors = tuple[float | None, float | None, float | None]

RESULTS_FINISHED: str = "Finished Normally"


@dataclass(frozen=True)
class ClassFigures:
    """What one session showed about a class.
    best_lap_s       float | None  s, pole or fastest lap of the class
    leader_pace_s    float | None  s/lap, clean pace of the class winner or leader
    energy_per_lap   float | None  fraction/lap, mean over the class's cars (engineer entries only)
    fuel_per_energy  float | None  litres per 1 % energy, mean over the class's cars (engineer entries only)
    """

    best_lap_s: float | None
    leader_pace_s: float | None
    energy_per_lap: float | None
    fuel_per_energy: float | None


@dataclass(frozen=True)
class SessionFigures:
    """The figures of one session of the player, one line of the history.
    session_id         str           results file stem, or the engineer's session id
    source             str           "results_file" | "engineer"
    started_unix       int           s unix time
    track              str
    kind               str           "practice" | "qualify" | "race" | ...
    car                str           car model (the key of the history)
    veh_name           str           vehicle name with team and number
    car_class          str
    best_lap_s         float | None  s
    clean_pace_s       float | None  s/lap, median of the clean laps
    clean_laps         int
    best_sectors_s     Sectors       s (S1, S2, S3)
    energy_per_lap     float | None  fraction/lap
    fuel_per_energy    float | None  litres per 1 % energy
    stint_laps         int | None    whole laps a full load lasts
    pit_lane_time_s    float | None  s
    pit_lane_length_m  float | None  m
    energy_rate_per_s  float | None  fraction/s
    fuel_rate_l_per_s  float | None  litres/s
    straights          tuple[tuple[float, float], ...]  (start m, end m)
    class_place        int | None
    grid_class_place   int | None
    laps               int
    pitstops           int
    finish_status      str | None
    classes            dict[str, ClassFigures]  class name -> figures
    complete           bool          the session was driven to its end: for a race the finish was seen (engineer) or
                                     the results file says the player finished; False for a race left, abandoned or
                                     not finished
    """

    session_id: str
    source: str
    started_unix: int
    track: str
    kind: str
    car: str
    veh_name: str
    car_class: str
    best_lap_s: float | None
    clean_pace_s: float | None
    clean_laps: int
    best_sectors_s: Sectors
    energy_per_lap: float | None
    fuel_per_energy: float | None
    stint_laps: int | None
    pit_lane_time_s: float | None
    pit_lane_length_m: float | None
    energy_rate_per_s: float | None
    fuel_rate_l_per_s: float | None
    straights: tuple[tuple[float, float], ...]
    class_place: int | None
    grid_class_place: int | None
    laps: int
    pitstops: int
    finish_status: str | None
    classes: dict[str, ClassFigures]
    complete: bool = True


@dataclass(frozen=True)
class CarHistory:
    """What the history knows about one car on one track.
    best_qualifying_lap_s  float | None  s, fastest over all qualifying sessions
    best_lap_s             float | None  s, fastest over all sessions
    race_pace_s            float | None  s/lap, median of the last sessions (races first)
    best_sectors_s         Sectors       s
    energy_per_lap         float | None  fraction/lap
    fuel_per_energy        float | None  litres per 1 % energy
    stint_laps             int | None
    pit_lane               PitLane | None
    energy_rate_per_s      float | None  fraction/s
    fuel_rate_l_per_s      float | None  litres/s
    straights              tuple[tuple[float, float], ...]  (start m, end m), of the newest session that has them
    sessions               int           sessions in the history
    last_update_unix       int           s unix time of the newest session
    last_results           tuple[tuple[str, int, int | None, int, str | None], ...]  (kind, started unix, class place,
                                         laps, finish status; "incomplete" for a session that was not driven to
                                         its end), newest first, at most 3
    """

    best_qualifying_lap_s: float | None
    best_lap_s: float | None
    race_pace_s: float | None
    best_sectors_s: Sectors
    energy_per_lap: float | None
    fuel_per_energy: float | None
    stint_laps: int | None
    pit_lane: PitLane | None
    energy_rate_per_s: float | None
    fuel_rate_l_per_s: float | None
    straights: tuple[tuple[float, float], ...]
    sessions: int
    last_update_unix: int
    last_results: tuple[tuple[str, int, int | None, int, str | None], ...]


@dataclass(frozen=True)
class ClassHistory:
    """What the history knows about one class on one track.
    pole_lap_s       float | None  s, fastest lap of the class over all qualifying sessions
    race_pace_s      float | None  s/lap of the class leader, median of the last sessions (races first)
    energy_per_lap   float | None  fraction/lap (from sessions the engineer watched)
    fuel_per_energy  float | None  litres per 1 % energy (from sessions the engineer watched)
    """

    pole_lap_s: float | None
    race_pace_s: float | None
    energy_per_lap: float | None
    fuel_per_energy: float | None


def figures_from_results(
    results: SessionResults, session_id: str, track_field: str, player_name: str
) -> SessionFigures | None:
    """History line of a results file. The clean pace is the median of the timed laps without lap 1, in-laps and
    the laps after them (the file cannot tell traffic or yellow laps apart). A race the file does not say the player
    finished is incomplete.
    In:  results      SessionResults
         session_id   str             the file's stem
         track_field  str             results field equal to the game's track name
         player_name  str             the player's driver name (isPlayer marks every human of an online session)
    Out:              SessionFigures | None  None when the driver is not in the file or drove no lap
    """
    player = next((d for d in results.drivers if d.name == player_name), None)
    if player is None or not player.lap_list:
        return None
    clean = clean_lap_times_s(player)
    return SessionFigures(
        session_id=session_id,
        source="results_file",
        started_unix=results.started_unix,
        track=getattr(results, TRACK_FIELDS[track_field]),
        kind=results.kind,
        car=player.car_type,
        veh_name=player.veh_name,
        car_class=player.car_class,
        best_lap_s=player.best_lap_s,
        clean_pace_s=statistics.median(clean) if clean else None,
        clean_laps=len(clean),
        best_sectors_s=(
            _lowest([lap.sector1_s for lap in player.lap_list]),
            _lowest([lap.sector2_s for lap in player.lap_list]),
            _lowest([lap.sector3_s for lap in player.lap_list]),
        ),
        energy_per_lap=None,
        fuel_per_energy=None,
        stint_laps=None,
        pit_lane_time_s=None,
        pit_lane_length_m=None,
        energy_rate_per_s=None,
        fuel_rate_l_per_s=None,
        straights=(),
        class_place=player.class_place,
        grid_class_place=player.class_grid_place,
        laps=player.laps,
        pitstops=player.pitstops,
        finish_status=player.finish_status,
        classes={name: _class_of_results(results, name) for name in {d.car_class for d in results.drivers}},
        complete=_results_complete(results.kind, player.finish_status),
    )


def figures_from_record(
    record: SessionRecord, summary: SessionSummary, car: str, default_rates: PitRates
) -> SessionFigures:
    """History line of a session the engineer watched: adds what results files lack (energy, pit lane, fill rates,
    straights, class usage). Fill rates are stored only when the player made a stop with a game estimate.
    In:  record         SessionRecord
         summary        SessionSummary  of the same session
         car            str             car model
         default_rates  PitRates        the configured rates the calibration starts from
    Out:                SessionFigures
    """
    player = summary.player
    lane = measured_pit_lane(record.pit_samples)
    rates = calibrate(record.pit_samples, default_rates)
    calibrated = any(s.is_player and s.estimate is not None for s in record.pit_samples)
    return SessionFigures(
        session_id=record.session_id,
        source="engineer",
        started_unix=int(record.started_unix),
        track=record.track,
        kind=record.kind,
        car=car,
        veh_name=record.vehicle_name,
        car_class=record.class_name,
        best_lap_s=player.best_lap_s,
        clean_pace_s=player.clean_pace_s,
        clean_laps=player.clean_laps,
        best_sectors_s=player.best_sectors_s,
        energy_per_lap=player.energy_per_lap,
        fuel_per_energy=player.fuel_per_energy,
        stint_laps=player.stint_laps,
        pit_lane_time_s=None if lane is None else lane.time_s,
        pit_lane_length_m=None if lane is None else lane.length_m,
        energy_rate_per_s=rates.energy_rate_per_s if calibrated else None,
        fuel_rate_l_per_s=rates.fuel_rate_l_per_s if calibrated else None,
        straights=record.straights,
        class_place=player.class_place,
        grid_class_place=player.grid_class_place,
        laps=player.laps,
        pitstops=len(player.pit_stops),
        finish_status=None,
        classes={name: _class_of_record(record, name) for name in {v.class_name for v in record.vehicles}},
        complete=record.kind != "race" or record.finished,
    )


def figures_from_dict(line: Mapping[str, Any]) -> SessionFigures:
    """History line from its stored JSON object. A line stored before the complete flag existed is complete, except
    a race of a results file the player did not finish.
    In:  line  Mapping[str, Any]  the fields of SessionFigures as written by dataclasses.asdict
    Out:       SessionFigures
    """
    values = dict(line)
    if "complete" not in values and values["source"] == "results_file":
        values["complete"] = _results_complete(values["kind"], values["finish_status"])
    values["best_sectors_s"] = tuple(values["best_sectors_s"])
    values["straights"] = tuple(tuple(s) for s in values["straights"])
    values["classes"] = {name: ClassFigures(**figures) for name, figures in values["classes"].items()}
    return SessionFigures(**values)


def car_history(entries: Sequence[SessionFigures], track: str, car: str, last_n: int) -> CarHistory | None:
    """What the history knows about a car on a track. Best laps and sectors are the minimum over all sessions; the
    rest is the median of the last sessions that have the value, race sessions first. Pace values come from
    results-file entries when there are any, else from the engineer's own, so a session is not counted twice.
    In:  entries  Sequence[SessionFigures]  the whole history
         track    str
         car      str                       car model
         last_n   int                       sessions a median is taken over
    Out:          CarHistory | None         None when the car has no session on the track
    """
    own = sorted((e for e in entries if e.track == track and e.car == car), key=lambda e: e.started_unix)
    if not own:
        return None
    from_files = [e for e in own if e.source == "results_file"]
    paced = from_files or own
    lane_time_s = _typical(own, lambda e: e.pit_lane_time_s, last_n)
    lane_length_m = _typical(own, lambda e: e.pit_lane_length_m, last_n)
    stint_laps = _typical(own, lambda e: e.stint_laps, last_n)
    return CarHistory(
        best_qualifying_lap_s=_lowest([e.best_lap_s for e in own if e.kind == "qualify"]),
        best_lap_s=_lowest([e.best_lap_s for e in own]),
        race_pace_s=_typical(paced, lambda e: e.clean_pace_s, last_n),
        best_sectors_s=(
            _lowest([e.best_sectors_s[0] for e in own]),
            _lowest([e.best_sectors_s[1] for e in own]),
            _lowest([e.best_sectors_s[2] for e in own]),
        ),
        energy_per_lap=_typical(own, lambda e: e.energy_per_lap, last_n),
        fuel_per_energy=_typical(own, lambda e: e.fuel_per_energy, last_n),
        stint_laps=None if stint_laps is None else int(stint_laps),
        pit_lane=None if lane_time_s is None or lane_length_m is None else PitLane(lane_time_s, lane_length_m),
        energy_rate_per_s=_typical(own, lambda e: e.energy_rate_per_s, last_n),
        fuel_rate_l_per_s=_typical(own, lambda e: e.fuel_rate_l_per_s, last_n),
        straights=next((e.straights for e in reversed(own) if e.straights), ()),
        sessions=len(paced),
        last_update_unix=own[-1].started_unix,
        last_results=tuple(
            (e.kind, e.started_unix, e.class_place, e.laps, e.finish_status if e.complete else "incomplete")
            for e in reversed(paced[-3:])
        ),
    )


def class_history(entries: Sequence[SessionFigures], track: str, class_name: str, last_n: int) -> ClassHistory | None:
    """What the history knows about a class on a track, from every session of the player there.
    In:  entries     Sequence[SessionFigures]  the whole history
         track       str
         class_name  str
         last_n      int                       sessions a median is taken over
    Out:             ClassHistory | None       None when no session on the track had the class
    """
    seen = sorted((e for e in entries if e.track == track and class_name in e.classes), key=lambda e: e.started_unix)
    if not seen:
        return None
    from_files = [e for e in seen if e.source == "results_file"]
    return ClassHistory(
        pole_lap_s=_lowest([e.classes[class_name].best_lap_s for e in seen if e.kind == "qualify"]),
        race_pace_s=_typical(from_files or seen, lambda e: e.classes[class_name].leader_pace_s, last_n),
        energy_per_lap=_typical(seen, lambda e: e.classes[class_name].energy_per_lap, last_n),
        fuel_per_energy=_typical(seen, lambda e: e.classes[class_name].fuel_per_energy, last_n),
    )


def car_key(
    vehicle_name: str, player_name: str, results: SessionResults | None, entries: Sequence[SessionFigures]
) -> str:
    """Key of the player's car in the history: the car model, which a vehicle name (team, number, season) does not
    give directly.
    In:  vehicle_name  str                       the player's vehicle name in shared memory
         player_name   str                       the player's driver name
         results       SessionResults | None     a results file of the event
         entries       Sequence[SessionFigures]  the whole history
    Out:               str  the car model of the player in the results file; else the model the history has stored
                            with this vehicle name; else the vehicle name itself
    """
    driver = None if results is None else next((d for d in results.drivers if d.name == player_name), None)
    if driver is not None:
        return driver.car_type
    return next((e.car for e in reversed(entries) if e.veh_name == vehicle_name), vehicle_name)


def clean_lap_times_s(driver: ResultDriver) -> list[float]:
    """Lap times of a driver's clean laps in a results file.
    In:  driver  ResultDriver
    Out:         list[float]  s, the timed laps without lap 1, in-laps and the laps after them
    """
    after_pit = {lap.num + 1 for lap in driver.lap_list if lap.is_pit}
    return [
        lap.lap_time_s
        for lap in driver.lap_list
        if lap.lap_time_s is not None and lap.num > 1 and not lap.is_pit and lap.num not in after_pit
    ]


def _results_complete(kind: str, finish_status: str | None) -> bool:
    """Whether a session of a results file was driven to its end.
    In:  kind           str         "practice" | "qualify" | "race" | ...
         finish_status  str | None  the player's FinishStatus in the file
    Out:                bool        False for a race the player did not finish normally (DNF, DQ, none); True for
                                    every other kind
    """
    return kind != "race" or finish_status == RESULTS_FINISHED


def _class_of_results(results: SessionResults, class_name: str) -> ClassFigures:
    """Figures of one class from a results file.
    In:  results     SessionResults
         class_name  str
    Out:             ClassFigures  without usage figures
    """
    drivers = [d for d in results.drivers if d.car_class == class_name]
    leader = min(drivers, key=lambda d: d.class_place)
    clean = clean_lap_times_s(leader)
    return ClassFigures(
        best_lap_s=_lowest([d.best_lap_s for d in drivers]),
        leader_pace_s=statistics.median(clean) if clean else None,
        energy_per_lap=None,
        fuel_per_energy=None,
    )


def _class_of_record(record: SessionRecord, class_name: str) -> ClassFigures:
    """Figures of one class from a session the engineer watched.
    In:  record      SessionRecord
         class_name  str
    Out:             ClassFigures
    """
    cars = [v for v in record.vehicles if v.class_name == class_name]
    measured = [record.estimates[v.slot_id] for v in cars if v.slot_id in record.estimates]
    leader = record.estimates.get(min(cars, key=lambda v: v.place).slot_id)
    return ClassFigures(
        best_lap_s=_lowest([v.best_lap_s for v in cars]),
        leader_pace_s=statistics.fmean(leader.clean_laps_s) if leader is not None and leader.clean_laps_s else None,
        energy_per_lap=_mean([e.energy_per_lap for e in measured]),
        fuel_per_energy=_mean([e.fuel_per_energy for e in measured]),
    )


def _typical(
    entries: Sequence[SessionFigures], value: Callable[[SessionFigures], float | None], last_n: int
) -> float | None:
    """Median of a value over the last sessions that have it; complete race sessions are used when any has it,
    the other kinds and the incomplete races only otherwise.
    In:  entries  Sequence[SessionFigures]                 oldest first
         value    Callable[[SessionFigures], float | None]  the value of one session
         last_n   int                                      sessions the median is taken over
    Out:          float | None                             None when no session has the value
    """
    races = [value(e) for e in entries if e.kind == "race" and e.complete and value(e) is not None]
    values = races or [value(e) for e in entries if value(e) is not None]
    return statistics.median(values[-last_n:]) if values else None


def _lowest(values: Sequence[float | None]) -> float | None:
    """Lowest of the values that exist.
    In:  values  Sequence[float | None]
    Out:         float | None  None without any
    """
    known = [v for v in values if v is not None]
    return min(known) if known else None


def _mean(values: Sequence[float | None]) -> float | None:
    """Mean of the values that exist.
    In:  values  Sequence[float | None]
    Out:         float | None  None without any
    """
    known = [v for v in values if v is not None]
    return statistics.fmean(known) if known else None
