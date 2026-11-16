import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from ai_race_engineer.strategy.field import CarEstimate
from ai_race_engineer.strategy.history import (
    CarHistory,
    ClassHistory,
    SessionFigures,
    car_history,
    car_key,
    class_history,
)
from ai_race_engineer.telemetry.results import TRACK_FIELDS, SessionResults
from ai_race_engineer.telemetry.snapshot import VehicleState

USAGE_SOURCES: tuple[str, ...] = ("measured", "remembered", "history_car", "game")


@dataclass(frozen=True)
class EventPriors:
    """What is known about pace and usage before a session has measured them.
    track              str
    car                str                        the player's car model (the key of the history)
    qualifying_best_s  Mapping[str, float]        driver name -> s, best qualifying lap of this event
    remembered         Mapping[tuple[str, str], CarEstimate]  (driver name, class) -> pace and usage measured in
                                                  earlier sessions of this process on this track (lap_time_s = median
                                                  of the clean laps of a practice or race session)
    car_history        CarHistory | None          the player's car on this track, from the history
    class_history      Mapping[str, ClassHistory]  class name -> figures on this track, from the history
    """

    track: str
    car: str
    qualifying_best_s: Mapping[str, float]
    remembered: Mapping[tuple[str, str], CarEstimate]
    car_history: CarHistory | None = None
    class_history: Mapping[str, ClassHistory] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorSource:
    """Where the pace and the usage of one car come from.
    pace    str  "measured" | "qualifying" | "remembered" | "best_lap" | "class_best" | "history_car" |
                 "history_class" | "none"
    energy  str  "measured" | "remembered" | "history_car" | "game" | "class" | "history_class" | "none"
    """

    pace: str
    energy: str


def event_priors(
    track: str,
    vehicle_name: str,
    player_name: str,
    results: SessionResults | None,
    remembered: Mapping[tuple[str, str], CarEstimate],
    entries: Sequence[SessionFigures],
    now_unix: float,
    track_field: str,
    last_n: int,
    max_qualifying_age_h: float,
    before_unix: float,
) -> EventPriors:
    """Priors of one session. Qualifying times are taken from the results only when they are a qualifying session of
    this track, contain the player's name and are not older than max_qualifying_age_h. Of the history only the
    sessions that started before before_unix are used, so a session never sees its own result or a later one (replay
    of a recording).
    In:  track                 str                        the game's track name
         vehicle_name          str                        the player's vehicle name
         player_name           str                        the player's driver name
         results               SessionResults | None      the results file the game points to, or the newest qualifying
         remembered            Mapping[tuple[str, str], CarEstimate]  (driver name, class) -> values of earlier
                                                          sessions on this track
         entries               Sequence[SessionFigures]   the whole history
         now_unix              float                      s unix time
         track_field           str                        results field equal to the game's track name: "TrackVenue" |
                                                          "TrackCourse" | "TrackEvent"
         last_n                int                        sessions the history figures are the median of
         max_qualifying_age_h  float                      hours
         before_unix           float                      s unix time; history sessions that started at or after it
                                                          are left out (the start of the running session, less the
                                                          margin by which its own results file may start earlier)
    Out:                       EventPriors
    """
    player = (
        None
        if results is None
        or results.kind != "qualify"
        or getattr(results, TRACK_FIELDS[track_field]) != track
        or now_unix - results.started_unix > max_qualifying_age_h * 3600
        else next((d for d in results.drivers if d.name == player_name), None)
    )
    earlier = [e for e in entries if e.started_unix < before_unix]
    car = car_key(vehicle_name, player_name, results, earlier)
    classes = {name for e in earlier if e.track == track for name in e.classes}
    return EventPriors(
        track=track,
        car=car,
        qualifying_best_s={}
        if player is None
        else {d.name: d.best_lap_s for d in results.drivers if d.best_lap_s is not None},
        remembered=remembered,
        car_history=car_history(earlier, track, car, last_n),
        class_history={name: class_history(earlier, track, name, last_n) for name in classes},
    )


def remember(
    remembered: Mapping[tuple[str, str], CarEstimate],
    vehicles: Sequence[VehicleState],
    estimates: Mapping[int, CarEstimate],
    kind: str,
) -> dict[tuple[str, str], CarEstimate]:
    """Add what a finished session measured to the remembered values; a value the session did not measure keeps the
    one remembered before. The pace of a qualifying session is not remembered: its clean laps include cool-down laps.
    In:  remembered  Mapping[tuple[str, str], CarEstimate]  (driver name, class) -> values remembered so far
         vehicles    Sequence[VehicleState]                 cars of the finished session
         estimates   Mapping[int, CarEstimate]              slot id -> what that session measured (no priors)
         kind        str                                    session kind, e.g. "practice" | "qualify" | "race"
    Out:             dict[tuple[str, str], CarEstimate]     (driver name, class) -> values; lap_time_s is the median
                                                            of the clean laps
    """
    result = dict(remembered)
    for vehicle in vehicles:
        key = (vehicle.driver_name, vehicle.class_name)
        new = estimates.get(vehicle.slot_id)
        old = result.get(key)
        if new is None:
            continue
        pace_s = statistics.median(new.clean_laps_s) if new.clean_laps_s and kind != "qualify" else None
        result[key] = CarEstimate(
            slot_id=vehicle.slot_id,
            lap_time_s=pace_s if pace_s is not None or old is None else old.lap_time_s,
            fuel_l_per_lap=new.fuel_l_per_lap if new.fuel_l_per_lap is not None or old is None else old.fuel_l_per_lap,
            energy_per_lap=new.energy_per_lap if new.energy_per_lap is not None or old is None else old.energy_per_lap,
            fuel_per_energy=new.fuel_per_energy
            if new.fuel_per_energy is not None or old is None
            else old.fuel_per_energy,
            clean_laps_s=new.clean_laps_s if new.clean_laps_s or old is None else old.clean_laps_s,
        )
    return result


def prior_pace_s(
    qualifying_s: float | None,
    remembered_s: float | None,
    best_lap_s: float | None,
    class_best_s: float | None,
    history_car_s: float | None,
    history_class_s: float | None,
    pace_factor: float,
) -> tuple[float | None, str]:
    """Race pace of a car while its measured pace is not in use, from the first source that has one.
    In:  qualifying_s     float | None  s, its best lap in this event's qualifying results
         remembered_s     float | None  s/lap, clean pace of an earlier practice or race session of this process
         best_lap_s       float | None  s, its best lap in this session
         class_best_s     float | None  s/lap, the fastest pace known in its class
         history_car_s    float | None  s/lap, the history's pace for this track and car (player only)
         history_class_s  float | None  s/lap, the history's pace for this track and class
         pace_factor      float         race pace = factor * a qualifying or best lap
    Out:                  tuple[float | None, str]  (s/lap, PriorSource.pace)
    """
    if qualifying_s is not None:
        return qualifying_s * pace_factor, "qualifying"
    if remembered_s is not None:
        return remembered_s, "remembered"
    if best_lap_s is not None:
        return best_lap_s * pace_factor, "best_lap"
    if class_best_s is not None:
        return class_best_s, "class_best"
    if history_car_s is not None:
        return history_car_s, "history_car"
    if history_class_s is not None:
        return history_class_s, "history_class"
    return None, "none"


def history_pace_s(race_pace_s: float | None, qualifying_lap_s: float | None, pace_factor: float) -> float | None:
    """Race pace the history gives for a car or a class: its race pace, else its best qualifying lap times the factor.
    In:  race_pace_s       float | None  s/lap
         qualifying_lap_s  float | None  s
         pace_factor       float         race pace = factor * a qualifying lap
    Out:                   float | None  s/lap; None when the history has neither
    """
    if race_pace_s is not None:
        return race_pace_s
    return None if qualifying_lap_s is None else qualifying_lap_s * pace_factor


def prior_energy_per_lap(
    remembered: float | None,
    history_car: float | None,
    game: float | None,
    class_usage: float | None,
    history_class: float | None,
) -> tuple[float | None, str]:
    """Energy use per lap of a car while its measured usage is not in use, from the first source that has one.
    In:  remembered     float | None  fraction/lap measured in an earlier session of this process
         history_car    float | None  fraction/lap, the history's figure for this track and car (player only)
         game           float | None  fraction/lap the game's garage screen shows (player only)
         class_usage    float | None  fraction/lap, the best-known figure of the car's class in this session
         history_class  float | None  fraction/lap, the history's figure for this track and class
    Out:                tuple[float | None, str]  (fraction/lap, PriorSource.energy)
    """
    if remembered is not None:
        return remembered, "remembered"
    if history_car is not None:
        return history_car, "history_car"
    if game is not None:
        return game, "game"
    if class_usage is not None:
        return class_usage, "class"
    if history_class is not None:
        return history_class, "history_class"
    return None, "none"


def class_usage(figures: Sequence[tuple[float | None, str]]) -> float | None:
    """Best-known energy use per lap of a class: the mean over its cars of the figures from the best source that any
    of them has (measured, then remembered, then the player's history, then the game's garage figure).
    In:  figures  Sequence[tuple[float | None, str]]  (fraction/lap, source) of each car of the class
    Out:          float | None                         fraction/lap; None when no car has a figure of its own
    """
    for source in USAGE_SOURCES:
        values = [value for value, origin in figures if origin == source and value is not None]
        if values:
            return sum(values) / len(values)
    return None


def fill_estimate(
    estimate: CarEstimate, pace_s: float | None, energy_per_lap: float | None, fuel_per_energy: float | None
) -> CarEstimate:
    """Estimate of one car with its missing fields filled from priors; measured fields are kept.
    In:  estimate         CarEstimate   what this session measured
         pace_s           float | None  s/lap used while the car has no clean lap
         energy_per_lap   float | None  fraction/lap used while no usage is measured
         fuel_per_energy  float | None  litres per 1 % energy used while no ratio is measured
    Out:                  CarEstimate   fuel per lap follows from energy per lap and the ratio when it is not measured
    """
    energy = estimate.energy_per_lap if estimate.energy_per_lap is not None else energy_per_lap
    ratio = estimate.fuel_per_energy if estimate.fuel_per_energy is not None else fuel_per_energy
    fuel = estimate.fuel_l_per_lap
    if fuel is None and energy is not None and ratio is not None:
        fuel = energy * 100 * ratio
    return replace(
        estimate,
        lap_time_s=estimate.lap_time_s if estimate.clean_laps_s else pace_s,
        fuel_l_per_lap=fuel,
        energy_per_lap=energy,
        fuel_per_energy=ratio,
    )
