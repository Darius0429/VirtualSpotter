import dataclasses
import json

import pytest

from ai_race_engineer.strategy.history import (
    ClassFigures,
    SessionFigures,
    car_history,
    car_key,
    class_history,
    figures_from_dict,
    figures_from_record,
    figures_from_results,
)
from ai_race_engineer.strategy.pit_time import PitLane, PitRates
from ai_race_engineer.strategy.session_log import SessionRecord
from ai_race_engineer.strategy.session_summary import summarise
from ai_race_engineer.telemetry.results import ResultDriver, ResultLap, SessionResults
from ai_race_engineer.telemetry.snapshot import VehicleState


def _lap(num: int, lap_time_s: float | None, is_pit: bool = False) -> ResultLap:
    """Results lap with sectors of a third of the lap each.
    In:  num         int
         lap_time_s  float | None  s
         is_pit      bool          in-lap
    Out:             ResultLap
    """
    third = None if lap_time_s is None else lap_time_s / 3
    return ResultLap(num, 1, 100.0 * num, lap_time_s, third, third, third, "Medium", "Medium", is_pit)


def _driver(
    name: str,
    class_place: int,
    laps: tuple[ResultLap, ...],
    car_class: str = "Hyper",
    finish_status: str = "Finished Normally",
) -> ResultDriver:
    """Results driver in "Car A" (the player's team) or "Car B".
    In:  name           str
         class_place    int
         laps           tuple[ResultLap, ...]
         car_class      str
         finish_status  str
    Out:                ResultDriver
    """
    times = [lap.lap_time_s for lap in laps if lap.lap_time_s is not None]
    car = "Car A" if name == "me" else "Car B"
    return ResultDriver(
        name,
        f"{car} #1",
        car,
        car_class,
        "1",
        "team",
        True,
        3,
        2,
        class_place,
        class_place,
        min(times, default=None),
        None,
        len(laps),
        sum(lap.is_pit for lap in laps),
        finish_status,
        laps,
    )


def _results(kind: str = "race", finish_status: str = "Finished Normally") -> SessionResults:
    """A session with the player P2 in class behind a leader, and a GT3 car without a lap.
    In:  kind           str
         finish_status  str  the player's
    Out:                SessionResults
    """
    mine = (_lap(1, 115.0), _lap(2, 106.0), _lap(3, 108.0), _lap(4, 120.0, True), _lap(5, 140.0), _lap(6, 107.0))
    leader = (_lap(1, 112.0), _lap(2, 105.0), _lap(3, 105.4), _lap(4, None))
    drivers = (
        _driver("leader", 1, leader),
        _driver("me", 2, mine, finish_status=finish_status),
        _driver("gt", 1, (), "GT3"),
    )
    return SessionResults("Venue", "Track", "Event", 5000.0, kind, 5000, 60.0, 0, drivers, ())


def _entry(
    session_id: str,
    started_unix: int,
    kind: str = "race",
    source: str = "results_file",
    best_lap_s: float | None = 106.0,
    clean_pace_s: float | None = 107.0,
    energy_per_lap: float | None = None,
    track: str = "Track",
    car: str = "Car A",
    classes: dict[str, ClassFigures] | None = None,
    complete: bool = True,
) -> SessionFigures:
    """History line of the player's car.
    In:  session_id      str
         started_unix    int
         kind            str
         source          str
         best_lap_s      float | None  s
         clean_pace_s    float | None  s/lap
         energy_per_lap  float | None  fraction/lap
         track           str
         car             str
         classes         dict[str, ClassFigures] | None
         complete        bool
    Out:                 SessionFigures
    """
    return SessionFigures(
        session_id,
        source,
        started_unix,
        track,
        kind,
        car,
        f"{car} #1",
        "Hyper",
        best_lap_s,
        clean_pace_s,
        10,
        (30.0, 40.0, 36.0),
        energy_per_lap,
        None,
        None,
        28.0 if energy_per_lap else None,
        400.0 if energy_per_lap else None,
        None,
        None,
        (),
        2,
        3,
        30,
        1,
        "Finished Normally",
        classes or {},
        complete,
    )


def test_figures_from_results_for_the_named_player() -> None:
    """The player is found by name; the clean pace is the median of the timed laps without lap 1, the in-lap and
    the lap after it; the usage fields stay empty."""
    figures = figures_from_results(_results(), "stem", "TrackCourse", "me")
    assert (figures.session_id, figures.source, figures.track, figures.kind) == (
        "stem",
        "results_file",
        "Track",
        "race",
    )
    assert (figures.car, figures.veh_name, figures.car_class) == ("Car A", "Car A #1", "Hyper")
    assert (figures.best_lap_s, figures.clean_pace_s, figures.clean_laps) == (106.0, 107.0, 3)
    assert figures.best_sectors_s == (pytest.approx(106.0 / 3),) * 3
    assert (figures.class_place, figures.grid_class_place, figures.laps, figures.pitstops) == (2, 2, 6, 1)
    assert (figures.energy_per_lap, figures.stint_laps, figures.straights) == (None, None, ())


def test_figures_from_results_carry_the_classes() -> None:
    """Each class has its best lap and the clean pace of its leader; a class without laps has neither."""
    classes = figures_from_results(_results(), "stem", "TrackCourse", "me").classes
    assert classes["Hyper"] == ClassFigures(105.0, pytest.approx(105.2), None, None)
    assert classes["GT3"] == ClassFigures(None, None, None, None)


def test_no_figures_without_the_player_or_a_lap() -> None:
    """A file the player is not in, or in which the player drove no lap, gives nothing."""
    assert figures_from_results(_results(), "stem", "TrackCourse", "nobody") is None
    assert figures_from_results(_results(), "stem", "TrackCourse", "gt") is None


def test_best_laps_are_the_minimum_and_the_rest_a_median_of_the_last_sessions() -> None:
    """One bad session among the last five does not move the race pace; the best lap is the fastest ever."""
    paces = (104.0, 107.0, 107.2, 130.0, 106.8, 107.4)
    entries = [_entry(f"r{i}", i, clean_pace_s=pace, best_lap_s=105.0 + i) for i, pace in enumerate(paces)]
    history = car_history(entries, "Track", "Car A", 5)
    assert history.race_pace_s == 107.2
    assert (history.best_lap_s, history.best_qualifying_lap_s) == (105.0, None)
    assert (history.sessions, history.last_update_unix) == (6, 5)
    assert [r[1] for r in history.last_results] == [5, 4, 3]


def test_race_sessions_come_first_and_file_entries_give_the_pace() -> None:
    """Race pace and energy use race sessions when there are any; the pace comes from results files when both
    sources have the session, the energy and the pit lane from the engineer's own lines."""
    entries = [
        _entry("p1", 1, "practice", clean_pace_s=110.0),
        _entry("q1", 2, "qualify", best_lap_s=104.5, clean_pace_s=105.0),
        _entry("r1", 3, clean_pace_s=107.0),
        _entry("e1", 3, source="engineer", clean_pace_s=109.0, energy_per_lap=0.045),
        _entry("e0", 1, "practice", source="engineer", clean_pace_s=111.0, energy_per_lap=0.05),
    ]
    history = car_history(entries, "Track", "Car A", 5)
    assert (history.race_pace_s, history.energy_per_lap) == (107.0, 0.045)
    assert history.best_qualifying_lap_s == 104.5
    assert history.pit_lane == PitLane(28.0, 400.0)
    assert history.sessions == 3
    only_practice = car_history(entries[:1], "Track", "Car A", 5)
    assert only_practice.race_pace_s == 110.0


def test_unknown_track_or_car_gives_no_history() -> None:
    """Nothing is known about a car or a class on a track without sessions."""
    entries = [_entry("r1", 1)]
    assert car_history(entries, "Other", "Car A", 5) is None
    assert car_history(entries, "Track", "Car B", 5) is None
    assert class_history(entries, "Track", "Hyper", 5) is None


def test_class_history_over_the_players_sessions() -> None:
    """The pole lap is the fastest qualifying lap of the class; pace and energy are medians."""
    entries = [
        _entry("q1", 1, "qualify", classes={"Hyper": ClassFigures(104.0, None, None, None)}),
        _entry("r1", 2, classes={"Hyper": ClassFigures(105.5, 106.0, None, None)}),
        _entry("e1", 2, source="engineer", classes={"Hyper": ClassFigures(105.5, 106.4, 0.044, 0.65)}),
    ]
    history = class_history(entries, "Track", "Hyper", 5)
    assert (history.pole_lap_s, history.race_pace_s) == (104.0, 106.0)
    assert (history.energy_per_lap, history.fuel_per_energy) == (0.044, 0.65)


def test_car_key_from_file_history_or_vehicle_name() -> None:
    """The car model comes from the player's entry in the results file, else from a history line with the same
    vehicle name, else it is the vehicle name."""
    entries = [_entry("r1", 1)]
    assert car_key("whatever", "me", _results(), entries) == "Car A"
    assert car_key("Car A #1", "nobody", _results(), entries) == "Car A"
    assert car_key("Car A #1", "me", None, entries) == "Car A"
    assert car_key("New Car #9", "me", None, entries) == "New Car #9"


def test_figures_round_trip_through_json() -> None:
    """A history line read back from its JSON form is equal to the original."""
    figures = figures_from_results(_results(), "stem", "TrackCourse", "me")
    assert figures_from_dict(json.loads(json.dumps(dataclasses.asdict(figures)))) == figures


def test_results_file_race_is_complete_only_when_finished_normally() -> None:
    """A race the file does not say the player finished is incomplete; other session kinds are always complete."""
    assert figures_from_results(_results(), "stem", "TrackCourse", "me").complete is True
    for status in ("DNF", "DQ", "None", ""):
        assert figures_from_results(_results(finish_status=status), "stem", "TrackCourse", "me").complete is False
        assert figures_from_results(_results("practice", status), "stem", "TrackCourse", "me").complete is True
        assert figures_from_results(_results("qualify", status), "stem", "TrackCourse", "me").complete is True


def test_figures_from_record_carry_the_finish() -> None:
    """A race the engineer watched is complete only when its finish was seen; other session kinds always are."""
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    me = VehicleState(**{**base, "is_player": True, "driver_name": "me", "class_name": "Hyper", "place": 1})
    record = SessionRecord(
        "id", 1000.0, "Track", "race", "me", "Car A #1", "Hyper", 60.0, True, (me,), 1, (), (), 0, (), {}, ()
    )
    rates = PitRates(3.0, 0.025, 1.0)
    for finished in (False, True):
        watched = dataclasses.replace(record, finished=finished)
        figures = figures_from_record(watched, summarise(watched, "Car A", 5), "Car A", rates)
        assert figures.complete is finished
    practice = dataclasses.replace(record, kind="practice", finished=False)
    assert figures_from_record(practice, summarise(practice, "Car A", 5), "Car A", rates).complete is True


def test_stored_line_without_the_flag() -> None:
    """A line stored before the flag existed loads as complete, except a results-file race the player did not
    finish; a stored flag is kept."""
    line = dataclasses.asdict(_entry("r1", 1))
    stored = dict(line)
    del line["complete"]
    assert figures_from_dict(line).complete is True
    assert figures_from_dict({**line, "finish_status": "DNF"}).complete is False
    assert figures_from_dict({**line, "finish_status": "None"}).complete is False
    assert figures_from_dict({**line, "finish_status": "DNF", "kind": "practice"}).complete is True
    assert figures_from_dict({**line, "finish_status": None, "source": "engineer"}).complete is True
    assert figures_from_dict({**stored, "finish_status": "DNF"}).complete is True
    assert figures_from_dict({**stored, "complete": False}).complete is False


def test_incomplete_race_counts_only_without_a_complete_one() -> None:
    """The typical values ignore an incomplete race while a complete one exists and fall back to it, like to a
    practice session, when none does."""
    entries = [
        _entry("r1", 1, clean_pace_s=107.0),
        _entry("r2", 2, clean_pace_s=120.0, complete=False),
        _entry("p1", 3, "practice", clean_pace_s=110.0),
    ]
    assert car_history(entries, "Track", "Car A", 5).race_pace_s == 107.0
    assert car_history(entries[1:], "Track", "Car A", 5).race_pace_s == 115.0


def test_last_results_show_an_incomplete_session() -> None:
    """The finish status of a session that was not driven to its end reads "incomplete"."""
    entries = [_entry("r1", 1), _entry("r2", 2, complete=False)]
    history = car_history(entries, "Track", "Car A", 5)
    assert [r[4] for r in history.last_results] == ["incomplete", "Finished Normally"]
