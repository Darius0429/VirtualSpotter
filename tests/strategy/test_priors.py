import dataclasses

import pytest

from ai_race_engineer.strategy.field import CarEstimate
from ai_race_engineer.strategy.history import ClassFigures, SessionFigures
from ai_race_engineer.strategy.priors import (
    class_usage,
    event_priors,
    fill_estimate,
    history_pace_s,
    prior_energy_per_lap,
    prior_pace_s,
    remember,
)
from ai_race_engineer.telemetry.results import ResultDriver, SessionResults
from ai_race_engineer.telemetry.snapshot import VehicleState

NOW_UNIX: float = 10000.0


def _driver(name: str, best_lap_s: float | None, car_type: str = "Car A") -> ResultDriver:
    """Results driver with a best lap.
    In:  name        str
         best_lap_s  float | None  s
         car_type    str
    Out:             ResultDriver
    """
    return ResultDriver(
        name, "veh", car_type, "Hyper", "1", "team", True, None, None, 1, 1, best_lap_s, None, 5, 0, "", ()
    )


def _results(kind: str = "qualify", track: str = "Track", started_unix: int = 9000) -> SessionResults:
    """Results of a session with the player ("me", 100.0 s), a rival (101.0 s) and a car without a lap.
    In:  kind          str
         track         str  TrackCourse
         started_unix  int  s unix time
    Out:               SessionResults
    """
    drivers = (_driver("me", 100.0), _driver("rival", 101.0, "Car B"), _driver("slow", None))
    return SessionResults("Venue", track, "Event", 5000.0, kind, started_unix, 60.0, 0, drivers, ())


def _priors(results: SessionResults | None, player_name: str = "me") -> dict[str, float]:
    """Qualifying times taken from results for a race on "Track" with a 6 h age limit.
    In:  results      SessionResults | None
         player_name  str
    Out:              dict[str, float]  driver name -> s
    """
    return dict(
        event_priors(
            "Track", "vehicle", player_name, results, {}, (), NOW_UNIX, "TrackCourse", 5, 6.0, NOW_UNIX
        ).qualifying_best_s
    )


def _entry(session_id: str, started_unix: int, clean_pace_s: float) -> SessionFigures:
    """History line of a race of "Car A" on "Track" with the Hyper class leader on the same pace.
    In:  session_id    str
         started_unix  int    s unix time
         clean_pace_s  float  s/lap
    Out:               SessionFigures
    """
    classes = {"Hyper": ClassFigures(None, clean_pace_s, None, None)}
    return SessionFigures(
        session_id,
        "results_file",
        started_unix,
        "Track",
        "race",
        "Car A",
        "vehicle",
        "Hyper",
        None,
        clean_pace_s,
        10,
        (None, None, None),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (),
        1,
        1,
        30,
        1,
        None,
        classes,
    )


def _vehicle(slot_id: int, driver_name: str, class_name: str = "Hyper") -> VehicleState:
    """Vehicle with a slot, a driver and a class.
    In:  slot_id      int
         driver_name  str
         class_name   str
    Out:              VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(slot_id=slot_id, driver_name=driver_name, class_name=class_name)
    return VehicleState(**base)


def test_qualifying_times_from_this_events_file() -> None:
    """A fresh qualifying file of this track with the player gives every timed driver's best lap and the car."""
    assert _priors(_results()) == {"me": 100.0, "rival": 101.0}
    priors = event_priors("Track", "vehicle", "me", _results(), {}, (), NOW_UNIX, "TrackCourse", 5, 6.0, NOW_UNIX)
    assert (priors.track, priors.car) == ("Track", "Car A")
    assert (priors.car_history, priors.class_history) == (None, {})


@pytest.mark.parametrize(
    "results",
    [None, _results(kind="race"), _results(track="Other"), _results(started_unix=int(NOW_UNIX) - 6 * 3600 - 1)],
)
def test_other_files_are_ignored(results: SessionResults | None) -> None:
    """No file, a race file, another track or a file older than the limit gives no qualifying times."""
    assert _priors(results) == {}


def test_file_without_the_player_is_ignored() -> None:
    """A qualifying file that does not contain the player's name is not this event's; the car is the vehicle name."""
    assert _priors(_results(), player_name="someone else") == {}
    priors = event_priors(
        "Track", "vehicle", "someone else", _results(), {}, (), NOW_UNIX, "TrackCourse", 5, 6.0, NOW_UNIX
    )
    assert priors.car == "vehicle"


def test_track_field_selects_the_compared_name() -> None:
    """The configured results field is the one compared with the game's track name."""
    assert event_priors(
        "Venue", "v", "me", _results(), {}, (), NOW_UNIX, "TrackVenue", 5, 6.0, NOW_UNIX
    ).qualifying_best_s
    assert not event_priors(
        "Venue", "v", "me", _results(), {}, (), NOW_UNIX, "TrackCourse", 5, 6.0, NOW_UNIX
    ).qualifying_best_s


def test_history_is_cut_off_before_the_session() -> None:
    """Only sessions that started before the given time count: the session's own result and later ones are left
    out, also for the car key and the classes."""
    entries = (_entry("old", 5000, 107.0), _entry("own", 9900, 105.0), _entry("later", 20000, 103.0))
    priors = event_priors("Track", "vehicle", "me", None, {}, entries, NOW_UNIX, "TrackCourse", 5, 6.0, 9900.0)
    assert (priors.car, priors.car_history.sessions, priors.car_history.race_pace_s) == ("Car A", 1, 107.0)
    assert priors.class_history["Hyper"].race_pace_s == 107.0
    none_yet = event_priors("Track", "vehicle", "me", None, {}, entries, NOW_UNIX, "TrackCourse", 5, 6.0, 5000.0)
    assert (none_yet.car, none_yet.car_history, none_yet.class_history) == ("vehicle", None, {})


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((100.0, 105.0, 102.0, 99.0, 98.0, 97.0), (101.0, "qualifying")),
        ((None, 105.0, 102.0, 99.0, 98.0, 97.0), (105.0, "remembered")),
        ((None, None, 102.0, 99.0, 98.0, 97.0), (103.02, "best_lap")),
        ((None, None, None, 99.0, 98.0, 97.0), (99.0, "class_best")),
        ((None, None, None, None, 98.0, 97.0), (98.0, "history_car")),
        ((None, None, None, None, None, 97.0), (97.0, "history_class")),
        ((None, None, None, None, None, None), (None, "none")),
    ],
)
def test_pace_order(values: tuple[float | None, ...], expected: tuple[float | None, str]) -> None:
    """The qualifying lap, then the remembered pace, the best lap, the best of the class and the history of the car
    and of the class; the factor scales only the qualifying and the best lap."""
    pace_s, source = prior_pace_s(*values, pace_factor=1.01)
    assert (pace_s, source) == (pytest.approx(expected[0]), expected[1])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.05, 0.045, 0.04, 0.03, 0.02), (0.05, "remembered")),
        ((None, 0.045, 0.04, 0.03, 0.02), (0.045, "history_car")),
        ((None, None, 0.04, 0.03, 0.02), (0.04, "game")),
        ((None, None, None, 0.03, 0.02), (0.03, "class")),
        ((None, None, None, None, 0.02), (0.02, "history_class")),
        ((None, None, None, None, None), (None, "none")),
    ],
)
def test_energy_order(values: tuple[float | None, ...], expected: tuple[float | None, str]) -> None:
    """Remembered usage, the history of the car, the game's garage figure, the figure of the class in this session,
    the history of the class, then none."""
    assert prior_energy_per_lap(*values) == expected


def test_history_pace_prefers_the_race_pace() -> None:
    """The history's race pace is used as it is; without one its qualifying lap times the factor."""
    assert history_pace_s(104.0, 100.0, 1.01) == 104.0
    assert history_pace_s(None, 100.0, 1.01) == pytest.approx(101.0)
    assert history_pace_s(None, None, 1.01) is None


def test_class_usage_is_the_mean_of_the_best_source() -> None:
    """Measured figures count before remembered ones and those before the garage figure; cars without one are left
    out."""
    figures = [(0.044, "measured"), (0.046, "measured"), (0.05, "remembered"), (0.06, "game"), (None, "none")]
    assert class_usage(figures) == pytest.approx(0.045)
    assert class_usage(figures[2:]) == 0.05
    assert class_usage(figures[3:]) == 0.06
    assert class_usage(figures[4:]) is None
    assert class_usage([(0.03, "class"), (0.02, "history_class")]) is None


def test_fill_estimate_fills_what_is_missing() -> None:
    """Without measurements the priors are used and the fuel per lap follows from energy and ratio."""
    filled = fill_estimate(CarEstimate(1, None, None, None, None), 100.0, 0.05, 0.8)
    assert filled == CarEstimate(1, 100.0, pytest.approx(4.0), 0.05, 0.8)


def test_fill_estimate_keeps_measured_fields() -> None:
    """Measured pace and usage are not replaced; a best-lap fallback without clean laps is."""
    measured = CarEstimate(1, 101.0, 3.5, 0.045, 0.78, (101.0,))
    assert fill_estimate(measured, 100.0, 0.05, 0.8) == measured
    best_lap_only = CarEstimate(1, 99.0, None, None, None)
    assert fill_estimate(best_lap_only, 100.0, None, None).lap_time_s == 100.0
    assert fill_estimate(best_lap_only, 100.0, 0.05, None).fuel_l_per_lap is None


def test_remember_keeps_the_median_pace_and_older_values() -> None:
    """A finished practice session's clean laps give the median as pace; a value it did not measure keeps the older
    one; drivers are kept apart by class."""
    vehicles = (_vehicle(3, "me"), _vehicle(4, "rival"))
    measured = {3: CarEstimate(3, 104.0, 3.5, 0.045, 0.78, (100.0, 120.0, 101.0)), 4: _empty(4)}
    first = remember({}, vehicles, measured, "practice")
    assert first[("me", "Hyper")].lap_time_s == 101.0
    assert first[("rival", "Hyper")].lap_time_s is None
    second = remember(first, vehicles, {3: CarEstimate(3, 99.0, None, None, None, ())}, "race")
    me = second[("me", "Hyper")]
    assert (me.lap_time_s, me.energy_per_lap, me.fuel_per_energy) == (101.0, 0.045, 0.78)
    assert second[("rival", "Hyper")] == first[("rival", "Hyper")]
    other_class = remember(second, (_vehicle(3, "me", "GT3"),), {3: _empty(3)}, "race")
    assert other_class[("me", "GT3")].lap_time_s is None
    assert other_class[("me", "Hyper")] == me


def test_qualifying_pace_is_not_remembered() -> None:
    """A qualifying session's clean laps include cool-down laps: its usage is remembered, its pace is not."""
    laps = (106.0, 125.0, 106.2, 125.4)
    kept = remember({}, (_vehicle(3, "me"),), {3: CarEstimate(3, 115.7, 3.5, 0.045, 0.78, laps)}, "qualify")
    assert (kept[("me", "Hyper")].lap_time_s, kept[("me", "Hyper")].energy_per_lap) == (None, 0.045)


def _empty(slot_id: int) -> CarEstimate:
    """Estimate of a car that measured nothing.
    In:  slot_id  int
    Out:          CarEstimate
    """
    return CarEstimate(slot_id, None, None, None, None)
