import dataclasses

import pytest

from ai_race_engineer.strategy.pit_time import PitStopSample
from ai_race_engineer.strategy.session_log import Note, PlayerLap, SessionRecord
from ai_race_engineer.strategy.session_summary import (
    results_of_session,
    summarise,
    summary_from_results,
    with_results,
)
from ai_race_engineer.telemetry.results import ResultDriver, ResultEvent, ResultLap, SessionResults
from ai_race_engineer.telemetry.snapshot import VehicleState


def _vehicle(slot_id: int, place: int, driver_name: str, class_name: str = "Hyper") -> VehicleState:
    """Car of the session with 30 laps and a best lap of 105 s plus its place.
    In:  slot_id      int
         place        int  overall position
         driver_name  str
         class_name   str
    Out:              VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        slot_id=slot_id,
        place=place,
        driver_name=driver_name,
        class_name=class_name,
        completed_laps=30,
        best_lap_s=105.0 + place,
        pitstops=1,
    )
    return VehicleState(**base)


def _lap(
    lap: int, lap_time_s: float, sectors: tuple[float, float, float], energy: float | None = 0.045, **flags: bool
) -> PlayerLap:
    """Lap of the player using 0.65 litres per 1 % energy.
    In:  lap         int
         lap_time_s  float                        s
         sectors     tuple[float, float, float]   s
         energy      float | None                 fraction used
         flags       bool                         timed, pitted, green overrides
    Out:             PlayerLap
    """
    values = {"timed": True, "pitted": False, "green": True, **flags}
    fuel = None if energy is None else energy * 65
    return PlayerLap(
        lap,
        100.0 * lap,
        lap_time_s,
        values["timed"],
        *sectors,
        energy,
        fuel,
        7,
        "Medium",
        "Medium",
        values["pitted"],
        values["green"],
    )


def _record() -> SessionRecord:
    """Race record: the player P7 of 9 in class from P9 on the grid, with a start lap, three clean laps, an
    invalidated lap, a pit lap and one stop.
    In:  -
    Out: SessionRecord
    """
    vehicles = tuple(_vehicle(i, i + 1, "me" if i == 6 else f"driver{i}") for i in range(9)) + (
        _vehicle(20, 10, "gt", "GT3"),
    )
    laps = (
        _lap(1, 114.0, (38.0, 48.0, 28.0), green=False),
        _lap(2, 107.0, (32.0, 48.0, 27.0)),
        _lap(3, 106.5, (32.2, 47.5, 26.8), 0.044),
        _lap(4, 108.0, (32.1, 48.4, 27.5), 0.046, timed=False),
        _lap(5, 111.0, (32.5, 48.5, 30.0), 0.042, pitted=True),
    )
    notes = (Note(450.0, 5, "Pit stop lap 5: 22.6 s stationary", True), Note(300.0, 3, "understeer", False))
    stops = (
        PitStopSample(3, False, 30.0, 400.0, 25.0, 0.6, 30.0, None),
        PitStopSample(6, True, 30.0, 400.0, 22.6, 0.55, 32.4, None),
    )
    return SessionRecord(
        "20260101_120000_race",
        1000.0,
        "Track",
        "race",
        "me",
        "my car",
        "Hyper",
        600.0,
        False,
        vehicles,
        9,
        laps,
        notes,
        4,
        stops,
        {},
        (),
    )


def _results() -> SessionResults:
    """Race results file with the player P2 in class from P4 on the grid, a contact, a penalty and a warning.
    In:  -
    Out: SessionResults
    """
    laps = tuple(
        ResultLap(num, 2, 100.0 * num, time_s, 32.0, 48.0, time_s - 80.0, "Hard", "Hard", num == 3)
        for num, time_s in ((1, 113.0), (2, 107.0), (3, 118.0), (4, 130.0), (5, 106.0))
    )
    drivers = (
        ResultDriver("leader", "v", "Car B", "Hyper", "2", "t", True, 1, 1, 1, 1, 105.0, 3600.0, 5, 1, "Finished", ()),
        ResultDriver(
            "me", "my car", "Car A", "Hyper", "1", "t", True, 5, 4, 3, 2, 106.0, 3650.0, 5, 1, "Finished", laps
        ),
    )
    events = (
        ResultEvent("incident", 143.0, "other", "other(3) reported contact (80.0) with another vehicle me(6)"),
        ResultEvent("incident", 200.0, "other", "other(3) reported contact (10.0) with another vehicle leader(1)"),
        ResultEvent("incident", 210.0, "meg", "meg(8) reported contact (10.0) with another vehicle some me(9)"),
        ResultEvent("penalty", 300.0, "me", "me received Drive Thru penalty"),
        ResultEvent("track_limits", 400.0, "me", "Warning"),
        ResultEvent("track_limits", 410.0, "leader", "Warning"),
    )
    return SessionResults("Venue", "Track", "Event", 5000.0, "race", 1010, 60.0, 0, drivers, events)


def test_summary_of_the_players_session() -> None:
    """Place, places gained, laps, best lap, theoretical best from the best sectors and the clean pace."""
    summary = summarise(_record(), "Car A", 3)
    player = summary.player
    assert (summary.session_id, summary.track, summary.kind, summary.car, summary.car_class) == (
        "20260101_120000_race",
        "Track",
        "race",
        "Car A",
        "Hyper",
    )
    assert (summary.complete, summary.yellow_flags, summary.time_s) == (False, 4, 600.0)
    assert (player.class_place, player.grid_class_place, player.positions_gained) == (7, 9, 2)
    assert (player.laps, player.best_lap_s) == (30, 112.0)
    assert player.best_sectors_s == (32.0, 47.5, 26.8)
    assert player.theoretical_best_s == pytest.approx(106.3)
    assert (player.clean_pace_s, player.clean_laps, player.invalid_laps) == (107.0, 3, 1)


def test_usage_and_pit_stops_of_the_session() -> None:
    """Energy per lap is the mean of the clean laps, the stint what a full load lasts; the player's stop carries
    the lap of its note."""
    player = summarise(_record(), "Car A", 3).player
    assert player.energy_per_lap == pytest.approx(0.045)
    assert player.fuel_per_energy == pytest.approx(0.65)
    assert player.stint_laps == 22
    (stop,) = player.pit_stops
    assert (stop.lap, stop.stationary_s, stop.energy_added, stop.fuel_added_l) == (5, 22.6, 0.55, 32.4)
    assert (player.incidents, player.penalties, player.track_limit_warnings) == (None, (), None)


def test_class_order_is_the_top_plus_the_neighbours() -> None:
    """Of nine cars in class the top three, the player and the cars directly ahead and behind are listed."""
    order = summarise(_record(), "Car A", 3).class_order
    assert [e.class_place for e in order] == [1, 2, 3, 6, 7, 8]
    assert (order[4].driver, order[4].laps, order[4].pitstops) == ("me", 30, 1)


def test_summary_from_a_results_file() -> None:
    """A file alone gives places, laps and a clean pace without lap 1, the in-lap and the lap after it; no usage."""
    summary = summary_from_results(_results(), "stem", "TrackCourse", 3, "me")
    player = summary.player
    assert (summary.session_id, summary.kind, summary.car, summary.complete, summary.yellow_flags) == (
        "stem",
        "race",
        "Car A",
        True,
        None,
    )
    assert (player.class_place, player.grid_class_place, player.positions_gained) == (2, 4, 2)
    assert (player.clean_pace_s, player.clean_laps, player.energy_per_lap) == (106.5, 2, None)
    assert [stop.lap for stop in player.pit_stops] == [3]
    assert (player.incidents, player.track_limit_warnings) == (1, 1)
    assert summary_from_results(_results(), "stem", "TrackCourse", 3, "nobody") is None


def test_results_add_incidents_penalties_and_their_notes() -> None:
    """The file adds the player's contacts, penalties, warnings and grid place, and a note for each contact and
    penalty in time order with the session's own notes."""
    joined = with_results(summarise(_record(), "Car A", 3), _results(), "me")
    player = joined.player
    assert (player.incidents, player.track_limit_warnings) == (1, 1)
    assert player.penalties == ("me received Drive Thru penalty",)
    assert (player.grid_class_place, player.positions_gained) == (4, -3)
    assert [(n.time_s, n.auto) for n in joined.notes] == [(143.0, True), (300.0, False), (300.0, True), (450.0, True)]
    assert joined.notes[0].text.startswith("Contact: ")
    assert joined.notes[2].text == "Penalty: me received Drive Thru penalty"


def test_contacts_are_matched_by_the_whole_name() -> None:
    """A contact counts for the player when the player reports it or is the other vehicle; a driver whose name only
    contains the player's does not."""
    reported = ResultEvent("incident", 500.0, "me", "me(6) reported contact (5.0) with Immovable")
    results = dataclasses.replace(_results(), events=(*_results().events, reported))
    joined = with_results(summarise(_record(), "Car A", 3), results, "me")
    assert joined.player.incidents == 2
    assert [n.time_s for n in joined.notes if n.text.startswith("Contact: ")] == [143.0, 500.0]


@pytest.mark.parametrize(
    ("kind", "start_unix", "expected"), [("race", 1000.0, True), ("race", 1400.0, False), ("qualify", 1000.0, False)]
)
def test_results_of_session(kind: str, start_unix: float, expected: bool) -> None:
    """A file belongs to a session of the same kind that started within the margin of the file's start."""
    assert results_of_session(_results(), kind, start_unix, 300.0) is expected
