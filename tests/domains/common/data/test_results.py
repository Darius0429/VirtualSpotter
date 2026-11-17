from pathlib import Path

from ai_race_engineer.domains.common.data.results import parse_results
from ai_race_engineer.domains.common.data.results_file import latest_results, read_results

HEADER: str = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE rF [
<!ENTITY rFEnt "rFactor Entity">
]>
<rFactorXML version="1.0">
<RaceResults>
<DateTime>1000</DateTime>
<TrackVenue>Test Venue</TrackVenue>
<TrackCourse>Test Course</TrackCourse>
<TrackEvent>6 Hours of Test</TrackEvent>
<TrackLength>5000.5</TrackLength>
<RaceLaps>0</RaceLaps>
<RaceTime>60</RaceTime>
"""
FOOTER: str = "</RaceResults>\n</rFactorXML>\n"
QUALIFY: str = (
    HEADER
    + """<Qualify>
<DateTime>1200</DateTime>
<Stream>
<ChatMessage et="5.0">Driver One: hello</ChatMessage>
<Score et="60.1">Driver One(1) lap=1 point=1</Score>
<Sector Driver="Driver One" ID="1" Sector="1" Class="Hyper" et="70.0">Driver One(1) set new best for sector 1</Sector>
</Stream>
<Driver>
<Name>Driver One</Name>
<VehName>Team A 2026 #1:WEC</VehName>
<CarType>Car A</CarType>
<CarClass>Hyper</CarClass>
<CarNumber>1</CarNumber>
<TeamName>Team A</TeamName>
<isPlayer>1</isPlayer>
<Position>1</Position>
<ClassPosition>1</ClassPosition>
<Lap num="1" p="2" et="--.---" fcompound="0,Medium" rcompound="0,Soft">--.----</Lap>
<Lap num="2" p="1" et="130.5" s1="30.0" s2="40.0" s3="30.5" fcompound="0,Medium" rcompound="0,Soft">100.5</Lap>
<Lap num="3" p="1" et="231.0" s1="31.0" s2="41.0" fcompound="0,Medium" rcompound="0,Soft" pit="1">--.----</Lap>
<BestLapTime>100.5</BestLapTime>
<Laps>3</Laps>
<Pitstops>1</Pitstops>
<FinishStatus>Finished Normally</FinishStatus>
</Driver>
<Driver>
<Name>Driver Two</Name>
<VehName>Team B 2026 #2:WEC</VehName>
<CarType>Car B</CarType>
<CarClass>GT3</CarClass>
<CarNumber>2</CarNumber>
<TeamName>Team B</TeamName>
<isPlayer>0</isPlayer>
<Position>2</Position>
<ClassPosition>1</ClassPosition>
<Laps>0</Laps>
<Pitstops>0</Pitstops>
<FinishStatus>None</FinishStatus>
</Driver>
</Qualify>
"""
    + FOOTER
)
RACE: str = (
    HEADER
    + """<Race>
<DateTime>2000</DateTime>
<Stream>
<Incident et="143.1">Driver Two(2) reported contact (84.50) with another vehicle Driver One(1)</Incident>
<Penalty Driver="Driver Two" ID="2" Penalty="Drive Thru" et="300.5">Driver Two received Drive Thru</Penalty>
<TrackLimits Driver="Driver One" ID="1" Lap="4" WarningPoints="1" CurrentPoints="1" et="410.0">Warning</TrackLimits>
<ChatMessage et="500.0">secret words</ChatMessage>
</Stream>
<Driver>
<Name>Driver One</Name>
<VehName>Team A 2026 #1:WEC</VehName>
<CarType>Car A</CarType>
<CarClass>Hyper</CarClass>
<CarNumber>1</CarNumber>
<TeamName>Team A</TeamName>
<isPlayer>1</isPlayer>
<GridPos>3</GridPos>
<Position>2</Position>
<ClassGridPos>2</ClassGridPos>
<ClassPosition>1</ClassPosition>
<Lap num="1" p="3" et="127.5" s1="38.0" s2="48.0" s3="27.0" fcompound="0,Hard" rcompound="0,Hard">113.0</Lap>
<BestLapTime>113.0</BestLapTime>
<FinishTime>3650.25</FinishTime>
<Laps>33</Laps>
<Pitstops>1</Pitstops>
<FinishStatus>Finished Normally</FinishStatus>
</Driver>
</Race>
"""
    + FOOTER
)


def test_qualifying_session_and_drivers() -> None:
    """The session kind, its own start time, the track fields and every driver are read."""
    results = parse_results(QUALIFY)
    assert (results.kind, results.started_unix) == ("qualify", 1200)
    assert (results.track_venue, results.track_course, results.track_event) == (
        "Test Venue",
        "Test Course",
        "6 Hours of Test",
    )
    assert (results.track_length_m, results.race_time_min, results.race_laps) == (5000.5, 60.0, 0)
    one, two = results.drivers
    assert (one.name, one.car_type, one.car_class, one.car_number, one.team_name) == (
        "Driver One",
        "Car A",
        "Hyper",
        "1",
        "Team A",
    )
    assert (one.is_player, two.is_player) == (True, False)
    assert (one.place, one.class_place, one.best_lap_s, one.laps, one.pitstops) == (1, 1, 100.5, 3, 1)
    assert (one.grid_place, one.class_grid_place, one.finish_time_s) == (None, None, None)
    assert (two.best_lap_s, two.lap_list) == (None, ())


def test_laps_untimed_missing_sectors_pit_and_compound() -> None:
    """An untimed lap has no time, a missing sector attribute is None, pit="1" marks the in-lap and the compound is
    the text after the comma."""
    first, second, third = parse_results(QUALIFY).drivers[0].lap_list
    assert (first.num, first.place, first.start_s, first.lap_time_s, first.sector1_s) == (1, 2, None, None, None)
    assert (second.start_s, second.lap_time_s) == (130.5, 100.5)
    assert (second.sector1_s, second.sector2_s, second.sector3_s) == (30.0, 40.0, 30.5)
    assert (third.lap_time_s, third.sector2_s, third.sector3_s, third.is_pit) == (None, 41.0, None, True)
    assert (second.compound_front, second.compound_rear, second.is_pit) == ("Medium", "Soft", False)


def test_race_grid_places_and_finish_time() -> None:
    """A race file carries the grid places and the finish time."""
    results = parse_results(RACE)
    driver = results.drivers[0]
    assert (results.kind, results.started_unix) == ("race", 2000)
    assert (driver.grid_place, driver.class_grid_place, driver.finish_time_s) == (3, 2, 3650.25)


def test_events_kept_without_chat_and_scores() -> None:
    """Incidents, penalties, track limits and sector bests are kept with their driver; chat and score lines are not."""
    events = parse_results(RACE).events
    assert [(e.kind, e.time_s, e.driver) for e in events] == [
        ("incident", 143.1, "Driver Two"),
        ("penalty", 300.5, "Driver Two"),
        ("track_limits", 410.0, "Driver One"),
    ]
    assert "secret" not in str(events)
    assert [e.kind for e in parse_results(QUALIFY).events] == ["sector"]


def test_broken_text_gives_none() -> None:
    """A half-written file and a document without a session give nothing."""
    assert parse_results(QUALIFY[:900]) is None
    assert parse_results(HEADER + FOOTER) is None


def test_read_results_and_latest_by_name(tmp_path: Path) -> None:
    """A missing file gives None; the newest file of a kind is picked by its name."""
    (tmp_path / "2026_01_01_10_00_00-11Q1.xml").write_text(QUALIFY.replace("<DateTime>1200", "<DateTime>1100"))
    (tmp_path / "2026_01_02_10_00_00-22Q1.xml").write_text(QUALIFY)
    (tmp_path / "2026_01_03_10_00_00-33R1.xml").write_text(RACE)
    assert read_results(tmp_path / "missing.xml") is None
    assert latest_results(tmp_path, "qualify").started_unix == 1200
    assert latest_results(tmp_path, "race").kind == "race"
    assert latest_results(tmp_path, "practice") is None


def test_file_with_an_empty_number_counts_as_unreadable(tmp_path: Path) -> None:
    """A well-formed file whose number field is empty (still being written) gives None instead of an error."""
    path = tmp_path / "2026_01_03_10_00_00-33R1.xml"
    path.write_text(RACE.replace("<Position>2</Position>", "<Position></Position>"))
    assert read_results(path) is None
    assert latest_results(tmp_path, "race") is None
