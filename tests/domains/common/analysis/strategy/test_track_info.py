import pytest

from ai_race_engineer.domains.common.analysis.strategy.track_info import track_infos

ALGARVE: dict[str, float] = {"length_m": 4635.3, "pit_lane_time_s": 31.0, "pit_entry_m": 4301.0, "pit_exit_m": 136.0}


def test_pit_lane_stretch_wraps_the_line() -> None:
    """The Algarve pit lane covers 470.3 m of lap from entry to exit across the start/finish line."""
    info = track_infos({"Algarve International Circuit": ALGARVE})["Algarve International Circuit"]
    assert info.length_m == 4635.3
    assert info.pit_lane.time_s == 31.0
    assert info.pit_lane.length_m == pytest.approx(470.3)


def test_entry_without_pit_keys_has_no_pit_lane() -> None:
    """A track with only a length has no pit lane."""
    assert track_infos({"Track": {"length_m": 5000.0}})["Track"].pit_lane is None


def test_straights_parsed() -> None:
    """Straights become (start_m, end_m) tuples."""
    info = track_infos({"Track": {"length_m": 5000.0, "straights": [[100.0, 900.0], [4800.0, 200.0]]}})["Track"]
    assert info.straights == ((100.0, 900.0), (4800.0, 200.0))


def test_straights_absent_gives_empty() -> None:
    """Without the straights key the tuple is empty."""
    assert track_infos({"Track": ALGARVE})["Track"].straights == ()
