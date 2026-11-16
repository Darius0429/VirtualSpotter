from datetime import datetime

from ai_race_engineer.strategy.timeline import (
    TimelineEntry,
    file_name,
    files_in_window,
    matching_file,
    select_entries,
)

START_UNIX: float = datetime(2026, 1, 2, 13, 4, 5).timestamp()


def _entries() -> list[TimelineEntry]:
    """Six entries on laps 1, 2, 2, 3, 5, 5, alternating "pit" and "flag".
    In:  -
    Out: list[TimelineEntry]
    """
    laps = (1, 2, 2, 3, 5, 5)
    return [TimelineEntry(10.0 * i, lap, "pit" if i % 2 else "flag", f"entry {i}", {}) for i, lap in enumerate(laps)]


def _texts(selection: tuple[tuple[TimelineEntry, ...], int]) -> tuple[list[str], int]:
    """A selection as its texts and the count left out.
    In:  selection  tuple[tuple[TimelineEntry, ...], int]
    Out:            tuple[list[str], int]
    """
    return [e.text for e in selection[0]], selection[1]


def test_lap_range_bounds_are_inclusive() -> None:
    """Entries of the first and the last lap of the range are returned."""
    assert _texts(select_entries(_entries(), 2, 3, "", 30)) == (["entry 1", "entry 2", "entry 3"], 0)


def test_zero_means_unbounded() -> None:
    """A bound of 0 leaves that side open."""
    assert _texts(select_entries(_entries(), 0, 0, "", 30)) == ([f"entry {i}" for i in range(6)], 0)
    assert _texts(select_entries(_entries(), 3, 0, "", 30)) == (["entry 3", "entry 4", "entry 5"], 0)
    assert _texts(select_entries(_entries(), 0, 2, "", 30)) == (["entry 0", "entry 1", "entry 2"], 0)


def test_kind_filter() -> None:
    """Only entries of the asked kind are returned."""
    assert _texts(select_entries(_entries(), 0, 0, "pit", 30)) == (["entry 1", "entry 3", "entry 5"], 0)
    assert _texts(select_entries(_entries(), 0, 0, "plan", 30)) == ([], 0)


def test_cap_keeps_the_newest_and_counts_the_rest() -> None:
    """Beyond the cap the newest entries are kept in time order and the older ones are counted."""
    assert _texts(select_entries(_entries(), 0, 0, "", 2)) == (["entry 4", "entry 5"], 4)
    assert _texts(select_entries(_entries(), 0, 0, "flag", 2)) == (["entry 2", "entry 4"], 1)


def test_empty_input() -> None:
    """No entries give an empty selection."""
    assert select_entries([], 0, 0, "", 30) == ((), 0)


def test_file_name_carries_the_local_start_stamp_and_the_kind() -> None:
    """The name is the session start in local time, the kind and the timeline suffix."""
    assert file_name("race", START_UNIX) == "20260102_130405_race.timeline.jsonl"


def test_matching_file_within_and_outside_the_margin() -> None:
    """A file of the same kind matches while its stamp is within the margin."""
    names = ["20260102_130405_race.timeline.jsonl", "20260102_130405_race.yaml"]
    assert matching_file(names, "race", START_UNIX + 100.0, 120.0) == names[0]
    assert matching_file(names, "race", START_UNIX - 120.0, 120.0) == names[0]
    assert matching_file(names, "race", START_UNIX + 121.0, 120.0) is None
    assert matching_file([], "race", START_UNIX, 120.0) is None


def test_matching_file_ignores_other_kinds_and_renamed_files() -> None:
    """Another session kind and a file set aside by a replay do not match."""
    names = ["20260102_130405_qualify.timeline.jsonl", "20260102_130405_race.timeline.1.jsonl"]
    assert matching_file(names, "race", START_UNIX, 120.0) is None
    assert matching_file(names, "qualify", START_UNIX, 120.0) == names[0]


def test_matching_file_is_the_nearest_of_two() -> None:
    """With two files inside the margin the one with the nearer stamp is taken."""
    names = ["20260102_130405_race.timeline.jsonl", "20260102_130505_race.timeline.jsonl"]
    assert matching_file(names, "race", START_UNIX + 40.0, 120.0) == names[1]
    assert matching_file(names, "race", START_UNIX + 20.0, 120.0) == names[0]


def test_files_in_window_are_of_the_kind_and_newest_first() -> None:
    """Files of the kind stamped inside the window, bounds included, are returned with the newest first."""
    names = [
        "20260102_130405_race.timeline.jsonl",
        "20260102_130505_race.timeline.jsonl",
        "20260102_131505_race.timeline.jsonl",
        "20260102_130455_qualify.timeline.jsonl",
        "20260102_130455_race.timeline.1.jsonl",
        "20260102_130455_race.yaml",
    ]
    assert files_in_window(names, "race", START_UNIX, START_UNIX + 60.0) == [names[1], names[0]]
    assert files_in_window(names, "race", START_UNIX + 1.0, START_UNIX + 59.0) == []
    assert files_in_window(names, "qualify", START_UNIX, START_UNIX + 60.0) == [names[3]]
