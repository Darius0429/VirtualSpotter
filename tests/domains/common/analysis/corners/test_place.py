from ai_race_engineer.domains.common.analysis.corners.place import place
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment

SEGMENTS: tuple[Segment, ...] = (
    Segment("straight 1", False, 0.0, 300.0, None),
    Segment("turn 2", True, 300.0, 600.0, 450.0),
    Segment("straight 3", False, 600.0, 1000.0, None),
)


def test_corner_is_split_into_entry_apex_and_exit() -> None:
    """Within the window around the apex it is the apex, before it the entry, after it the exit."""
    assert place(SEGMENTS, 350.0, 20.0) == "entry of turn 2"
    assert place(SEGMENTS, 431.0, 20.0) == "apex of turn 2"
    assert place(SEGMENTS, 470.0, 20.0) == "apex of turn 2"
    assert place(SEGMENTS, 471.0, 20.0) == "exit of turn 2"


def test_straight_is_named_as_the_map_names_it() -> None:
    """A straight has no parts."""
    assert place(SEGMENTS, 700.0, 20.0) == "straight 3"


def test_without_a_map_there_is_no_place() -> None:
    """An empty track map gives an empty text."""
    assert place((), 350.0, 20.0) == ""
