import numpy as np
import pytest

from ai_race_engineer.strategy.lap_segments import lap_samples
from ai_race_engineer.strategy.track_map import Segment, curvature_per_m, track_map
from tests.oval import LENGTH_M, RADIUS_M, TURN_1_M, TURN_2_M, oval_lap

EDGE_TOLERANCE_M: float = 30.0


def _map(merge_gap_m: float = 50.0) -> tuple[Segment, ...]:
    """Track map of the oval with a 100 m entry pad and a 50 m exit pad.
    In:  merge_gap_m  float  m
    Out:              tuple[Segment, ...]
    """
    samples = lap_samples([s.player for s in oval_lap(0)], 30.0)
    return track_map(
        samples.distance_m,
        samples.x_m,
        samples.z_m,
        LENGTH_M,
        step_m=5.0,
        smooth_m=55.0,
        max_radius_m=400.0,
        min_turn_deg=18.0,
        entry_pad_m=100.0,
        exit_pad_m=50.0,
        merge_gap_m=merge_gap_m,
    )


def test_curvature_of_a_circle_is_one_over_its_radius() -> None:
    """A closed circle of 100 m radius has a curvature of 0.01 per m everywhere."""
    angle = np.linspace(0.0, 2 * np.pi, 126, endpoint=False)
    step_m = 2 * np.pi * RADIUS_M / 126
    kappa = curvature_per_m(RADIUS_M * np.sin(angle), -RADIUS_M * np.cos(angle), step_m, 55.0)
    assert np.abs(kappa) == pytest.approx(1 / RADIUS_M, rel=0.02)


def test_oval_is_split_into_two_turns_and_three_straights() -> None:
    """The segments cover the lap in driving order: the padded turns with the straights between them, the
    start/finish straight in two parts."""
    segments = _map()
    assert [s.name for s in segments] == ["straight 1", "turn 1", "straight 2", "turn 2", "straight 3"]
    assert [s.corner for s in segments] == [False, True, False, True, False]
    assert segments[0].start_m == 0.0
    assert segments[-1].end_m == LENGTH_M
    assert all(a.end_m == b.start_m for a, b in zip(segments, segments[1:], strict=False))
    for turn, (start_m, end_m) in ((segments[1], TURN_1_M), (segments[3], TURN_2_M)):
        assert turn.start_m == pytest.approx(start_m - 100.0, abs=EDGE_TOLERANCE_M)
        assert turn.end_m == pytest.approx(end_m + 50.0, abs=EDGE_TOLERANCE_M)
        assert start_m < turn.apex_m < end_m
    assert segments[2].apex_m is None


def test_turns_closer_than_the_merge_gap_become_one() -> None:
    """With a merge gap longer than the back straight the two turns are one segment."""
    segments = _map(merge_gap_m=600.0)
    assert [s.name for s in segments] == ["straight 1", "turn 1", "straight 2"]
    assert segments[1].end_m == pytest.approx(TURN_2_M[1] + 50.0, abs=EDGE_TOLERANCE_M)
