import math

import pytest

from ai_race_engineer.strategy.gaps import lapped_status, relative_distance_m, relative_time_gap_s


def test_relative_distance_ahead_and_behind() -> None:
    """Positive ahead, negative behind."""
    assert relative_distance_m(6000.0, 1000.0, 1200.0) == pytest.approx(200.0)
    assert relative_distance_m(6000.0, 1000.0, 700.0) == pytest.approx(-300.0)


def test_relative_distance_wraps_across_line() -> None:
    """Cars either side of the start/finish line are close, not a lap apart."""
    assert relative_distance_m(6000.0, 5900.0, 100.0) == pytest.approx(200.0)
    assert relative_distance_m(6000.0, 100.0, 5900.0) == pytest.approx(-200.0)


def test_relative_distance_exactly_half_lap() -> None:
    """Exactly half a lap stays within +/- track/2."""
    assert abs(relative_distance_m(6000.0, 0.0, 3000.0)) == pytest.approx(3000.0)


def test_time_gap() -> None:
    """Distance over mean speed, sign ignored."""
    assert relative_time_gap_s(-100.0, 40.0, 60.0) == pytest.approx(2.0)


def test_time_gap_both_stopped() -> None:
    """Both cars stopped gives infinity."""
    assert relative_time_gap_s(100.0, 0.0, 0.0) == math.inf


@pytest.mark.parametrize(
    ("other", "player", "expected"),
    [(11.0, 10.0, 1), (10.99, 10.0, 0), (9.0, 10.0, -1), (9.01, 10.0, 0), (10.5, 10.0, 0)],
)
def test_lapped_status_boundaries(other: float, player: float, expected: int) -> None:
    """+1 / -1 from exactly one lap apart, 0 within a lap."""
    assert lapped_status(other, player) == expected
