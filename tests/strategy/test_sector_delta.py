import pytest

from ai_race_engineer.strategy.sector_delta import splits


def test_splits_of_a_complete_lap() -> None:
    """Sector 2 and 3 come from the cumulative times."""
    assert splits(32.25, 80.5, 107.0) == (32.25, 48.25, 26.5)


@pytest.mark.parametrize(
    ("sector1_s", "sector2_cum_s", "lap_s", "expected"),
    [
        (32.5, None, 107.0, (32.5, None, None)),
        (None, 80.5, 107.0, (None, None, 26.5)),
        (32.5, 80.5, None, (32.5, 48.0, None)),
        (None, None, None, (None, None, None)),
    ],
)
def test_splits_with_missing_times(
    sector1_s: float | None, sector2_cum_s: float | None, lap_s: float | None, expected: tuple[float | None, ...]
) -> None:
    """A sector that cannot be derived from the times the game gives is None."""
    assert splits(sector1_s, sector2_cum_s, lap_s) == expected
