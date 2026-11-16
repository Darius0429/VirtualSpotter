import pytest

from ai_race_engineer.strategy.sector_delta import sector_delta, splits


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


def test_deltas_totals_and_worst_sector() -> None:
    """Deltas are player minus other; the total is their sum; the worst sector is the largest loss on the last lap."""
    delta = sector_delta((32.5, 48.5, 26.5), (32.25, 48.0, 26.25), (32.0, 48.75, 26.0), (32.0, 47.5, 26.0))
    assert delta.last_delta_s == (0.5, -0.25, 0.5)
    assert delta.best_delta_s == (0.25, 0.5, 0.25)
    assert delta.last_total_delta_s == 0.75
    assert delta.best_total_delta_s == 1.0
    assert delta.worst_sector == 1


def test_worst_sector_is_the_largest_loss() -> None:
    """Of several losing sectors the one with the most time lost is named."""
    assert sector_delta((32.5, 49.0, 26.5), (None,) * 3, (32.0, 48.0, 26.25), (None,) * 3).worst_sector == 2


def test_missing_sector_gives_no_delta_and_no_total() -> None:
    """A sector either car has no time for has no delta, and the total needs all three."""
    delta = sector_delta((32.5, None, None), (32.25, 48.0, 26.25), (32.0, 48.75, 26.0), (None, 47.5, 26.0))
    assert delta.last_delta_s == (0.5, None, None)
    assert delta.last_total_delta_s is None
    assert delta.best_delta_s == (None, 0.5, 0.25)
    assert delta.best_total_delta_s is None


def test_worst_sector_falls_back_to_the_best_sectors() -> None:
    """Without a loss on the last lap (no times) the best sectors decide."""
    none = (None, None, None)
    assert sector_delta(none, (32.25, 48.0, 26.5), none, (32.0, 48.0, 26.0)).worst_sector == 3


def test_no_worst_sector_when_faster_everywhere() -> None:
    """A player who loses nowhere has no worst sector."""
    delta = sector_delta((32.0, 48.0, 26.0), (32.0, 48.0, 26.0), (32.5, 48.5, 26.5), (32.25, 48.0, 26.0))
    assert delta.worst_sector is None
