import pytest

from ai_race_engineer.strategy.sectors import sector_deltas_s, sector_splits_s


def test_splits_sum_to_lap() -> None:
    """The three splits add up to the lap time."""
    assert sum(sector_splits_s(30.0, 65.0, 100.0)) == pytest.approx(100.0)


def test_split_arithmetic() -> None:
    """S2 = cumulative - S1, S3 = lap - cumulative."""
    assert sector_splits_s(30.0, 65.0, 100.0) == pytest.approx((30.0, 35.0, 35.0))


def test_delta_signs() -> None:
    """Negative delta means the other car is faster in that sector."""
    assert sector_deltas_s((29.5, 35.0, 35.2), (30.0, 34.8, 35.2)) == pytest.approx((-0.5, 0.2, 0.0))
