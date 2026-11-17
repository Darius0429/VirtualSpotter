import pytest

from ai_race_engineer.domains.common.analysis.strategy.fuel import LapConsumption, amount_to_add, amount_to_finish


def test_no_seed_first_update_is_none() -> None:
    """Without a seed nothing is known before a lap is measured."""
    assert LapConsumption(window_laps=3).update(0, 50.0) is None


def test_seed_until_first_measured_lap() -> None:
    """The seed is returned until a lap is measured, then the measurement."""
    fuel = LapConsumption(window_laps=3, seed_per_lap=4.0)
    assert fuel.update(0, 50.0) == 4.0
    assert fuel.update(0, 48.0) == 4.0
    assert fuel.update(1, 47.0) == pytest.approx(3.0)


def test_one_lap_measured() -> None:
    """One completed lap gives its consumption."""
    fuel = LapConsumption(window_laps=3)
    fuel.update(0, 50.0)
    assert fuel.update(1, 46.5) == pytest.approx(3.5)


def test_window_drops_oldest() -> None:
    """Only the last window_laps laps are averaged."""
    fuel = LapConsumption(window_laps=2)
    fuel.update(0, 50.0)
    fuel.update(1, 40.0)
    fuel.update(2, 38.0)
    assert fuel.update(3, 36.0) == pytest.approx(2.0)


def test_same_lap_repeated_calls_add_nothing() -> None:
    """Calls within one lap do not add measurements."""
    fuel = LapConsumption(window_laps=3)
    fuel.update(0, 50.0)
    fuel.update(1, 47.0)
    assert fuel.update(1, 46.0) == pytest.approx(3.0)
    assert fuel.update(1, 45.0) == pytest.approx(3.0)


def test_refuel_lap_skipped() -> None:
    """A lap where the amount rose is not measured."""
    fuel = LapConsumption(window_laps=3)
    fuel.update(0, 50.0)
    fuel.update(1, 47.0)
    assert fuel.update(2, 90.0) == pytest.approx(3.0)
    assert fuel.update(3, 86.0) == pytest.approx(3.5)


def test_lap_decrease_resets_keeps_seed() -> None:
    """A lower lap count (new session) clears measurements but keeps the seed."""
    fuel = LapConsumption(window_laps=3, seed_per_lap=4.0)
    fuel.update(0, 50.0)
    fuel.update(1, 47.0)
    assert fuel.update(0, 60.0) == 4.0
    assert fuel.update(1, 58.0) == pytest.approx(2.0)


def test_energy_fraction() -> None:
    """Works with virtual energy fractions."""
    energy = LapConsumption(window_laps=3)
    energy.update(0, 1.0)
    energy.update(1, 0.969)
    assert energy.update(2, 0.938) == pytest.approx(0.031)


def test_amount_to_finish() -> None:
    """Per-lap amount times laps."""
    assert amount_to_finish(3.0, 10.5) == pytest.approx(31.5)


def test_amount_to_add_positive_and_surplus() -> None:
    """Positive when short, negative when in surplus."""
    assert amount_to_add(31.5, 20.0) == pytest.approx(11.5)
    assert amount_to_add(31.5, 40.0) == pytest.approx(-8.5)
