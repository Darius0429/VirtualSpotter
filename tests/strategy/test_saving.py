import pytest

from ai_race_engineer.strategy.pit_time import PitRates
from ai_race_engineer.strategy.saving import (
    SavingNeed,
    check_saving,
    cost_s_per_energy,
    saving_for_extra_lap,
    saving_to_reach,
    saving_to_shorten_stop,
    saving_to_skip_stop,
)

RATES: PitRates = PitRates(fuel_rate_l_per_s=3.0, energy_rate_per_s=0.025, scale=1.0, base_s=1.0)
LAPS: list[tuple[float, float]] = [(107.0, 0.046), (107.2, 0.045), (107.4, 0.044), (107.6, 0.043), (107.8, 0.042)]


def test_saving_to_reach_laps() -> None:
    """13.7 laps on board against 14 needed is a saving of 2.1 % of the usage."""
    need = saving_to_reach(13.7 * 0.045, 0.045, 14.0)
    assert need.saving_fraction == pytest.approx(0.0214, abs=1e-4)
    assert need.usage_per_lap == pytest.approx(13.7 * 0.045 / 14)
    assert need.saving_per_lap == pytest.approx(0.045 - 13.7 * 0.045 / 14)
    assert need.laps == 14.0


def test_saving_for_the_extra_lap() -> None:
    """With 10.6 laps on board 0.3 laps into the lap, the car stops after 10 more crossings; one more needs the
    energy to last 10.7 laps."""
    need = saving_for_extra_lap(10.6 * 0.045, 0.045, 0.3)
    assert need.laps == pytest.approx(10.7)
    assert need.saving_fraction == pytest.approx(1 - 10.6 / 10.7)


def test_skipping_the_only_stop() -> None:
    """One stop left: the energy on board must last all remaining laps."""
    need = saving_to_skip_stop(0.5, 0.045, 1.0, 12.5, 1)
    assert need.usage_per_lap == pytest.approx(0.04)
    assert need.saving_fraction == pytest.approx(1 - 0.04 / 0.045)
    assert need.laps == 12.5


def test_skipping_one_of_two_stops() -> None:
    """Two stops left over 40.5 laps on 80 %: with one stop the best split is 17.5 + 23 laps, limited by a full load
    lasting the 23 laps (18.5 + 22 would be limited by the first stint at a lower usage)."""
    need = saving_to_skip_stop(0.8, 0.045, 1.0, 40.5, 2)
    assert need.usage_per_lap == pytest.approx(1 / 23)
    assert need.laps == 40.5


def test_no_stop_nothing_to_skip() -> None:
    """Without a stop there is nothing to skip."""
    assert saving_to_skip_stop(1.0, 0.045, 1.0, 10.0, 0) is None


def test_shortening_the_stop_by_two_seconds() -> None:
    """2 s less at 2.5 % per second is 5 % not added, spread over 20 laps: 0.25 % per lap."""
    need = saving_to_shorten_stop(0.045, 20.0, 2.0, RATES)
    assert need.saving_per_lap == pytest.approx(0.0025)
    assert need.usage_per_lap == pytest.approx(0.0425)
    assert need.saving_fraction == pytest.approx(0.0025 / 0.045)


def test_cost_from_the_slope_of_the_laps() -> None:
    """Laps that are 0.2 s slower per 0.1 % saved cost 200 s per unit of energy."""
    assert cost_s_per_energy(LAPS, 5, 0.01) == pytest.approx(200.0)


def test_cost_unknown_with_too_few_laps_or_too_little_spread() -> None:
    """Fewer laps than needed, or laps that all use about the same, give no cost."""
    assert cost_s_per_energy(LAPS[:4], 5, 0.01) is None
    assert cost_s_per_energy([(107.0 + i * 0.1, 0.045 - i * 0.00001) for i in range(5)], 5, 0.01) is None


def test_cost_unknown_when_the_laps_show_none() -> None:
    """Laps that got faster with less energy (traffic, mistakes on the thirsty laps) give no cost instead of a gain."""
    assert cost_s_per_energy([(lap_s, 0.09 - used) for lap_s, used in LAPS], 5, 0.01) is None


@pytest.mark.parametrize(("fraction", "within"), [(0.05, True), (0.051, False)])
def test_limit_of_what_is_realistic(fraction: float, within: bool) -> None:
    """A saving up to the limit of 5 % of the usage is realistic, above it is not."""
    need = SavingNeed(0.045 * (1 - fraction), 0.045 * fraction, fraction, 10.0)
    assert check_saving(need, 0.05, [], 5, 0.01).within_limit is within


def test_lowest_usage_shown_and_cost() -> None:
    """A target at or above the lowest lap driven is within what was shown; the cost is the slope times the saving.
    Without laps both are unknown."""
    shown = check_saving(SavingNeed(0.043, 0.002, 0.044, 10.0), 0.05, LAPS, 5, 0.01)
    assert shown.within_shown is True
    assert shown.cost_s_per_lap == pytest.approx(0.4)
    assert check_saving(SavingNeed(0.041, 0.004, 0.089, 10.0), 0.05, LAPS, 5, 0.01).within_shown is False
    unknown = check_saving(SavingNeed(0.043, 0.002, 0.044, 10.0), 0.05, [], 5, 0.01)
    assert (unknown.within_shown, unknown.cost_s_per_lap) == (None, None)
