import pytest

from ai_race_engineer.domains.common.analysis.strategy.pit_time import (
    PitLane,
    PitModel,
    PitRates,
    PitStopSample,
    calibrate,
    fill_time_s,
    lane_loss_s,
    measured_pit_lane,
    service_time_s,
)
from ai_race_engineer.domains.common.data.pit_estimate import PitEstimate

DEFAULT: PitRates = PitRates(fuel_rate_l_per_s=1.8, energy_rate_per_s=0.0193, scale=1.55)
RATES: PitRates = PitRates(fuel_rate_l_per_s=2.0, energy_rate_per_s=0.02, scale=1.5)
BASE_RATES: PitRates = PitRates(fuel_rate_l_per_s=2.0, energy_rate_per_s=0.02, scale=1.5, base_s=1.0)


def _sample(
    is_player: bool = True,
    lane_time_s: float = 31.0,
    lane_length_m: float = 470.0,
    stationary_s: float = 6.8,
    energy_added: float = 0.085,
    fuel_added_l: float = 7.9,
    estimate: PitEstimate | None = None,
) -> PitStopSample:
    """Pit stop sample with the owner's measured stop as defaults.
    In:  is_player      bool
         lane_time_s    float               s
         lane_length_m  float               m
         stationary_s   float               s
         energy_added   float               fraction
         fuel_added_l   float               litres
         estimate       PitEstimate | None
    Out:                PitStopSample
    """
    return PitStopSample(
        0 if is_player else 5, is_player, lane_time_s, lane_length_m, stationary_s, energy_added, fuel_added_l, estimate
    )


@pytest.mark.parametrize(("lap_time_s", "expected_s"), [(100.0, 31.0 - 47.03 / 463.53 * 100.0), (120.0, 18.8249)])
def test_lane_loss_is_lane_time_minus_race_pace_over_the_stretch(lap_time_s: float, expected_s: float) -> None:
    """31.0 s in the lane against the car's own pace over 470.3 of 4635.3 m."""
    pit = PitModel(DEFAULT, PitLane(31.0, 470.3), 4635.3)
    assert lane_loss_s(pit, lap_time_s) == pytest.approx(expected_s, abs=1e-3)


def test_service_time_is_the_slower_of_energy_and_fuel() -> None:
    """Both fill at once, so the longer one counts, times the scale."""
    assert service_time_s(0.1, 4.0, RATES) == pytest.approx(1.5 * 5.0)
    assert service_time_s(0.02, 8.0, RATES) == pytest.approx(1.5 * 4.0)


def test_service_time_zero_when_nothing_to_add() -> None:
    """Nothing or a surplus to add takes no time."""
    assert service_time_s(0.0, 0.0, RATES) == 0.0
    assert service_time_s(-0.2, -5.0, RATES) == 0.0


def test_calibrate_without_samples_returns_default() -> None:
    """No data keeps every default."""
    assert calibrate([], DEFAULT) == DEFAULT


def test_calibrate_rates_from_player_estimate() -> None:
    """Rates are the added amounts over the game's estimated times."""
    rates = calibrate([_sample(estimate=PitEstimate(fuel_s=4.0, energy_s=4.4, total_s=4.4))], DEFAULT)
    assert rates.energy_rate_per_s == pytest.approx(0.085 / 4.4)
    assert rates.fuel_rate_l_per_s == pytest.approx(7.9 / 4.0)


def test_calibrate_scale_from_player_stop() -> None:
    """6.8 s measured against the game's 4.4 s gives a scale of 1.55."""
    rates = calibrate([_sample(estimate=PitEstimate(fuel_s=4.0, energy_s=4.4, total_s=4.4))], DEFAULT)
    assert rates.scale == pytest.approx(1.55, abs=0.01)


def test_calibrate_scale_without_estimate_uses_default_rates() -> None:
    """A player stop without an estimate still calibrates the scale, against the default rates."""
    rates = calibrate([_sample(stationary_s=10.0, energy_added=0.0965, fuel_added_l=0.0)], DEFAULT)
    assert (rates.fuel_rate_l_per_s, rates.energy_rate_per_s) == (1.8, 0.0193)
    assert rates.scale == pytest.approx(2.0)


def test_calibrate_scale_excludes_stop_with_other_work() -> None:
    """A stop whose estimate total exceeds max(fuel, ve) had tyres or repairs and does not set the scale."""
    rates = calibrate(
        [_sample(stationary_s=30.0, estimate=PitEstimate(fuel_s=4.0, energy_s=4.4, total_s=25.0))], DEFAULT
    )
    assert rates.scale == DEFAULT.scale
    assert rates.energy_rate_per_s == pytest.approx(0.085 / 4.4)


def test_calibrate_ignores_other_cars() -> None:
    """Other cars' samples change nothing."""
    other = _sample(is_player=False, stationary_s=20.0, estimate=PitEstimate(fuel_s=1.0, energy_s=1.0, total_s=1.0))
    assert calibrate([other], DEFAULT) == DEFAULT


def test_measured_pit_lane_averages_any_car() -> None:
    """Lane time and stretch are the means over all passes."""
    lane = measured_pit_lane([_sample(lane_time_s=30.0, lane_length_m=460.0), _sample(False, 32.0, 480.0)])
    assert lane == PitLane(31.0, 470.0)


def test_measured_pit_lane_none_when_empty() -> None:
    """No passes, no lane."""
    assert measured_pit_lane([]) is None


def test_fill_time_is_the_filling_without_the_base() -> None:
    """The filling time is what the service time was before the fixed time per stop: the slower fill times the
    scale, 0 for nothing or a surplus."""
    assert fill_time_s(0.1, 4.0, BASE_RATES) == pytest.approx(1.5 * 5.0)
    assert fill_time_s(0.02, 8.0, BASE_RATES) == pytest.approx(1.5 * 4.0)
    assert fill_time_s(-0.2, -5.0, BASE_RATES) == 0.0


def test_service_time_adds_the_base_once_times_the_scale() -> None:
    """A stop that adds energy or fuel takes the filling plus the fixed second, both scaled."""
    assert service_time_s(0.1, 4.0, BASE_RATES) == pytest.approx(1.5 * (5.0 + 1.0))


def test_service_time_has_no_base_when_nothing_is_added() -> None:
    """Nothing or a surplus to add takes no time, also with a base."""
    assert service_time_s(0.0, 0.0, BASE_RATES) == 0.0
    assert service_time_s(-0.2, -5.0, BASE_RATES) == 0.0


def test_calibrate_subtracts_and_returns_the_base() -> None:
    """The fixed second is taken off the game's fill times before the rates are formed, kept in the result and
    part of the model time the scale is measured against."""
    default = PitRates(fuel_rate_l_per_s=1.8, energy_rate_per_s=0.0193, scale=1.55, base_s=1.0)
    stop = _sample(stationary_s=5.4, estimate=PitEstimate(fuel_s=5.0, energy_s=5.4, total_s=5.4))
    rates = calibrate([stop], default)
    assert rates.energy_rate_per_s == pytest.approx(0.085 / 4.4)
    assert rates.fuel_rate_l_per_s == pytest.approx(7.9 / 4.0)
    assert rates.base_s == 1.0
    assert rates.scale == pytest.approx(1.0)
    assert calibrate([], default) == default
