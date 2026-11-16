import pytest

from ai_race_engineer.strategy.lap_segments import lap_samples
from tests.oval import FAST_MPS, LENGTH_M, RADIUS_M, SLOW_MPS, oval_lap

LAP_TIME_S: float = 40.0


def test_samples_end_at_the_track_length_and_the_lap_time() -> None:
    """Distance and time start at 0 and are scaled to the track length and the timed lap."""
    samples = lap_samples([s.player for s in oval_lap(0)], LENGTH_M, LAP_TIME_S)
    assert (samples.distance_m[0], samples.time_s[0]) == (0.0, 0.0)
    assert samples.distance_m[-1] == pytest.approx(LENGTH_M)
    assert samples.time_s[-1] == pytest.approx(LAP_TIME_S)
    assert samples.distance_m[40] == pytest.approx(40 * 60.0 * 0.05, abs=1.0)


def test_accelerations_are_in_g_with_braking_negative() -> None:
    """Cornering at 30 m/s on a 100 m radius is 0.92 g; braking from 60 m/s is 1.83 g and negative."""
    samples = lap_samples([s.player for s in oval_lap(0)], LENGTH_M, LAP_TIME_S)
    assert samples.lat_g.max() == pytest.approx(SLOW_MPS**2 / RADIUS_M / 9.81)
    assert samples.lon_g.min() == pytest.approx(-FAST_MPS * (FAST_MPS - SLOW_MPS) / 100.0 / 9.81, rel=0.02)


def test_states_polled_twice_from_one_tick_are_dropped() -> None:
    """A state repeated because the telemetry had not advanced adds no sample."""
    players = [s.player for s in oval_lap(0)]
    doubled = [p for player in players for p in (player, player)]
    assert lap_samples(doubled, LENGTH_M, LAP_TIME_S).distance_m.size == len(players)


def test_wheel_ratio_is_one_without_wheel_telemetry() -> None:
    """A source without wheel speeds gives freely rolling wheels."""
    samples = lap_samples([s.player for s in oval_lap(0)], LENGTH_M, LAP_TIME_S)
    assert samples.wheel_ratio.shape == (samples.distance_m.size, 4)
    assert samples.wheel_ratio == pytest.approx(1.0)
