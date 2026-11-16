import dataclasses

import numpy as np
import pytest

from ai_race_engineer.strategy.corner_card import CardLimits, SegmentCard, lap_cards
from ai_race_engineer.strategy.lap_segments import LapSamples, lap_samples
from ai_race_engineer.strategy.lap_summary import lap_shares, theoretical_best_s
from ai_race_engineer.strategy.track_map import Segment
from tests.oval import ARC_M, FUEL_PER_LAP_L, LENGTH_M, TURN_1_M, oval_lap

LAP_TIME_S: float = 40.0
NATURAL_LAP_S: float = 38.2
SEGMENTS: tuple[Segment, ...] = (
    Segment("straight 1", False, 0.0, 100.0, None),
    Segment("turn 1", True, 100.0, 650.0, 400.0),
    Segment("straight 2", False, 650.0, LENGTH_M, None),
)
LIMITS: CardLimits = CardLimits(
    brake_on=0.05,
    throttle_on=0.05,
    full_throttle=0.95,
    throttle_lift=0.05,
    turn_in_g=0.5,
    lock_ratio=0.85,
    spin_ratio=1.08,
    search_m=150.0,
    after_exit_m=150.0,
)


def _samples() -> LapSamples:
    """The oval lap.
    In:  -
    Out: LapSamples
    """
    return lap_samples([s.player for s in oval_lap(0)], LENGTH_M, LAP_TIME_S)


def _cards(samples: LapSamples) -> tuple[SegmentCard, ...]:
    """The lap split into SEGMENTS.
    In:  samples  LapSamples
    Out:          tuple[SegmentCard, ...]
    """
    return lap_cards(samples, SEGMENTS, LIMITS)


def test_segment_times_fuel_and_energy_add_up_to_the_lap() -> None:
    """The segments share out the whole lap."""
    cards = _cards(_samples())
    assert sum(c.time_s for c in cards) == pytest.approx(LAP_TIME_S)
    assert sum(c.fuel_l for c in cards) == pytest.approx(FUEL_PER_LAP_L, abs=0.01)
    assert sum(c.energy for c in cards) == pytest.approx(FUEL_PER_LAP_L / 100, abs=0.0001)


def test_straight_has_speeds_but_no_corner_phases() -> None:
    """A straight has speeds and usage only."""
    straight = _cards(_samples())[0]
    assert (straight.entry, straight.mid, straight.exit, straight.time_to_apex_s) == (None, None, None, None)
    assert (straight.min_kph, straight.lowest_gear) == (pytest.approx(216.0), 6)


def test_corner_entry() -> None:
    """The braking is found before the corner and measured to the apex of the map."""
    entry = _cards(_samples())[1].entry
    assert entry.brake_point_m == pytest.approx(TURN_1_M[0] - 100.0, abs=4.0)
    assert entry.brake_to_apex_m == pytest.approx(400.0 - entry.brake_point_m)
    assert entry.speed_at_brake_kph == pytest.approx(216.0, abs=3.0)
    assert entry.lift_coast_m == 0.0
    assert entry.peak_decel_g == pytest.approx(1.83, abs=0.05)
    assert entry.time_to_peak_decel_s == pytest.approx(0.0, abs=0.06)
    assert entry.braking_m == pytest.approx(100.0, abs=6.0)
    assert entry.release_to_apex_m == pytest.approx(150.0, abs=4.0)
    assert entry.trail_brake_m == 0.0
    assert (entry.downshifts, entry.lockup_m, entry.lockup_axle) == (1, 0.0, "")


def test_corner_mid() -> None:
    """The minimum speed, its place against the apex, the cornering load and the coasting."""
    mid = _cards(_samples())[1].mid
    assert mid.min_speed_kph == pytest.approx(108.0)
    assert mid.min_speed_m == pytest.approx(TURN_1_M[0], abs=4.0)
    assert mid.min_speed_to_apex_m == pytest.approx(150.0, abs=4.0)
    assert mid.peak_lat_g == pytest.approx(0.92, abs=0.01)
    assert mid.coast_s == pytest.approx(ARC_M / 2 / 30.0 * LAP_TIME_S / NATURAL_LAP_S, abs=0.15)
    assert mid.countersteer_s == 0.0


def test_corner_exit() -> None:
    """Throttle pick-up and full throttle are measured from the apex of the map; the speed is taken again after the
    corner."""
    card = _cards(_samples())[1]
    assert card.exit.throttle_on_after_apex_m == pytest.approx(TURN_1_M[0] + ARC_M / 2 - 400.0, abs=4.0)
    assert card.exit.full_throttle_after_apex_m == card.exit.throttle_on_after_apex_m
    assert card.exit.throttle_acceptance == pytest.approx(1.0)
    assert (card.exit.part_throttle_s, card.exit.throttle_lifts, card.exit.wheelspin_s) == (0.0, 0, 0.0)
    assert card.exit.after_exit_kph == pytest.approx(216.0)
    assert card.time_to_apex_s == pytest.approx(8.4, abs=0.3)


def test_corner_taken_flat_has_no_entry_and_no_throttle_points() -> None:
    """Without braking and without lifting there is no entry and no pick-up."""
    samples = _samples()
    flat = dataclasses.replace(samples, brake=np.zeros_like(samples.brake), throttle=np.ones_like(samples.throttle))
    card = _cards(flat)[1]
    assert card.entry is None
    assert (card.exit.throttle_on_after_apex_m, card.exit.full_throttle_after_apex_m) == (None, None)


def test_locking_and_spinning_wheels_are_found() -> None:
    """A front wheel far slower than the car under braking is a lock-up; a rear wheel far faster after the
    minimum speed is wheelspin."""
    samples = _samples()
    ratio = samples.wheel_ratio.copy()
    ratio[(samples.distance_m > 180) & (samples.distance_m < 220), 0] = 0.6
    ratio[(samples.distance_m > 420) & (samples.distance_m < 480), 3] = 1.3
    card = _cards(dataclasses.replace(samples, wheel_ratio=ratio))[1]
    assert card.entry.lockup_m == pytest.approx(40.0, abs=6.0)
    assert card.entry.lockup_axle == "front"
    assert card.exit.wheelspin_s == pytest.approx(60.0 / 30.0, abs=0.2)


def test_lap_shares_and_theoretical_best() -> None:
    """The pedal shares add up to the lap; the theoretical best takes the best pass of every segment."""
    samples = _samples()
    shares = lap_shares(samples, LIMITS)
    assert shares.full_throttle + shares.braking + shares.coasting == pytest.approx(1.0)
    assert shares.braking == pytest.approx(2 * (100.0 / 43.0) / NATURAL_LAP_S, rel=0.1)
    fast, slow = _cards(samples), _cards(lap_samples([s.player for s in oval_lap(0)], LENGTH_M, 42.0))
    mixed = (slow[0], fast[1], slow[2])
    assert theoretical_best_s([mixed, slow]) == pytest.approx(slow[0].time_s + fast[1].time_s + slow[2].time_s)


def test_braking_for_the_corner_before_is_not_this_corners_brake_point() -> None:
    """A dab on the brake followed by full throttle belongs to the corner before; a second application after
    coasting belongs to this corner."""
    samples = _samples()
    in_search = (samples.distance_m > 20.0) & (samples.distance_m < 40.0)
    dab = dataclasses.replace(samples, brake=np.where(in_search, 1.0, samples.brake))
    assert _cards(dab)[1].entry.brake_point_m == pytest.approx(TURN_1_M[0] - 100.0, abs=4.0)
    gap = (samples.distance_m > 190.0) & (samples.distance_m < 210.0)
    twice = dataclasses.replace(samples, brake=np.where(gap, 0.0, samples.brake))
    assert _cards(twice)[1].entry.brake_point_m == pytest.approx(TURN_1_M[0] - 100.0, abs=4.0)
