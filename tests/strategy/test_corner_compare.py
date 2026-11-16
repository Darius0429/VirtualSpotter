import dataclasses

import pytest

from ai_race_engineer.strategy.corner_card import lap_cards
from ai_race_engineer.strategy.corner_compare import FindingLimits, compare, line_wider_m
from ai_race_engineer.strategy.lap_segments import LapSamples, lap_samples
from tests.oval import LENGTH_M, oval_lap
from tests.strategy.test_corner_card import LIMITS, SEGMENTS

FINDINGS: FindingLimits = FindingLimits(distance_m=5.0, speed_kph=2.0, decel_g=0.1, time_s=0.15)


def _lap(lap_time_s: float, brake_zone_m: float) -> LapSamples:
    """An oval lap braked over a given distance.
    In:  lap_time_s    float  s
         brake_zone_m  float  m
    Out:               LapSamples
    """
    return lap_samples([s.player for s in oval_lap(0, brake_zone_m=brake_zone_m)], LENGTH_M, lap_time_s)


def test_earlier_and_softer_braking_is_found_and_costs_time_before_the_apex() -> None:
    """Braking 20 m earlier and softer loses time on the way to the apex and none after it."""
    best, mine = _lap(38.2, 100.0), _lap(38.2 * 1.006, 120.0)
    delta = compare(lap_cards(mine, SEGMENTS, LIMITS)[1], lap_cards(best, SEGMENTS, LIMITS)[1], 0.0, FINDINGS)
    assert delta.deltas["brake_point_m"] == pytest.approx(-20.0, abs=4.0)
    assert delta.findings[0].startswith("braked ") and delta.findings[0].endswith(" m earlier")
    assert delta.findings[1] == "braked softer, 1.5 g against 1.8 g"
    assert delta.time_s > 0.0
    assert delta.to_apex_s + delta.from_apex_s == pytest.approx(delta.time_s)
    assert delta.to_apex_s > 5 * abs(delta.from_apex_s)


def test_same_lap_has_no_findings() -> None:
    """A lap compared with itself differs nowhere."""
    card = lap_cards(_lap(38.2, 100.0), SEGMENTS, LIMITS)[1]
    delta = compare(card, card, 0.0, FINDINGS)
    assert (delta.time_s, delta.findings) == (0.0, ())
    assert set(delta.deltas.values()) == {0.0}


def test_straight_is_compared_on_time_and_speed_only() -> None:
    """A straight has no apex split and no line."""
    fast, slow = (lap_cards(_lap(t, 100.0), SEGMENTS, LIMITS)[0] for t in (38.2, 40.0))
    delta = compare(slow, fast, None, FINDINGS)
    assert (delta.to_apex_s, delta.from_apex_s, delta.line_wider_m) == (None, None, None)
    assert delta.time_s == pytest.approx(slow.time_s - fast.time_s)


def test_line_further_from_the_inside_is_wider() -> None:
    """At the apex of turn 1 the inside is towards negative x: a line 3 m further in x is 3 m wider, one 3 m
    less is tighter."""
    best = _lap(38.2, 100.0)
    wide, tight = (dataclasses.replace(best, x_m=best.x_m + shift) for shift in (3.0, -3.0))
    assert line_wider_m(wide, best, 400.0) == pytest.approx(3.0, abs=0.1)
    assert line_wider_m(tight, best, 400.0) == pytest.approx(-3.0, abs=0.1)
    delta = compare(*(lap_cards(lap, SEGMENTS, LIMITS)[1] for lap in (wide, best)), 3.0, FINDINGS)
    assert "3.0 m wider at the apex" in delta.findings
