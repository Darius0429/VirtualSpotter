import pytest

from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitLane, PitModel, PitRates
from ai_race_engineer.domains.common.analysis.strategy.race_finish import (
    CarInput,
    flag_time_s,
    laps_in_time,
    project_race,
    stops_needed,
    time_for_laps_s,
)

PIT: PitModel = PitModel(
    PitRates(fuel_rate_l_per_s=2.0, energy_rate_per_s=0.02, scale=1.0), PitLane(30.0, 500.0), 5000.0
)


def _car(
    slot_id: int,
    progress_laps: float,
    virtual_energy: float | None = None,
    energy_per_lap: float | None = None,
    fuel_l: float | None = None,
    fuel_l_per_lap: float | None = None,
    fuel_capacity_l: float | None = None,
    fuel_per_energy: float | None = None,
    lap_time_s: float = 100.0,
    finished: bool = False,
    class_name: str = "Hyper",
) -> CarInput:
    """Projection input; without resources the car's usage is unknown. The lane loss is 20 s at a 100 s lap.
    In:  slot_id          int
         progress_laps    float         laps
         virtual_energy   float | None  fraction
         energy_per_lap   float | None  fraction/lap
         fuel_l           float | None  litres
         fuel_l_per_lap   float | None  litres/lap
         fuel_capacity_l  float | None  litres
         fuel_per_energy  float | None  litres per 1 % energy
         lap_time_s       float         s/lap
         finished         bool
         class_name       str
    Out:                  CarInput
    """
    return CarInput(
        slot_id,
        class_name,
        progress_laps,
        lap_time_s,
        finished,
        False,
        fuel_l,
        fuel_capacity_l,
        virtual_energy,
        fuel_l_per_lap,
        energy_per_lap,
        fuel_per_energy,
    )


@pytest.mark.parametrize("fuel_l", [200.0, 12.0])
def test_strategy_follows_energy_whatever_the_fuel(fuel_l: float) -> None:
    """Stint, laps on board, stops and pit lap come from energy, with fuel lasting longer or shorter."""
    car = _car(1, 0.0, 0.5, 0.125, fuel_l, 2.0, 100.0, 0.8)
    mine = project_race([car], None, 10, PIT).cars[1]
    assert mine.resource == "energy"
    assert mine.laps_on_board == 4.0
    assert mine.stops_remaining == 1
    assert mine.pit_lap == 4
    assert stops_needed(car, 12.0) == 1
    assert stops_needed(car, 12.5) == 2


def test_fuel_runs_out_first_is_only_a_warning() -> None:
    """Fuel for fewer laps than energy sets the warning and a negative margin; stops are unchanged."""
    mine = project_race([_car(1, 0.0, 0.5, 0.125, 6.0, 2.0, 100.0, 0.8)], None, 10, PIT).cars[1]
    assert mine.fuel_runs_out_first
    assert mine.fuel_laps_on_board == 3.0
    assert mine.fuel_margin_laps == -1.0
    assert (mine.stops_remaining, mine.pit_lap) == (1, 4)


def test_fuel_lasting_longer_gives_positive_margin() -> None:
    """Fuel for more laps than energy is no warning."""
    mine = project_race([_car(1, 0.0, 0.5, 0.125, 20.0, 2.0, 100.0, 0.8)], None, 10, PIT).cars[1]
    assert not mine.fuel_runs_out_first
    assert mine.fuel_margin_laps == 6.0


def test_fuel_fallback_without_virtual_energy() -> None:
    """A car without virtual energy is planned on fuel."""
    mine = project_race([_car(1, 0.0, fuel_l=20.0, fuel_l_per_lap=2.5, fuel_capacity_l=50.0)], None, 10, PIT).cars[1]
    assert mine.resource == "fuel"
    assert mine.laps_on_board == 8.0
    assert (mine.stops_remaining, mine.pit_lap) == (1, 8)
    assert mine.energy_to_add is None
    assert mine.fuel_to_add_l == 5.0
    assert mine.fuel_margin_laps is None
    assert mine.finish_time_s == pytest.approx(1000.0 + 20.0 + 2.5)


def test_usage_unknown_gives_no_stops() -> None:
    """Without usage there is no resource, no stop and nothing to add."""
    mine = project_race([_car(1, 0.0)], None, 10, PIT).cars[1]
    assert mine.resource is None
    assert (mine.stops_remaining, mine.laps_on_board, mine.pit_lap) == (0, None, None)
    assert (mine.energy_to_add, mine.fuel_to_add_l) == (None, None)
    assert mine.finish_time_s == 1000.0


def test_fuel_to_add_follows_energy_to_add() -> None:
    """Fuel to add is the energy to add in percent times the litres per percent."""
    mine = project_race([_car(1, 0.0, 0.5, 0.125, 20.0, 2.0, 100.0, 0.8)], None, 10, PIT).cars[1]
    assert mine.energy_to_add == 0.75
    assert mine.fuel_to_add_l == pytest.approx(0.75 * 100 * 0.8)
    assert mine.finish_time_s == pytest.approx(1000.0 + 20.0 + 37.5)


def test_net_leader_is_the_car_without_a_stop() -> None:
    """P1 with a stop left loses the lead to P6 without one; P6 sets the flag."""
    first = _car(1, 10.5, 0.125, 0.125)
    sixth = _car(6, 10.3, 1.0, 0.125)
    projection = project_race([first, sixth], 300.0, None, PIT)
    assert projection.leader_slot_id == 6
    assert projection.flag_time_s == pytest.approx(370.0)
    assert (projection.cars[6].place, projection.cars[1].place) == (1, 2)
    assert projection.cars[6].laps_remaining == pytest.approx(3.7)
    assert projection.cars[1].laps_remaining == pytest.approx(3.5)
    assert projection.cars[1].stops_remaining == 1


def test_timed_leader_crossing_before_zero_drives_one_more_lap() -> None:
    """A crossing 1 s before the clock runs out is not the last one."""
    projection = project_race([_car(1, 0.5)], 51.0, None, PIT)
    assert projection.flag_time_s == pytest.approx(150.0)
    assert projection.cars[1].laps_remaining == pytest.approx(1.5)


def test_timed_leader_crossing_after_zero_finishes_there() -> None:
    """The first crossing after the clock runs out is the finish."""
    projection = project_race([_car(1, 0.5)], 49.0, None, PIT)
    assert projection.flag_time_s == pytest.approx(50.0)
    assert projection.cars[1].laps_remaining == pytest.approx(0.5)


def test_lap_limited_race() -> None:
    """Without a clock the flag falls when the first car completes max_laps."""
    assert flag_time_s([_car(1, 0.5), _car(2, 0.25)], None, 3, PIT) == (250.0, 1)


@pytest.mark.parametrize(("remaining_s", "expected_s"), [(49.0, 50.0), (1000.0, 250.0)])
def test_both_limits_earlier_flag_wins(remaining_s: float, expected_s: float) -> None:
    """With a clock and a lap limit the earlier flag counts."""
    assert flag_time_s([_car(1, 0.5)], remaining_s, 3, PIT)[0] == pytest.approx(expected_s)


def test_finished_car_ends_the_race() -> None:
    """Once a car has finished the flag is out: it has no laps left, a lapped car finishes on its next crossing."""
    projection = project_race([_car(1, 10.0, finished=True), _car(2, 8.4)], -20.0, None, PIT)
    assert (projection.flag_time_s, projection.leader_slot_id) == (0.0, 1)
    assert projection.cars[1].laps_remaining == 0.0
    assert projection.cars[2].laps_remaining == pytest.approx(0.6)
    assert projection.cars[2].finish_time_s == pytest.approx(60.0)
    assert (projection.cars[1].place, projection.cars[2].place) == (1, 2)


def test_lapped_car_finishes_on_first_crossing_after_the_flag() -> None:
    """A slower car does not complete the leader's distance: it stops at its first crossing after the flag."""
    projection = project_race([_car(1, 5.0), _car(2, 3.9, lap_time_s=110.0)], None, 8, PIT)
    assert projection.flag_time_s == 300.0
    assert projection.cars[2].laps_remaining == pytest.approx(3.1)
    assert projection.cars[2].finish_time_s == pytest.approx(341.0)


def test_class_places_rank_within_class() -> None:
    """Projected class places count only cars of the same class."""
    cars = [_car(1, 5.0), _car(2, 4.5, class_name="GT3"), _car(3, 4.0), _car(4, 3.0, class_name="GT3")]
    projection = project_race(cars, None, 8, PIT)
    assert {slot_id: (c.place, c.class_place) for slot_id, c in projection.cars.items()} == {
        1: (1, 1),
        2: (2, 1),
        3: (3, 2),
        4: (4, 2),
    }


@pytest.mark.parametrize("laps", [0.5, 1.0, 1.5, 5.0, 9.0, 9.5, 12.0, 20.0])
def test_laps_in_time_inverts_time_for_laps(laps: float) -> None:
    """Driving the time needed for n laps covers n laps, on both sides of a stop."""
    car = _car(1, 0.0, 0.125, 0.125, fuel_per_energy=0.8)
    assert laps_in_time(car, time_for_laps_s(car, laps, PIT), PIT) == pytest.approx(laps)


def test_car_stays_at_the_pit_lap_during_its_stop() -> None:
    """While the time of a stop runs, the car makes no progress past the lap where the stop is due."""
    car = _car(1, 0.0, 0.125, 0.125)
    assert laps_in_time(car, 110.0, PIT) == pytest.approx(1.0)


@pytest.mark.parametrize(("fuel_l", "warns"), [(7.5, True), (7.625, False), (8.5, False)])
def test_fuel_warning_needs_more_than_the_tolerance(fuel_l: float, warns: bool) -> None:
    """With 4 laps of energy and a 0.2 lap tolerance, fuel for 3.75 laps warns and fuel for 3.8125 laps does not;
    the margin itself is always reported."""
    mine = project_race([_car(1, 0.0, 0.5, 0.125, fuel_l, 2.0, 100.0, 0.8)], None, 10, PIT, 0.2).cars[1]
    assert mine.fuel_runs_out_first is warns
    assert mine.fuel_margin_laps == fuel_l / 2.0 - 4.0


def test_usage_above_capacity_stops_every_lap() -> None:
    """A car using 120 % of energy per lap has a stint of one lap instead of none: no division by zero."""
    car = _car(1, 0.0, 0.6, 1.2)
    assert stops_needed(car, 0.5) == 0
    assert stops_needed(car, 3.5) == 3
    mine = project_race([car], None, 4, PIT).cars[1]
    assert (mine.laps_on_board, mine.stops_remaining, mine.pit_lap) == (0.5, 4, 0)
    assert laps_in_time(car, time_for_laps_s(car, 2.5, PIT), PIT) == pytest.approx(2.5)


BASE_PIT: PitModel = PitModel(
    PitRates(fuel_rate_l_per_s=2.0, energy_rate_per_s=0.02, scale=1.0, base_s=3.0), PitLane(30.0, 500.0), 5000.0
)


@pytest.mark.parametrize(("laps", "expected_s"), [(1.0, 100.0), (5.0, 548.0), (12.0, 1314.75)])
def test_base_time_counts_once_per_stop(laps: float, expected_s: float) -> None:
    """With one lap on board and 8-lap stints: no stop, no base; each stop adds the lane loss and the 3 s base,
    the filling time follows the total amount."""
    assert time_for_laps_s(_car(1, 0.0, 0.125, 0.125), laps, BASE_PIT) == pytest.approx(expected_s)


@pytest.mark.parametrize("laps", [0.5, 1.0, 1.5, 5.0, 9.0, 9.5, 12.0, 20.0])
def test_laps_in_time_inverts_time_for_laps_with_a_base(laps: float) -> None:
    """With a fixed time per stop, driving the time needed for n laps still covers n laps on both sides of a stop."""
    car = _car(1, 0.0, 0.125, 0.125, fuel_per_energy=0.8)
    assert laps_in_time(car, time_for_laps_s(car, laps, BASE_PIT), BASE_PIT) == pytest.approx(laps)


def test_bahrain_hour_end_to_end() -> None:
    """The owner's example: 60 minutes at 1:47.1 using 4.5 % per lap, 28.0 s lane against 8.3 s at race pace, 1.0 s
    plus 0.4 s per 1 %: 34 laps with one stop of 22.2 s standing, 3683.3 s in total."""
    pit = PitModel(PitRates(3.0, 0.025, 1.0, base_s=1.0), PitLane(28.0, 417.4), 5386.8)
    mine = project_race([_car(1, 0.0, 1.0, 0.045, lap_time_s=107.1)], 3600.0, None, pit).cars[1]
    assert (mine.laps_remaining, mine.stops_remaining, mine.pit_lap) == (34.0, 1, 22)
    assert mine.energy_to_add == pytest.approx(0.53)
    assert mine.finish_time_s == pytest.approx(3683.3, abs=0.05)
