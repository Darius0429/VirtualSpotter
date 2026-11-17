import pytest

from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitLane, PitModel, PitRates
from ai_race_engineer.domains.common.analysis.strategy.race_finish import CarInput, project_race
from ai_race_engineer.domains.common.analysis.strategy.race_plan import RacePlan, min_start_amount, min_stops, race_plan

PIT: PitModel = PitModel(
    PitRates(fuel_rate_l_per_s=3.0, energy_rate_per_s=0.025, scale=1.0, base_s=1.0), PitLane(28.0, 417.4), 5386.8
)


def _plan(car: CarInput, remaining_s: float | None = 3600.0, max_laps: int | None = None) -> RacePlan | None:
    """Plan of a car racing alone.
    In:  car          CarInput
         remaining_s  float | None  s on the session clock
         max_laps     int | None    laps
    Out:              RacePlan | None
    """
    return race_plan(car, project_race([car], remaining_s, max_laps, PIT).cars[car.slot_id])


def _energy_car(virtual_energy: float = 1.0, energy_per_lap: float | None = 0.045) -> CarInput:
    """Energy car on the grid at a 107.1 s lap, using 0.7 litres per 1 % energy.
    In:  virtual_energy  float         fraction on board
         energy_per_lap  float | None  fraction/lap
    Out:                 CarInput
    """
    return CarInput(1, "Hyper", 0.0, 107.1, False, False, 70.0, 100.0, virtual_energy, None, energy_per_lap, 0.7)


def test_one_hour_race_from_a_full_start() -> None:
    """The owner's example: 60 minutes at 1:47.1 using 4.5 % per lap is 34 laps with one stop at the end of lap 22,
    53 % to add and at least 54 % at the start to keep it at one stop."""
    plan = _plan(_energy_car())
    assert (plan.race_laps, plan.stint_laps, plan.stops, plan.pit_lap) == (34, 22, 1, 22)
    assert plan.amount_on_board == 1.0
    assert plan.to_add_total == pytest.approx(0.53)
    assert plan.first_stop_add == pytest.approx(0.53)
    assert plan.fuel_to_add_l == pytest.approx(53 * 0.7)
    assert (plan.min_stops, plan.min_start_amount) == (1, pytest.approx(0.54))
    assert plan.average_lap_s == pytest.approx(108.33, abs=0.01)


def test_start_on_ten_percent() -> None:
    """A start on 10 % pits after 2 laps for a full load and needs a second stop."""
    plan = _plan(_energy_car(virtual_energy=0.10))
    assert (plan.stops, plan.pit_lap, plan.min_stops) == (2, 2, 1)
    assert plan.first_stop_add == pytest.approx(1.0 - 0.01)
    assert plan.to_add_total == pytest.approx(plan.race_laps * 0.045 - 0.10)


def test_no_stop_needed() -> None:
    """A race the load on board covers has no pit lap, nothing to add at a stop and a surplus in total."""
    plan = _plan(_energy_car(), remaining_s=None, max_laps=10)
    assert (plan.race_laps, plan.stops, plan.pit_lap, plan.first_stop_add) == (10, 0, None, 0.0)
    assert plan.to_add_total == pytest.approx(-0.55)
    assert (plan.min_stops, plan.min_start_amount) == (0, pytest.approx(0.45))


def test_unknown_usage_gives_no_plan() -> None:
    """Without a usage there is no plan."""
    assert _plan(_energy_car(energy_per_lap=None)) is None


def test_fuel_car_is_planned_in_litres() -> None:
    """A car without virtual energy is planned on its fuel: 30 laps at 2.5 litres with a 50 litre tank."""
    car = CarInput(1, "GT3", 0.0, 100.0, False, False, 50.0, 50.0, None, 2.5, None, None)
    plan = _plan(car, remaining_s=None, max_laps=30)
    assert (plan.stint_laps, plan.stops, plan.pit_lap) == (20, 1, 20)
    assert (plan.amount_on_board, plan.to_add_total, plan.first_stop_add) == (50.0, 25.0, 25.0)
    assert plan.fuel_to_add_l == 25.0


@pytest.mark.parametrize(
    ("laps", "stops", "start"), [(22.0, 0, 0.99), (22.5, 1, 0.0225), (44.0, 1, 0.99), (45.0, 2, 0.045)]
)
def test_min_stops_and_start_amount(laps: float, stops: int, start: float) -> None:
    """With 22-lap stints at 4.5 % per lap: stops with a full start, and the start amount that still allows them."""
    assert min_stops(laps, 0.045, 1.0) == stops
    assert min_start_amount(laps, 0.045, 1.0) == pytest.approx(start)
