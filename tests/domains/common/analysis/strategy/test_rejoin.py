import pytest

from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitLane, PitModel, PitRates
from ai_race_engineer.domains.common.analysis.strategy.race_finish import CarInput
from ai_race_engineer.domains.common.analysis.strategy.rejoin import predict_rejoin, stop_amounts

PIT: PitModel = PitModel(
    PitRates(fuel_rate_l_per_s=2.0, energy_rate_per_s=0.02, scale=1.0), PitLane(30.0, 500.0), 5000.0
)


def _car(
    slot_id: int, progress_laps: float, class_name: str = "Hyper", fuel_per_energy: float | None = None
) -> CarInput:
    """Energy car at a 100 s lap with 25 % on board using 12.5 % per lap. The lane loss is 20 s.
    In:  slot_id          int
         progress_laps    float         laps
         class_name       str
         fuel_per_energy  float | None  litres per 1 % energy
    Out:                  CarInput
    """
    return CarInput(
        slot_id, class_name, progress_laps, 100.0, False, False, 30.0, 100.0, 0.25, 2.0, 0.125, fuel_per_energy
    )


@pytest.mark.parametrize(
    ("laps_remaining", "energy", "fuel_l"), [(20.0, 0.75, 37.5), (3.0, 0.125, 6.25), (1.0, 0.0, 0.0)]
)
def test_stop_amounts_energy_car(laps_remaining: float, energy: float, fuel_l: float) -> None:
    """An energy car takes what it needs to the flag, at most a full charge, never less than nothing."""
    assert stop_amounts(_car(1, 0.0, fuel_per_energy=0.5), laps_remaining) == (energy, fuel_l)


@pytest.mark.parametrize(("laps_remaining", "fuel_l"), [(40.0, 40.0), (6.0, 5.0), (2.0, 0.0)])
def test_stop_amounts_fuel_car(laps_remaining: float, fuel_l: float) -> None:
    """A car without virtual energy takes fuel, capped by the tank and by the need."""
    car = CarInput(1, "GT3", 0.0, 100.0, False, False, 10.0, 50.0, None, 2.5, None, None)
    assert stop_amounts(car, laps_remaining) == (0.0, fuel_l)


def test_rejoin_place_and_gaps() -> None:
    """A 26.25 s stop drops the car 0.2625 laps: places count cars ahead, gaps go to the class neighbours."""
    others = [_car(2, 10.6), _car(3, 10.3), _car(4, 10.1), _car(5, 10.25, "GT3")]
    rejoin = predict_rejoin(_car(1, 10.5), 3.0, others, PIT, 3.0)
    assert (rejoin.lane_loss_s, rejoin.service_s, rejoin.stop_time_s) == (20.0, 6.25, 26.25)
    assert (rejoin.energy_to_add, rejoin.fuel_to_add_l) == (0.125, 0.0)
    assert (rejoin.place, rejoin.class_place) == (4, 3)
    assert (rejoin.ahead_slot_id, rejoin.behind_slot_id) == (3, 4)
    assert rejoin.gap_ahead_s == pytest.approx(6.25)
    assert rejoin.gap_behind_s == pytest.approx(13.75)


def test_rejoin_for_another_car() -> None:
    """The same prediction works for any car, with the player among the others."""
    rejoin = predict_rejoin(_car(3, 10.3), 3.0, [_car(1, 10.5), _car(4, 10.1)], PIT, 3.0)
    assert (rejoin.slot_id, rejoin.place, rejoin.class_place) == (3, 3, 3)
    assert (rejoin.ahead_slot_id, rejoin.behind_slot_id) == (4, None)
    assert rejoin.gap_ahead_s == pytest.approx(6.25)
    assert rejoin.gap_behind_s is None


def test_traffic_counts_cars_on_the_road_with_wrap_around() -> None:
    """Cars of any class within the window of the rejoin point count, also a lap down."""
    others = [_car(3, 10.3), _car(5, 10.25, "GT3"), _car(6, 9.24, "GT3"), _car(7, 8.7375)]
    assert predict_rejoin(_car(1, 10.5), 3.0, others, PIT, 3.0).traffic == 2


def test_service_time_of_the_stop_includes_the_base() -> None:
    """With a fixed second per stop the 6.25 s of filling become 7.25 s standing still."""
    pit = PitModel(PitRates(2.0, 0.02, 1.0, base_s=1.0), PitLane(30.0, 500.0), 5000.0)
    rejoin = predict_rejoin(_car(1, 10.5), 3.0, [], pit, 3.0)
    assert (rejoin.service_s, rejoin.stop_time_s) == (7.25, 27.25)
