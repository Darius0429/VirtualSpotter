import pytest

from ai_race_engineer.strategy.race_finish import CarInput
from ai_race_engineer.strategy.traffic import find_traffic, packs_ahead

TRACK_M: float = 5000.0


def _car(
    slot_id: int,
    progress_laps: float,
    lap_time_s: float,
    class_name: str = "GT3",
    in_pits: bool = False,
    finished: bool = False,
) -> CarInput:
    """Car at a race distance with a pace and no resources.
    In:  slot_id        int
         progress_laps  float  laps
         lap_time_s     float  s/lap
         class_name     str
         in_pits        bool
         finished       bool
    Out:                CarInput
    """
    return CarInput(
        slot_id, class_name, progress_laps, lap_time_s, finished, in_pits, None, None, None, None, None, None
    )


PLAYER: CarInput = _car(0, 20.5, 100.0, "Hyper")


def test_slower_car_ahead_is_caught() -> None:
    """A GT3 0.1 lap up the road lapping 10 s slower is reached after 1.1 laps: on lap 22, 60 % round the lap."""
    (car,), caught_by = find_traffic(PLAYER, [_car(5, 18.6, 110.0)], TRACK_M, 3.0)
    assert caught_by == ()
    assert (car.slot_id, car.class_name) == (5, "GT3")
    assert car.gap_s == pytest.approx(10.0)
    assert car.closing_s_per_lap == 10.0
    assert car.laps_to_meet == pytest.approx(1.1)
    assert car.time_to_meet_s == pytest.approx(110.0)
    assert car.meet_lap == 22
    assert car.meet_lap_fraction == pytest.approx(0.6)
    assert car.meet_lap_distance_m == pytest.approx(3000.0)


def test_faster_car_behind_catches_the_player() -> None:
    """A faster-class car 0.05 lap behind lapping 5 s quicker arrives after one lap of the player."""
    catching, (car,) = find_traffic(_car(0, 20.5, 100.0, "GT3"), [_car(7, 30.45, 95.0, "Hyper")], TRACK_M, 3.0)
    assert catching == ()
    assert car.slot_id == 7
    assert car.gap_s == pytest.approx(5.0)
    assert car.closing_s_per_lap == 5.0
    assert car.time_to_meet_s == pytest.approx(95.0)
    assert car.laps_to_meet == pytest.approx(0.95)
    assert car.meet_lap == 22
    assert car.meet_lap_fraction == pytest.approx(0.45)


def test_only_cars_within_the_horizon_nearest_first() -> None:
    """Cars met within 3 laps are listed by time to meet; one further away is left out."""
    others = [_car(5, 18.7, 110.0), _car(6, 17.6, 105.0), _car(7, 18.9, 110.0)]
    catching, _ = find_traffic(PLAYER, others, TRACK_M, 3.0)
    assert [c.slot_id for c in catching] == [6, 5]
    assert [c.laps_to_meet for c in catching] == pytest.approx([2.1, 2.2])


def test_cars_that_are_not_traffic() -> None:
    """Same-class cars, cars in the pits, finished cars and cars the player does not close on are left out."""
    others = [
        _car(1, 20.6, 110.0, "Hyper"),
        _car(2, 18.6, 110.0, in_pits=True),
        _car(3, 18.6, 110.0, finished=True),
        _car(4, 18.6, 100.0),
    ]
    assert find_traffic(PLAYER, others, TRACK_M, 3.0) == ((), ())


def test_slower_cars_ahead_are_grouped_into_packs() -> None:
    """Two GT3 cars 50 m apart are one pack, a third 500 m further is its own; the nearest pack comes first with the
    meeting of its first car."""
    others = [_car(5, 18.60, 110.0), _car(6, 18.61, 110.0), _car(7, 18.71, 110.0)]
    first, second = packs_ahead(PLAYER, others, TRACK_M, 3.0, 100.0)
    assert (first.slot_ids, first.class_name, first.backmarkers) == ((5, 6), "GT3", False)
    assert first.laps_to_meet == pytest.approx(1.1)
    assert first.meet_lap == 22
    assert second.slot_ids == (7,)


def test_lapped_cars_of_the_own_class_are_backmarkers() -> None:
    """A slower car of the player's class that is behind in the race is a backmarker; one ahead in the race, a
    faster one and one in the pits are not traffic."""
    others = [
        _car(5, 18.6, 104.0, "Hyper"),
        _car(6, 20.6, 104.0, "Hyper"),
        _car(7, 18.6, 99.0, "Hyper"),
        _car(8, 18.6, 110.0, in_pits=True),
    ]
    (pack,) = packs_ahead(PLAYER, others, TRACK_M, 5.0, 100.0)
    assert (pack.slot_ids, pack.backmarkers, pack.class_name) == ((5,), True, "Hyper")


def test_packs_beyond_the_horizon_are_left_out() -> None:
    """A car reached after 1.1 laps is not within one lap."""
    assert packs_ahead(PLAYER, [_car(5, 18.6, 110.0)], TRACK_M, 1.0, 100.0) == ()
