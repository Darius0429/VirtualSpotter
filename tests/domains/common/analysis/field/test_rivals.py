import dataclasses

import pytest

from ai_race_engineer.domains.common.analysis.strategy.race_finish import CarProjection, RaceProjection
from ai_race_engineer.domains.common.analysis.field.rivals import Rival, find_rivals, laps_to_catch
from ai_race_engineer.domains.common.data.snapshot import VehicleState

TRACK_M: float = 1000.0


def _vehicle(
    slot_id: int,
    place: int,
    progress_laps: float,
    class_name: str = "Hyper",
    time_behind_leader_s: float = 0.0,
    laps_behind_leader: int = 0,
) -> VehicleState:
    """Vehicle at a race distance.
    In:  slot_id               int
         place                 int    overall position
         progress_laps         float  laps (completed laps + lap fraction)
         class_name            str
         time_behind_leader_s  float  s, the game's gap to the leader
         laps_behind_leader    int    laps
    Out:                       VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        slot_id=slot_id,
        place=place,
        class_name=class_name,
        driver_name=f"driver{slot_id}",
        completed_laps=int(progress_laps),
        lap_distance_m=(progress_laps - int(progress_laps)) * TRACK_M,
        time_behind_leader_s=time_behind_leader_s,
        laps_behind_leader=laps_behind_leader,
    )
    return VehicleState(**base)


def _projection(flag_progress_laps: dict[int, float]) -> RaceProjection:
    """Projection with the given progress at the flag and neutral other values.
    In:  flag_progress_laps  dict[int, float]  slot id -> laps covered when the flag falls
    Out:                     RaceProjection
    """
    cars = {
        slot_id: CarProjection(
            slot_id, "Hyper", 5.0, 500.0, laps, 1, slot_id, None, 0, None, None, None, None, None, None, False
        )
        for slot_id, laps in flag_progress_laps.items()
    }
    return RaceProjection(500.0, 1, cars)


def _find(
    far_flag_progress_laps: float = 15.0,
    strategy_slot_ids: tuple[int, ...] = (),
    since_s: float = 0.0,
    before_green: bool = False,
) -> dict[int, Rival]:
    """Rivals of a player in class P3 between four class cars and a GT3 car, with a 0.25 s pace window, a 15 s finish
    window, a 20 s exit window and a 60 s minimum membership, at session time 1000 s. The player is 91 s behind the
    leader; slot 3 is a lap down on the leader.
    In:  far_flag_progress_laps  float            laps at the flag of slot 4 (the player has 15.5; 0.01 lap = 1 s)
         strategy_slot_ids       tuple[int, ...]  strategy rivals of the previous call
         since_s                 float            s session time since when they are strategy rivals
         before_green            bool             the race has not started
    Out:                         dict[int, Rival]  slot id -> rival
    """
    player = _vehicle(0, 4, 10.5, time_behind_leader_s=91.0)
    vehicles = [
        _vehicle(1, 1, 11.4),
        _vehicle(9, 2, 11.0, "GT3", 40.0),
        _vehicle(2, 3, 10.6, time_behind_leader_s=79.5),
        player,
        _vehicle(3, 5, 10.4, time_behind_leader_s=3.0, laps_behind_leader=1),
        _vehicle(4, 6, 10.0, time_behind_leader_s=141.0),
    ]
    lap_times_s = {0: 100.0, 1: 100.5, 9: 100.0, 2: 101.0, 3: 100.25, 4: 103.0}
    projection = _projection({0: 15.5, 1: 15.625, 9: 15.5, 2: 16.0, 3: 15.0, 4: far_flag_progress_laps})
    since = dict.fromkeys(strategy_slot_ids, since_s)
    found = find_rivals(
        player, vehicles, lap_times_s, projection, TRACK_M, 0.25, 15.0, 20.0, since, 1000.0, 60.0, before_green
    )
    return {r.slot_id: r for r in found}


def test_other_classes_and_unrelated_cars_are_ignored() -> None:
    """Only cars of the player's class in some group are rivals; the result is in class order."""
    assert list(_find()) == [1, 2, 3]


def test_ahead_and_behind_are_the_class_neighbours() -> None:
    """The neighbours by class place, with the GT3 car in between overall not counted."""
    rivals = _find()
    assert rivals[2].groups == ("ahead",)
    assert rivals[2].class_place == 2
    assert rivals[3].class_place == 4


def test_current_gap_is_the_games_gap_on_the_same_lap() -> None:
    """For two cars on the same lap of the leader the gap is the difference of the game's gaps to the leader."""
    rivals = _find()
    assert rivals[2].gap_s == 11.5
    assert rivals[1].gap_s == 91.0


def test_current_gap_from_race_distance_across_laps() -> None:
    """When the cars are on different laps of the leader the gap comes from race distance and the player's pace."""
    assert _find()[3].gap_s == pytest.approx(-10.0)


def test_pace_window_edge_and_two_groups() -> None:
    """A pace difference equal to the window still counts; a car can be in two groups."""
    rivals = _find()
    assert rivals[3].groups == ("behind", "pace")
    assert rivals[3].pace_delta_s == 0.25
    assert "pace" not in rivals[1].groups


def test_strategy_rival_far_away_on_track() -> None:
    """A car 91 s up the road that is projected within 15 s at the flag is a strategy rival."""
    rival = _find()[1]
    assert rival.groups == ("strategy",)
    assert rival.gap_s == 91.0
    assert rival.projected_gap_s == pytest.approx(13.5)


def test_projected_gap_is_anchored_on_the_games_gap() -> None:
    """For cars on the same lap of the leader the projected gap is the game's gap now plus what the car gains by the
    flag: 11.5 s now, 0.1 lap ahead now and 0.5 lap ahead at the flag, so 40 s gained."""
    assert _find()[2].projected_gap_s == pytest.approx(51.5)


def test_projected_gap_from_race_distance_across_laps() -> None:
    """When the game's gap is not usable the projected gap is the race-distance difference at the flag."""
    assert _find()[3].projected_gap_s == pytest.approx(-50.0)


@pytest.mark.parametrize("flag_progress_laps", [15.355, 15.34, 15.36])
def test_strategy_rival_stays_between_the_windows(flag_progress_laps: float) -> None:
    """A strategy rival oscillating between 14.5 and 16 s at the flag stays one."""
    assert _find(flag_progress_laps, strategy_slot_ids=(4,))[4].groups == ("strategy",)


def test_car_between_the_windows_does_not_become_a_strategy_rival() -> None:
    """A car 16 s away at the flag that was not a strategy rival is not one; at 14.5 s it becomes one."""
    assert 4 not in _find(15.34)
    assert _find(15.355)[4].groups == ("strategy",)


def test_strategy_rival_leaves_beyond_the_exit_window() -> None:
    """More than 20 s away at the flag a strategy rival is dropped."""
    assert _find(15.3, strategy_slot_ids=(4,))[4].groups == ("strategy",)
    assert 4 not in _find(15.29, strategy_slot_ids=(4,))


def test_laps_to_catch() -> None:
    """The gap closes when the car ahead is the slower one."""
    assert laps_to_catch(10.0, 0.5) == 20.0
    assert laps_to_catch(-10.0, -0.5) == 20.0


@pytest.mark.parametrize(("gap_s", "pace_delta_s"), [(10.0, -0.5), (-10.0, 0.5), (5.0, 0.0)])
def test_laps_to_catch_none_when_not_closing(gap_s: float, pace_delta_s: float) -> None:
    """A faster car ahead, a slower car behind or equal pace never meet."""
    assert laps_to_catch(gap_s, pace_delta_s) is None


def test_strategy_rival_stays_for_the_minimum_time() -> None:
    """A car that became a strategy rival less than 60 s ago stays one although it is 50 s away at the flag; from
    60 s on the exit window applies."""
    assert _find(15.0, strategy_slot_ids=(4,), since_s=941.0)[4].groups == ("strategy",)
    assert 4 not in _find(15.0, strategy_slot_ids=(4,), since_s=940.0)
    assert _find(15.3, strategy_slot_ids=(4,), since_s=940.0)[4].groups == ("strategy",)


def test_before_the_green_there_is_no_gap_and_the_projection_alone_counts() -> None:
    """On the grid the game's gaps and lap distances are ignored: every current gap is 0 and the projected gap is the
    difference at the flag (slot 2: 0.5 laps ahead at 100 s per lap)."""
    rivals = _find(before_green=True)
    assert [r.gap_s for r in rivals.values()] == [0.0] * len(rivals)
    assert rivals[2].projected_gap_s == pytest.approx(50.0)
