import dataclasses

import pytest

from ai_race_engineer.strategy.field import FieldTracker
from ai_race_engineer.strategy.pit_log import PitStopLog
from ai_race_engineer.strategy.pit_time import PitLane, PitRates
from ai_race_engineer.strategy.prior_field import PriorFieldTracker
from ai_race_engineer.strategy.priors import EventPriors
from ai_race_engineer.strategy.race_state import RaceTracker
from ai_race_engineer.strategy.track_info import TrackInfo
from ai_race_engineer.telemetry.race_snapshot import CarResources, LapSectors, RaceSnapshot, TrackFlags
from ai_race_engineer.telemetry.snapshot import PlayerState, SessionState, SessionType, VehicleState

DEFAULT_RATES: PitRates = PitRates(fuel_rate_l_per_s=2.0, energy_rate_per_s=0.02, scale=1.0)
DEFAULT_LANE: PitLane = PitLane(30.0, 500.0)
TRACK_LANE: PitLane = PitLane(25.0, 400.0)
UNSET_REMAINING_S: float = -2147483000.0
UNSET_MAX_LAPS: int = 2147483647


def _vehicle(
    slot_id: int, best_lap_s: float | None = 100.0, in_pits: bool = False, lap_distance_m: float = 0.0
) -> VehicleState:
    """Hyper car at the start of its first lap; slot 0 is the player.
    In:  slot_id         int
         best_lap_s      float | None  s
         in_pits         bool
         lap_distance_m  float         m
    Out:                 VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        slot_id=slot_id,
        is_player=slot_id == 0,
        place=slot_id + 1,
        class_name="Hyper",
        driver_name=f"driver{slot_id}",
        best_lap_s=best_lap_s,
        last_lap_s=None,
        in_pits=in_pits,
        lap_distance_m=lap_distance_m,
        speed_mps=50.0,
    )
    return VehicleState(**base)


def _snapshot(
    elapsed_s: float = 100.0,
    vehicles: tuple[VehicleState, ...] = (),
    session_type: SessionType = SessionType.RACE,
    remaining_s: float = 250.0,
    max_laps: int = UNSET_MAX_LAPS,
    track_name: str = "Track",
    finish_status: int = 0,
    game_phase: int = 5,
) -> RaceSnapshot:
    """Race snapshot of the given cars (default: the player alone with a 100 s best lap) on a 1000 m track.
    In:  elapsed_s      float                     s
         vehicles       tuple[VehicleState, ...]  the player first
         session_type   SessionType
         remaining_s    float                     s
         max_laps       int                       laps
         track_name     str
         finish_status  int                       of every car
         game_phase     int                       mGamePhase
    Out:                RaceSnapshot
    """
    cars = vehicles or (_vehicle(0),)
    session = SessionState(session_type, elapsed_s, remaining_s, max_laps, track_name, 1000.0, 20.0, 20.0)
    player = PlayerState(cars[0], 50.0, 0.5, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, "", "", (0.0,) * 4)
    resources = tuple(CarResources(v.slot_id, 50.0, 100.0, 0.5, finish_status) for v in cars)
    return RaceSnapshot(session, player, cars, resources, TrackFlags(game_phase, 0, (11, 11, 11), 0))


def _tracker(tracks: dict[str, TrackInfo] | None = None, field: FieldTracker | None = None) -> RaceTracker:
    """Tracker that recomputes every second of session time, without a game estimate.
    In:  tracks  dict[str, TrackInfo] | None  track table (default: empty)
         field   FieldTracker | None          field tracker (default: one without priors)
    Out:         RaceTracker
    """
    return RaceTracker(
        field or FieldTracker(3, 3),
        PitStopLog(0.5),
        tracks or {},
        DEFAULT_RATES,
        DEFAULT_LANE,
        lambda: None,
        pace_window_s=0.3,
        finish_window_s=15.0,
        finish_window_exit_s=20.0,
        min_rival_s=60.0,
        traffic_window_s=3.0,
        period_s=1.0,
        fuel_margin_laps=0.2,
    )


def test_state_in_a_race() -> None:
    """In a race with a lap time for the player there is a state holding the projection."""
    tracker = _tracker()
    tracker.update(_snapshot())
    assert tracker.state.time_s == 100.0
    assert tracker.state.player_slot_id == 0
    assert tracker.state.projection.flag_time_s == pytest.approx(300.0)
    assert tracker.state.set_fuel_ratio is None
    assert tracker.state.traffic_window_s == 3.0


def test_no_state_outside_a_race() -> None:
    """Practice has no race state."""
    tracker = _tracker()
    tracker.update(_snapshot(session_type=SessionType.PRACTICE))
    assert tracker.state is None


def test_no_state_without_a_player_lap_time() -> None:
    """Without a lap time the player cannot be projected."""
    tracker = _tracker()
    tracker.update(_snapshot(vehicles=(_vehicle(0, best_lap_s=None),)))
    assert tracker.state is None


def test_state_recomputed_once_per_period() -> None:
    """Within period_s of session time the state object is kept; after it a new one is made."""
    tracker = _tracker()
    tracker.update(_snapshot(100.0))
    first = tracker.state
    tracker.update(_snapshot(100.8))
    assert tracker.state is first
    tracker.update(_snapshot(101.0))
    assert tracker.state.time_s == 101.0


def test_session_clock_going_back_recomputes() -> None:
    """A restarted session clock does not wait for the old time to come round."""
    tracker = _tracker()
    tracker.update(_snapshot(100.0))
    tracker.update(_snapshot(5.0))
    assert tracker.state.time_s == 5.0


def test_unset_clock_is_no_time_limit() -> None:
    """The game's unset end time is ignored: the flag comes from the lap limit."""
    tracker = _tracker()
    tracker.update(_snapshot(remaining_s=UNSET_REMAINING_S, max_laps=5))
    assert tracker.state.projection.flag_time_s == pytest.approx(500.0)


def test_unset_lap_limit_is_no_lap_limit() -> None:
    """The game's unset lap limit is ignored: the flag comes from the clock."""
    tracker = _tracker()
    tracker.update(_snapshot(remaining_s=250.0, max_laps=UNSET_MAX_LAPS))
    assert tracker.state.projection.flag_time_s == pytest.approx(300.0)


def test_no_state_without_any_limit() -> None:
    """With neither limit set there is nothing to project to."""
    tracker = _tracker()
    tracker.update(_snapshot(remaining_s=UNSET_REMAINING_S, max_laps=UNSET_MAX_LAPS))
    assert tracker.state is None


def test_cars_out_of_the_race_are_not_projected() -> None:
    """A car without a lap time is left out; a disqualified player ends the state."""
    tracker = _tracker()
    tracker.update(_snapshot(vehicles=(_vehicle(0), _vehicle(1), _vehicle(2, best_lap_s=None))))
    assert [c.slot_id for c in tracker.state.cars] == [0, 1]
    tracker.update(_snapshot(102.0, finish_status=3))
    assert tracker.state is None


def test_lane_from_the_track_table_first() -> None:
    """A track with a pit lane in the table uses it."""
    tracker = _tracker({"Track": TrackInfo(1000.0, TRACK_LANE, ())})
    tracker.update(_snapshot())
    assert tracker.state.pit.lane == TRACK_LANE
    assert tracker.state.pit.rates == DEFAULT_RATES
    assert tracker.state.pit.track_length_m == 1000.0


def test_default_lane_without_table_entry_or_measurement() -> None:
    """An unknown track with no measured pass uses the default lane, also when its table entry has no pit lane."""
    for tracks in ({}, {"Track": TrackInfo(1000.0, None, ())}):
        tracker = _tracker(tracks)
        tracker.update(_snapshot())
        assert tracker.state.pit.lane == DEFAULT_LANE


def test_measured_lane_before_the_default() -> None:
    """On an unknown track a measured pass of any car replaces the default lane."""
    tracker = _tracker()
    tracker.update(_snapshot(100.0, (_vehicle(0), _vehicle(1, lap_distance_m=900.0))))
    tracker.update(_snapshot(101.0, (_vehicle(0), _vehicle(1, in_pits=True, lap_distance_m=950.0))))
    tracker.update(_snapshot(121.0, (_vehicle(0), _vehicle(1, lap_distance_m=50.0))))
    assert tracker.state.pit.lane == PitLane(20.0, 100.0)


def test_state_is_kept_after_the_session_is_over() -> None:
    """When the game phase is session over the state is computed once more and then no longer replaced."""
    tracker = _tracker()
    tracker.update(_snapshot(100.0))
    assert not tracker.finished
    tracker.update(_snapshot(100.4, game_phase=8))
    final = tracker.state
    assert final.time_s == 100.4
    assert tracker.finished
    tracker.update(_snapshot(130.0, game_phase=8))
    assert tracker.state is final


def test_state_is_kept_after_the_player_finished() -> None:
    """Once the player has taken the flag the state is final, also while the phase is still green."""
    tracker = _tracker()
    tracker.update(_snapshot(100.0))
    tracker.update(_snapshot(101.0, finish_status=1))
    final = tracker.state
    assert tracker.finished
    assert final.projection.cars[0].laps_remaining == 0.0
    tracker.update(_snapshot(140.0, finish_status=1))
    assert tracker.state is final


def test_new_race_after_a_finished_one_is_tracked_again() -> None:
    """A race that is running again after a finished one gets new states."""
    tracker = _tracker()
    tracker.update(_snapshot(500.0, game_phase=8))
    tracker.update(_snapshot(5.0))
    assert tracker.state.time_s == 5.0
    assert not tracker.finished


def test_fuel_tolerance_reaches_the_projection() -> None:
    """The configured fuel margin decides the warning: 50 L at 30 L per lap is 0.33 laps short of the energy."""
    tracker = _tracker()
    vehicle = _vehicle(0)
    tracker.update(_snapshot(100.0, (vehicle,)))
    crossed = dataclasses.replace(vehicle, lap_start_s=100.0, completed_laps=1)
    tracker.update(_snapshot(100.0, (crossed,)))
    lapped = dataclasses.replace(vehicle, lap_start_s=200.0, completed_laps=2, last_lap_s=100.0)
    snapshot = _snapshot(200.0, (lapped,))
    used = dataclasses.replace(snapshot, resources=(CarResources(0, 20.0, 100.0, 0.25, 0),))
    tracker.update(used)
    mine = tracker.state.projection.cars[0]
    assert mine.fuel_margin_laps == pytest.approx(20.0 / 30.0 - 1.0)
    assert mine.fuel_runs_out_first


def test_state_carries_estimates_and_sectors() -> None:
    """The pace estimates and last-lap sectors of all cars are part of the state for the tools."""
    tracker = _tracker()
    snapshot = _snapshot(vehicles=(_vehicle(0), _vehicle(1)))
    sectors = (LapSectors(0, 30.0, 65.0), LapSectors(1, None, None))
    tracker.update(dataclasses.replace(snapshot, sectors=sectors))
    assert tracker.state.sectors == {0: sectors[0], 1: sectors[1]}
    assert tracker.state.estimates[1].lap_time_s == 100.0


def test_race_length_before_the_green_comes_from_the_game() -> None:
    """On the grid the session clock counts down to the start; the race length is the game's remaining time."""
    tracker = _tracker()
    tracker.update(dataclasses.replace(_snapshot(remaining_s=60.0, game_phase=0), time_remaining_s=250.0))
    assert tracker.state.projection.flag_time_s == pytest.approx(300.0)


@pytest.mark.parametrize("game_s", [0.0, -5.0, UNSET_REMAINING_S])
def test_game_clock_without_time_before_the_green_is_no_time_limit(game_s: float) -> None:
    """Before the green a remaining time of 0 or below (a lap-limited race) is no time limit: the flag comes from
    the lap limit."""
    tracker = _tracker()
    snapshot = _snapshot(remaining_s=60.0, max_laps=5, game_phase=0)
    tracker.update(dataclasses.replace(snapshot, time_remaining_s=game_s))
    assert tracker.state.projection.flag_time_s == pytest.approx(500.0)


def test_game_clock_is_not_used_after_the_green() -> None:
    """Under green the session's own remaining time counts, whatever the game's field holds."""
    tracker = _tracker()
    tracker.update(dataclasses.replace(_snapshot(remaining_s=250.0), time_remaining_s=0.0))
    assert tracker.state.projection.flag_time_s == pytest.approx(300.0)
    assert not tracker.state.before_green


def test_car_without_a_lap_is_at_zero_before_the_green() -> None:
    """On the grid and the formation lap a car that has not completed a lap is at 0 laps wherever it stands; under
    green its lap distance counts."""
    tracker = _tracker()
    grid = (_vehicle(0, lap_distance_m=-200.0), _vehicle(1, lap_distance_m=300.0))
    tracker.update(_snapshot(vehicles=grid, game_phase=3))
    assert [c.progress_laps for c in tracker.state.cars] == [0.0, 0.0]
    assert tracker.state.before_green
    assert [(r.gap_s, r.projected_gap_s) for r in tracker.state.rivals] == [(0.0, 0.0)]
    tracker.update(_snapshot(101.0, vehicles=grid, game_phase=5))
    assert [c.progress_laps for c in tracker.state.cars] == [-0.2, 0.3]


def test_state_exists_when_the_field_gives_a_lap_time() -> None:
    """A player without any lap has a state once the field tracker fills the pace from a prior."""
    priors = EventPriors("Track", "car", {"driver0": 100.0}, {})
    tracker = _tracker(field=PriorFieldTracker(3, 3, 1.0, lambda: priors, lambda: None, 5))
    tracker.update(_snapshot(vehicles=(_vehicle(0, best_lap_s=None),), game_phase=0))
    assert tracker.state.cars[0].lap_time_s == 100.0
    assert tracker.state.projection.cars[0].laps_remaining == 3.0


@pytest.mark.parametrize("phase", [4, 9])
def test_never_before_the_green_again_once_it_was_shown(phase: int) -> None:
    """A pre-green phase in the middle of a race does not put the race back before the green: the cars keep their
    distance and the session's own clock counts (250 s left: the leader, 0.3 laps into a 100 s lap, takes the flag
    after 270 s), not the game's 900 s."""
    tracker = _tracker()
    cars = (_vehicle(0, lap_distance_m=200.0), _vehicle(1, lap_distance_m=300.0))
    tracker.update(_snapshot(100.0, cars, game_phase=phase))
    assert tracker.state.before_green
    assert not tracker.green_seen
    tracker.update(_snapshot(101.0, cars, game_phase=5))
    assert tracker.green_seen
    tracker.update(dataclasses.replace(_snapshot(102.0, cars, game_phase=phase), time_remaining_s=900.0))
    assert not tracker.state.before_green
    assert [c.progress_laps for c in tracker.state.cars] == [0.2, 0.3]
    assert tracker.state.projection.flag_time_s == pytest.approx(270.0)


@pytest.mark.parametrize(
    "new",
    [
        {"elapsed_s": 5.0},
        {"elapsed_s": 200.0, "track_name": "Other"},
    ],
)
def test_new_session_can_be_before_the_green_again(new: dict[str, object]) -> None:
    """A restarted clock or another track is a new session: its grid is before the green again."""
    tracker = _tracker()
    tracker.update(_snapshot(100.0, game_phase=5))
    tracker.update(_snapshot(game_phase=9, **new))
    assert tracker.state.before_green
    assert not tracker.green_seen


def test_another_session_type_forgets_the_green() -> None:
    """The green of a practice session does not count for the race that follows."""
    tracker = _tracker()
    tracker.update(_snapshot(100.0, session_type=SessionType.PRACTICE, game_phase=5))
    assert tracker.green_seen
    tracker.update(_snapshot(200.0, game_phase=3))
    assert tracker.state.before_green
