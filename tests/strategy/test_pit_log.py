import dataclasses

from ai_race_engineer.strategy.pit_log import PitStopLog
from ai_race_engineer.strategy.pit_time import PitStopSample
from ai_race_engineer.telemetry.pit_estimate import PitEstimate
from ai_race_engineer.telemetry.race_snapshot import CarResources
from ai_race_engineer.telemetry.snapshot import VehicleState

TRACK_M: float = 1000.0


def _vehicle(
    slot_id: int,
    in_pits: bool = False,
    lap_distance_m: float = 500.0,
    speed_mps: float = 20.0,
    completed_laps: int = 3,
    in_garage: bool = False,
) -> VehicleState:
    """Vehicle with the given pit values; slot 0 is the player.
    In:  slot_id         int
         in_pits         bool
         lap_distance_m  float  m
         speed_mps       float  m/s
         completed_laps  int    laps
         in_garage       bool
    Out:                 VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        slot_id=slot_id,
        is_player=slot_id == 0,
        in_pits=in_pits,
        lap_distance_m=lap_distance_m,
        speed_mps=speed_mps,
        completed_laps=completed_laps,
        in_garage=in_garage,
    )
    return VehicleState(**base)


def _step(
    log: PitStopLog,
    time_s: float,
    car: VehicleState,
    fuel_l: float = 10.0,
    energy: float = 0.25,
    estimate: PitEstimate | None = None,
) -> tuple[PitStopSample, ...]:
    """One update with a single car's resources; a player on track is added when the car is not the player.
    In:  log       PitStopLog
         time_s    float               s session clock
         car       VehicleState
         fuel_l    float               litres of the car
         energy    float               fraction of the car
         estimate  PitEstimate | None
    Out:           tuple[PitStopSample, ...]
    """
    vehicles = [car] if car.is_player else [_vehicle(0), car]
    return log.update(vehicles, [CarResources(car.slot_id, fuel_l, 100.0, energy, 0)], TRACK_M, time_s, estimate)


def _pass(log: PitStopLog, slot_id: int, estimate: PitEstimate | None = None) -> tuple[PitStopSample, ...]:
    """A full pass: in at 1 s and 950 m, standing still for two ticks while filling, out at 6 s and 50 m.
    In:  log       PitStopLog
         slot_id   int
         estimate  PitEstimate | None  the game's estimate at pit entry
    Out:           tuple[PitStopSample, ...]
    """
    _step(log, 0.0, _vehicle(slot_id, False, 900.0))
    _step(log, 1.0, _vehicle(slot_id, True, 950.0), estimate=estimate)
    _step(log, 2.0, _vehicle(slot_id, True, 980.0))
    _step(log, 3.0, _vehicle(slot_id, True, 990.0, 0.0))
    _step(log, 4.0, _vehicle(slot_id, True, 990.0, 0.0), 30.0, 0.75)
    _step(log, 5.0, _vehicle(slot_id, True, 10.0), 30.0, 0.75)
    return _step(log, 6.0, _vehicle(slot_id, False, 50.0), 30.0, 0.75)


def test_full_pass_gives_one_sample() -> None:
    """Lane time excludes the stationary time; the stretch wraps the line; amounts are exit minus entry."""
    (sample,) = _pass(PitStopLog(0.5), 5)
    assert (sample.slot_id, sample.is_player) == (5, False)
    assert sample.lane_time_s == 3.0
    assert sample.lane_length_m == 100.0
    assert sample.stationary_s == 2.0
    assert sample.energy_added == 0.5
    assert sample.fuel_added_l == 20.0
    assert sample.estimate is None


def test_entry_not_seen_gives_no_sample() -> None:
    """A car first seen in the pits is not measured."""
    log = PitStopLog(0.5)
    _step(log, 0.0, _vehicle(5, True, 950.0))
    assert _step(log, 6.0, _vehicle(5, False, 50.0)) == ()


def test_drive_through_has_zero_stationary_time() -> None:
    """A pass without standing still has no stationary time."""
    log = PitStopLog(0.5)
    _step(log, 0.0, _vehicle(5, False, 900.0))
    _step(log, 1.0, _vehicle(5, True, 950.0))
    _step(log, 2.0, _vehicle(5, True, 990.0))
    (sample,) = _step(log, 3.0, _vehicle(5, False, 50.0))
    assert sample.stationary_s == 0.0
    assert sample.lane_time_s == 2.0


def test_player_estimate_at_entry_is_attached() -> None:
    """The player's sample keeps the game's estimate from the moment of pit entry."""
    entry = PitEstimate(fuel_s=4.0, energy_s=4.4, total_s=4.4)
    (sample,) = _pass(PitStopLog(0.5), 0, entry)
    assert sample.is_player
    assert sample.estimate == entry


def test_lap_count_decrease_clears_the_log() -> None:
    """The player's lap count going down (new session) forgets all samples."""
    log = PitStopLog(0.5)
    assert len(_pass(log, 5)) == 1
    assert _step(log, 7.0, _vehicle(0, completed_laps=0)) == ()


def test_visit_through_the_garage_gives_no_sample() -> None:
    """A stop during which the car is seen in the garage (ESC to the pits) is not a measured pass."""
    log = PitStopLog(0.5)
    _step(log, 0.0, _vehicle(0, False, 900.0))
    _step(log, 1.0, _vehicle(0, True, 950.0))
    _step(log, 2.0, _vehicle(0, True, 990.0, 0.0, in_garage=True))
    _step(log, 60.0, _vehicle(0, True, 990.0, 0.0), 100.0, 1.0)
    assert _step(log, 70.0, _vehicle(0, False, 50.0), 100.0, 1.0) == ()
    assert len(_pass(log, 5)) == 1
