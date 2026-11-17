import math

from ai_race_engineer.domains.common.data.snapshot import VehicleState


def lap_fraction(lap_distance_m: float, track_length_m: float) -> float:
    """Fraction of the current lap completed.
    In:  lap_distance_m  float  m from the start/finish line
         track_length_m  float  m
    Out:                 float  0-1
    """
    return lap_distance_m / track_length_m


def leader_laps_remaining(
    remaining_s: float, max_laps: int, leader_progress_laps: float, leader_lap_time_s: float
) -> float:
    """Laps the overall leader still drives until the chequered flag: the first line crossing after the clock reaches 0
    (a crossing exactly at 0 still needs one more lap), or max_laps, whichever comes first.
    In:  remaining_s           float  s on the session clock
         max_laps              int    laps
         leader_progress_laps  float  laps (completed laps + lap fraction)
         leader_lap_time_s     float  s/lap
    Out:                       float  laps
    """
    timed = math.floor(leader_progress_laps + max(remaining_s, 0.0) / leader_lap_time_s) + 1 - leader_progress_laps
    return min(timed, max_laps - leader_progress_laps)


def laps_remaining(
    remaining_s: float,
    max_laps: int,
    leader_progress_laps: float,
    leader_lap_time_s: float,
    player_progress_laps: float,
    player_lap_time_s: float,
    player_is_leader: bool,
) -> float:
    """Laps the player still drives: until the first line crossing after the leader takes the flag.
    In:  remaining_s           float  s on the session clock
         max_laps              int    laps
         leader_progress_laps  float  laps (completed laps + lap fraction)
         leader_lap_time_s     float  s/lap
         player_progress_laps  float  laps (completed laps + lap fraction)
         player_lap_time_s     float  s/lap
         player_is_leader      bool
    Out:                       float  laps
    """
    leader_laps = leader_laps_remaining(remaining_s, max_laps, leader_progress_laps, leader_lap_time_s)
    if player_is_leader:
        return leader_laps
    time_to_flag_s = leader_laps * leader_lap_time_s
    return math.ceil(player_progress_laps + time_to_flag_s / player_lap_time_s) - player_progress_laps


def reference_lap_time_s(vehicle: VehicleState) -> float | None:
    """Lap time used to project laps remaining: the car's best lap.
    Future: replace with a linear regression over recent lap times.
    In:  vehicle  VehicleState
    Out:          float | None  s; None when the car has no best lap
    """
    return vehicle.best_lap_s
