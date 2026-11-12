import math


def relative_distance_m(track_length_m: float, player_distance_m: float, other_distance_m: float) -> float:
    """Signed track distance from the player to another car, wrapped to half a lap.
    In:  track_length_m     float  m
         player_distance_m  float  m from the start/finish line
         other_distance_m   float  m from the start/finish line
    Out:                    float  m in [-track/2, track/2), positive = other car ahead
    """
    half = track_length_m / 2
    return (other_distance_m - player_distance_m + half) % track_length_m - half


def relative_time_gap_s(relative_distance_m: float, other_speed_mps: float, player_speed_mps: float) -> float:
    """Time gap from a track distance and the mean speed of both cars.
    In:  relative_distance_m  float  m (sign ignored)
         other_speed_mps      float  m/s
         player_speed_mps     float  m/s
    Out:                      float  s; inf when both cars are stopped
    """
    mean_speed = (other_speed_mps + player_speed_mps) / 2
    return abs(relative_distance_m) / mean_speed if mean_speed > 0 else math.inf


def lapped_status(other_progress_laps: float, player_progress_laps: float) -> int:
    """Whether another car is a lap or more ahead of or behind the player.
    In:  other_progress_laps   float  laps (completed laps + lap fraction)
         player_progress_laps  float  laps (completed laps + lap fraction)
    Out:                       int    +1 at least a lap ahead, -1 at least a lap behind, else 0
    """
    diff = other_progress_laps - player_progress_laps
    if diff >= 1:
        return 1
    if diff <= -1:
        return -1
    return 0
