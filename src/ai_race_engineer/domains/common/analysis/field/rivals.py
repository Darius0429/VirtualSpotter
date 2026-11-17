import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ai_race_engineer.domains.common.analysis.pace.laps import lap_fraction
from ai_race_engineer.domains.common.analysis.strategy.race_finish import RaceProjection
from ai_race_engineer.domains.common.data.snapshot import VehicleState


@dataclass(frozen=True)
class Rival:
    """A car of the player's class worth watching.
    slot_id                int           game slot id
    driver_name            str
    groups                 tuple[str, ...]  any of "ahead", "behind", "pace", "strategy"
    class_place            int           current position in class, 1-based
    gap_s                  float         s, positive = rival ahead in race distance (the game's own gap when both
                                         cars are on the same lap of the leader)
    lap_fraction           float         0-1 track position
    in_pits                bool
    pitstops               int           count
    lap_time_s             float         s/lap, recent pace
    best_lap_s             float | None  s
    pace_delta_s           float         s/lap, rival - player (negative = rival faster)
    projected_class_place  int           position in class at the flag, 1-based
    projected_gap_s        float         s at the flag, positive = rival ahead: the current gap plus what the rival
                                         gains or loses on the player until the flag (pace and remaining stops)
    stops_remaining        int           stops
    laps_on_board          float | None  laps on its strategy resource
    pit_lap                int | None    completed-lap count at which it must pit
    """

    slot_id: int
    driver_name: str
    groups: tuple[str, ...]
    class_place: int
    gap_s: float
    lap_fraction: float
    in_pits: bool
    pitstops: int
    lap_time_s: float
    best_lap_s: float | None
    pace_delta_s: float
    projected_class_place: int
    projected_gap_s: float
    stops_remaining: int
    laps_on_board: float | None
    pit_lap: int | None


def find_rivals(
    player: VehicleState,
    vehicles: Sequence[VehicleState],
    lap_times_s: Mapping[int, float],
    projection: RaceProjection,
    track_length_m: float,
    pace_window_s: float,
    finish_window_s: float,
    finish_window_exit_s: float,
    strategy_since_s: Mapping[int, float],
    time_s: float,
    min_rival_s: float,
    before_green: bool = False,
) -> tuple[Rival, ...]:
    """Pick the player's rivals: class neighbours, cars on the same pace and cars finishing close to the player. A
    strategy rival stays one for at least min_rival_s, and after that until its projected gap is beyond the exit
    window. Before the green there is no gap yet: the current gap of every car is 0 and the projected gap comes from
    the projection alone (the game's gaps and lap distances on the grid and the formation lap mean nothing).
    In:  player                VehicleState
         vehicles              Sequence[VehicleState]  all cars
         lap_times_s           Mapping[int, float]     slot id -> s/lap, for the projected cars (the player included)
         projection            RaceProjection
         track_length_m        float                   m
         pace_window_s         float              s/lap; within this of the player's pace = "pace"
         finish_window_s       float              s; within this of the player at the flag a car becomes "strategy"
         finish_window_exit_s  float              s; beyond this at the flag a strategy rival stops being one
         strategy_since_s      Mapping[int, float]  slot id -> s session time since when the car is a strategy
                                                    rival, for the strategy rivals of the previous call
         time_s                float              s session time now
         min_rival_s           float              s a strategy rival stays one whatever its projected gap
         before_green          bool               the race has not started
    Out:                       tuple[Rival, ...]  in class order
    """
    in_class = sorted(
        (v for v in vehicles if v.class_name == player.class_name and v.slot_id in lap_times_s), key=lambda v: v.place
    )
    player_place = in_class.index(player) + 1
    player_lap_s = lap_times_s[player.slot_id]
    player_progress = player.completed_laps + lap_fraction(player.lap_distance_m, track_length_m)
    player_flag_progress = projection.cars[player.slot_id].flag_progress_laps
    rivals: list[Rival] = []
    for class_place, vehicle in enumerate(in_class, start=1):
        projected = projection.cars[vehicle.slot_id]
        pace_delta_s = lap_times_s[vehicle.slot_id] - player_lap_s
        fraction = lap_fraction(vehicle.lap_distance_m, track_length_m)
        distance_gap_s = 0.0 if before_green else (vehicle.completed_laps + fraction - player_progress) * player_lap_s
        gap_s = 0.0 if before_green else _gap_s(player, vehicle, distance_gap_s)
        flag_gap_s = (projected.flag_progress_laps - player_flag_progress) * player_lap_s
        projected_gap_s = gap_s + flag_gap_s - distance_gap_s
        window_s = _strategy_window_s(
            strategy_since_s.get(vehicle.slot_id), time_s, min_rival_s, finish_window_s, finish_window_exit_s
        )
        groups = _groups(class_place - player_place, pace_delta_s, projected_gap_s, pace_window_s, window_s)
        if vehicle.slot_id == player.slot_id or not groups:
            continue
        rivals.append(
            Rival(
                slot_id=vehicle.slot_id,
                driver_name=vehicle.driver_name,
                groups=groups,
                class_place=class_place,
                gap_s=gap_s,
                lap_fraction=fraction,
                in_pits=vehicle.in_pits,
                pitstops=vehicle.pitstops,
                lap_time_s=lap_times_s[vehicle.slot_id],
                best_lap_s=vehicle.best_lap_s,
                pace_delta_s=pace_delta_s,
                projected_class_place=projected.class_place,
                projected_gap_s=projected_gap_s,
                stops_remaining=projected.stops_remaining,
                laps_on_board=projected.laps_on_board,
                pit_lap=projected.pit_lap,
            )
        )
    return tuple(rivals)


def laps_to_catch(gap_s: float, pace_delta_s: float) -> float | None:
    """Laps until the gap between a rival and the player closes at the current pace difference.
    In:  gap_s         float  s, positive = rival ahead
         pace_delta_s  float  s/lap, rival - player
    Out:               float | None  laps; None when the gap is not closing
    """
    return gap_s / pace_delta_s if gap_s * pace_delta_s > 0 else None


def _strategy_window_s(
    since_s: float | None, time_s: float, min_rival_s: float, finish_window_s: float, finish_window_exit_s: float
) -> float:
    """Projected gap within which a car counts as a strategy rival.
    In:  since_s               float | None  s session time since when it is a strategy rival; None when it is not
         time_s                float         s session time now
         min_rival_s           float         s a strategy rival stays one whatever its projected gap
         finish_window_s       float         s window to become a strategy rival
         finish_window_exit_s  float         s window a strategy rival must leave
    Out:                       float         s; inf during the minimum membership time
    """
    if since_s is None:
        return finish_window_s
    return math.inf if 0 <= time_s - since_s < min_rival_s else finish_window_exit_s


def _gap_s(player: VehicleState, vehicle: VehicleState, distance_gap_s: float) -> float:
    """Current gap from the player to another car: the game's own time gap when both are on the same lap of the
    leader, else the gap from race distance.
    In:  player          VehicleState
         vehicle         VehicleState  the other car
         distance_gap_s  float         s, race-distance difference times the player's lap time
    Out:                 float         s, positive = the other car is ahead
    """
    if vehicle.laps_behind_leader != player.laps_behind_leader:
        return distance_gap_s
    return player.time_behind_leader_s - vehicle.time_behind_leader_s


def _groups(
    places_behind: int, pace_delta_s: float, projected_gap_s: float, pace_window_s: float, finish_window_s: float
) -> tuple[str, ...]:
    """Rival groups a car of the player's class belongs to.
    In:  places_behind    int    class places behind the player (-1 = directly ahead, 1 = directly behind)
         pace_delta_s     float  s/lap, car - player
         projected_gap_s  float  s at the flag, positive = car ahead
         pace_window_s    float  s/lap
         finish_window_s  float  s
    Out:                  tuple[str, ...]  any of "ahead", "behind", "pace", "strategy"
    """
    groups: list[str] = []
    if places_behind == -1:
        groups.append("ahead")
    if places_behind == 1:
        groups.append("behind")
    if abs(pace_delta_s) <= pace_window_s:
        groups.append("pace")
    if abs(places_behind) != 1 and abs(projected_gap_s) <= finish_window_s:
        groups.append("strategy")
    return tuple(groups)
