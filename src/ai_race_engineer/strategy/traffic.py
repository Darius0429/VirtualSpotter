import math
from collections.abc import Sequence
from dataclasses import dataclass

from ai_race_engineer.strategy.race_finish import CarInput


@dataclass(frozen=True)
class TrafficCar:
    """A car of another class the player will meet on the road.
    slot_id              int    game slot id
    class_name           str
    gap_s                float  s on the road now: track position between the two cars times the player's lap time
    closing_s_per_lap    float  s per lap the gap shrinks (difference of the two lap-time means)
    laps_to_meet         float  laps the player drives until the cars meet
    time_to_meet_s       float  s until the cars meet
    meet_lap             int    lap number on which they meet, in the driver's counting (current lap = completed + 1)
    meet_lap_fraction    float  0-1, where on that lap
    meet_lap_distance_m  float  m from the start/finish line
    """

    slot_id: int
    class_name: str
    gap_s: float
    closing_s_per_lap: float
    laps_to_meet: float
    time_to_meet_s: float
    meet_lap: int
    meet_lap_fraction: float
    meet_lap_distance_m: float


def find_traffic(
    player: CarInput, others: Sequence[CarInput], track_length_m: float, horizon_laps: float
) -> tuple[tuple[TrafficCar, ...], tuple[TrafficCar, ...]]:
    """Cars of other classes the player meets within a horizon: slower ones ahead on the road that the player
    reaches, and faster ones behind that reach the player. Cars in the pits or finished are left out.
    In:  player          CarInput
         others          Sequence[CarInput]  all other cars
         track_length_m  float               m
         horizon_laps    float               laps of the player within which the cars must meet
    Out:                 tuple[tuple[TrafficCar, ...], tuple[TrafficCar, ...]]  (cars the player catches, cars that
                                                                                catch the player), nearest first
    """
    on_road = [o for o in others if o.class_name != player.class_name and not o.in_pits and not o.finished]
    catching = [_meeting(player, o, True, track_length_m) for o in on_road if o.lap_time_s > player.lap_time_s]
    caught = [_meeting(player, o, False, track_length_m) for o in on_road if o.lap_time_s < player.lap_time_s]
    return _within(catching, horizon_laps), _within(caught, horizon_laps)


def _meeting(player: CarInput, other: CarInput, other_is_ahead: bool, track_length_m: float) -> TrafficCar:
    """Where and when the player and another car meet on the road at their current paces.
    In:  player          CarInput
         other           CarInput
         other_is_ahead  bool      True: the slower car ahead is caught; False: the faster car behind catches up
         track_length_m  float     m
    Out:                 TrafficCar
    """
    ahead, behind = (other, player) if other_is_ahead else (player, other)
    gap_laps = (ahead.progress_laps - behind.progress_laps) % 1.0
    time_s = gap_laps / (1 / behind.lap_time_s - 1 / ahead.lap_time_s)
    laps = time_s / player.lap_time_s
    fraction = (player.progress_laps + laps) % 1.0
    return TrafficCar(
        slot_id=other.slot_id,
        class_name=other.class_name,
        gap_s=gap_laps * player.lap_time_s,
        closing_s_per_lap=abs(other.lap_time_s - player.lap_time_s),
        laps_to_meet=laps,
        time_to_meet_s=time_s,
        meet_lap=math.floor(player.progress_laps + laps) + 1,
        meet_lap_fraction=fraction,
        meet_lap_distance_m=fraction * track_length_m,
    )


def _within(cars: Sequence[TrafficCar], horizon_laps: float) -> tuple[TrafficCar, ...]:
    """The meetings within the horizon, nearest first.
    In:  cars          Sequence[TrafficCar]
         horizon_laps  float                 laps
    Out:               tuple[TrafficCar, ...]
    """
    return tuple(sorted((c for c in cars if c.laps_to_meet <= horizon_laps), key=lambda c: c.time_to_meet_s))
