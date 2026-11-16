import dataclasses
import math

from ai_race_engineer.telemetry.race_snapshot import CarTyres, RaceSnapshot, TrackFlags
from ai_race_engineer.telemetry.snapshot import PlayerState, SessionState, SessionType, VehicleState

STRAIGHT_M: float = 500.0
RADIUS_M: float = 100.0
ARC_M: float = math.pi * RADIUS_M
LENGTH_M: float = 2 * STRAIGHT_M + 2 * ARC_M
TURN_1_M: tuple[float, float] = (STRAIGHT_M / 2, STRAIGHT_M / 2 + ARC_M)
TURN_2_M: tuple[float, float] = (1.5 * STRAIGHT_M + ARC_M, 1.5 * STRAIGHT_M + 2 * ARC_M)
BRAKE_ZONE_M: float = 100.0
FAST_MPS: float = 60.0
SLOW_MPS: float = 30.0
FUEL_PER_LAP_L: float = 2.0
POLL_S: float = 0.05


def oval_lap(
    completed_laps: int,
    last_lap_s: float | None = None,
    in_pits: bool = False,
    track_name: str = "Oval",
    brake_zone_m: float = BRAKE_ZONE_M,
) -> list[RaceSnapshot]:
    """Snapshots of one lap of an oval (two 500 m straights, two 100 m radius half circles, the line in the middle
    of a straight) polled at 20 Hz: 60 m/s on the straights, braking over the last brake_zone_m before a turn,
    30 m/s through it at 0.92 g with the throttle back on from the middle of the turn.
    In:  completed_laps  int           laps completed during this lap
         last_lap_s      float | None  s the game shows as the last lap time during this lap
         in_pits         bool          the car is in the pits for the whole lap
         track_name      str
         brake_zone_m    float         m braked before each turn
    Out:                 list[RaceSnapshot]
    """
    snapshots: list[RaceSnapshot] = []
    distance_m = 0.0
    while distance_m < LENGTH_M:
        snapshots.append(_snapshot(distance_m, completed_laps, last_lap_s, in_pits, track_name, brake_zone_m))
        distance_m += _speed_mps(distance_m, brake_zone_m) * POLL_S
    return snapshots


def _snapshot(
    distance_m: float,
    completed_laps: int,
    last_lap_s: float | None,
    in_pits: bool,
    track_name: str,
    brake_zone_m: float,
) -> RaceSnapshot:
    """Snapshot of the player at a lap distance.
    In:  distance_m      float         m
         completed_laps  int
         last_lap_s      float | None  s
         in_pits         bool
         track_name      str
         brake_zone_m    float         m
    Out:                 RaceSnapshot
    """
    to_turn_m = min((start - distance_m for start, _ in (TURN_1_M, TURN_2_M) if start > distance_m), default=LENGTH_M)
    turn = next(((start, end) for start, end in (TURN_1_M, TURN_2_M) if start <= distance_m < end), None)
    braking = to_turn_m <= brake_zone_m
    speed_mps = _speed_mps(distance_m, brake_zone_m)
    coasting = turn is not None and distance_m < sum(turn) / 2
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        is_player=True,
        completed_laps=completed_laps,
        last_lap_s=last_lap_s,
        in_pits=in_pits,
        position_xz_m=_position_xz_m(distance_m),
        speed_mps=speed_mps,
    )
    vehicle = VehicleState(**base)
    fuel_l = 50.0 - FUEL_PER_LAP_L * (completed_laps + distance_m / LENGTH_M)
    player = PlayerState(
        vehicle=vehicle,
        fuel_l=fuel_l,
        virtual_energy=fuel_l / 100,
        lap_invalidated=False,
        throttle=0.0 if braking or coasting else 1.0,
        brake=1.0 if braking else 0.0,
        clutch=0.0,
        steering=0.3 if turn else 0.0,
        rpm=0.0,
        gear=3 if turn else 6,
        accel_lateral_mps2=speed_mps**2 / RADIUS_M if turn else 0.0,
        accel_longitudinal_mps2=speed_mps * (FAST_MPS - SLOW_MPS) / brake_zone_m if braking else 0.0,
        accel_vertical_mps2=0.0,
        tyre_compound_front="Medium",
        tyre_compound_rear="Medium",
        tyre_surface_temp_c=(80.4, 81.0, 85.0, 86.0),
    )
    session = SessionState(SessionType.RACE, 0.0, 3600.0, 1000, track_name, LENGTH_M, 30.0, 20.0)
    return RaceSnapshot(
        session,
        player,
        (vehicle,),
        resources=(),
        flags=TrackFlags(0, 0, (0, 0, 0), 0),
        tyres=(CarTyres(0, "Medium", "Medium", (0.9, 0.91, 0.95, 0.96)),),
    )


def _speed_mps(distance_m: float, brake_zone_m: float) -> float:
    """Speed at a lap distance.
    In:  distance_m    float  m
         brake_zone_m  float  m braked before each turn
    Out:               float  m/s
    """
    for start, end in (TURN_1_M, TURN_2_M):
        if start <= distance_m < end:
            return SLOW_MPS
        if 0 < start - distance_m <= brake_zone_m:
            return SLOW_MPS + (FAST_MPS - SLOW_MPS) * (start - distance_m) / brake_zone_m
    return FAST_MPS


def _position_xz_m(distance_m: float) -> tuple[float, float]:
    """World position at a lap distance.
    In:  distance_m  float  m
    Out:             tuple[float, float]  m (x, z)
    """
    half = STRAIGHT_M / 2
    if distance_m < TURN_1_M[0]:
        return (distance_m, 0.0)
    if distance_m < TURN_1_M[1]:
        angle = (distance_m - TURN_1_M[0]) / RADIUS_M
        return (half + RADIUS_M * math.sin(angle), RADIUS_M - RADIUS_M * math.cos(angle))
    if distance_m < TURN_2_M[0]:
        return (half - (distance_m - TURN_1_M[1]), 2 * RADIUS_M)
    if distance_m < TURN_2_M[1]:
        angle = (distance_m - TURN_2_M[0]) / RADIUS_M
        return (-half - RADIUS_M * math.sin(angle), RADIUS_M + RADIUS_M * math.cos(angle))
    return (-half + (distance_m - TURN_2_M[1]), 0.0)


def with_rival(
    snapshots: list[RaceSnapshot], driver_name: str, last_lap_s: float | None, shift_x_m: float = 0.0
) -> list[RaceSnapshot]:
    """The same snapshots with a second car that drives the player's lap, moved sideways in x.
    In:  snapshots    list[RaceSnapshot]  one lap of the player
         driver_name  str                 driver of the second car (slot 1)
         last_lap_s   float | None        s the game shows as that car's last lap time
         shift_x_m    float               m added to the car's x
    Out:              list[RaceSnapshot]
    """
    with_car: list[RaceSnapshot] = []
    for snapshot in snapshots:
        mine = snapshot.player.vehicle
        rival = dataclasses.replace(
            mine,
            slot_id=1,
            is_player=False,
            driver_name=driver_name,
            last_lap_s=last_lap_s,
            position_xz_m=(mine.position_xz_m[0] + shift_x_m, mine.position_xz_m[1]),
        )
        with_car.append(dataclasses.replace(snapshot, vehicles=(mine, rival)))
    return with_car
