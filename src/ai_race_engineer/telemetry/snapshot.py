from dataclasses import dataclass
from enum import IntEnum


class SessionType(IntEnum):
    """Kind of session."""

    TEST_DAY = 0
    PRACTICE = 1
    QUALIFY = 2
    WARMUP = 3
    RACE = 4


@dataclass(frozen=True)
class SessionState:
    """Session-wide state.
    session_type    SessionType
    elapsed_s       float  s
    remaining_s     float  s (negative after the clock ran out)
    max_laps        int    laps
    track_name      str
    track_length_m  float  m
    track_temp_c    float  degC
    ambient_temp_c  float  degC
    """

    session_type: SessionType
    elapsed_s: float
    remaining_s: float
    max_laps: int
    track_name: str
    track_length_m: float
    track_temp_c: float
    ambient_temp_c: float


@dataclass(frozen=True)
class VehicleState:
    """Scoring and motion state of one car.
    slot_id                int           game slot id
    is_player              bool
    driver_name            str
    vehicle_name           str
    class_name             str
    place                  int           1-based overall position
    completed_laps         int           laps
    lap_distance_m         float         m from the start/finish line
    lap_start_s            float         s session time the lap started
    sector_index           int           0, 1, 2
    best_lap_s             float | None  s
    last_lap_s             float | None  s
    best_sector1_s         float | None  s
    best_sector2_cum_s     float | None  s (sector 1 + sector 2)
    time_behind_next_s     float         s
    laps_behind_next       int           laps
    time_behind_leader_s   float         s
    laps_behind_leader     int           laps
    in_pits                bool
    in_garage              bool
    pit_state              int           0 none, 1 request, 2 entering, 3 stopped, 4 exiting
    pitstops               int           count
    position_xz_m          tuple[float, float]  m world (x, z)
    position_y_m           float         m world height
    heading_rad            float         rad, forward = (sin h, cos h) in world (x, z)
    speed_mps              float         m/s
    """

    slot_id: int
    is_player: bool
    driver_name: str
    vehicle_name: str
    class_name: str
    place: int
    completed_laps: int
    lap_distance_m: float
    lap_start_s: float
    sector_index: int
    best_lap_s: float | None
    last_lap_s: float | None
    best_sector1_s: float | None
    best_sector2_cum_s: float | None
    time_behind_next_s: float
    laps_behind_next: int
    time_behind_leader_s: float
    laps_behind_leader: int
    in_pits: bool
    in_garage: bool
    pit_state: int
    pitstops: int
    position_xz_m: tuple[float, float]
    position_y_m: float
    heading_rad: float
    speed_mps: float


@dataclass(frozen=True)
class WheelState:
    """Telemetry of one wheel of the player's car.
    speed_mps       float  m/s rolling speed of the tyre (rotation x radius), forward positive
    grip_fraction   float  0-1 share of the contact patch that is sliding, as the game approximates it
    surface_temp_c  tuple[float, float, float]  degC of the tread: left, centre, right as seen from behind
    carcass_temp_c  float  degC
    pressure_kpa    float  kPa
    brake_temp_c    float  degC
    """

    speed_mps: float
    grip_fraction: float
    surface_temp_c: tuple[float, float, float]
    carcass_temp_c: float
    pressure_kpa: float
    brake_temp_c: float


@dataclass(frozen=True)
class PlayerState:
    """Player-only telemetry.
    vehicle                  VehicleState
    fuel_l                   float         litres
    virtual_energy           float | None  fraction 0-1 (LMU only)
    lap_invalidated          bool | None   (LMU only)
    throttle                 float         0-1
    brake                    float         0-1
    clutch                   float         0-1
    steering                 float         -1 (left) to 1 (right)
    rpm                      float         1/min
    gear                     int           -1 reverse, 0 neutral, 1+ forward
    accel_lateral_mps2       float         m/s^2 (local x)
    accel_longitudinal_mps2  float         m/s^2 (local z)
    accel_vertical_mps2      float         m/s^2 (local y)
    tyre_compound_front      str
    tyre_compound_rear       str
    tyre_surface_temp_c      tuple[float, float, float, float]  degC (FL, FR, RL, RR)
    wheels                   tuple[WheelState, ...]  FL, FR, RL, RR; empty when the source has none
    """

    vehicle: VehicleState
    fuel_l: float
    virtual_energy: float | None
    lap_invalidated: bool | None
    throttle: float
    brake: float
    clutch: float
    steering: float
    rpm: float
    gear: int
    accel_lateral_mps2: float
    accel_longitudinal_mps2: float
    accel_vertical_mps2: float
    tyre_compound_front: str
    tyre_compound_rear: str
    tyre_surface_temp_c: tuple[float, float, float, float]
    wheels: tuple[WheelState, ...] = ()


@dataclass(frozen=True)
class Snapshot:
    """One consistent frame of game state.
    session   SessionState
    player    PlayerState
    vehicles  tuple[VehicleState, ...]  all cars, including the player
    """

    session: SessionState
    player: PlayerState
    vehicles: tuple[VehicleState, ...]
