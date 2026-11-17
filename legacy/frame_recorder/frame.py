from dataclasses import dataclass

from ai_race_engineer.domains.common.data.snapshot import SessionType


@dataclass(frozen=True)
class SessionInfo:
    """Identity of a recorded session (written to session.yaml); None where an h5 import lacks the value.
    game            str
    track_name      str
    track_length_m  float | None  m
    vehicle_name    str
    class_name      str | None
    session_type    SessionType
    session_number  int | None    raw game session 0-13 (0 test day, 1-4 practice, 5-8 qualify, 9 warmup, 10-13 race)
    max_laps        int | None    laps
    """

    game: str
    track_name: str
    track_length_m: float | None
    vehicle_name: str
    class_name: str | None
    session_type: SessionType
    session_number: int | None
    max_laps: int | None


@dataclass(frozen=True)
class PlayerSample:
    """Player telemetry row (player_telemetry_100hz).
    time_s          float  s telemetry clock (mElapsedTime)
    throttle        float  0-1 unfiltered
    brake           float  0-1 unfiltered
    clutch          float  0-1 unfiltered
    steering        float  -1 (left) to 1 (right) unfiltered
    gear            int    -1 reverse, 0 neutral, 1+ forward
    rpm             float  1/min
    speed_mps       float  m/s
    pos_x_m         float  m world
    pos_y_m         float  m world height
    pos_z_m         float  m world
    heading_rad     float  rad, forward = (sin h, cos h) in world (x, z)
    accel_x_mps2    float  m/s^2 local x (lateral)
    accel_y_mps2    float  m/s^2 local y (vertical)
    accel_z_mps2    float  m/s^2 local z (longitudinal)
    lap_number      int    mLapNumber
    lap_distance_m  float  m from the start/finish line (scoring value, steps at the scoring rate)
    """

    time_s: float
    throttle: float
    brake: float
    clutch: float
    steering: float
    gear: int
    rpm: float
    speed_mps: float
    pos_x_m: float
    pos_y_m: float
    pos_z_m: float
    heading_rad: float
    accel_x_mps2: float
    accel_y_mps2: float
    accel_z_mps2: float
    lap_number: int
    lap_distance_m: float


@dataclass(frozen=True)
class CarSample:
    """Position row of one car, including the player (cars_positions_50hz, long format).
    time_s          float  s player telemetry clock
    slot_id         int    game slot id
    pos_x_m         float  m world
    pos_y_m         float  m world height
    pos_z_m         float  m world
    heading_rad     float  rad, forward = (sin h, cos h) in world (x, z)
    speed_mps       float  m/s
    lap_distance_m  float  m from the start/finish line
    """

    time_s: float
    slot_id: int
    pos_x_m: float
    pos_y_m: float
    pos_z_m: float
    heading_rad: float
    speed_mps: float
    lap_distance_m: float


@dataclass(frozen=True)
class StateSample:
    """Player and session state row (player_state_5hz), one per scoring update.
    time_s                 float         s player telemetry clock when the scoring update was seen
    session_time_s         float         s session clock (mCurrentET)
    remaining_s            float         s (negative after the clock ran out)
    game_phase             int           mGamePhase
    yellow_flag_state      int           mYellowFlagState (full-course yellow)
    sector_flag_1          int           mSectorFlag[0] (local yellow)
    sector_flag_2          int           mSectorFlag[1]
    sector_flag_3          int           mSectorFlag[2]
    player_flag            int           0 green, 6 blue
    fuel_l                 float         litres
    virtual_energy         float | None  fraction 0-1 (LMU only)
    lap_invalidated        bool | None   (LMU only)
    tyre_compound_front    str
    tyre_compound_rear     str
    tyre_temp_c_fl         float         degC mean surface temperature
    tyre_temp_c_fr         float         degC
    tyre_temp_c_rl         float         degC
    tyre_temp_c_rr         float         degC
    tyre_wear_fl           float         fraction (1 new)
    tyre_wear_fr           float         fraction
    tyre_wear_rl           float         fraction
    tyre_wear_rr           float         fraction
    tyre_pressure_kpa_fl   float         kPa
    tyre_pressure_kpa_fr   float         kPa
    tyre_pressure_kpa_rl   float         kPa
    tyre_pressure_kpa_rr   float         kPa
    dent_0 .. dent_7       int           0 none, 1 some, 2 more (8 locations around the car)
    overheating            bool
    detached               bool          any part (besides wheels) detached
    track_temp_c           float         degC
    ambient_temp_c         float         degC
    raining                float         0-1
    path_wetness_avg       float         0-1
    """

    time_s: float
    session_time_s: float
    remaining_s: float
    game_phase: int
    yellow_flag_state: int
    sector_flag_1: int
    sector_flag_2: int
    sector_flag_3: int
    player_flag: int
    fuel_l: float
    virtual_energy: float | None
    lap_invalidated: bool | None
    tyre_compound_front: str
    tyre_compound_rear: str
    tyre_temp_c_fl: float
    tyre_temp_c_fr: float
    tyre_temp_c_rl: float
    tyre_temp_c_rr: float
    tyre_wear_fl: float
    tyre_wear_fr: float
    tyre_wear_rl: float
    tyre_wear_rr: float
    tyre_pressure_kpa_fl: float
    tyre_pressure_kpa_fr: float
    tyre_pressure_kpa_rl: float
    tyre_pressure_kpa_rr: float
    dent_0: int
    dent_1: int
    dent_2: int
    dent_3: int
    dent_4: int
    dent_5: int
    dent_6: int
    dent_7: int
    overheating: bool
    detached: bool
    track_temp_c: float
    ambient_temp_c: float
    raining: float
    path_wetness_avg: float


@dataclass(frozen=True)
class CarStatus:
    """Scoring row of one car, including the player (cars_state_5hz, long format); fields as in VehicleState.
    time_s                float         s player telemetry clock when the scoring update was seen
    slot_id               int           game slot id
    is_player             bool
    driver_name           str
    vehicle_name          str
    class_name            str
    place                 int           1-based overall position
    completed_laps        int           laps
    lap_distance_m        float         m from the start/finish line
    lap_start_s           float         s session time the lap started
    sector_index          int           0, 1, 2
    best_lap_s            float | None  s
    last_lap_s            float | None  s
    best_sector1_s        float | None  s
    best_sector2_cum_s    float | None  s (sector 1 + sector 2)
    time_behind_next_s    float         s
    laps_behind_next      int           laps
    time_behind_leader_s  float         s
    laps_behind_leader    int           laps
    in_pits               bool
    in_garage             bool
    pit_state             int           0 none, 1 request, 2 entering, 3 stopped, 4 exiting
    pitstops              int           count
    last_sector1_s        float | None  s
    last_sector2_cum_s    float | None  s (sector 1 + sector 2)
    """

    time_s: float
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
    last_sector1_s: float | None
    last_sector2_cum_s: float | None


@dataclass(frozen=True)
class Scoring:
    """Rows of one scoring update.
    state  StateSample
    cars   tuple[CarStatus, ...]  all cars, including the player
    """

    state: StateSample
    cars: tuple[CarStatus, ...]


@dataclass(frozen=True)
class Frame:
    """Rows of one player telemetry tick.
    session  SessionInfo
    player   PlayerSample
    cars     tuple[CarSample, ...]  all cars, including the player
    scoring  Scoring | None         set only when the scoring clock advanced
    """

    session: SessionInfo
    player: PlayerSample
    cars: tuple[CarSample, ...]
    scoring: Scoring | None
