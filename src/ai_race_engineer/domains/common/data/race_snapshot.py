from dataclasses import dataclass

from ai_race_engineer.domains.common.data.snapshot import Snapshot

GAME_PHASE_GREEN: int = 5
GAME_PHASE_FULL_COURSE_YELLOW: int = 6
GAME_PHASE_STOPPED: int = 7
GAME_PHASE_OVER: int = 8
FLAG_BLUE: int = 6
SECTOR_FLAG_CLEAR: int = 11


@dataclass(frozen=True)
class CarResources:
    """Fuel, virtual energy and finish status of one car with telemetry.
    slot_id          int           game slot id
    fuel_l           float         litres
    fuel_capacity_l  float         litres
    virtual_energy   float | None  fraction 0-1 (None when the game has none)
    finish_status    int           0 none, 1 finished, 2 DNF, 3 DQ
    last_impact_s          float   s session time of the car's last impact (0 before the first)
    last_impact_magnitude  float   the game's magnitude of that impact
    """

    slot_id: int
    fuel_l: float
    fuel_capacity_l: float
    virtual_energy: float | None
    finish_status: int
    last_impact_s: float = 0.0
    last_impact_magnitude: float = 0.0


@dataclass(frozen=True)
class LapSectors:
    """Sector times of one car: of its last completed lap and of the lap it is on.
    slot_id             int           game slot id
    last_sector1_s      float | None  s; None when the game has no valid time
    last_sector2_cum_s  float | None  s (sector 1 + sector 2); None when the game has no valid time
    sector1_s           float | None  s sector 1 of the lap in progress; None until it is finished or when not valid
    sector2_cum_s       float | None  s (sector 1 + sector 2) of the lap in progress; None until sector 2 is finished
    """

    slot_id: int
    last_sector1_s: float | None
    last_sector2_cum_s: float | None
    sector1_s: float | None = None
    sector2_cum_s: float | None = None


@dataclass(frozen=True)
class CarTyres:
    """Tyres of one car with telemetry.
    slot_id         int    game slot id
    compound_front  str    e.g. "Medium"
    compound_rear   str
    wear            tuple[float, float, float, float]  fraction left (FL, FR, RL, RR), 1 = new (the game reports it
                                                       for the player's car; other cars stay at 1)
    """

    slot_id: int
    compound_front: str
    compound_rear: str
    wear: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackFlags:
    """Flag state of the track and the player.
    game_phase         int                   mGamePhase (5 green, 6 full-course yellow, 7 stopped, 8 over)
    yellow_flag_state  int                   mYellowFlagState (full-course only)
    sector_flags       tuple[int, int, int]  mSectorFlag[0..2] of sector 1, 2, 3: 1 yellow, 11 clear
    player_flag        int                   player's mFlag: 0 green, 6 blue
    """

    game_phase: int
    yellow_flag_state: int
    sector_flags: tuple[int, int, int]
    player_flag: int


@dataclass(frozen=True)
class PenaltyState:
    """Track-limit count and penalties of the player as the game holds them; a value the source lacks is None.
    track_limit_steps  int | None  steps collected (points x steps_per_point)
    steps_per_point    int | None  steps that make one track-limit point
    steps_per_penalty  int | None  steps at which the game gives a penalty
    penalties          int | None  penalties outstanding
    """

    track_limit_steps: int | None
    steps_per_point: int | None
    steps_per_penalty: int | None
    penalties: int | None


@dataclass(frozen=True)
class RaceSnapshot(Snapshot):
    """Snapshot extended with every car's resources, the flags and every car's last-lap sector times.
    resources  tuple[CarResources, ...]  cars with telemetry
    flags      TrackFlags
    sectors    tuple[LapSectors, ...]    all cars
    time_remaining_s  float | None       s of session time left as the game counts it (mSessionTimeRemaining: the
                                         race length on the grid and the formation lap); None when the source has none
    tyres      tuple[CarTyres, ...]      cars with telemetry
    penalties  PenaltyState | None       track limits and penalties of the player; None = the source has none
    """

    resources: tuple[CarResources, ...]
    flags: TrackFlags
    sectors: tuple[LapSectors, ...] = ()
    time_remaining_s: float | None = None
    tyres: tuple[CarTyres, ...] = ()
    penalties: PenaltyState | None = None
