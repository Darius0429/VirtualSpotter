from dataclasses import dataclass

from ai_race_engineer.domains.common.data.race_snapshot import GAME_PHASE_GREEN, GAME_PHASE_OVER, RaceSnapshot
from ai_race_engineer.domains.common.data.snapshot import SessionType, VehicleState

LAP: str = "lap"
OUT_LAP: str = "out lap"
IN_LAP: str = "in lap"
INVALID_LAP: str = "invalid lap"
ABORTED_LAP: str = "aborted lap"
RETIRED: str = "retired"
FIRST_LAP: str = "first lap"
FINISHED: int = 1


@dataclass(frozen=True)
class LapEnd:
    """One lap, or part of a lap, a car has just ended.
    vehicle  VehicleState  the car when it ended
    lap      int           lap number as the car's driver counts it
    name     str           "lap" = from line to line with a time; "out lap" = from leaving the pits to the line;
                           "in lap" = from the line, or from leaving the pits, to entering the pits; "invalid lap" =
                           from line to line, flagged by the game for track limits and not timed; "aborted lap" =
                           cut short by a jump to the garage; "retired" = the same in a race before the car has
                           finished; "first lap" = the first lap of a race, from the grid to the line
    time_s   float         s; the game's lap time for "lap" and "first lap", else the session time it took
    """

    vehicle: VehicleState
    lap: int
    name: str
    time_s: float


@dataclass(frozen=True)
class _Run:
    """A lap a car is driving on track.
    lap        int    lap number as the car's driver counts it
    started_s  float  s session time it left the pits or crossed the line
    out        bool   it started by leaving the pits
    """

    lap: int
    started_s: float
    out: bool


class LapTypeTracker:
    """Says what every lap of every car was: a timed lap, an out-lap, an in-lap, a lap from line to line flagged by
    the game for track limits and not timed, or a lap cut short by a jump to the garage (retiring, in a race). A lap
    a car is already on when it is first seen is left out, and so is a jump to the garage before a race has started,
    after a session is over or of a car that has finished.
    ended     tuple[LapEnd, ...]        the laps that ended at the last snapshot (replaced, never mutated)
    _runs     dict[int, _Run]           slot id -> the lap the car is driving on track
    _seen     dict[int, VehicleState]   slot id -> the car at the last snapshot
    _time_s   float                     s session time of the last snapshot
    _race     bool                      the session of the last snapshot is a race
    _player   int | None                slot id of the player at the last snapshot; None before the first
    _counts   bool | None               the game is timing the player's lap in progress (None when the source does
                                        not say)
    """

    def __init__(self) -> None:
        """Start without cars.
        In:  -
        Out: None
        """
        self.ended: tuple[LapEnd, ...] = ()
        self._runs: dict[int, _Run] = {}
        self._seen: dict[int, VehicleState] = {}
        self._time_s: float = 0.0
        self._race: bool = False
        self._player: int | None = None
        self._counts: bool | None = None

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer). A session clock that goes back starts again from nothing.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        now_s = snapshot.session.elapsed_s
        if now_s < self._time_s:
            self._runs, self._seen = {}, {}
        self._time_s = now_s
        race = snapshot.session.session_type is SessionType.RACE
        phase = snapshot.flags.game_phase
        idle = phase == GAME_PHASE_OVER or (race and phase < GAME_PHASE_GREEN)
        finished = {car.slot_id for car in snapshot.resources if car.finish_status == FINISHED}
        ended = [self._update_car(v, now_s, race, idle or v.slot_id in finished) for v in snapshot.vehicles]
        self.ended = tuple(end for end in ended if end is not None)
        self._seen = {vehicle.slot_id: vehicle for vehicle in snapshot.vehicles}
        self._race, player = race, snapshot.player.vehicle
        self._player, self._counts = player.slot_id, player.lap_time_counts

    def in_progress(self) -> str | None:
        """Provisional type of the lap the player is driving.
        In:  -
        Out: str | None  "out lap" (left the pits) | "first lap" (race lap 1 from the grid) | "invalid lap" (a lap from
                         the line the game is not timing) | "lap"; None in the pits or garage, or when the lap was not
                         seen from its start
        """
        run = self._runs.get(self._player)
        if run is None:
            return None
        if run.out:
            return FIRST_LAP if self._race and run.lap == 1 else OUT_LAP
        return INVALID_LAP if self._counts is False else LAP

    def _update_car(self, vehicle: VehicleState, now_s: float, race: bool, finished: bool) -> LapEnd | None:
        """Feed one car's state.
        In:  vehicle   VehicleState
             now_s     float         s session time
             race      bool          the session is a race
             finished  bool          the car has finished the session, or the session is over or a race not yet
                                     started: a jump to the garage then is not a lap cut short
        Out:           LapEnd | None  the lap the car has just ended, if any
        """
        before, run = self._seen.get(vehicle.slot_id), self._runs.get(vehicle.slot_id)
        if before is None:
            return None
        pitted, was_pitted = vehicle.in_pits or vehicle.in_garage, before.in_pits or before.in_garage
        if pitted:
            self._runs.pop(vehicle.slot_id, None)
            jumped = vehicle.in_garage and not before.in_pits
            name = (RETIRED if race else ABORTED_LAP) if jumped else IN_LAP
            ignored = run is None or (jumped and finished)
            return None if ignored else LapEnd(vehicle, run.lap, name, now_s - run.started_s)
        if was_pitted:
            self._runs[vehicle.slot_id] = _Run(vehicle.completed_laps + 1, now_s, True)
            return None
        if vehicle.completed_laps != before.completed_laps + 1:
            return None
        self._runs[vehicle.slot_id] = _Run(vehicle.completed_laps + 1, now_s, False)
        if run is None:
            return None
        if run.out and race and run.lap == 1:
            return LapEnd(vehicle, 1, FIRST_LAP, vehicle.last_lap_s or now_s - run.started_s)
        if run.out:
            return LapEnd(vehicle, run.lap, OUT_LAP, now_s - run.started_s)
        if vehicle.last_lap_s is None:
            return LapEnd(vehicle, run.lap, INVALID_LAP, now_s - run.started_s)
        return LapEnd(vehicle, run.lap, LAP, vehicle.last_lap_s)
