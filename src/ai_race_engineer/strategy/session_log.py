import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from ai_race_engineer.strategy.field import CarEstimate
from ai_race_engineer.strategy.pit_time import PitStopSample
from ai_race_engineer.strategy.race_state import FINISH_STATUS_FINISHED
from ai_race_engineer.strategy.sector_delta import splits
from ai_race_engineer.telemetry.race_snapshot import GAME_PHASE_GREEN, GAME_PHASE_OVER, RaceSnapshot
from ai_race_engineer.telemetry.snapshot import SessionState, VehicleState


@dataclass(frozen=True)
class PlayerLap:
    """One completed lap of the player.
    lap             int           lap number as the driver counts it, 1-based
    start_s         float         s session time the lap started
    lap_time_s      float         s, from the line-crossing times, so an invalidated lap has one too
    timed           bool          the game gave the lap a valid time
    sector1_s       float | None  s
    sector2_s       float | None  s
    sector3_s       float | None  s
    energy_used     float | None  fraction of virtual energy; None when unknown or when the amount rose
    fuel_used_l     float | None  litres; None when unknown or when the amount rose
    class_place     int           position in class at the end of the lap
    compound_front  str
    compound_rear   str
    pitted          bool          the car was in the pits during the lap
    green           bool          the lap started at a line crossing and the game phase was green for the whole lap
                                  (clean = green and not pitted)
    """

    lap: int
    start_s: float
    lap_time_s: float
    timed: bool
    sector1_s: float | None
    sector2_s: float | None
    sector3_s: float | None
    energy_used: float | None
    fuel_used_l: float | None
    class_place: int
    compound_front: str
    compound_rear: str
    pitted: bool
    green: bool


@dataclass(frozen=True)
class Note:
    """One note of the session.
    time_s  float  s session time
    lap     int    lap it belongs to, as the driver counts laps
    text    str
    auto    bool   written by the engineer itself (False = dictated by the driver)
    """

    time_s: float
    lap: int
    text: str
    auto: bool


@dataclass(frozen=True)
class SessionRecord:
    """Everything recorded about one session.
    session_id        str                        "<YYYYmmdd_HHMMSS>_<kind>", wall time of the first snapshot
    started_unix      float                      s unix time of the first snapshot
    track             str
    kind              str                        "test_day" | "practice" | "qualify" | "warmup" | "race"
    player_name       str
    vehicle_name      str
    class_name        str
    time_s            float                      s session time of the latest snapshot
    ended             bool                       the session is over
    vehicles          tuple[VehicleState, ...]   all cars at the latest snapshot
    grid_class_place  int                        the player's place in class at the first snapshot
    laps              tuple[PlayerLap, ...]      the player's completed laps
    notes             tuple[Note, ...]
    yellow_flags      int                        times a sector flag turned yellow
    pit_samples       tuple[PitStopSample, ...]  measured pit-lane passes of any car
    estimates         Mapping[int, CarEstimate]  slot id -> pace and usage measured in the session (no priors)
    straights         tuple[tuple[float, float], ...]  (start m, end m) of the learned straights
    finished          bool                       the finish was seen: the player took the chequered flag or the game
                                                 phase was over
    """

    session_id: str
    started_unix: float
    track: str
    kind: str
    player_name: str
    vehicle_name: str
    class_name: str
    time_s: float
    ended: bool
    vehicles: tuple[VehicleState, ...]
    grid_class_place: int
    laps: tuple[PlayerLap, ...]
    notes: tuple[Note, ...]
    yellow_flags: int
    pit_samples: tuple[PitStopSample, ...]
    estimates: Mapping[int, CarEstimate]
    straights: tuple[tuple[float, float], ...]
    finished: bool = False


class SessionLog:
    """Keeps the record of the running session: the player's laps, notes, yellow flags; a finished session's record
    is handed over on a queue. Lists are replaced, never mutated (read from other threads); update, record and
    add_note hold one lock, so a record is never built halfway through a session change and no note is lost.
    ended                 queue.Queue[SessionRecord]                  records of sessions that are over
    _pit_samples          Callable[[], tuple[PitStopSample, ...]]     the pit log's measured passes
    _estimates            Callable[[], Mapping[int, CarEstimate]]     what the field tracker measured (no priors)
    _straights            Callable[[], tuple[tuple[float, float], ...]]  the learned straights
    _sector_yellow_value  int                       mSectorFlag value that means yellow
    _now_unix             Callable[[], float]       s unix time now
    _session              SessionState | None       session of the latest snapshot; None before the first
    _session_id           str
    _started_unix         float                     s unix time of the session's first snapshot
    _player               VehicleState | None       the player at the session's first snapshot
    _grid_class_place     int
    _vehicles             tuple[VehicleState, ...]  all cars at the latest snapshot
    _laps                 tuple[PlayerLap, ...]
    _notes                tuple[Note, ...]
    _yellow_flags         int
    _sector_flags         tuple[int, int, int]      sector flags at the previous snapshot
    _player_stops         int                       pit samples of the player already noted
    _lap_start_s          float                     s session time the lap in progress started
    _completed_laps       int                       the player's completed laps at the previous snapshot
    _start_energy         float | None              fraction at the start of the lap in progress
    _start_fuel_l         float | None              litres at the start of the lap in progress
    _pitted               bool                      the player was in the pits during the lap in progress
    _green                bool                      the phase was green during the whole lap in progress so far
    _finished             bool                      the finish of the session has been seen
    _lock                 threading.Lock            held by update, record and add_note
    """

    def __init__(
        self,
        pit_samples: Callable[[], tuple[PitStopSample, ...]],
        estimates: Callable[[], Mapping[int, CarEstimate]],
        straights: Callable[[], tuple[tuple[float, float], ...]],
        sector_yellow_value: int,
        now_unix: Callable[[], float],
    ) -> None:
        """Store the sources read when a record is built.
        In:  pit_samples          Callable[[], tuple[PitStopSample, ...]]        measured pit-lane passes so far
             estimates            Callable[[], Mapping[int, CarEstimate]]        slot id -> measured pace and usage
             straights            Callable[[], tuple[tuple[float, float], ...]]  (start m, end m) per learned straight
             sector_yellow_value  int                                            mSectorFlag value that means yellow
             now_unix             Callable[[], float]                            s unix time now (of the recording in
                                                                                 a replay)
        Out:                      None
        """
        self.ended: queue.Queue[SessionRecord] = queue.Queue()
        self._pit_samples: Callable[[], tuple[PitStopSample, ...]] = pit_samples
        self._estimates: Callable[[], Mapping[int, CarEstimate]] = estimates
        self._straights: Callable[[], tuple[tuple[float, float], ...]] = straights
        self._sector_yellow_value: int = sector_yellow_value
        self._now_unix: Callable[[], float] = now_unix
        self._session: SessionState | None = None
        self._session_id: str = ""
        self._started_unix: float = 0.0
        self._player: VehicleState | None = None
        self._grid_class_place: int = 0
        self._vehicles: tuple[VehicleState, ...] = ()
        self._laps: tuple[PlayerLap, ...] = ()
        self._notes: tuple[Note, ...] = ()
        self._yellow_flags: int = 0
        self._sector_flags: tuple[int, int, int] = (0, 0, 0)
        self._player_stops: int = 0
        self._lap_start_s: float = 0.0
        self._completed_laps: int = 0
        self._start_energy: float | None = None
        self._start_fuel_l: float | None = None
        self._pitted: bool = False
        self._green: bool = False
        self._finished: bool = False
        self._lock: threading.Lock = threading.Lock()

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer; runs before the race tracker so a finished session's pit samples and
        estimates are still there). A new session is a change of session type or track, or the session clock going
        backwards; the old record is put on ended first. A lap is stored when the player's lap start time changes
        and the lap count went up; only such a line crossing starts a lap that can be clean (the game also sets the
        lap start time at the green without a crossing, so the lap of the start is never clean).
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        with self._lock:
            session, player = snapshot.session, snapshot.player.vehicle
            if _is_new_session(self._session, session):
                if self._session is not None:
                    self.ended.put(self._record(ended=True))
                self._start(snapshot)
            self._session, self._vehicles = session, snapshot.vehicles
            self._yellow_flags += sum(
                now == self._sector_yellow_value != before
                for now, before in zip(snapshot.flags.sector_flags, self._sector_flags, strict=True)
            )
            self._sector_flags = snapshot.flags.sector_flags
            self._note_stops(player.completed_laps)
            if player.lap_start_s != self._lap_start_s:
                crossed = player.completed_laps > self._completed_laps
                if crossed:
                    self._laps = (*self._laps, self._lap(snapshot))
                self._lap_start_s = player.lap_start_s
                self._start_energy, self._start_fuel_l = snapshot.player.virtual_energy, snapshot.player.fuel_l
                self._pitted, self._green = False, crossed
            self._completed_laps = player.completed_laps
            self._pitted = self._pitted or player.in_pits
            self._green = self._green and snapshot.flags.game_phase == GAME_PHASE_GREEN
            self._finished = self._finished or _finish_seen(snapshot)

    def record(self) -> SessionRecord | None:
        """The running session, built on request.
        In:  -
        Out: SessionRecord | None  None before the first snapshot
        """
        with self._lock:
            return None if self._session is None else self._record(ended=False)

    def clean_laps(self) -> tuple[tuple[float, float], ...]:
        """The player's clean laps (green, no pits) of the running session that have an energy figure.
        In:  -
        Out: tuple[tuple[float, float], ...]  (lap time s, energy fraction used) per lap, oldest first
        """
        return tuple(
            (lap.lap_time_s, lap.energy_used)
            for lap in self._laps
            if lap.green and not lap.pitted and lap.energy_used is not None
        )

    def add_note(self, text: str) -> Note:
        """Add a note of the driver to the running session (called from the ask thread).
        In:  text  str
        Out:       Note  stamped with the session time and the lap in progress
        """
        with self._lock:
            elapsed_s = 0.0 if self._session is None else self._session.elapsed_s
            note = Note(elapsed_s, self._completed_laps + 1, text, False)
            self._notes = (*self._notes, note)
            return note

    def _start(self, snapshot: RaceSnapshot) -> None:
        """Begin the record of a new session; the lap in progress has no usage figure.
        In:  snapshot  RaceSnapshot  the session's first snapshot
        Out:           None
        """
        player = snapshot.player.vehicle
        self._started_unix = self._now_unix()
        stamp = datetime.fromtimestamp(self._started_unix).strftime("%Y%m%d_%H%M%S")
        self._session_id = f"{stamp}_{snapshot.session.session_type.name.lower()}"
        self._player = player
        self._grid_class_place = _class_place(player, snapshot.vehicles)
        self._laps, self._notes, self._yellow_flags = (), (), 0
        self._sector_flags = snapshot.flags.sector_flags
        self._player_stops = sum(s.is_player for s in self._pit_samples())
        self._lap_start_s, self._completed_laps = player.lap_start_s, player.completed_laps
        self._start_energy, self._start_fuel_l = None, None
        self._pitted, self._green = False, False
        self._finished = False

    def _lap(self, snapshot: RaceSnapshot) -> PlayerLap:
        """The lap the player just completed.
        In:  snapshot  RaceSnapshot  first snapshot after the line crossing
        Out:           PlayerLap
        """
        player = snapshot.player.vehicle
        sectors = next((s for s in snapshot.sectors if s.slot_id == player.slot_id), None)
        sector_s = (
            (None, None, None)
            if sectors is None
            else splits(sectors.last_sector1_s, sectors.last_sector2_cum_s, player.last_lap_s)
        )
        return PlayerLap(
            lap=player.completed_laps,
            start_s=self._lap_start_s,
            lap_time_s=player.lap_start_s - self._lap_start_s,
            timed=player.last_lap_s is not None,
            sector1_s=sector_s[0],
            sector2_s=sector_s[1],
            sector3_s=sector_s[2],
            energy_used=_used(self._start_energy, snapshot.player.virtual_energy),
            fuel_used_l=_used(self._start_fuel_l, snapshot.player.fuel_l),
            class_place=_class_place(player, snapshot.vehicles),
            compound_front=snapshot.player.tyre_compound_front,
            compound_rear=snapshot.player.tyre_compound_rear,
            pitted=self._pitted or player.in_pits,
            green=self._green and snapshot.flags.game_phase == GAME_PHASE_GREEN,
        )

    def _note_stops(self, completed_laps: int) -> None:
        """Add an automatic note for every pit sample of the player that is new.
        In:  completed_laps  int  the player's completed laps now
        Out:                 None
        """
        stops = [s for s in self._pit_samples() if s.is_player]
        for stop in stops[self._player_stops :]:
            text = (
                f"Pit stop lap {completed_laps}: {stop.stationary_s:.1f} s stationary, "
                f"plus {stop.energy_added * 100:.0f} percent energy, plus {stop.fuel_added_l:.1f} litres"
            )
            self._notes = (*self._notes, Note(self._session.elapsed_s, completed_laps, text, True))
        self._player_stops = len(stops)

    def _record(self, ended: bool) -> SessionRecord:
        """The record of the session as it stands.
        In:  ended  bool  the session is over
        Out:        SessionRecord
        """
        return SessionRecord(
            session_id=self._session_id,
            started_unix=self._started_unix,
            track=self._session.track_name,
            kind=self._session.session_type.name.lower(),
            player_name=self._player.driver_name,
            vehicle_name=self._player.vehicle_name,
            class_name=self._player.class_name,
            time_s=self._session.elapsed_s,
            ended=ended,
            vehicles=self._vehicles,
            grid_class_place=self._grid_class_place,
            laps=self._laps,
            notes=self._notes,
            yellow_flags=self._yellow_flags,
            pit_samples=self._pit_samples(),
            estimates=self._estimates(),
            straights=self._straights(),
            finished=self._finished,
        )


def _is_new_session(prev: SessionState | None, cur: SessionState) -> bool:
    """Whether the current session differs from the previous one (the rule the event monitor uses).
    In:  prev  SessionState | None  None before the first snapshot
         cur   SessionState
    Out:       bool  True on the first snapshot, a session type or track change, or the session clock going backwards
    """
    return (
        prev is None
        or cur.session_type != prev.session_type
        or cur.track_name != prev.track_name
        or cur.elapsed_s < prev.elapsed_s
    )


def _class_place(player: VehicleState, vehicles: Sequence[VehicleState]) -> int:
    """The player's position in class.
    In:  player    VehicleState
         vehicles  Sequence[VehicleState]  all cars
    Out:           int  1-based
    """
    return 1 + sum(v.class_name == player.class_name and v.place < player.place for v in vehicles)


def _used(start: float | None, now: float | None) -> float | None:
    """Amount used over a lap.
    In:  start  float | None  amount at the start of the lap
         now    float | None  amount at its end
    Out:        float | None  None when either is unknown or the amount rose (refill)
    """
    return start - now if start is not None and now is not None and now <= start else None


def _finish_seen(snapshot: RaceSnapshot) -> bool:
    """Whether a snapshot shows the finish of the session.
    In:  snapshot  RaceSnapshot
    Out:           bool  True when the game phase is over or the player has taken the chequered flag
    """
    slot_id = snapshot.player.vehicle.slot_id
    return snapshot.flags.game_phase == GAME_PHASE_OVER or any(
        r.slot_id == slot_id and r.finish_status == FINISH_STATUS_FINISHED for r in snapshot.resources
    )
