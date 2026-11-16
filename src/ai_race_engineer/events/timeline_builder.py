from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace

from ai_race_engineer.events.text import lap_time_text
from ai_race_engineer.strategy.pit_time import PitStopSample
from ai_race_engineer.strategy.race_finish import strategy_resource
from ai_race_engineer.strategy.race_plan import race_plan
from ai_race_engineer.strategy.race_state import FINISH_STATUS_FINISHED, PRE_GREEN_PHASES, RaceState
from ai_race_engineer.strategy.rejoin import predict_rejoin
from ai_race_engineer.strategy.rivals import Rival
from ai_race_engineer.strategy.timeline import Fields, TimelineEntry
from ai_race_engineer.telemetry.race_snapshot import (
    GAME_PHASE_FULL_COURSE_YELLOW,
    GAME_PHASE_GREEN,
    SECTOR_FLAG_CLEAR,
    CarResources,
    RaceSnapshot,
)
from ai_race_engineer.telemetry.snapshot import SessionState, SessionType, VehicleState

VIA_TEXT: dict[str, str] = {
    "track": "on track",
    "your_stop": "during your pit stop",
    "their_stop": "while {driver} was in the pits",
}
BASIS_TEXT: dict[str, str] = {"pre_race": "pre-race figures", "measured": "measured laps"}
PLAN_ORDER: tuple[str, ...] = ("race_laps", "stops", "pit_lap", "first_stop_add", "projected_class_place")


class TimelineBuilder:
    """Detects the events of a session for its timeline by comparing every snapshot and race state with the previous
    ones. No I/O and no clock other than the session time; a replay gives the same entries as the live session only
    at low replay speeds, because the replay samples the recording by wall time and so sees fewer snapshots the
    faster it runs. A pit-out is paired with its pit-in only within one run of the engineer: after a restart the
    pit-out of a car that was in the pits has no entry.
    _sector_yellow_value   int                    mSectorFlag value that means yellow
    _sector_of_index       Sequence[int]          sector number (1-3) of mSectorFlag[0], [1], [2]
    _yellow_clear_hold_s   float                  session s a sector must stay clear before the clear is recorded
    _position_hold_s       float                  session s a swap in class must hold before it is recorded
    _min_best_lap_gain_s   float                  s a best lap must improve on the last recorded one
    _min_energy_change     float                  fraction of energy the first-stop amount must move to be a change
    _min_fuel_change_l     float                  litres the first-stop amount of a fuel car must move to be a change
    _rival_groups          frozenset[str]         rival groups whose members are recorded
    _plan_place_hold_laps  int                    consecutive lap checks a new projected place must hold
    _rival_hold_s          float                  session s a car must be in or out of the rival set to be recorded
    _previous              SessionState | None    session of the previous snapshot; None before the first
    _phase                 int                    mGamePhase at the previous snapshot
    _green_seen            bool                   the green has been shown in this session
    _finish_recorded       bool                   the finish of this session has been recorded (or was seen at first)
    _yellow                list[bool]             per flag index: the sector is followed as yellow
    _yellow_recorded       list[bool]             per flag index: a yellow entry was made for the yellow followed
    _yellow_since_s        list[float]            per flag index: s session time the sector turned yellow
    _clear_since_s         list[float | None]     per flag index: s session time since a recorded yellow reads clear
    _fcy_since_s           float                  s session time the full course yellow started
    _class_slots           frozenset[int]         slot ids of the player's class at the previous snapshot
    _ahead                 frozenset[int]         slot ids of the class cars recorded as ahead of the player
    _class_place           int                    the player's recorded place in class
    _swaps                 dict[int, tuple[float, str]]  slot id -> (s session time, via) of a swap not yet recorded
    _in_pits               dict[int, bool]        slot id -> in pits at the previous snapshot
    _pit_ins               dict[int, tuple[int | None, int]]  slot id of a car with a recorded pit-in -> (predicted
                                                  place in class, number of pit samples at the pit-in)
    _class_best_s          float | None           s, the last recorded fastest lap of the class
    _player_best_s         float | None           s, the last recorded best lap of the player
    _state                 RaceState | None       race state at the previous snapshot
    _watched               dict[int, str] | None  slot id -> driver name of the recorded rivals; None before the set
    _rival_since_s         dict[int, float]       slot id -> s session time since a car differs from the recorded set
    _plan_figures          _PlanFigures | None    the last recorded plan; None before the first
    _plan_laps             int                    the player's completed laps when the plan was last compared
    _pre_race              bool                   the last recorded plan was on the pre-race figures
    _plan_stops_done       int                    pit-lane passes of the player when the plan was last compared
    _place_candidate       int                    projected place seen at the last lap check that differed
    _place_checks          int                    consecutive lap checks the candidate place has been seen
    """

    def __init__(
        self,
        sector_yellow_value: int,
        sector_of_index: Sequence[int],
        yellow_clear_hold_s: float,
        position_hold_s: float,
        min_best_lap_gain_s: float,
        min_energy_change: float,
        rival_groups: Collection[str],
        plan_place_hold_laps: int,
        rival_hold_s: float,
        min_fuel_change_l: float,
    ) -> None:
        """Set how the flags are read and the thresholds.
        In:  sector_yellow_value  int              mSectorFlag value that means yellow
             sector_of_index      Sequence[int]    sector number (1-3) of mSectorFlag[0], [1], [2]
             yellow_clear_hold_s  float            session s a sector must stay clear before the clear is recorded
             position_hold_s      float            session s a swap in class must hold before it is recorded
             min_best_lap_gain_s  float            s a best lap must improve on the last recorded one
             min_energy_change    float            fraction of energy the first-stop amount must move to count as
                                                   a plan change
             rival_groups         Collection[str]  rival groups whose members are recorded
             plan_place_hold_laps  int             consecutive lap checks a new projected place must hold before it
                                                   is a plan change
             rival_hold_s         float            session s a car must be in, or out of, the rival set before that
                                                   is recorded
             min_fuel_change_l    float            litres the first-stop amount of a car without virtual energy
                                                   must move to count as a plan change
        Out:                      None
        """
        self._sector_yellow_value: int = sector_yellow_value
        self._sector_of_index: Sequence[int] = sector_of_index
        self._yellow_clear_hold_s: float = yellow_clear_hold_s
        self._position_hold_s: float = position_hold_s
        self._min_best_lap_gain_s: float = min_best_lap_gain_s
        self._min_energy_change: float = min_energy_change
        self._min_fuel_change_l: float = min_fuel_change_l
        self._rival_groups: frozenset[str] = frozenset(rival_groups)
        self._plan_place_hold_laps: int = plan_place_hold_laps
        self._rival_hold_s: float = rival_hold_s
        self._previous: SessionState | None = None
        self._phase: int = 0
        self._green_seen: bool = False
        self._finish_recorded: bool = False
        self._yellow: list[bool] = [False, False, False]
        self._yellow_recorded: list[bool] = [False, False, False]
        self._yellow_since_s: list[float] = [0.0, 0.0, 0.0]
        self._clear_since_s: list[float | None] = [None, None, None]
        self._fcy_since_s: float = 0.0
        self._class_slots: frozenset[int] = frozenset()
        self._ahead: frozenset[int] = frozenset()
        self._class_place: int = 0
        self._swaps: dict[int, tuple[float, str]] = {}
        self._in_pits: dict[int, bool] = {}
        self._pit_ins: dict[int, tuple[int | None, int]] = {}
        self._class_best_s: float | None = None
        self._player_best_s: float | None = None
        self._state: RaceState | None = None
        self._watched: dict[int, str] | None = None
        self._rival_since_s: dict[int, float] = {}
        self._plan_figures: _PlanFigures | None = None
        self._plan_laps: int = 0
        self._pre_race: bool = True
        self._plan_stops_done: int = 0
        self._place_candidate: int = 0
        self._place_checks: int = 0

    def update(
        self,
        snapshot: RaceSnapshot,
        state: RaceState | None,
        samples: Sequence[PitStopSample],
        pre_race: bool,
        finished: bool,
        plan_recorded: bool | None = False,
        rivals_recorded: bool | None = False,
    ) -> tuple[TimelineEntry, ...]:
        """Entries for what changed since the previous snapshot. A new session (first snapshot, session type or
        track change, clock going backwards) clears everything and yields the "start" entry first; flags, places,
        pits and best laps are then taken as they are without an entry. Rival and plan entries are made only when
        the race state is a new one. The initial plan and the rival set are not made again when the session's
        timeline already has them (an engineer started again in the session), and wait while that is not known.
        In:  snapshot         RaceSnapshot
             state            RaceState | None         the race tracker's state after this snapshot
             samples          Sequence[PitStopSample]  all pit-lane passes measured so far
             pre_race         bool                     the pre-race figures are in use
             finished         bool                     the race is over for the player
             plan_recorded    bool | None              the session's timeline already has a plan entry; None = not
                                                       known yet
             rivals_recorded  bool | None              the session's timeline already has a rival entry; None = not
                                                       known yet
        Out:                  tuple[TimelineEntry, ...]  in the order they are to be stored
        """
        session, previous = snapshot.session, self._previous
        entries: list[TimelineEntry] = []
        if (
            previous is None
            or session.session_type != previous.session_type
            or session.track_name != previous.track_name
            or session.elapsed_s < previous.elapsed_s
        ):
            entries.append(self._start(snapshot, finished))
        else:
            entries.extend(self._session(snapshot, finished))
            entries.extend(self._flags(snapshot))
            entries.extend(self._positions(snapshot))
            entries.extend(self._pits(snapshot, state, samples))
            entries.extend(self._best_laps(snapshot))
        if state is not None and state is not self._state:
            entries.extend(self._rivals(snapshot, state, pre_race, rivals_recorded))
            entries.extend(self._plan(snapshot, state, samples, pre_race, plan_recorded))
        self._previous, self._phase, self._state = session, snapshot.flags.game_phase, state
        self._green_seen = self._green_seen or snapshot.flags.game_phase == GAME_PHASE_GREEN
        return tuple(entries)

    def _start(self, snapshot: RaceSnapshot, finished: bool) -> TimelineEntry:
        """Forget the previous session, take the first snapshot of the new one as it is and make its "start" entry.
        In:  snapshot  RaceSnapshot  the session's first snapshot
             finished  bool          the race is over for the player
        Out:           TimelineEntry
        """
        session, player = snapshot.session, snapshot.player.vehicle
        in_class = [v for v in snapshot.vehicles if v.class_name == player.class_name]
        yellow = [value == self._sector_yellow_value for value in snapshot.flags.sector_flags]
        self._green_seen, self._finish_recorded = False, _player_finished(snapshot, finished)
        self._yellow, self._yellow_since_s = yellow, [session.elapsed_s] * 3
        self._yellow_recorded = [False, False, False]
        self._clear_since_s, self._fcy_since_s = [None, None, None], session.elapsed_s
        self._class_slots = frozenset(v.slot_id for v in in_class)
        self._ahead = frozenset(v.slot_id for v in in_class if v.place < player.place)
        self._class_place, self._swaps = len(self._ahead) + 1, {}
        self._in_pits, self._pit_ins = {v.slot_id: v.in_pits for v in snapshot.vehicles}, {}
        self._class_best_s = min((v.best_lap_s for v in in_class if v.best_lap_s is not None), default=None)
        self._player_best_s = player.best_lap_s
        self._state, self._watched, self._plan_figures = None, None, None
        kind = session.session_type.name.lower()
        return _entry(
            snapshot,
            "session",
            f"Engineer watching: {kind.replace('_', ' ')} at {session.track_name}, lap {player.completed_laps + 1}, "
            f"P{self._class_place} in class",
            "start",
            {"kind": kind, "track": session.track_name, "class_place": self._class_place},
        )

    def _session(self, snapshot: RaceSnapshot, finished: bool) -> list[TimelineEntry]:
        """The green flag of a race start and the finish of a race. The finish is the player's own chequered flag,
        not the leader's: the game phase is over from the leader's flag on, while the player still completes the
        lap. Its entry is stamped with the last lap the player completed.
        In:  snapshot  RaceSnapshot
             finished  bool          the race is over for the player
        Out:           list[TimelineEntry]
        """
        player = snapshot.player.vehicle
        if snapshot.session.session_type != SessionType.RACE:
            return []
        place = _class_place(player, snapshot.vehicles)
        entries: list[TimelineEntry] = []
        if snapshot.flags.game_phase == GAME_PHASE_GREEN and self._phase in PRE_GREEN_PHASES and not self._green_seen:
            text = f"Green flag, race started from P{place} in class"
            entries.append(_entry(snapshot, "session", text, "green", {"class_place": place}))
        if _player_finished(snapshot, finished) and not self._finish_recorded:
            self._finish_recorded = True
            laps = player.completed_laps
            text = f"Race finished P{place} in class after {laps} laps"
            entry = _entry(snapshot, "session", text, "finish", {"class_place": place, "laps": laps})
            entries.append(replace(entry, lap=laps))
        return entries

    def _flags(self, snapshot: RaceSnapshot) -> list[TimelineEntry]:
        """Sector yellows raised and cleared, and a full course yellow starting and ending. Sector flags are
        recorded only while the game phase is green; in any other phase they are followed without an entry. Only
        the clear value held for the hold clears a yellow with an entry, and only a yellow that has an entry (one
        first seen outside green, or already up at the session start, clears without one); any other value ends it
        without one, and a yellow raised again within the hold is not recorded again.
        In:  snapshot  RaceSnapshot
        Out:           list[TimelineEntry]
        """
        time_s, phase = snapshot.session.elapsed_s, snapshot.flags.game_phase
        entries: list[TimelineEntry] = []
        for index, (sector, value) in enumerate(zip(self._sector_of_index, snapshot.flags.sector_flags, strict=True)):
            since_s = self._clear_since_s[index]
            self._clear_since_s[index] = None
            if value == self._sector_yellow_value:
                if not self._yellow[index]:
                    self._yellow[index], self._yellow_since_s[index] = True, time_s
                    self._yellow_recorded[index] = phase == GAME_PHASE_GREEN
                    if phase == GAME_PHASE_GREEN:
                        fields = {"sector": sector, "player_sector": snapshot.player.vehicle.sector_index + 1}
                        entries.append(_entry(snapshot, "flag", f"Yellow flag in sector {sector}", "yellow", fields))
            elif value != SECTOR_FLAG_CLEAR or phase != GAME_PHASE_GREEN:
                self._yellow[index] = False
            elif self._yellow[index] and since_s is not None and time_s - since_s >= self._yellow_clear_hold_s:
                self._yellow[index] = False
                if self._yellow_recorded[index]:
                    duration_s = since_s - self._yellow_since_s[index]
                    text = f"Sector {sector} clear after {duration_s:.0f} seconds"
                    fields = {"sector": sector, "duration_s": duration_s}
                    entries.append(_entry(snapshot, "flag", text, "clear", fields))
            elif self._yellow[index]:
                self._clear_since_s[index] = time_s if since_s is None else since_s
        if phase == GAME_PHASE_FULL_COURSE_YELLOW != self._phase:
            self._fcy_since_s = time_s
            entries.append(_entry(snapshot, "flag", "Full course yellow", "fcy_start", {}))
        if self._phase == GAME_PHASE_FULL_COURSE_YELLOW and phase == GAME_PHASE_GREEN:
            duration_s = time_s - self._fcy_since_s
            text = f"Green flag after {duration_s:.0f} seconds of full course yellow"
            entries.append(_entry(snapshot, "flag", text, "fcy_end", {"duration_s": duration_s}))
        return entries

    def _positions(self, snapshot: RaceSnapshot) -> list[TimelineEntry]:
        """Places gained and lost in class, in a race after the green: a class car that changed side relative to
        the player is recorded once the new order has held for the hold, with how it happened as seen at the first
        snapshot of the swap. A place that changed without a car changing side (a car left or joined) is recorded
        without a cause.
        In:  snapshot  RaceSnapshot
        Out:           list[TimelineEntry]
        """
        time_s, player = snapshot.session.elapsed_s, snapshot.player.vehicle
        in_class = {v.slot_id: v for v in snapshot.vehicles if v.class_name == player.class_name}
        slots = frozenset(in_class)
        ahead = frozenset(slot_id for slot_id, v in in_class.items() if v.place < player.place)
        recorded = (self._ahead & slots) | (ahead - self._class_slots)
        self._class_slots = slots
        if snapshot.session.session_type != SessionType.RACE or not self._green_seen:
            self._ahead, self._class_place, self._swaps = ahead, len(ahead) + 1, {}
            return []
        swapped = ahead ^ recorded
        self._swaps = {
            slot_id: self._swaps.get(slot_id, (time_s, _via(player, in_class[slot_id]))) for slot_id in swapped
        }
        entries: list[TimelineEntry] = []
        held = sorted(s for s, (since_s, _) in self._swaps.items() if time_s - since_s >= self._position_hold_s)
        for slot_id in held:
            gained, via = slot_id in recorded, self._swaps.pop(slot_id)[1]
            recorded = recorded ^ {slot_id}
            entries.append(_swap_entry(snapshot, in_class[slot_id].driver_name, gained, len(recorded) + 1, via))
        place = len(recorded) + 1
        if place != self._class_place and not entries:
            fields = {"driver": None, "gained": place < self._class_place, "class_place": place, "via": None}
            entries.append(_entry(snapshot, "position", f"Now P{place} in class", "place", fields))
        self._ahead, self._class_place = recorded, place
        return entries

    def _pits(
        self, snapshot: RaceSnapshot, state: RaceState | None, samples: Sequence[PitStopSample]
    ) -> list[TimelineEntry]:
        """Pit entries and exits of the player and of the recorded rival groups, until the player has finished. A
        pit-in is a car entering the pit lane from the track (not one in the garage); its pit-out is paired with
        the pass measured meanwhile and the place predicted at the pit-in. A car that left the session is forgotten.
        In:  snapshot  RaceSnapshot
             state     RaceState | None         the race tracker's state after this snapshot
             samples   Sequence[PitStopSample]  all pit-lane passes measured so far
        Out:           list[TimelineEntry]
        """
        rivals = set() if state is None else {r.slot_id for r in state.rivals if self._rival_groups & set(r.groups)}
        resources = {r.slot_id: r for r in snapshot.resources}
        entries: list[TimelineEntry] = []
        for vehicle in () if self._finish_recorded else snapshot.vehicles:
            slot_id = vehicle.slot_id
            entered = vehicle.in_pits and self._in_pits.get(slot_id) is False and not vehicle.in_garage
            if entered and (vehicle.is_player or slot_id in rivals):
                stop_s, predicted = _prediction(state, slot_id)
                self._pit_ins[slot_id] = (predicted, len(samples))
                place = _class_place(vehicle, snapshot.vehicles)
                entries.append(_pit_in_entry(snapshot, vehicle, place, stop_s, predicted))
            elif not vehicle.in_pits and slot_id in self._pit_ins:
                predicted, count = self._pit_ins.pop(slot_id)
                sample = next((s for s in samples[count:] if s.slot_id == slot_id), None)
                place = _class_place(vehicle, snapshot.vehicles)
                entries.append(_pit_out_entry(snapshot, vehicle, place, sample, resources.get(slot_id), predicted))
        self._in_pits = {v.slot_id: v.in_pits for v in snapshot.vehicles}
        self._pit_ins = {slot_id: pit_in for slot_id, pit_in in self._pit_ins.items() if slot_id in self._in_pits}
        return entries

    def _best_laps(self, snapshot: RaceSnapshot) -> list[TimelineEntry]:
        """A new fastest lap of the player's class and a new personal best, each when it improves on the last
        recorded one by the gain threshold; one entry when the player sets the fastest lap of the class.
        In:  snapshot  RaceSnapshot
        Out:           list[TimelineEntry]
        """
        player = snapshot.player.vehicle
        timed = [v for v in snapshot.vehicles if v.class_name == player.class_name and v.best_lap_s is not None]
        holder = min(timed, key=lambda v: v.best_lap_s, default=None)
        entries: list[TimelineEntry] = []
        if holder is not None and self._improved(holder.best_lap_s, self._class_best_s):
            self._class_best_s = holder.best_lap_s
            text = f"Fastest lap in class: {holder.driver_name}, {lap_time_text(holder.best_lap_s)}"
            fields = {"driver": holder.driver_name, "lap_s": holder.best_lap_s, "is_player": holder.is_player}
            entries.append(_entry(snapshot, "best_lap", text, "class_best", fields))
            if holder.is_player:
                self._player_best_s = holder.best_lap_s
        if player.best_lap_s is not None and self._improved(player.best_lap_s, self._player_best_s):
            self._player_best_s = player.best_lap_s
            text = f"Personal best, {lap_time_text(player.best_lap_s)}"
            fields = {"driver": player.driver_name, "lap_s": player.best_lap_s, "is_player": True}
            entries.append(_entry(snapshot, "best_lap", text, "personal_best", fields))
        return entries

    def _rivals(
        self, snapshot: RaceSnapshot, state: RaceState, pre_race: bool, recorded: bool | None
    ) -> list[TimelineEntry]:
        """The rival set at the first state that is neither before the green nor on the pre-race figures, then
        every car that has been in it, or out of it, for the rival hold. A car that leaves and returns within the
        hold, or stays while its groups change, gives no entry. When the session's timeline already has rival
        entries the set is taken as it is without an entry.
        In:  snapshot  RaceSnapshot
             state     RaceState     a new race state
             pre_race  bool          the pre-race figures are in use
             recorded  bool | None   the session's timeline already has a rival entry; None = not known yet
        Out:           list[TimelineEntry]
        """
        time_s = snapshot.session.elapsed_s
        rivals = {r.slot_id: r for r in state.rivals if self._rival_groups & set(r.groups)}
        if self._watched is None:
            if state.before_green or pre_race or recorded is None:
                return []
            self._watched, self._rival_since_s = {slot_id: r.driver_name for slot_id, r in rivals.items()}, {}
            if recorded:
                return []
            listed = ", ".join(f"{r.driver_name} {self._groups(r)}" for r in rivals.values()) or "none"
            fields = {"drivers": ", ".join(self._watched.values())}
            return [_entry(snapshot, "rival", f"Rivals: {listed}", "set", fields)]
        self._rival_since_s = {
            slot_id: self._rival_since_s.get(slot_id, time_s) for slot_id in rivals.keys() ^ self._watched.keys()
        }
        held = sorted(s for s, since_s in self._rival_since_s.items() if time_s - since_s >= self._rival_hold_s)
        watched = dict(self._watched)
        entries: list[TimelineEntry] = []
        for slot_id in held:
            del self._rival_since_s[slot_id]
            if slot_id in rivals:
                watched[slot_id] = rivals[slot_id].driver_name
                entries.append(self._new_rival_entry(snapshot, rivals[slot_id]))
            else:
                name = watched.pop(slot_id)
                entries.append(_entry(snapshot, "rival", f"{name} is no longer a rival", "gone", {"driver": name}))
        self._watched = watched
        return entries

    def _plan(
        self,
        snapshot: RaceSnapshot,
        state: RaceState,
        samples: Sequence[PitStopSample],
        pre_race: bool,
        recorded: bool | None,
    ) -> list[TimelineEntry]:
        """The plan at the first race state that has a usage figure for the player (at the green at the latest),
        again in full when the pre-race figures give way to the measured laps, and its changes at the first new
        state after each lap the player completes, once the player is out of the pits. What follows from a stop
        the player has completed since the last check (one stop fewer, the next pit lap and amount) is taken over
        without an entry. A projected place is a change only once it has held for the place hold. A pit lap and a
        first-stop amount that are gone are named in the fields only: the text then says "no stop". When the
        session's timeline already has plan entries the first plan is taken as it is without an entry.
        In:  snapshot  RaceSnapshot
             state     RaceState                a new race state
             samples   Sequence[PitStopSample]  all pit-lane passes measured so far
             pre_race  bool                     the pre-race figures are in use
             recorded  bool | None              the session's timeline already has a plan entry; None = not known yet
        Out:           list[TimelineEntry]
        """
        player = snapshot.player.vehicle
        laps, stops_done = player.completed_laps, sum(s.is_player for s in samples)
        figures, old = _plan_figures(state, pre_race), self._plan_figures
        if old is None and (recorded is None or (figures.first_stop_add is None and state.before_green)):
            return []
        if old is None and recorded:
            self._plan_figures, self._plan_laps, self._pre_race = figures, laps, pre_race
            self._plan_stops_done, self._place_checks = stops_done, 0
            return []
        if old is None or (self._pre_race and not pre_race):
            event, lead, changed = ("initial", f"Plan ({BASIS_TEXT[figures.basis]})", None)
            if old is not None:
                event, lead = "basis", "Plan now from measured laps"
            text = f"{lead}: {', '.join(part for part in _plan_parts(figures) if part)}"
            self._place_checks = 0
        elif laps > self._plan_laps and not player.in_pits:
            if stops_done > self._plan_stops_done:
                old = replace(old, stops=figures.stops, pit_lap=figures.pit_lap, first_stop_add=figures.first_stop_add)
            place = self._held_place(figures.projected_class_place, old.projected_class_place)
            figures = replace(figures, projected_class_place=place)
            min_change = self._min_fuel_change_l if figures.unit == "l" else self._min_energy_change
            names = _changed(figures, old, min_change)
            if not names:
                self._plan_figures, self._plan_laps, self._plan_stops_done = old, laps, stops_done
                return []
            parts, was = _plan_parts(figures), _plan_parts(old)
            text = "Plan change: " + ", ".join(
                f"{parts[i]} (was {_was(was[i])})" for i, name in enumerate(PLAN_ORDER) if name in names and parts[i]
            )
            event, changed = "change", ", ".join(names)
        else:
            return []
        self._plan_figures, self._plan_laps, self._pre_race, self._plan_stops_done = figures, laps, pre_race, stops_done
        fields = _plan_fields(figures) if changed is None else {**_plan_fields(figures), "changed": changed}
        return [_entry(snapshot, "plan", text, event, fields)]

    def _held_place(self, place: int, recorded: int) -> int:
        """The projected place to compare at a lap check: a new place counts only once it has been the same for the
        place hold of consecutive checks.
        In:  place     int  projected place in class now
             recorded  int  projected place in class of the last recorded plan
        Out:           int  place when it equals the recorded one or has held, else recorded
        """
        if place == recorded:
            self._place_checks = 0
            return recorded
        self._place_checks = self._place_checks + 1 if place == self._place_candidate else 1
        self._place_candidate = place
        return place if self._place_checks >= self._plan_place_hold_laps else recorded

    def _new_rival_entry(self, snapshot: RaceSnapshot, rival: Rival) -> TimelineEntry:
        """Entry for a car that became a rival.
        In:  snapshot  RaceSnapshot
             rival     Rival
        Out:           TimelineEntry
        """
        side = "ahead" if rival.gap_s > 0 else "behind"
        text = (
            f"{rival.driver_name} became a rival ({self._groups(rival)}), P{rival.class_place} in class, "
            f"{abs(rival.gap_s):.1f} seconds {side}"
        )
        fields = {
            "driver": rival.driver_name,
            "groups": self._groups(rival),
            "class_place": rival.class_place,
            "gap_s": rival.gap_s,
        }
        return _entry(snapshot, "rival", text, "new", fields)

    def _improved(self, lap_s: float, recorded_s: float | None) -> bool:
        """Whether a best lap is to be recorded.
        In:  lap_s       float         s
             recorded_s  float | None  s, the last recorded one; None without one
        Out:             bool          True without a recorded one or when faster by the gain threshold
        """
        return recorded_s is None or recorded_s - lap_s >= self._min_best_lap_gain_s

    def _groups(self, rival: Rival) -> str:
        """The recorded groups a rival belongs to.
        In:  rival  Rival
        Out:        str    e.g. "ahead" or "behind, strategy"
        """
        return ", ".join(group for group in rival.groups if group in self._rival_groups)


@dataclass(frozen=True)
class _PlanFigures:
    """What is compared and recorded of the player's plan.
    basis                  str           "pre_race" | "measured"
    race_laps              int           laps the player's race will have
    stops                  int           stops still to make
    pit_lap                int | None    lap at the end of which to pit, as the driver counts; None without a stop
    first_stop_add         float | None  fraction of energy (litres for a fuel car) added at the first stop; None
                                         when the usage is unknown
    unit                   str           "percent" | "l", unit first_stop_add is recorded in
    projected_class_place  int           place in class at the flag
    """

    basis: str
    race_laps: int
    stops: int
    pit_lap: int | None
    first_stop_add: float | None
    unit: str
    projected_class_place: int


def _entry(snapshot: RaceSnapshot, kind: str, text: str, event: str, fields: Fields) -> TimelineEntry:
    """Entry stamped with the session time and the player's lap in progress.
    In:  snapshot  RaceSnapshot
         kind      str     entry kind
         text      str     short sentence
         event     str     sub-kind, stored as fields["event"]
         fields    Fields  the other structured values
    Out:           TimelineEntry
    """
    lap = snapshot.player.vehicle.completed_laps + 1
    return TimelineEntry(snapshot.session.elapsed_s, lap, kind, text, {"event": event, **fields})


def _player_finished(snapshot: RaceSnapshot, finished: bool) -> bool:
    """Whether the player has finished the race.
    In:  snapshot  RaceSnapshot
         finished  bool          the race is over for the player, as the race tracker sees it
    Out:           bool          the player's own finish status when the player has resources, else finished
    """
    slot_id = snapshot.player.vehicle.slot_id
    status = next((r.finish_status for r in snapshot.resources if r.slot_id == slot_id), None)
    return finished if status is None else status == FINISH_STATUS_FINISHED


def _class_place(vehicle: VehicleState, vehicles: Sequence[VehicleState]) -> int:
    """Position of a car in its class.
    In:  vehicle   VehicleState
         vehicles  Sequence[VehicleState]  all cars
    Out:           int  1-based
    """
    return 1 + sum(v.class_name == vehicle.class_name and v.place < vehicle.place for v in vehicles)


def _via(player: VehicleState, other: VehicleState) -> str:
    """How a swap with another car is happening.
    In:  player  VehicleState
         other   VehicleState  the car changing side
    Out:         str  "your_stop" with the player in the pits, else "their_stop" with the other car there, else
                      "track"
    """
    if player.in_pits:
        return "your_stop"
    return "their_stop" if other.in_pits else "track"


def _swap_entry(snapshot: RaceSnapshot, driver: str, gained: bool, place: int, via: str) -> TimelineEntry:
    """Entry for a place gained from or lost to one car.
    In:  snapshot  RaceSnapshot
         driver    str   the other car's driver
         gained    bool  the player passed the car
         place     int   the player's place in class after the swap
         via       str   "track" | "your_stop" | "their_stop"
    Out:           TimelineEntry
    """
    how = VIA_TEXT[via].format(driver=driver)
    text = (
        f"Passed {driver} for P{place} in class, {how}"
        if gained
        else f"Lost P{place - 1} to {driver}, now P{place} in class, {how}"
    )
    fields = {"driver": driver, "gained": gained, "class_place": place, "via": via}
    return _entry(snapshot, "position", text, "swap", fields)


def _prediction(state: RaceState | None, slot_id: int) -> tuple[float | None, int | None]:
    """Stop time and rejoin place of a car pitting now, from the race state.
    In:  state    RaceState | None
         slot_id  int
    Out:          tuple[float | None, int | None]  (s lost in total, place in class after the stop); (None, None)
                                                   without a race state or for a car that is not projected
    """
    car = None if state is None else next((c for c in state.cars if c.slot_id == slot_id), None)
    if car is None:
        return None, None
    rejoin = predict_rejoin(
        car,
        state.projection.cars[slot_id].laps_remaining,
        [c for c in state.cars if c.slot_id != slot_id],
        state.pit,
        state.traffic_window_s,
    )
    return rejoin.stop_time_s, rejoin.class_place


def _pit_in_entry(
    snapshot: RaceSnapshot, vehicle: VehicleState, place: int, stop_s: float | None, predicted: int | None
) -> TimelineEntry:
    """Entry for a car entering the pit lane.
    In:  snapshot   RaceSnapshot
         vehicle    VehicleState  the car
         place      int           its place in class
         stop_s     float | None  s the stop is predicted to cost in total, pit lane included; None when unknown
         predicted  int | None    place in class predicted after the stop; None when unknown
    Out:            TimelineEntry
    """
    text = f"{'You pit' if vehicle.is_player else vehicle.driver_name + ' pits'} from P{place}"
    if predicted is not None:
        text += f", expected time loss {stop_s:.0f} seconds, rejoin P{predicted} in class"
    fields = {
        "driver": vehicle.driver_name,
        "is_player": vehicle.is_player,
        "class_place": place,
        "predicted_stop_s": stop_s,
        "predicted_class_place": predicted,
    }
    return _entry(snapshot, "pit", text, "in", fields)


def _pit_out_entry(
    snapshot: RaceSnapshot,
    vehicle: VehicleState,
    place: int,
    sample: PitStopSample | None,
    resources: CarResources | None,
    predicted: int | None,
) -> TimelineEntry:
    """Entry for a car back on track after a recorded pit-in.
    In:  snapshot    RaceSnapshot
         vehicle     VehicleState          the car
         place       int                   its place in class at the pit exit
         sample      PitStopSample | None  the measured pass; None when it was dropped
         resources   CarResources | None   None for a car without telemetry (its amounts are then unknown)
         predicted   int | None            place in class predicted at the pit-in; None when unknown
    Out:             TimelineEntry
    """
    known = sample is not None and resources is not None
    energy = sample.energy_added * 100 if known and resources.virtual_energy is not None else None
    subject = "You are" if vehicle.is_player else vehicle.driver_name
    stood = "of the pits" if sample is None else f"after {sample.stationary_s:.1f} seconds stationary"
    parts = [f"{subject} out {stood}"]
    if energy is not None:
        parts.append(f"plus {energy:.0f} percent energy")
    parts.append(f"rejoined P{place} in class")
    if predicted is not None:
        parts.append(f"predicted P{predicted}")
    fields = {
        "driver": vehicle.driver_name,
        "is_player": vehicle.is_player,
        "stationary_s": None if sample is None else sample.stationary_s,
        "energy_added_percent": energy,
        "fuel_added_l": sample.fuel_added_l if known else None,
        "class_place": place,
        "predicted_class_place": predicted,
    }
    return _entry(snapshot, "pit", ", ".join(parts), "out", fields)


def _plan_figures(state: RaceState, pre_race: bool) -> _PlanFigures:
    """The player's plan in a race state.
    In:  state     RaceState
         pre_race  bool       the pre-race figures are in use
    Out:           _PlanFigures
    """
    car = next(c for c in state.cars if c.slot_id == state.player_slot_id)
    mine = state.projection.cars[state.player_slot_id]
    plan, resource = race_plan(car, mine), strategy_resource(car)
    return _PlanFigures(
        basis="pre_race" if pre_race else "measured",
        race_laps=round(car.progress_laps + mine.laps_remaining),
        stops=mine.stops_remaining,
        pit_lap=None if plan is None else plan.pit_lap,
        first_stop_add=None if plan is None else plan.first_stop_add,
        unit="l" if resource is not None and resource[0] == "fuel" else "percent",
        projected_class_place=mine.class_place,
    )


def _plan_parts(figures: _PlanFigures) -> list[str]:
    """The plan as parts of a sentence, one per compared figure (race laps, stops, pit lap, first-stop amount,
    projected place).
    In:  figures  _PlanFigures
    Out:          list[str]  five parts; an empty one for a figure there is nothing to say about
    """
    stops = "no stop" if figures.stops == 0 else f"{figures.stops} stop{'s' if figures.stops > 1 else ''}"
    amount = "" if figures.first_stop_add is None or figures.pit_lap is None else _amount_text(figures)
    return [
        f"{figures.race_laps} laps",
        stops,
        "" if figures.pit_lap is None else f"pit end of lap {figures.pit_lap}",
        amount,
        f"finish P{figures.projected_class_place} in class",
    ]


def _amount_text(figures: _PlanFigures) -> str:
    """The first-stop amount as text.
    In:  figures  _PlanFigures  with a first-stop amount
    Out:          str           e.g. "add 62 percent", "add 35 litres"
    """
    if figures.unit == "l":
        return f"add {figures.first_stop_add:.0f} litres"
    return f"add {figures.first_stop_add * 100:.0f} percent"


def _was(part: str) -> str:
    """The earlier value of a plan part, for "(was ...)".
    In:  part  str  a part from _plan_parts; "" when there was nothing
    Out:       str  the part without its leading words, e.g. "17" from "pit end of lap 17"; "none" for an empty one
    """
    for lead in ("pit end of lap ", "add ", "finish "):
        part = part.removeprefix(lead)
    return part.removesuffix(" in class") or "none"


def _changed(figures: _PlanFigures, old: _PlanFigures, min_change: float) -> list[str]:
    """Names of the plan figures that differ from the recorded plan.
    In:  figures     _PlanFigures  the plan now
         old         _PlanFigures  the last recorded plan
         min_change  float         fraction of energy, or litres for a fuel car, the first-stop amount must move
    Out:             list[str]     any of "race_laps", "stops", "pit_lap", "first_stop_add",
                                   "projected_class_place", in that order
    """
    now, before = figures.first_stop_add, old.first_stop_add
    moved = (now is None) != (before is None) or (now is not None and abs(now - before) >= min_change)
    return [
        name
        for name in PLAN_ORDER
        if (moved if name == "first_stop_add" else getattr(figures, name) != getattr(old, name))
    ]


def _plan_fields(figures: _PlanFigures) -> Fields:
    """The plan as entry fields.
    In:  figures  _PlanFigures
    Out:          Fields  units in the keys
    """
    scale = 1.0 if figures.unit == "l" else 100.0
    return {
        "basis": figures.basis,
        "race_laps": figures.race_laps,
        "stops": figures.stops,
        "pit_lap": figures.pit_lap,
        f"first_stop_add_{figures.unit}": None if figures.first_stop_add is None else figures.first_stop_add * scale,
        "projected_class_place": figures.projected_class_place,
    }
