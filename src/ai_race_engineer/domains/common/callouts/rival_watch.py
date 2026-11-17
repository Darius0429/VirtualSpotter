import math
from collections import deque
from collections.abc import Collection

from ai_race_engineer.domains.common.callouts.base import Callout, Event, Priority
from ai_race_engineer.domains.common.callouts.text import lap_time_text
from ai_race_engineer.domains.common.analysis.pace.laps import lap_fraction
from ai_race_engineer.domains.common.analysis.strategy.race_finish import CarInput
from ai_race_engineer.domains.common.analysis.field.race_state import RaceState, RaceTracker
from ai_race_engineer.domains.common.analysis.strategy.rejoin import Rejoin, predict_rejoin
from ai_race_engineer.domains.common.analysis.field.rivals import Rival
from ai_race_engineer.domains.common.data.snapshot import Snapshot


class RivalWatch(Event):
    """Tells the driver what the rivals do: new and gone rivals, stops, position swaps, pace and low energy or fuel.
    enabled           bool                      whether callouts are given (switched by the driver through the agent)
    _tracker          RaceTracker               source of the race state
    _pace_change_s    float                     s/lap a rival's pace must move to be announced
    _low_laps         float                     laps on board at or below which a rival must pit soon
    _announce_groups  Collection[str]           rival groups that are watched
    _time_s           float | None              session clock of the last state diffed
    _rivals           dict[int, Rival] | None   slot id -> rival at the last diff; None until seeded
    _ahead            dict[int, bool]           slot id -> rival was ahead of the player at the last diff
    _pace_s           dict[int, float]          slot id -> s/lap pace last announced
    _min_rival_s      float                     s within which a car is not announced as a new rival again
    _start_laps       int                       laps the player completes before the first callout
    _settling         bool                      the last state diffed was before the race had settled
    _announced_s      dict[int, float]          slot id -> s session time it was last announced as a new rival
    _pit_progress     dict[int, float]          slot id -> laps of race distance when the car was last seen in the pits
    _low              set[int]                  slot ids warned as low in their current stint (re-armed by a pit
                                                visit or by laps on board above low_laps)
    _pending          deque[Callout]            callouts not yet returned
    """

    name = "rival_watch"

    def __init__(
        self,
        tracker: RaceTracker,
        pace_change_s: float,
        low_laps: float,
        announce_groups: Collection[str],
        min_rival_s: float,
        start_laps: int,
    ) -> None:
        """Set the state source and thresholds.
        In:  tracker          RaceTracker
             pace_change_s    float            s/lap a rival's pace must move to be announced
             low_laps         float            laps on board at or below which a rival must pit soon
             announce_groups  Collection[str]  rival groups that are watched ("ahead", "behind", "pace", "strategy")
             min_rival_s      float            s within which a car is not announced as a new rival again
             start_laps       int              laps the player completes before the first callout
        Out:                  None
        """
        self.enabled: bool = True
        self._tracker: RaceTracker = tracker
        self._pace_change_s: float = pace_change_s
        self._low_laps: float = low_laps
        self._announce_groups: Collection[str] = announce_groups
        self._min_rival_s: float = min_rival_s
        self._start_laps: int = start_laps
        self._settling: bool = True
        self._announced_s: dict[int, float] = {}
        self._pit_progress: dict[int, float] = {}
        self._time_s: float | None = None
        self._rivals: dict[int, Rival] | None = None
        self._ahead: dict[int, bool] = {}
        self._pace_s: dict[int, float] = {}
        self._low: set[int] = set()
        self._pending: deque[Callout] = deque()

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Diff the rivals of each new race state and return one queued callout; nothing while disabled, before the
        player has completed start_laps laps or once the race is over for the player. Until the race has settled the
        rivals are followed silently, and so is the first state after it, in which every pace switches from the
        pre-race figure to the measured one: the first callout is about a real change after that. Low and new-rival
        marks made meanwhile are dropped with the callouts, so a rival that is low at the release is still called.
        In:  snapshot  Snapshot  (unused: the rivals come from the tracker's state)
             now_s     float     s monotonic clock (unused)
        Out:           Callout | None  kind "rival_new", "rival_gone", "rival_pit", "rival_out", "rival_position",
                                       "rival_pace" or "rival_low"; facts slot_id
        """
        state = self._tracker.state
        if state is not None and state.time_s != self._time_s:
            self._time_s = state.time_s
            settling = self._is_settling(state)
            self._diff(state)
            if settling or self._settling:
                self._pending.clear()
                self._low, self._announced_s = set(), {}
                self._pace_s = {slot_id: rival.lap_time_s for slot_id, rival in (self._rivals or {}).items()}
            self._settling = settling
        if not self.enabled or self._tracker.finished:
            self._pending.clear()
        return self._pending.popleft() if self._pending else None

    def reset(self) -> None:
        """Forget the rivals and pending callouts for a new session; the switch keeps its position.
        In:  -
        Out: None
        """
        self._time_s, self._rivals, self._settling = None, None, True
        self._ahead, self._pace_s, self._low, self._announced_s, self._pit_progress = {}, {}, set(), {}, {}
        self._pending.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Switch the callouts on or off.
        In:  enabled  bool
        Out:          None
        """
        self.enabled = enabled

    def _diff(self, state: RaceState) -> None:
        """Queue callouts for what changed since the last state; the first state only seeds. A class neighbour
        replaced by a position change gives one position callout, not a gone and a new one. A car in the pits or on
        its out-lap does not become a strategy rival until that lap is over; one that already is a rival stays.
        In:  state  RaceState
        Out:        None
        """
        progress = {
            v.slot_id: v.completed_laps + lap_fraction(v.lap_distance_m, state.pit.track_length_m)
            for v in state.vehicles
        }
        self._pit_progress.update({v.slot_id: progress[v.slot_id] for v in state.vehicles if v.in_pits})
        known = self._rivals or {}
        rivals = {
            r.slot_id: r
            for r in state.rivals
            if any(g in self._announce_groups for g in r.groups)
            and (
                r.slot_id in known
                or not self._is_strategy(r)
                or self._rivals is None
                or progress[r.slot_id] - self._pit_progress.get(r.slot_id, -math.inf) >= 1.0
            )
        }
        places = {v.slot_id: v.place for v in state.vehicles}
        ahead = {slot_id: places[slot_id] < places[state.player_slot_id] for slot_id in rivals}
        previous, self._rivals = self._rivals, rivals
        previous_ahead, self._ahead = self._ahead, ahead
        if previous is None:
            self._pace_s = {slot_id: r.lap_time_s for slot_id, r in rivals.items()}
            return
        flipped = [s for s in rivals if s in previous and ahead[s] != previous_ahead[s]]
        self._pending.extend(_position(rivals[slot_id], ahead[slot_id]) for slot_id in flipped)
        self._pending.extend(
            _gone(old) for slot_id, old in previous.items() if slot_id not in rivals and self._is_strategy(old)
        )
        for slot_id, rival in rivals.items():
            if slot_id not in previous:
                self._pace_s[slot_id] = rival.lap_time_s
                if self._is_strategy(rival):
                    if not 0 <= state.time_s - self._announced_s.get(slot_id, -math.inf) < self._min_rival_s:
                        self._announced_s[slot_id] = state.time_s
                        self._pending.append(_new(rival))
                elif not flipped:
                    self._pending.append(_neighbour(rival))
                continue
            old = previous[slot_id]
            if rival.in_pits and not old.in_pits:
                self._pending.append(_pit(rival, _rejoin(state, slot_id)))
            if old.in_pits and not rival.in_pits:
                self._pending.append(_out(rival))
            is_best = rival.best_lap_s is not None and old.best_lap_s is not None and rival.best_lap_s < old.best_lap_s
            if is_best or abs(rival.lap_time_s - self._pace_s[slot_id]) >= self._pace_change_s:
                self._pace_s[slot_id] = rival.lap_time_s
                self._pending.append(_pace(rival, is_best))
            if rival.laps_on_board is not None:
                if rival.pitstops > old.pitstops or rival.in_pits or rival.laps_on_board > self._low_laps:
                    self._low.discard(slot_id)
                is_low = rival.stops_remaining >= 1 and rival.laps_on_board <= self._low_laps and not rival.in_pits
                if is_low and slot_id not in self._low:
                    self._low.add(slot_id)
                    self._pending.append(_low(rival))

    def _is_settling(self, state: RaceState) -> bool:
        """Whether the race has not settled yet: before the green or in the player's first laps.
        In:  state  RaceState
        Out:        bool
        """
        laps = next(v.completed_laps for v in state.vehicles if v.slot_id == state.player_slot_id)
        return state.before_green or laps < self._start_laps

    def _is_strategy(self, rival: Rival) -> bool:
        """Whether a rival is watched for more than being the class neighbour.
        In:  rival  Rival
        Out:        bool  True when it is in a watched group other than "ahead" and "behind"
        """
        return any(g in self._announce_groups and g not in ("ahead", "behind") for g in rival.groups)


def _rejoin(state: RaceState, slot_id: int) -> Rejoin:
    """Predicted rejoin of a car that pits now.
    In:  state    RaceState
         slot_id  int        the car that pits
    Out:          Rejoin
    """
    car: CarInput = next(c for c in state.cars if c.slot_id == slot_id)
    return predict_rejoin(
        car,
        state.projection.cars[slot_id].laps_remaining,
        [c for c in state.cars if c.slot_id != slot_id],
        state.pit,
        state.traffic_window_s,
    )


def _new(rival: Rival) -> Callout:
    """Callout for a car that became a rival.
    In:  rival  Rival
    Out:        Callout  kind "rival_new"
    """
    text = f"New rival {rival.driver_name}, P{rival.class_place} in class, {_gap_text(rival.gap_s)}"
    return Callout("rival_new", text, Priority.STRATEGY, {"slot_id": rival.slot_id, "gap_s": rival.gap_s})


def _neighbour(rival: Rival) -> Callout:
    """Callout for a car that became the class neighbour through a position change among the other cars.
    In:  rival  Rival  in group "ahead" or "behind"
    Out:        Callout  kind "rival_position"
    """
    side = "ahead" if "ahead" in rival.groups else "behind"
    text = f"{rival.driver_name} is now the car {side}, {abs(rival.gap_s):.1f} seconds"
    return Callout("rival_position", text, Priority.STRATEGY, {"slot_id": rival.slot_id, "gap_s": rival.gap_s})


def _gone(rival: Rival) -> Callout:
    """Callout for a car that is no longer a rival.
    In:  rival  Rival  as last seen
    Out:        Callout  kind "rival_gone"
    """
    return Callout(
        "rival_gone", f"{rival.driver_name} is no longer a rival", Priority.STRATEGY, {"slot_id": rival.slot_id}
    )


def _pit(rival: Rival, rejoin: Rejoin) -> Callout:
    """Callout for a rival entering the pits, with its expected stop and rejoin.
    In:  rival   Rival
         rejoin  Rejoin  predicted for this stop
    Out:         Callout  kind "rival_pit"
    """
    gap_s = rival.gap_s - rejoin.stop_time_s
    text = (
        f"{rival.driver_name} pits, about {rejoin.stop_time_s:.0f} seconds, "
        f"rejoins P{rejoin.class_place} in class, {_gap_text(gap_s)}"
    )
    facts = {
        "slot_id": rival.slot_id,
        "stop_time_s": rejoin.stop_time_s,
        "rejoin_class_place": rejoin.class_place,
        "rejoin_gap_s": gap_s,
    }
    return Callout("rival_pit", text, Priority.STRATEGY, facts)


def _out(rival: Rival) -> Callout:
    """Callout for a rival leaving the pits.
    In:  rival  Rival
    Out:        Callout  kind "rival_out"
    """
    text = f"{rival.driver_name} is out of the pits, {_gap_text(rival.gap_s)}"
    return Callout("rival_out", text, Priority.STRATEGY, {"slot_id": rival.slot_id, "gap_s": rival.gap_s})


def _position(rival: Rival, is_ahead: bool) -> Callout:
    """Callout for the order between a rival and the player flipping.
    In:  rival     Rival
         is_ahead  bool   the rival is now ahead of the player
    Out:           Callout  kind "rival_position"
    """
    text = f"{rival.driver_name} is now ahead of you" if is_ahead else f"You are now ahead of {rival.driver_name}"
    return Callout("rival_position", text, Priority.STRATEGY, {"slot_id": rival.slot_id, "ahead": int(is_ahead)})


def _pace(rival: Rival, is_personal_best: bool) -> Callout:
    """Callout for a rival's personal best or changed pace.
    In:  rival             Rival
         is_personal_best  bool   the rival just improved its best lap
    Out:                   Callout  kind "rival_pace"
    """
    if is_personal_best:
        text = f"{rival.driver_name} personal best, {lap_time_text(rival.best_lap_s)}"
    else:
        side = "slower" if rival.pace_delta_s > 0 else "faster"
        text = (
            f"{rival.driver_name} pace {lap_time_text(rival.lap_time_s)}, {abs(rival.pace_delta_s):.1f} {side} than you"
        )
    facts = {"slot_id": rival.slot_id, "lap_time_s": rival.lap_time_s, "pace_delta_s": rival.pace_delta_s}
    return Callout("rival_pace", text, Priority.STRATEGY, facts)


def _low(rival: Rival) -> Callout:
    """Callout for a rival that must pit soon.
    In:  rival  Rival  with laps_on_board
    Out:        Callout  kind "rival_low"
    """
    text = f"{rival.driver_name} has {rival.laps_on_board:.1f} laps on board, must pit"
    return Callout(
        "rival_low", text, Priority.STRATEGY, {"slot_id": rival.slot_id, "laps_on_board": rival.laps_on_board}
    )


def _gap_text(gap_s: float) -> str:
    """Gap to the player as speech text.
    In:  gap_s  float  s, positive = the other car is ahead
    Out:        str    e.g. "3.2 seconds ahead"
    """
    return f"{abs(gap_s):.1f} seconds {'ahead' if gap_s > 0 else 'behind'}"
