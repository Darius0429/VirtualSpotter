from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ai_race_engineer.domains.common.analysis.corners.place import place
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot
from ai_race_engineer.domains.common.data.snapshot import SessionState

TRACK_LIMITS: str = "track_limits"
PENALTY: str = "penalty"


@dataclass(frozen=True)
class TrackLimitEvent:
    """One track-limit offence the game counted.
    time_s          float  s session time the step was counted
    lap             int    lap as the driver counts it, on which the car left the track
    steps           int    steps gained
    lap_distance_m  float  m where the car left the track (where the step was counted when that is unknown)
    place           str    "exit of turn 2"; "" without a map
    """

    time_s: float
    lap: int
    steps: int
    lap_distance_m: float
    place: str


@dataclass(frozen=True)
class TrackLimitStatus:
    """Track limits and penalties of the player now.
    points          float | None  track-limit points held (steps / steps per point)
    penalty_points  float | None  points at which a penalty comes (steps per penalty / steps per point)
    penalties       int | None    penalties outstanding
    events          tuple[TrackLimitEvent, ...]  oldest first
    """

    points: float | None
    penalty_points: float | None
    penalties: int | None
    events: tuple[TrackLimitEvent, ...]


class TrackLimitTracker:
    """Follows the track-limit count and the penalties the game holds for the player: every step the count goes up
    is kept as an offence with the place where the car left the track and written to the session timeline as
    "track_limits"; every change of the number of outstanding penalties is written as "penalty".
    status          TrackLimitStatus | None  the count, the limit, the penalties and the offences of the session
                                             (replaced, never mutated; None while the source has none)
    _record         Callable[[str, str, Mapping[str, object]], object]  adds a timeline entry (kind, text, fields)
    _segments       Callable[[], Sequence[Segment]]  the track map, empty until known
    _apex_window_m  float                    m either side of the apex that reads "apex of"
    _session        SessionState | None      session of the last snapshot with a count; None before the first
    _steps          int                      steps at the last snapshot
    _penalties      int                      outstanding penalties at the last snapshot
    _counts         bool | None              the game was timing the player's lap at the last snapshot
    _left_m         float | None             m lap distance where the game stopped timing the lap; None when it is
                                             timing, in the pits, more than a lap later, or after the offence
                                             was counted
    _left_lap       int                      laps the player had completed when the game stopped timing the lap
    _events         tuple[TrackLimitEvent, ...]  the offences of the session, oldest first
    """

    def __init__(
        self,
        record: Callable[[str, str, Mapping[str, object]], object],
        segments: Callable[[], Sequence[Segment]],
        apex_window_m: float,
    ) -> None:
        """Store where the entries go and how a place is named.
        In:  record         Callable[[str, str, Mapping[str, object]], object]  adds a timeline entry
             segments       Callable[[], Sequence[Segment]]  the track map, empty until known
             apex_window_m  float  m either side of the apex that reads "apex of"
        Out:                None
        """
        self.status: TrackLimitStatus | None = None
        self._record: Callable[[str, str, Mapping[str, object]], object] = record
        self._segments: Callable[[], Sequence[Segment]] = segments
        self._apex_window_m: float = apex_window_m
        self._session: SessionState | None = None
        self._steps: int = 0
        self._penalties: int = 0
        self._counts: bool | None = None
        self._left_m: float | None = None
        self._left_lap: int = 0
        self._events: tuple[TrackLimitEvent, ...] = ()

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer, after the timeline). A new session (session type or track change, or
        the session clock going back) clears the offences, and its first snapshot only notes the values. The place of
        an offence is where the game stopped timing the lap, as the count goes up some seconds later; when the game
        times the lap again, in the pits (no lap is timed there) and more than a lap later the place is forgotten.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        state = snapshot.penalties
        if state is None:
            self.status = None
            return
        session, player = snapshot.session, snapshot.player.vehicle
        before = self._session
        fresh = (
            before is None
            or session.session_type != before.session_type
            or session.track_name != before.track_name
            or session.elapsed_s < before.elapsed_s
        )
        steps, penalties = state.track_limit_steps or 0, state.penalties or 0
        points = _ratio(state.track_limit_steps, state.steps_per_point)
        limit = _ratio(state.steps_per_penalty, state.steps_per_point)
        if fresh:
            self._events, self._counts, self._left_m = (), None, None
        if self._counts is True and player.lap_time_counts is False:
            self._left_m, self._left_lap = player.lap_distance_m, player.completed_laps
        if player.lap_time_counts is True or player.in_pits or player.completed_laps > self._left_lap + 1:
            self._left_m = None
        if not fresh and steps > self._steps:
            left_m = player.lap_distance_m if self._left_m is None else self._left_m
            lap = (player.completed_laps if self._left_m is None else self._left_lap) + 1
            self._left_m = None
            where = place(self._segments(), left_m, self._apex_window_m)
            event = TrackLimitEvent(session.elapsed_s, lap, steps - self._steps, left_m, where)
            self._events = (*self._events, event)
            fields = {
                "steps": event.steps,
                "points": points,
                "penalty_points": limit,
                "place": where,
                "lap_distance_m": left_m,
            }
            self._record(TRACK_LIMITS, track_limit_text(where, points, limit), fields)
        if not fresh and penalties != self._penalties:
            self._record(PENALTY, penalty_text(penalties), {"penalties": penalties})
        self._session, self._steps, self._penalties = session, steps, penalties
        self._counts = player.lap_time_counts
        self.status = TrackLimitStatus(points, limit, state.penalties, self._events)


def track_limit_text(place: str, points: float | None, penalty_points: float | None) -> str:
    """Sentence for a track-limit offence.
    In:  place           str           "exit of turn 2"; "" when unknown
         points          float | None  track-limit points held now
         penalty_points  float | None  points at which a penalty comes
    Out:                 str  e.g. "Track limits at exit of turn 2: 0.25 of 5 points"
    """
    where = f" at {place}" if place else ""
    count = "" if points is None or penalty_points is None else f": {points:g} of {penalty_points:g} points"
    return f"Track limits{where}{count}"


def penalty_text(penalties: int) -> str:
    """Sentence for the number of outstanding penalties.
    In:  penalties  int  penalties outstanding
    Out:            str  "No penalty outstanding" | "1 penalty outstanding" | "2 penalties outstanding"
    """
    if penalties == 0:
        return "No penalty outstanding"
    return f"{penalties} {'penalty' if penalties == 1 else 'penalties'} outstanding"


def _ratio(amount: int | None, per: int | None) -> float | None:
    """One count in units of another.
    In:  amount  int | None  steps
         per     int | None  steps that make one unit
    Out:         float | None  None when the source lacks either, or gives no steps per unit
    """
    return None if amount is None or not per else amount / per
