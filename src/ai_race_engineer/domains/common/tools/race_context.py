from collections.abc import Callable

from ai_race_engineer.domains.common.analysis.field.race_state import RaceTracker
from ai_race_engineer.domains.common.analysis.corners.segment_tracker import SegmentTracker
from ai_race_engineer.domains.common.analysis.session.track_limits import TrackLimitStatus, penalty_text
from ai_race_engineer.domains.common.data.snapshot import SessionType


class RaceContext:
    """One line that says where the driver is right now, given to the agent with every question so it can speak
    of this lap, the last lap and the place on track without looking it up.
    _tracker   RaceTracker     source of the laps left
    _segments  SegmentTracker  source of the latest snapshot and the track map
    _lap_type  Callable[[], str | None]               provisional type of the lap in progress; None when not known
    _limits    Callable[[], TrackLimitStatus | None]  the player's track limits and penalties
    """

    def __init__(
        self,
        tracker: RaceTracker,
        segments: SegmentTracker,
        lap_type: Callable[[], str | None],
        limits: Callable[[], TrackLimitStatus | None],
    ) -> None:
        """Store the sources.
        In:  tracker   RaceTracker
             segments  SegmentTracker
             lap_type  Callable[[], str | None]               provisional type of the lap in progress
             limits    Callable[[], TrackLimitStatus | None]  the player's track limits and penalties
        Out:           None
        """
        self._tracker: RaceTracker = tracker
        self._segments: SegmentTracker = segments
        self._lap_type: Callable[[], str | None] = lap_type
        self._limits: Callable[[], TrackLimitStatus | None] = limits

    def driver_names(self) -> str:
        """The family names of the drivers in the session, for the speech recognition to lean towards.
        In:  -
        Out: str  e.g. "Drivers: Gamble, Riberas, Vanthoor."; "" without a session
        """
        snapshot = self._segments.latest
        if snapshot is None:
            return ""
        names = sorted({v.driver_name.split()[-1] for v in snapshot.vehicles if v.driver_name.strip()})
        return f"Drivers: {', '.join(names)}."

    def __call__(self) -> str:
        """Where the driver is now.
        In:  -
        Out: str  e.g. "Now: race, on lap 3 (lap), 3.4 laps to go, P2 in class, at turn 7, 61 percent energy, track
                  limits 1.25 of 5 points, 1 penalty outstanding."; the lap type when it is known, the track limits
                  in a race only and when points or penalties are above zero
        """
        snapshot = self._segments.latest
        if snapshot is None:
            return "Now: no session running."
        vehicle = snapshot.player.vehicle
        in_class = sorted((v for v in snapshot.vehicles if v.class_name == vehicle.class_name), key=lambda v: v.place)
        place = next(index for index, v in enumerate(in_class, start=1) if v.slot_id == vehicle.slot_id)
        where = next(
            (s.name for s in self._segments.segments if s.start_m <= vehicle.lap_distance_m < s.end_m), "the lap"
        )
        lap_type = self._lap_type()
        lap = f"on lap {vehicle.completed_laps + 1}" + (f" ({lap_type})" if lap_type else "")
        parts = [
            snapshot.session.session_type.name.lower().replace("_", " "),
            "in the pits" if vehicle.in_pits else lap,
        ]
        state = self._tracker.state
        projection = None if state is None else state.projection.cars.get(state.player_slot_id)
        if state is not None and state.before_green:
            parts.append("before the start")
        if projection is not None:
            parts.append(f"{projection.laps_remaining:.1f} laps to go")
        parts += [f"P{place} in class", f"at {where}"]
        if snapshot.player.virtual_energy is not None:
            parts.append(f"{snapshot.player.virtual_energy * 100:.0f} percent energy")
        limits = self._limits()
        if limits is not None and snapshot.session.session_type is SessionType.RACE:
            parts += _limit_parts(limits)
        return "Now: " + ", ".join(parts) + "."


def _limit_parts(limits: TrackLimitStatus) -> list[str]:
    """The track limits and penalties worth a word in the line.
    In:  limits  TrackLimitStatus
    Out:         list[str]  e.g. ["track limits 1.25 of 5 points", "1 penalty outstanding"]; a part is left out
                            when its count is zero or unknown
    """
    parts: list[str] = []
    if limits.points and limits.penalty_points:
        parts.append(f"track limits {limits.points:g} of {limits.penalty_points:g} points")
    if limits.penalties:
        parts.append(penalty_text(limits.penalties))
    return parts
