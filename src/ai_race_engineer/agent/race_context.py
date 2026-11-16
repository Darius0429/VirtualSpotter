from ai_race_engineer.strategy.race_state import RaceTracker
from ai_race_engineer.strategy.segment_tracker import SegmentTracker


class RaceContext:
    """One line that says where the driver is right now, given to the agent with every question so it can speak
    of this lap, the last lap and the place on track without looking it up.
    _tracker   RaceTracker     source of the laps left
    _segments  SegmentTracker  source of the latest snapshot and the track map
    """

    def __init__(self, tracker: RaceTracker, segments: SegmentTracker) -> None:
        """Store the sources.
        In:  tracker   RaceTracker
             segments  SegmentTracker
        Out:           None
        """
        self._tracker: RaceTracker = tracker
        self._segments: SegmentTracker = segments

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
        Out: str  e.g. "Now: race, on lap 3, 3.4 laps to go, P2 in class, at turn 7, 61 percent energy."
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
        parts = [
            snapshot.session.session_type.name.lower().replace("_", " "),
            "in the pits" if vehicle.in_pits else f"on lap {vehicle.completed_laps + 1}",
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
        return "Now: " + ", ".join(parts) + "."
