from ai_race_engineer.events.base import Callout, Event, Priority
from ai_race_engineer.strategy.race_state import RaceTracker
from ai_race_engineer.strategy.traffic import TrafficPack, cars_on_track, packs_ahead
from ai_race_engineer.telemetry.snapshot import Snapshot

REARM_FACTOR: float = 2.0


class TrafficAhead(Event):
    """Says once when the player is about to reach slower traffic: a pack or a single car of another class, or
    backmarkers of the player's own class. The sentence is fixed; nothing is asked of a model. One sentence covers
    one run of traffic of a kind: cars of that kind that come within reach while earlier ones are still ahead are
    not said again.
    _tracker       RaceTracker                       source of the race state
    _horizon_laps  float                             laps within which the traffic must be reached to be said
    _pack_gap_m    float                             m two cars may be apart on the road and still be one pack
    _said          dict[tuple[bool, str], set[int]]  (backmarkers, class) -> slot ids announced and still near
    _time_s        float                             s session clock of the last state looked at
    """

    name = "traffic"

    def __init__(self, tracker: RaceTracker, horizon_laps: float, pack_gap_m: float) -> None:
        """Store the state source and the distances.
        In:  tracker       RaceTracker
             horizon_laps  float        laps within which the traffic must be reached to be said
             pack_gap_m    float        m two cars may be apart on the road and still be one pack
        Out:               None
        """
        self._tracker: RaceTracker = tracker
        self._horizon_laps: float = horizon_laps
        self._pack_gap_m: float = pack_gap_m
        self._said: dict[tuple[bool, str], set[int]] = {}
        self._time_s: float = -1.0

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Callout for the nearest pack of a kind of which nothing is announced and still ahead. Traffic counts as
        still ahead until it is further than twice the horizon away, which it is after it has been passed. Cars
        standing in their garage are no traffic.
        In:  snapshot  Snapshot  (unused: the cars come from the tracker's state)
             now_s     float     s monotonic clock (unused)
        Out:           Callout | None  kind "traffic"; facts class_name, cars, backmarkers, laps_to_meet
        """
        state = self._tracker.state
        if state is None or state.before_green or self._tracker.finished or state.time_s == self._time_s:
            return None
        self._time_s = state.time_s
        player = next(car for car in state.cars if car.slot_id == state.player_slot_id)
        others = cars_on_track([car for car in state.cars if car.slot_id != state.player_slot_id], state.vehicles)
        length_m = state.pit.track_length_m
        near = packs_ahead(player, others, length_m, self._horizon_laps * REARM_FACTOR, self._pack_gap_m)
        still = {slot_id for pack in near for slot_id in pack.slot_ids}
        self._said = {kind: slots & still for kind, slots in self._said.items() if slots & still}
        callout = None
        for pack in packs_ahead(player, others, length_m, self._horizon_laps, self._pack_gap_m):
            kind = (pack.backmarkers, pack.class_name)
            if kind not in self._said and callout is None:
                facts: dict[str, str | float | int] = {
                    "class_name": pack.class_name,
                    "cars": len(pack.slot_ids),
                    "backmarkers": int(pack.backmarkers),
                    "laps_to_meet": pack.laps_to_meet,
                }
                callout = Callout("traffic", traffic_text(pack), Priority.STRATEGY, facts)
            self._said.setdefault(kind, set()).update(pack.slot_ids)
        return callout

    def reset(self) -> None:
        """Forget what was announced.
        In:  -
        Out: None
        """
        self._said = {}
        self._time_s = -1.0


def traffic_text(pack: TrafficPack) -> str:
    """Sentence for a pack of slower cars ahead.
    In:  pack  TrafficPack
    Out:       str  "GT traffic in the next lap", "GT3 car up ahead", "Backmarkers in the next lap" or
                    "Backmarker up ahead"; a class not named GT is said by its name
    """
    several = len(pack.slot_ids) > 1
    if pack.backmarkers:
        return "Backmarkers in the next lap" if several else "Backmarker up ahead"
    if several:
        return f"{'GT' if pack.class_name.startswith('GT') else pack.class_name} traffic in the next lap"
    return f"{pack.class_name} car up ahead"
