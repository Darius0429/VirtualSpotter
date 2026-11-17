import dataclasses
from collections.abc import Callable

from ai_race_engineer.domains.common.analysis.field.race_state import RaceTracker
from ai_race_engineer.domains.common.analysis.field.traffic import cars_on_track, packs_ahead

UNAVAILABLE: dict[str, object] = {"available": False}


class TrafficTools:
    """Agent tool for when the driver reaches slower traffic, worked out from the pace of the cars.
    _tracker       RaceTracker  source of the latest race state
    _horizon_laps  float        laps within which traffic is reported
    _pack_gap_m    float        m two cars may be apart on the road and still be one pack
    """

    def __init__(self, tracker: RaceTracker, horizon_laps: float, pack_gap_m: float) -> None:
        """Store the state source and the distances.
        In:  tracker       RaceTracker
             horizon_laps  float        laps within which traffic is reported
             pack_gap_m    float        m two cars may be apart on the road and still be one pack
        Out:               None
        """
        self._tracker: RaceTracker = tracker
        self._horizon_laps: float = horizon_laps
        self._pack_gap_m: float = pack_gap_m

    def traffic_ahead(self) -> dict[str, object]:
        """When the driver reaches slower traffic, worked out from the pace of every car: packs of cars of another
        class (GT3 traffic for a prototype) and backmarkers of the driver's own class, nearest first. A pack is
        cars that follow each other closely; a single car is a pack of one. For each pack: how many cars, the gap
        on the road now, the laps and the time until the driver reaches it, and the lap and the place on the lap
        where that happens. Use it for any question about when traffic is coming.
        In:  -
        Out: dict[str, object]  {packs: [{backmarkers, class_name, cars, gap_s, laps_to_meet, time_to_meet_s,
                                meet_lap, meet_lap_distance_m}], looked_ahead_laps}; {"available": False} without a
                                race state
        """
        state = self._tracker.state
        if state is None:
            return UNAVAILABLE
        packs = packs_ahead(
            next(car for car in state.cars if car.slot_id == state.player_slot_id),
            cars_on_track([car for car in state.cars if car.slot_id != state.player_slot_id], state.vehicles),
            state.pit.track_length_m,
            self._horizon_laps,
            self._pack_gap_m,
        )
        rows = [{**dataclasses.asdict(pack), "cars": len(pack.slot_ids)} for pack in packs]
        return {
            "packs": [
                {key: round(value, 1) if isinstance(value, float) else value for key, value in row.items()}
                for row in ({k: v for k, v in row.items() if k != "slot_ids"} for row in rows)
            ],
            "looked_ahead_laps": self._horizon_laps,
        }

    def tools(self) -> list[Callable[..., object]]:
        """The functions an agent may call.
        In:  -
        Out: list[Callable[..., object]]  traffic_ahead
        """
        return [self.traffic_ahead]
