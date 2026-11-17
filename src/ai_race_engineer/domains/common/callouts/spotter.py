import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.domains.common.callouts.base import Callout, Event, Priority
from ai_race_engineer.domains.common.analysis.field.proximity import to_player_frame
from ai_race_engineer.domains.common.data.snapshot import Snapshot

type Sides = tuple[bool, bool]


class Spotter(Event):
    """Calls cars alongside the player (left, right, three wide) and when each side is clear again.
    _car_length_m            float        m
    _car_width_m             float        m
    _lateral_range_m         float        m
    _longitudinal_overlap_m  float        m
    _clear_hold_s            float        s
    _repeat_suppress_s       float        s
    _min_speed_mps           float        m/s
    _announced               Sides        (left, right) currently announced as occupied
    _last_seen_s             list[float]  s monotonic, last time each side was occupied (left, right)
    _announced_at_s          list[float]  s monotonic, time each side was announced (left, right)
    """

    name = "spotter"

    def __init__(
        self,
        car_length_m: float,
        car_width_m: float,
        lateral_range_m: float,
        longitudinal_overlap_m: float,
        clear_hold_s: float,
        repeat_suppress_s: float,
        min_speed_mps: float,
    ) -> None:
        """Set the geometry and timing.
        In:  car_length_m            float  m centre-to-centre distance at which bumpers meet
             car_width_m             float  m; cars with |lateral| < width/2 are not beside
             lateral_range_m         float  m max |lateral| still counted as beside
             longitudinal_overlap_m  float  m margin added to car_length_m
             clear_hold_s            float  s a side must be empty before "clear"
             repeat_suppress_s       float  s min time from a side's "car" call to its "clear"
             min_speed_mps           float  m/s player speed below which the spotter is silent
        Out:                         None
        """
        self._car_length_m: float = car_length_m
        self._car_width_m: float = car_width_m
        self._lateral_range_m: float = lateral_range_m
        self._longitudinal_overlap_m: float = longitudinal_overlap_m
        self._clear_hold_s: float = clear_hold_s
        self._repeat_suppress_s: float = repeat_suppress_s
        self._min_speed_mps: float = min_speed_mps
        self._announced: Sides = (False, False)
        self._last_seen_s: list[float] = [0.0, 0.0]
        self._announced_at_s: list[float] = [0.0, 0.0]

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Spotter call when the announced sides change.
        In:  snapshot  Snapshot
             now_s     float     s monotonic clock
        Out:           Callout | None  kind "spotter"; facts left, right (1 occupied, 0 clear)
        """
        me = snapshot.player.vehicle
        if me.speed_mps < self._min_speed_mps or me.in_pits:
            self.reset()
            return None
        others = [v.position_xz_m for v in snapshot.vehicles if not v.is_player and not v.in_pits and not v.in_garage]
        offsets = to_player_frame(me.position_xz_m, me.heading_rad, np.array(others, dtype=np.float64).reshape(-1, 2))
        return self._step(self._sides(offsets), now_s)

    def reset(self) -> None:
        """Forget announced sides and timers.
        In:  -
        Out: None
        """
        self._announced = (False, False)
        self._last_seen_s = [0.0, 0.0]
        self._announced_at_s = [0.0, 0.0]

    def _sides(self, offsets: NDArray[np.float64]) -> Sides:
        """Which sides have a car alongside.
        In:  offsets  NDArray[float64]  (n, 2) m [lateral (+ right), longitudinal (+ ahead)]
        Out:          Sides             (left occupied, right occupied)
        """
        lateral = np.abs(offsets[:, 0])
        beside = (
            (np.abs(offsets[:, 1]) < self._car_length_m + self._longitudinal_overlap_m)
            & (lateral >= self._car_width_m / 2)
            & (lateral <= self._lateral_range_m)
        )
        return bool(np.any(beside & (offsets[:, 0] < 0))), bool(np.any(beside & (offsets[:, 0] > 0)))

    def _step(self, seen: Sides, now_s: float) -> Callout | None:
        """Update the announced sides (occupied at once, clear after hold and suppress times) and call changes.
        A side that newly fills while the other side is empty this tick drops the other side's hold,
        so a lone car is never called "three wide".
        In:  seen   Sides  (left, right) occupied this tick
             now_s  float  s monotonic clock
        Out:        Callout | None
        """
        new = list(self._announced)
        for side in (0, 1):
            if seen[side]:
                if not self._announced[side]:
                    self._announced_at_s[side] = now_s
                self._last_seen_s[side] = now_s
                new[side] = True
            elif (
                self._announced[side]
                and now_s - self._last_seen_s[side] >= self._clear_hold_s
                and now_s - self._announced_at_s[side] >= self._repeat_suppress_s
            ):
                new[side] = False
        for side in (0, 1):
            if seen[side] and not self._announced[side] and not seen[1 - side]:
                new[1 - side] = False
        old, self._announced = self._announced, (new[0], new[1])
        if old == self._announced:
            return None
        return Callout(
            "spotter", _TEXT[(old, self._announced)], Priority.SPOTTER, {"left": int(new[0]), "right": int(new[1])}
        )


_TEXT: dict[tuple[Sides, Sides], str] = {
    ((False, False), (True, False)): "car left",
    ((False, False), (False, True)): "car right",
    ((False, False), (True, True)): "three wide",
    ((True, False), (True, True)): "three wide",
    ((True, False), (False, False)): "clear left",
    ((True, False), (False, True)): "car right",
    ((False, True), (True, True)): "three wide",
    ((False, True), (False, False)): "clear right",
    ((False, True), (True, False)): "car left",
    ((True, True), (True, False)): "clear right",
    ((True, True), (False, True)): "clear left",
    ((True, True), (False, False)): "clear",
}
