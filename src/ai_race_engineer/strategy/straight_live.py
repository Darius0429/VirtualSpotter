from ai_race_engineer.strategy.straight import StraightDetector, on_straight
from ai_race_engineer.telemetry.snapshot import Snapshot


class LiveStraightDetector(StraightDetector):
    """Straight detection from the player's steering, lateral acceleration and speed; needs no per-track setup.
    _max_steering      float         |steering| limit, fraction of lock
    _max_lateral_mps2  float         m/s^2
    _min_speed_mps     float         m/s
    _hold_s            float         s the car must have driven straight
    _since_s           float | None  s session clock since when the car drives straight; None in a corner
    _straight          bool          latest answer (read from other threads)
    """

    def __init__(self, max_steering: float, max_lateral_mps2: float, min_speed_mps: float, hold_s: float) -> None:
        """Set the thresholds.
        In:  max_steering      float  |steering| limit, fraction of lock
             max_lateral_mps2  float  |lateral acceleration| limit, m/s^2
             min_speed_mps     float  m/s
             hold_s            float  s the car must have driven straight
        Out:                   None
        """
        self._max_steering: float = max_steering
        self._max_lateral_mps2: float = max_lateral_mps2
        self._min_speed_mps: float = min_speed_mps
        self._hold_s: float = hold_s
        self._since_s: float | None = None
        self._straight: bool = False

    def update(self, snapshot: Snapshot) -> None:
        """Feed one snapshot.
        In:  snapshot  Snapshot
        Out:           None
        """
        player = snapshot.player
        now_s = snapshot.session.elapsed_s
        if not on_straight(
            player.steering,
            player.accel_lateral_mps2,
            player.vehicle.speed_mps,
            self._max_steering,
            self._max_lateral_mps2,
            self._min_speed_mps,
        ):
            self._since_s = None
        elif self._since_s is None:
            self._since_s = now_s
        self._straight = self._since_s is not None and now_s - self._since_s >= self._hold_s

    def is_straight(self) -> bool:
        """Whether the car has driven straight for hold_s.
        In:  -
        Out: bool
        """
        return self._straight
