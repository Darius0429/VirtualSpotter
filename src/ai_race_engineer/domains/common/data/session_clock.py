from ai_race_engineer.domains.common.data.snapshot import Snapshot


class SessionClock:
    """Holds the session time of the latest snapshot, for blocks outside the telemetry thread.
    _time_s  float  s session clock (read from other threads)
    """

    def __init__(self) -> None:
        """Start at zero.
        In:  -
        Out: None
        """
        self._time_s: float = 0.0

    def update(self, snapshot: Snapshot) -> None:
        """Store the snapshot's session time (source observer).
        In:  snapshot  Snapshot
        Out:           None
        """
        self._time_s = snapshot.session.elapsed_s

    def time_s(self) -> float:
        """Session time of the latest snapshot.
        In:  -
        Out: float  s
        """
        return self._time_s
