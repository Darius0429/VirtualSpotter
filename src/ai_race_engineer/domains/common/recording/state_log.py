from ai_race_engineer.domains.common.recording.run_log import RunLog
from ai_race_engineer.domains.common.analysis.field.race_state import RaceState, RaceTracker
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot


class StateLog:
    """Writes the race state to the run log every period_s of session time.
    _log       RunLog
    _tracker   RaceTracker        source of the race state
    _period_s  float              session s between records
    _state     RaceState | None   the state written last
    _time_s    float | None       s session time of the last record; None before the first
    """

    def __init__(self, log: RunLog, tracker: RaceTracker, period_s: float) -> None:
        """Store the log, the state source and the period.
        In:  log       RunLog
             tracker   RaceTracker
             period_s  float  session s between records
        Out:           None
        """
        self._log: RunLog = log
        self._tracker: RaceTracker = tracker
        self._period_s: float = period_s
        self._state: RaceState | None = None
        self._time_s: float | None = None

    def update(self, snapshot: RaceSnapshot) -> None:
        """Queue the tracker's state for the log when it is a new one and period_s of session time has passed, or
        the session clock went back (source observer, after the tracker).
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        state, now_s = self._tracker.state, snapshot.session.elapsed_s
        if state is None or state is self._state:
            return
        if self._time_s is not None and 0 <= now_s - self._time_s < self._period_s:
            return
        self._state, self._time_s = state, now_s
        session = snapshot.session
        self._log.write("state", {"track": session.track_name, "session_type": session.session_type, "state": state})
