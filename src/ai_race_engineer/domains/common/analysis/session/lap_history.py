from collections.abc import Callable, Sequence

from ai_race_engineer.domains.common.recording.run_log import RunLog
from ai_race_engineer.domains.common.analysis.session.lap_digest import lap_digest
from ai_race_engineer.domains.common.analysis.field.race_state import RaceTracker
from ai_race_engineer.domains.common.analysis.session.timeline import TimelineEntry
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot
from ai_race_engineer.domains.common.data.snapshot import SessionType


class LapHistory:
    """Keeps the facts of every race lap the player finished: the player's class with places, lap times and gaps,
    so trends over the laps can be read; each lap is also written to the run log (stream "lap_history"). No model
    is involved.
    laps         tuple[dict[str, object], ...]  the digest of every finished race lap, oldest first (replaced, never
                                                mutated)
    _tracker     RaceTracker                    source of the race state
    _entries     Callable[[], Sequence[TimelineEntry]]  the session timeline
    _loss_s      float                          s a lap must be slower than a car's pace to be marked lost
    _trend_laps  int                            laps back over which the gap trend is given
    _log         RunLog
    _completed   int | None                     the player's completed laps at the last snapshot
    """

    def __init__(
        self,
        tracker: RaceTracker,
        entries: Callable[[], Sequence[TimelineEntry]],
        loss_s: float,
        trend_laps: int,
        log: RunLog,
    ) -> None:
        """Store the sources and the settings.
        In:  tracker     RaceTracker
             entries     Callable[[], Sequence[TimelineEntry]]  the session timeline
             loss_s      float  s a lap must be slower than a car's pace to be marked lost
             trend_laps  int    laps back over which the gap trend is given
             log         RunLog
        Out:             None
        """
        self.laps: tuple[dict[str, object], ...] = ()
        self._tracker: RaceTracker = tracker
        self._entries: Callable[[], Sequence[TimelineEntry]] = entries
        self._loss_s: float = loss_s
        self._trend_laps: int = trend_laps
        self._log: RunLog = log
        self._completed: int | None = None

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer, runs after the race tracker and the timeline): add the digest of a
        lap the player just finished in a race; a lap count that goes down starts a new history.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        state = self._tracker.state
        if state is None or snapshot.session.session_type is not SessionType.RACE or state.before_green:
            return
        completed = snapshot.player.vehicle.completed_laps
        if self._completed is not None and completed < self._completed:
            self.laps = ()
        elif self._completed is not None and completed == self._completed + 1:
            digest = lap_digest(state, self.laps, self._entries(), completed, self._loss_s, self._trend_laps)
            self.laps = (*self.laps, digest)
            self._log.write("lap_history", digest)
        self._completed = completed
