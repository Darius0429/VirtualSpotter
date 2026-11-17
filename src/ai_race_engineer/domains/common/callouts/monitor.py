import queue
import time
from collections.abc import Collection, Mapping, Sequence

from ai_race_engineer.domains.common.callouts.base import Callout, Event
from ai_race_engineer.domains.common.analysis.strategy.consumption import ConsumptionTracker
from ai_race_engineer.domains.common.data.base import TelemetrySource
from ai_race_engineer.domains.common.data.snapshot import SessionState


class EventMonitor:
    """Real-time loop: read telemetry, update consumption, run the events enabled for the session, queue callouts.
    _source        TelemetrySource
    _events        Sequence[Event]
    _enabled       Mapping[str, Collection[str]]  event name -> SessionType names it runs in
    _consumption   ConsumptionTracker
    _callouts      queue.Queue[Callout]
    _poll_hz       float                          Hz
    _last_session  SessionState | None            session of the last non-empty read
    """

    def __init__(
        self,
        source: TelemetrySource,
        events: Sequence[Event],
        enabled: Mapping[str, Collection[str]],
        consumption: ConsumptionTracker,
        callouts: queue.Queue[Callout],
        poll_hz: float,
    ) -> None:
        """Store the blocks to wire together.
        In:  source       TelemetrySource                opened by the caller
             events       Sequence[Event]
             enabled      Mapping[str, Collection[str]]  event name -> SessionType names
             consumption  ConsumptionTracker             shared with events that need it
             callouts     queue.Queue[Callout]           output queue
             poll_hz      float                          Hz
        Out:              None
        """
        self._source: TelemetrySource = source
        self._events: Sequence[Event] = events
        self._enabled: Mapping[str, Collection[str]] = enabled
        self._consumption: ConsumptionTracker = consumption
        self._callouts: queue.Queue[Callout] = callouts
        self._poll_hz: float = poll_hz
        self._last_session: SessionState | None = None

    def run(self) -> None:
        """Poll forever at poll_hz (run in a daemon thread).
        In:  -
        Out: None
        """
        while True:
            self.step(time.monotonic())
            time.sleep(1 / self._poll_hz)

    def step(self, now_s: float) -> None:
        """Process one telemetry read.
        In:  now_s  float  s monotonic clock
        Out:        None
        """
        snapshot = self._source.read()
        if snapshot is None:
            return
        if _is_new_session(self._last_session, snapshot.session):
            self._consumption.reset()
            for event in self._events:
                event.reset()
        self._last_session = snapshot.session
        self._consumption.update(snapshot.player)
        for event in self._events:
            if snapshot.session.session_type.name in self._enabled[event.name]:
                callout = event.check(snapshot, now_s)
                if callout is not None:
                    self._callouts.put(callout)


def _is_new_session(prev: SessionState | None, cur: SessionState) -> bool:
    """Whether the current session differs from the previous one.
    In:  prev  SessionState | None  None before the first read
         cur   SessionState
    Out:       bool  True on first read, session type or track change, or session clock going backwards
    """
    return (
        prev is None
        or cur.session_type != prev.session_type
        or cur.track_name != prev.track_name
        or cur.elapsed_s < prev.elapsed_s
    )
