import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ai_race_engineer.domains.common.analysis.session.timeline_builder import TimelineBuilder
from ai_race_engineer.domains.common.store.timeline_store import TimelineStore
from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitStopSample
from ai_race_engineer.domains.common.analysis.field.race_state import RaceTracker
from ai_race_engineer.domains.common.analysis.session.timeline import TimelineEntry
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot

STOP: None = None
CLOSE_TIMEOUT_S: float = 5.0

logger = logging.getLogger(__name__)


class TimelineRecorder:
    """Keeps the timeline of the running session in memory and in its file: the builder's entries from every
    snapshot plus the notes and questions added by the agent. Building runs in the telemetry thread without I/O; one
    writer thread does all file access.
    _builder      TimelineBuilder
    _store        TimelineStore
    _tracker      RaceTracker                              source of the race state and the finish
    _pit_samples  Callable[[], tuple[PitStopSample, ...]]  the pit log's measured passes
    _pre_race     Callable[[], bool]                       the pre-race figures are in use
    _now_unix     Callable[[], float]                      s unix time now
    _entries      tuple[TimelineEntry, ...]                entries of the running session (replaced, never mutated)
    _time_s       float                                    s session time of the latest snapshot
    _lap          int                                      the player's lap in progress at the latest snapshot
    _earlier      tuple[bool, bool] | None                 whether the session's file already had a plan and a rival
                                                           set entry; None until the writer has tried to open it
    _opens        int                                      sessions started so far (number of the latest open request)
    _work         queue.Queue[_Open | tuple[TimelineEntry, ...] | None]  file work waiting for the writer
    _lock         threading.Lock                           guards _entries (telemetry, ask and writer thread)
    _writer       threading.Thread                         the writer thread
    """

    def __init__(
        self,
        builder: TimelineBuilder,
        store: TimelineStore,
        tracker: RaceTracker,
        pit_samples: Callable[[], tuple[PitStopSample, ...]],
        pre_race: Callable[[], bool],
        now_unix: Callable[[], float],
    ) -> None:
        """Store the blocks and start the writer thread.
        In:  builder      TimelineBuilder
             store        TimelineStore
             tracker      RaceTracker
             pit_samples  Callable[[], tuple[PitStopSample, ...]]  measured pit-lane passes so far (non-blocking)
             pre_race     Callable[[], bool]                       the pre-race figures are in use (non-blocking)
             now_unix     Callable[[], float]                      s unix time now (of the recording in a replay)
        Out:              None
        """
        self._builder: TimelineBuilder = builder
        self._store: TimelineStore = store
        self._tracker: RaceTracker = tracker
        self._pit_samples: Callable[[], tuple[PitStopSample, ...]] = pit_samples
        self._pre_race: Callable[[], bool] = pre_race
        self._now_unix: Callable[[], float] = now_unix
        self._entries: tuple[TimelineEntry, ...] = ()
        self._time_s: float = 0.0
        self._lap: int = 1
        self._earlier: tuple[bool, bool] | None = None
        self._opens: int = 0
        self._work: queue.Queue[_Open | tuple[TimelineEntry, ...] | None] = queue.Queue()
        self._lock: threading.Lock = threading.Lock()
        self._writer: threading.Thread = threading.Thread(target=self._write_forever, daemon=True)
        self._writer.start()

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer, runs after the race tracker): keep and queue the entries for what
        changed. The "start" entry of a new session empties the entries in memory and asks the writer for the
        session's file first; until the writer has looked for that file the builder holds the initial plan and the
        rival set back, and it does not repeat them when the file already has them (a restarted engineer).
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        session = snapshot.session
        plan_recorded, rivals_recorded = self._earlier or (None, None)
        entries = self._builder.update(
            snapshot,
            self._tracker.state,
            self._pit_samples(),
            self._pre_race(),
            self._tracker.finished,
            plan_recorded,
            rivals_recorded,
        )
        self._time_s, self._lap = session.elapsed_s, snapshot.player.vehicle.completed_laps + 1
        if not entries:
            return
        is_start = entries[0].kind == "session" and entries[0].fields.get("event") == "start"
        with self._lock:
            self._entries = entries if is_start else (*self._entries, *entries)
            if is_start:
                self._earlier, self._opens = None, self._opens + 1
                kind = session.session_type.name.lower()
                self._work.put(_Open(kind, self._now_unix() - session.elapsed_s, session.elapsed_s, self._opens))
            self._work.put(entries)

    def add(self, kind: str, text: str, fields: Mapping[str, str | float | int | bool | None]) -> TimelineEntry:
        """Add an entry stamped with the latest session time and lap (called from the ask thread; never blocks).
        In:  kind    str      e.g. "note", "question"
             text    str      short sentence
             fields  Mapping[str, str | float | int | bool | None]  structured values
        Out:         TimelineEntry  the stored entry
        """
        entry = TimelineEntry(self._time_s, self._lap, kind, text, dict(fields))
        with self._lock:
            self._entries = (*self._entries, entry)
            self._work.put((entry,))
        return entry

    def entries(self) -> tuple[TimelineEntry, ...]:
        """The entries of the running session, oldest first (read from other threads).
        In:  -
        Out: tuple[TimelineEntry, ...]
        """
        return self._entries

    def close(self) -> None:
        """Write what is queued and stop the writer thread.
        In:  -
        Out: None
        """
        self._work.put(STOP)
        self._writer.join(CLOSE_TIMEOUT_S)

    def _write_forever(self) -> None:
        """Writer thread: open the file of each new session, putting the entries already in it in front of those in
        memory, and append every batch. A disk error is logged once per failure and the entries stay in memory; a
        failed open is tried again with every later batch of the session (the batches that came meanwhile are not
        in the file). After the first try, failed or not, the builder is told whether the file had a plan and a
        rival set.
        In:  -
        Out: None
        """
        request, path = None, None
        while True:
            work = self._work.get()
            if work is STOP:
                return
            if isinstance(work, _Open):
                request, path = work, None
            earlier: tuple[TimelineEntry, ...] = ()
            try:
                if path is None and request is not None:
                    path, earlier = self._store.open(request.kind, request.started_unix, request.time_s)
                if path is not None and not isinstance(work, _Open):
                    self._store.append(path, work)
            except OSError as error:
                logger.warning("timeline not written: %s", error)
            with self._lock:
                if request is not None and request.number == self._opens:
                    self._entries = (*earlier, *self._entries)
                    known = self._earlier or (False, False)
                    self._earlier = (
                        known[0] or any(e.kind == "plan" for e in earlier),
                        known[1] or any(e.kind == "rival" for e in earlier),
                    )


@dataclass(frozen=True)
class _Open:
    """Request to open the timeline file of a new session.
    kind          str    "practice" | "qualify" | "race" | ...
    started_unix  float  s unix time the session started (now minus the session time)
    time_s        float  s session time of the session's first snapshot
    number        int    count of the sessions started so far, this one included
    """

    kind: str
    started_unix: float
    time_s: float
    number: int
