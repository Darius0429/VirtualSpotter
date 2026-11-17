import dataclasses
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_race_engineer.domains.common.store.history_store import HistoryStore
from ai_race_engineer.domains.common.recording.run_log import RunLog
from ai_race_engineer.domains.common.store.session_store import SessionStore
from ai_race_engineer.domains.common.analysis.field.field import CarEstimate
from ai_race_engineer.domains.common.analysis.session.history import SessionFigures, car_key, figures_from_record, figures_from_results
from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitRates
from ai_race_engineer.domains.common.analysis.field.priors import EventPriors, event_priors, remember
from ai_race_engineer.domains.common.analysis.session.session_log import SessionLog, SessionRecord
from ai_race_engineer.domains.common.analysis.session.session_summary import (
    SessionSummary,
    results_of_session,
    summarise,
    summary_from_results,
    with_results,
)
from ai_race_engineer.domains.common.data.results import SessionResults
from ai_race_engineer.domains.common.data.results_file import latest_results, read_results

logger = logging.getLogger(__name__)


class SessionKeeper:
    """The one background step for session files, history and priors, so the telemetry loop never touches a file:
    it reads the results file the game points to, keeps the priors of the running session, rewrites the running
    session's summary file and, when a session ends, writes its final summary and adds its figures to the history.
    step and close hold one lock, so the final write never runs beside a step; after close a step does nothing.
    _session_log           SessionLog
    _sessions              SessionStore
    _history               HistoryStore
    _results_path          Callable[[], Path | None]  the results file the game points to
    _default_rates         PitRates                   the configured pit rates
    _log                   RunLog
    _refresh_s             float                      s between rewrites of the running session's summary
    _class_top             int                        cars of the class listed in a summary
    _track_field           str                        results field equal to the game's track name
    _history_sessions      int                        sessions the history figures are the median of
    _max_qualifying_age_h  float                      hours
    _now_unix              Callable[[], float]        s unix time now (of the recording in a replay)
    _period_s              float                      s between checks
    _results_margin_s      float                      s a results file's start may differ from the session's
    _green_seen            Callable[[], bool]         the green has been shown in the running session
    _entries               tuple[SessionFigures, ...] | None  the history; None until loaded
    _remembered            dict[str, dict[tuple[str, str], CarEstimate]]  track -> (driver, class) -> values of
                                                      sessions that ended in this process
    _session_id            str | None                 session the priors were made for
    _priors                EventPriors | None         replaced, never mutated (read from other threads)
    _results               SessionResults | None      the file at results_path, as last read
    _stamp                 tuple[Path, float] | None  path and modification time (s unix) of the results file when it
                                                      was last read; None without a path or file
    _previous_started_unix  float                     s unix time the previous session of this process started
                                                      (0 = there was none)
    _written_s             float                      s monotonic clock of the last summary rewrite
    _lock                  threading.Lock             held by step and close
    _closed                bool                       close has run
    """

    def __init__(
        self,
        session_log: SessionLog,
        sessions: SessionStore,
        history: HistoryStore,
        results_path: Callable[[], Path | None],
        default_rates: PitRates,
        log: RunLog,
        refresh_s: float,
        class_top: int,
        track_field: str,
        history_sessions: int,
        max_qualifying_age_h: float,
        now_unix: Callable[[], float],
        period_s: float,
        results_margin_s: float,
        green_seen: Callable[[], bool],
    ) -> None:
        """Store the blocks and parameters.
        In:  session_log           SessionLog
             sessions              SessionStore
             history               HistoryStore
             results_path          Callable[[], Path | None]  the results file the game points to (None = none)
             default_rates         PitRates
             log                   RunLog
             refresh_s             float                      s between rewrites of the running summary
             class_top             int                        cars of the class listed in a summary
             track_field           str                        "TrackVenue" | "TrackCourse" | "TrackEvent"
             history_sessions      int                        sessions the history figures are the median of
             max_qualifying_age_h  float                      hours after which a qualifying file is not this event's
             now_unix              Callable[[], float]        s unix time now
             period_s              float                      s between checks
             results_margin_s      float                      s a results file's start may differ from the session's
             green_seen            Callable[[], bool]         the green has been shown in the running session
                                                              (non-blocking)
        Out:                       None
        """
        self._session_log: SessionLog = session_log
        self._sessions: SessionStore = sessions
        self._history: HistoryStore = history
        self._results_path: Callable[[], Path | None] = results_path
        self._default_rates: PitRates = default_rates
        self._log: RunLog = log
        self._refresh_s: float = refresh_s
        self._class_top: int = class_top
        self._track_field: str = track_field
        self._history_sessions: int = history_sessions
        self._max_qualifying_age_h: float = max_qualifying_age_h
        self._now_unix: Callable[[], float] = now_unix
        self._period_s: float = period_s
        self._results_margin_s: float = results_margin_s
        self._green_seen: Callable[[], bool] = green_seen
        self._entries: tuple[SessionFigures, ...] | None = None
        self._remembered: dict[str, dict[tuple[str, str], CarEstimate]] = {}
        self._session_id: str | None = None
        self._priors: EventPriors | None = None
        self._results: SessionResults | None = None
        self._stamp: tuple[Path, float] | None = None
        self._previous_started_unix: float = 0.0
        self._written_s: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False

    def run(self) -> None:
        """Check forever every period_s (run in a daemon thread).
        In:  -
        Out: None
        """
        while True:
            self.step()
            time.sleep(self._period_s)

    def step(self) -> None:
        """One check: load the history once; finish the sessions that ended; read the results file and replace the
        priors when a session ended or a new one runs, and until the green also whenever the path or the
        modification time of the results file changed (the file is never read again while it is unchanged); every
        refresh_s rewrite the running session's summary file.
        In:  -
        Out: None
        """
        with self._lock:
            if self._closed:
                return
            if self._entries is None:
                self._entries = self._history.load()
            ended = False
            while not self._session_log.ended.empty():
                record = self._session_log.ended.get()
                self._remembered[record.track] = remember(
                    self._remembered.get(record.track, {}), record.vehicles, record.estimates, record.kind
                )
                self._finish(record)
                self._previous_started_unix = record.started_unix
                ended = True
            record = self._session_log.record()
            if record is None:
                return
            path = self._results_path()
            stamp = _stamp(path)
            changed = ended or record.session_id != self._session_id
            if changed or (stamp != self._stamp and not self._green_seen()):
                self._session_id, self._stamp = record.session_id, stamp
                self._priors = event_priors(
                    record.track,
                    record.vehicle_name,
                    record.player_name,
                    self._qualifying(record, path),
                    self._remembered.get(record.track, {}),
                    self._entries,
                    self._now_unix(),
                    self._track_field,
                    self._history_sessions,
                    self._max_qualifying_age_h,
                    record.started_unix - self._results_margin_s,
                )
            if time.monotonic() - self._written_s >= self._refresh_s:
                self._written_s = time.monotonic()
                self._write(summarise(record, self._car(record), self._class_top))

    def close(self) -> None:
        """Store the sessions that ended since the last step, then the running session as it stands: its summary
        file and its figures in the history (called on a normal shutdown, also after Ctrl+C); waits for a step in
        progress.
        In:  -
        Out: None
        """
        with self._lock:
            self._closed = True
            while not self._session_log.ended.empty():
                self._finish(self._session_log.ended.get())
            record = self._session_log.record()
            if record is not None:
                self._finish(record)

    def priors(self) -> EventPriors | None:
        """Priors of the running session.
        In:  -
        Out: EventPriors | None  None until the first session has been seen
        """
        return self._priors

    def entries(self) -> tuple[SessionFigures, ...]:
        """The history as loaded and extended in this process.
        In:  -
        Out: tuple[SessionFigures, ...]
        """
        return self._entries or ()

    def summary(self) -> SessionSummary | None:
        """Summary of the running session, built now.
        In:  -
        Out: SessionSummary | None  None before the first snapshot
        """
        record = self._session_log.record()
        return None if record is None else summarise(record, self._car(record), self._class_top)

    def past_summary(self, kind: str, skip_running: bool) -> dict[str, Any] | None:
        """Summary of the newest session of a kind on the running session's track: the running session itself when
        it is of that kind and not left out, else the stored file of the newest session that started before the
        running one, else one built from the game's results file (the one it points to when its kind matches, else
        the newest of that kind in the same folder).
        In:  kind          str   "practice" | "qualify" | "race"
             skip_running  bool  leave the running session out
        Out:               dict[str, Any] | None  the SessionSummary fields; None without such a session
        """
        record = self._session_log.record()
        if record is None:
            return None
        if kind == record.kind and not skip_running:
            running = summarise(record, self._car(record), self._class_top)
            return json.loads(json.dumps(dataclasses.asdict(running)))
        stored = self._sessions.latest(record.track, kind, record.started_unix)
        if stored is not None:
            return stored
        results = self.past_results(kind)
        summary = (
            None
            if results is None
            else summary_from_results(results, "results_file", self._track_field, self._class_top, record.player_name)
        )
        return None if summary is None else json.loads(json.dumps(dataclasses.asdict(summary)))

    def past_results(self, kind: str) -> SessionResults | None:
        """The game's results of the newest session of a kind: the file it points to when its kind matches, else the
        newest file of that kind in the same folder.
        In:  kind  str  "practice" | "qualify" | "race"
        Out:       SessionResults | None  None without a path or such a file
        """
        path = self._results_path()
        if path is None:
            return None
        results = read_results(path)
        if results is not None and results.kind == kind:
            return results
        return latest_results(path.parent, kind)

    def _qualifying(self, record: SessionRecord, path: Path | None) -> SessionResults | None:
        """Read the results file the game points to and give the qualifying results to take the priors from. The
        file's figures go into the history once, and only when it is not the running session's own file (that one is
        added when the session has ended, so a file still being written is never stored). The newest qualifying file
        of the folder is taken instead of the pointed file only when it did not start before the pointed file or
        before the previous session of this process: an older one belongs to an earlier event on the track.
        In:  record  SessionRecord  the running session
             path    Path | None    the results file the game points to
        Out:         SessionResults | None  the pointed file when it is a qualifying session, else the newest
                                            qualifying file in the same folder when it is recent enough; None
                                            without a path or such a file
        """
        if path is None:
            return None
        self._results = read_results(path)
        earliest_unix = self._previous_started_unix - self._results_margin_s
        if self._results is not None:
            if not results_of_session(self._results, record.kind, record.started_unix, self._results_margin_s):
                self._add(figures_from_results(self._results, path.stem, self._track_field, record.player_name))
            if self._results.kind == "qualify":
                return self._results
            earliest_unix = max(earliest_unix, self._results.started_unix)
        fallback = latest_results(path.parent, "qualify")
        return fallback if fallback is not None and fallback.started_unix >= earliest_unix else None

    def _finish(self, record: SessionRecord) -> None:
        """Store a session: its summary file, joined with the game's results file when that is this session's, and
        its figures in the history; the results file's own figures only once the session has ended.
        In:  record  SessionRecord
        Out:         None
        """
        car = self._car(record)
        summary = summarise(record, car, self._class_top)
        path = self._results_path()
        results = None if path is None else read_results(path)
        if results is not None and results_of_session(
            results, record.kind, record.started_unix, self._results_margin_s
        ):
            summary = with_results(summary, results, record.player_name)
            if record.ended:
                self._add(figures_from_results(results, path.stem, self._track_field, record.player_name))
        self._write(summary)
        if record.laps:
            self._add(figures_from_record(record, summary, car, self._default_rates))

    def _car(self, record: SessionRecord) -> str:
        """Car model of the player in a session.
        In:  record  SessionRecord
        Out:         str  from the results file, else from the history, else the vehicle name
        """
        return car_key(record.vehicle_name, record.player_name, self._results, self._entries or ())

    def _add(self, figures: SessionFigures | None) -> None:
        """Append a session's figures to the history unless it is already there; a disk error is logged.
        In:  figures  SessionFigures | None  None = nothing to add
        Out:          None
        """
        entries = self._entries or ()
        if figures is None or any(e.session_id == figures.session_id for e in entries):
            return
        self._entries = (*entries, figures)
        try:
            self._history.append(figures)
        except OSError as error:
            logger.warning("history not written: %s", error)

    def _write(self, summary: SessionSummary) -> None:
        """Write a summary file and note it in the run log; a disk error is logged.
        In:  summary  SessionSummary
        Out:          None
        """
        try:
            path = self._sessions.write(summary)
        except OSError as error:
            logger.warning("session file not written: %s", error)
            return
        self._log.write("sessions", {"session_id": summary.session_id, "path": path, "complete": summary.complete})


def _stamp(path: Path | None) -> tuple[Path, float] | None:
    """What tells one state of a results file from another, without reading it.
    In:  path  Path | None  the results file the game points to
    Out:       tuple[Path, float] | None  (path, modification time s unix); None without a path or file
    """
    try:
        return None if path is None else (path, path.stat().st_mtime)
    except OSError:
        return None
