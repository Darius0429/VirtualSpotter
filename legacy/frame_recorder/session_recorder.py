import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

from ai_race_engineer.domains.common.recording.layout import TABLES, session_dir
from ai_race_engineer.domains.common.recording.parquet_session import ParquetSessionWriter
from ai_race_engineer.domains.common.recording.session_change import is_new_session
from ai_race_engineer.domains.common.analysis.pace.lap_table import lap_table
from ai_race_engineer.domains.common.data.frame import Frame, SessionInfo
from ai_race_engineer.domains.common.data.frame_source import FrameSource

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Records one session folder per game session: the poll thread only polls and buffers, one writer thread
    does all disk work (batches every flush_s, session open/close, laps) in order.
    _source        FrameSource           opened by the caller
    _writer        ParquetSessionWriter  used by the writer thread only
    _root          Path                  root of the session folders
    _poll_hz       float                 Hz
    _flush_s       float                 s between batch writes
    _stop          threading.Event       set by stop()
    _work          queue.Queue[Callable[[], None] | None]  disk work for the writer thread (None ends it)
    _worker        threading.Thread      writer thread
    _session       SessionInfo | None    poll thread: session being recorded (None between sessions)
    _last_time_s   float                 poll thread: s telemetry clock of the last frame
    _last_flush_s  float                 poll thread: s monotonic clock of the last hand-over
    _buffers       dict[str, list[object]]  poll thread: table name -> rows since the last hand-over
    _folder        Path | None           writer thread: folder of the session being written
    _writing       bool                  writer thread: False after a disk failure until the next session
    """

    def __init__(
        self, source: FrameSource, writer: ParquetSessionWriter, root: Path, poll_hz: float, flush_s: float
    ) -> None:
        """Store the source, writer and timing.
        In:  source   FrameSource           opened by the caller
             writer   ParquetSessionWriter
             root     Path                  root of the session folders
             poll_hz  float                 Hz
             flush_s  float                 s between batch writes
        Out:          None
        """
        self._source: FrameSource = source
        self._writer: ParquetSessionWriter = writer
        self._root: Path = root
        self._poll_hz: float = poll_hz
        self._flush_s: float = flush_s
        self._stop: threading.Event = threading.Event()
        self._work: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._worker: threading.Thread = threading.Thread(target=self._write_loop)
        self._session: SessionInfo | None = None
        self._last_time_s: float = 0.0
        self._last_flush_s: float = time.monotonic()
        self._buffers: dict[str, list[object]] = _empty_buffers()
        self._folder: Path | None = None
        self._writing: bool = False

    def run(self) -> None:
        """Poll until stop() (or Ctrl+C); on exit the open session is closed and all queued disk work is finished.
        In:  -
        Out: None
        """
        self._worker.start()
        try:
            while not self._stop.is_set():
                frame = self._source.read_new()
                if frame is not None:
                    self._record(frame)
                if time.monotonic() - self._last_flush_s >= self._flush_s:
                    self._hand_over()
                time.sleep(1 / self._poll_hz)
        finally:
            self._end_session()
            self._work.put(None)
            self._worker.join()

    def stop(self) -> None:
        """Ask run() to finish and wait until all queued disk work is written.
        In:  -
        Out: None
        """
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join()

    def _record(self, frame: Frame) -> None:
        """Start a new session if needed and buffer the frame's rows.
        In:  frame  Frame
        Out:        None
        """
        if is_new_session(self._session, frame.session, self._last_time_s, frame.player.time_s):
            self._end_session()
            started = datetime.now()
            self._session = frame.session
            self._work.put(
                partial(self._open_session, session_dir(self._root, started, frame.session), frame.session, started)
            )
        self._last_time_s = frame.player.time_s
        self._buffers["player_telemetry_100hz"].append(frame.player)
        self._buffers["cars_positions_50hz"].extend(frame.cars)
        if frame.scoring is not None:
            self._buffers["player_state_5hz"].append(frame.scoring.state)
            self._buffers["cars_state_5hz"].extend(frame.scoring.cars)

    def _hand_over(self) -> None:
        """Pass the buffered rows to the writer thread.
        In:  -
        Out: None
        """
        rows, self._buffers = self._buffers, _empty_buffers()
        self._last_flush_s = time.monotonic()
        self._work.put(partial(self._flush, rows))

    def _end_session(self) -> None:
        """Hand over the last rows of the open session and queue its close.
        In:  -
        Out: None
        """
        if self._session is None:
            return
        self._hand_over()
        self._work.put(self._close_session)
        self._session = None

    def _write_loop(self) -> None:
        """Writer thread: run queued disk work in order until None.
        In:  -
        Out: None
        """
        while (work := self._work.get()) is not None:
            work()

    def _open_session(self, folder: Path, info: SessionInfo, started: datetime) -> None:
        """Writer thread: create the session folder; a disk failure disables writing for this session.
        In:  folder   Path         session folder
             info     SessionInfo
             started  datetime     local wall-clock start of the recording
        Out:          None
        """
        self._folder = folder
        try:
            self._writer.open(folder, info, started)
            self._writing = True
            logger.info("recording %s", folder)
        except OSError as err:
            self._writing = False
            logger.error("cannot record %s: %s", folder, err)

    def _flush(self, rows: dict[str, list[object]]) -> None:
        """Writer thread: append rows to the session's tables; a disk failure disables writing for this session.
        In:  rows  dict[str, list[object]]  table name -> row dataclasses
        Out:       None
        """
        if not self._writing:
            return
        try:
            for table, table_rows in rows.items():
                if table_rows:
                    self._writer.append(table, pd.DataFrame([vars(row) for row in table_rows]))
        except OSError as err:
            self._writing = False
            logger.error("stopped recording %s: %s", self._folder, err)

    def _close_session(self) -> None:
        """Writer thread: close the part files and write the laps table (skipped without any scoring update).
        In:  -
        Out: None
        """
        try:
            self._writer.close()
            if self._writing and not (self._folder / "cars_state_5hz").exists():
                logger.info("no scoring update, no laps: %s", self._folder)
            elif self._writing:
                cars_state = pd.read_parquet(self._folder / "cars_state_5hz")
                self._writer.write("laps", lap_table(cars_state, pd.read_parquet(self._folder / "player_state_5hz")))
                logger.info("closed %s", self._folder)
        except OSError as err:
            logger.error("cannot finish %s: %s", self._folder, err)
        self._writing = False


def _empty_buffers() -> dict[str, list[object]]:
    """One empty row list per streamed table.
    In:  -
    Out: dict[str, list[object]]  table name -> []
    """
    return {table: [] for table in TABLES if table != "laps"}
