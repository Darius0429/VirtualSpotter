import contextlib
import dataclasses
import io
import json
import logging
import queue
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ai_race_engineer.strategy.lap_segments import LapSamples, LineSamples
from ai_race_engineer.strategy.segment_records import Record

logger = logging.getLogger(__name__)

STOP: None = None
CLOSE_TIMEOUT_S: float = 5.0
PLAIN: tuple[str, ...] = (
    "kind",
    "session",
    "slot_id",
    "driver",
    "class_name",
    "me",
    "lap",
    "name",
    "time_s",
    "finished_s",
)
JSON: tuple[str, ...] = ("figures", "vs")
COLUMNS: str = ", ".join((*PLAIN, *JSON))
SCHEMA: str = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS times (
    id INTEGER PRIMARY KEY, kind TEXT, session INTEGER, slot_id INTEGER, driver TEXT, class_name TEXT, me INTEGER,
    lap INTEGER, name TEXT, time_s REAL, finished_s REAL, figures TEXT, vs TEXT
);
CREATE INDEX IF NOT EXISTS times_of_car ON times (me, driver, session, lap);
CREATE TABLE IF NOT EXISTS telemetry (id INTEGER PRIMARY KEY, arrays BLOB);
"""


class SegmentStore:
    """The one list of everything timed in a run, as an SQLite file: a row for every segment, sector and lap of
    every car the moment it is finished, the player's with the differences to the reference laps, for each of the
    player's segments its telemetry and for every full lap of every car its line. Nothing is kept in memory; every
    question is a query on the file, during the run and afterwards. Rows are written by one thread so that adding
    never waits for the disk.
    session   int               number of the running session within the file, from 1; 0 before the first
    _path     Path              the file
    _records  queue.Queue[Record | None]  records waiting to be written
    _writer   threading.Thread  the writing thread
    """

    def __init__(self, path: Path) -> None:
        """Open the file of a run, new or of an earlier run, and start the writing thread; a folder or file that
        cannot be made raises OSError or sqlite3.Error.
        In:  path  Path  the SQLite file; its folder is created
        Out:       None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(sqlite3.connect(path)) as db:
            db.executescript(SCHEMA)
            self.session: int = db.execute("SELECT COALESCE(MAX(session), 0) FROM times").fetchone()[0]
        self._path: Path = path
        self._records: queue.Queue[Record | None] = queue.Queue()
        self._writer: threading.Thread = threading.Thread(target=self._write_forever, daemon=True)
        self._writer.start()

    def start_session(self) -> None:
        """Count a new session; the rows of the sessions before stay.
        In:  -
        Out: None
        """
        self.session += 1

    def append(self, record: Record) -> None:
        """Queue one record for the file; never blocks.
        In:  record  Record
        Out:         None
        """
        self._records.put(record)

    def select(self, where: str, values: Sequence[object] = ()) -> list[Record]:
        """The records that match a condition, without their telemetry; everything added before the call is in.
        In:  where   str               SQL condition on the columns id, kind, session, slot_id, driver, class_name,
                                       me, lap, name, time_s and finished_s, with ? for each value
             values  Sequence[object]  the values
        Out:         list[Record]      oldest first; empty when the file cannot be read
        """
        rows = self._query(f"SELECT {COLUMNS} FROM times WHERE {where} ORDER BY id", values)
        return [
            Record(
                **{**dict(zip(PLAIN, row, strict=False)), "me": bool(row[PLAIN.index("me")])},
                **{name: json.loads(text) for name, text in zip(JSON, row[len(PLAIN) :], strict=True)},
            )
            for row in rows
        ]

    def telemetry(self, record: Record) -> Record:
        """A record with its arrays: a segment of the player with its telemetry, the differences per sample and the
        lines of the player's own reference laps; a full lap of any car with its line.
        In:  record  Record  from select
        Out:         Record  the same with samples, traces and lines; unchanged when the file has none for it
        """
        where = "session = ? AND me = ? AND driver = ? AND kind = ? AND lap = ? AND name = ?"
        values = (record.session, record.me, record.driver, record.kind, record.lap, record.name)
        rows = self._query(f"SELECT arrays FROM telemetry WHERE id IN (SELECT id FROM times WHERE {where})", values)
        return dataclasses.replace(record, **_unpacked(rows[-1][0])) if rows else record

    def close(self) -> None:
        """Write what is queued and stop the writing thread.
        In:  -
        Out: None
        """
        if self._writer.is_alive():
            self._records.put(STOP)
            self._writer.join(CLOSE_TIMEOUT_S)

    def __enter__(self) -> "SegmentStore":
        """Use as a context manager.
        In:  -
        Out: SegmentStore  this store
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the store.
        In:  exc  object  exception info (ignored)
        Out:      None
        """
        self.close()

    def _query(self, sql: str, values: Sequence[object]) -> list[tuple[object, ...]]:
        """Rows of a query, after everything queued has been written; a file that cannot be read is logged.
        In:  sql     str
             values  Sequence[object]
        Out:         list[tuple[object, ...]]  empty when the file cannot be read
        """
        if self._writer.is_alive():
            self._records.join()
        try:
            with contextlib.closing(sqlite3.connect(self._path)) as db:
                return db.execute(sql, values).fetchall()
        except sqlite3.Error as error:
            logger.warning("segment store %s not read: %s", self._path, error)
            return []

    def _write_forever(self) -> None:
        """Writing thread: take the queued records in batches and add them to the file. A batch that cannot be
        written (disk error, locked file, a record that cannot be stored) is logged and left out; the thread carries
        on, so that nothing ever waits for a record that will not come.
        In:  -
        Out: None
        """
        with contextlib.closing(sqlite3.connect(self._path)) as db:
            while True:
                batch = [self._records.get()]
                while not self._records.empty():
                    batch.append(self._records.get())
                try:
                    for record in batch:
                        if record is not STOP:
                            _insert(db, record)
                    db.commit()
                except Exception as error:
                    logger.warning("segment store %s: %d records not written: %r", self._path, len(batch), error)
                    with contextlib.suppress(sqlite3.Error):
                        db.rollback()
                finally:
                    for _ in batch:
                        self._records.task_done()
                if STOP in batch:
                    return


def _insert(db: sqlite3.Connection, record: Record) -> None:
    """Add one record to the file, with its arrays when it has any.
    In:  db      sqlite3.Connection
         record  Record
    Out:         None
    """
    values = [getattr(record, name) for name in PLAIN]
    texts = [json.dumps(getattr(record, name), default=lambda value: value.item()) for name in JSON]
    marks = ", ".join("?" * (len(PLAIN) + len(JSON)))
    row = db.execute(f"INSERT INTO times ({COLUMNS}) VALUES ({marks})", [*values, *texts])
    if record.samples is not None or record.lines:
        db.execute("INSERT INTO telemetry (id, arrays) VALUES (?, ?)", (row.lastrowid, _packed(record)))


def _packed(record: Record) -> bytes:
    """The arrays of a record as one compressed block, the numbers in single precision.
    In:  record  Record  with samples or lines
    Out:         bytes
    """
    arrays: dict[str, np.ndarray] = {}
    if record.samples is not None:
        arrays.update({f"samples/{f.name}": getattr(record.samples, f.name) for f in dataclasses.fields(LapSamples)})
    for key, trace in record.traces.items():
        arrays.update({f"traces/{key}/{name}": values for name, values in trace.items()})
    for key, line in record.lines.items():
        arrays.update({f"lines/{key}/{f.name}": getattr(line, f.name) for f in dataclasses.fields(LineSamples)})
    block = io.BytesIO()
    small = {
        path: values.astype(np.float32) if values.dtype == np.float64 else values for path, values in arrays.items()
    }
    np.savez_compressed(block, **small)
    return block.getvalue()


def _unpacked(block: bytes) -> dict[str, object]:
    """The arrays of a record from their compressed block.
    In:  block  bytes
    Out:        dict[str, object]  {samples: LapSamples | None, traces: {reference: {name: array}}, lines: {key:
                                   LineSamples}}
    """
    groups: dict[str, dict[str, dict[str, np.ndarray]]] = {"traces": {}, "lines": {}}
    samples: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(block)) as arrays:
        for path in arrays.files:
            group, *key, name = path.split("/")
            values = arrays[path].astype(np.float64) if arrays[path].dtype == np.float32 else arrays[path]
            (groups[group].setdefault(key[0], {}) if key else samples)[name] = values
    lines = {key: LineSamples(**line) for key, line in groups["lines"].items()}
    return {"samples": LapSamples(**samples) if samples else None, "traces": groups["traces"], "lines": lines}
