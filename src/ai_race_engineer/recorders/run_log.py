import dataclasses
import itertools
import json
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import IO

STOP: None = None
CLOSE_TIMEOUT_S: float = 5.0
MAX_ERROR_CHARS: int = 200


class RunLog:
    """Debug log of one process run: JSON lines per stream in data/logs/<YYYYmmdd_HHMMSS>/<stream>.jsonl. Writing is
    non-blocking; one writer thread converts and stores the records.
    _folder          Path                 folder of this run (created on the first write)
    _enabled         bool                 False = write does nothing; set False when the writer thread ends
    _session_time_s  Callable[[], float]  s session time now
    _records         queue.Queue[tuple[str, dict[str, object]] | None]  stamped records waiting for the writer
    _sequence        itertools.count[int]  record counter
    _writer          threading.Thread     the writer thread
    """

    def __init__(self, root: Path, enabled: bool, session_time_s: Callable[[], float]) -> None:
        """Name the run folder after the start time and start the writer thread.
        In:  root            Path                 folder that holds one folder per run
             enabled         bool                 False = nothing is written and no folder is made
             session_time_s  Callable[[], float]  s session time now (non-blocking)
        Out:                 None
        """
        self._folder: Path = root / time.strftime("%Y%m%d_%H%M%S")
        self._enabled: bool = enabled
        self._session_time_s: Callable[[], float] = session_time_s
        self._records: queue.Queue[tuple[str, dict[str, object]] | None] = queue.Queue()
        self._sequence: itertools.count[int] = itertools.count()
        self._writer: threading.Thread = threading.Thread(target=self._write_forever, daemon=True)
        if enabled:
            self._writer.start()

    def write(self, stream: str, record: Mapping[str, object]) -> None:
        """Queue one record for a stream, stamped with wall time, session time and a sequence number; never blocks.
        In:  stream  str                   file name without extension, e.g. "callouts"
             record  Mapping[str, object]  values; dataclasses, enums and paths are converted
                                           by the writer thread
        Out:         None
        """
        if not self._enabled:
            return
        stamp = {"seq": next(self._sequence), "wall_s": time.time(), "session_s": self._session_time_s()}
        self._records.put((stream, {**stamp, **record}))

    def queue_size(self) -> int:
        """Records waiting to be written.
        In:  -
        Out: int
        """
        return self._records.qsize()

    def close(self) -> None:
        """Write what is queued and stop the writer thread.
        In:  -
        Out: None
        """
        if self._writer.is_alive():
            self._records.put(STOP)
            self._writer.join(CLOSE_TIMEOUT_S)

    def __enter__(self) -> "RunLog":
        """Use as a context manager.
        In:  -
        Out: RunLog  this log
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the log.
        In:  exc  object  exception info (ignored)
        Out:      None
        """
        self.close()

    def _write_forever(self) -> None:
        """Writer thread: take the queued records in batches, append each as one JSON line to its stream's file and
        flush. A disk error is reported once on the console and ends the writing; however the thread ends, writing
        is switched off so the queue is no longer filled.
        In:  -
        Out: None
        """
        files: dict[str, IO[str]] = {}
        try:
            while True:
                batch = [self._records.get()]
                while not self._records.empty():
                    batch.append(self._records.get())
                for item in batch:
                    if item is STOP:
                        return
                    stream, record = item
                    if stream not in files:
                        self._folder.mkdir(parents=True, exist_ok=True)
                        files[stream] = (self._folder / f"{stream}.jsonl").open("a", encoding="utf-8")
                    files[stream].write(_line(record) + "\n")
                for file in files.values():
                    file.flush()
        except OSError as error:
            print(f"run log stopped: {error}", flush=True)
        finally:
            self._enabled = False
            for file in files.values():
                file.close()


def _line(record: Mapping[str, object]) -> str:
    """One record as a JSON line. A record that cannot be converted or encoded (a value that refers to itself, text
    that is not valid Unicode) is replaced by its stamp and a short text of the failure, so one bad record never
    stops the log.
    In:  record  Mapping[str, object]  stamped record
    Out:         str                   JSON text without the line end, encodable as UTF-8
    """
    try:
        text = json.dumps(_plain(record), ensure_ascii=False)
        text.encode("utf-8")
        return text
    except Exception as error:
        stamp = {key: record.get(key) for key in ("seq", "wall_s", "session_s")}
        return json.dumps({**stamp, "unwritable_record": repr(error)[:MAX_ERROR_CHARS]})


def _plain(value: object) -> object:
    """Value as JSON-ready data.
    In:  value  object  dataclass, enum, path, mapping, sequence, set or plain value
    Out:        object  dicts with text keys, lists, text, numbers, booleans and None; an enum becomes its name, a
                        number that is not finite None, anything unknown its text
    """
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(item) for item in value]
    return str(value)
