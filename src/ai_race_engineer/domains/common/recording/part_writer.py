import multiprocessing as mp
import signal
import time
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

type Rows = list[dict[str, object]]
type ToColumns = Callable[[str, Rows], dict[str, object]]

END: str = "end"
STOP: None = None


class PartWriter:
    """Writes buffered rows as rolling Parquet part files <folder>/<table>/part-NNN.parquet in a separate process fed
    through a queue, so converting and writing never slow the polling loop (no shared GIL). A hard kill loses at
    most the open part of each table.
    _queue    mp.Queue    buffers, folder changes and the stop mark for the writer process
    _process  mp.Process  the writer process
    """

    def __init__(self, to_columns: ToColumns, part_s: float) -> None:
        """Prepare the writer process; start runs it.
        In:  to_columns  ToColumns  (table name, its buffered rows) -> column name -> values; a module-level function
             part_s      float      s after which a table's part file is closed and the next one started
        Out:             None
        """
        self._queue: mp.Queue = mp.Queue()
        self._process: mp.Process = mp.Process(
            target=_write_parts, args=(self._queue, to_columns, part_s), daemon=False
        )

    def start(self) -> None:
        """Start the writer process; it runs until close.
        In:  -
        Out: None
        """
        self._process.start()

    def open(self, folder: Path) -> None:
        """Close the open part files and write what follows into a new folder; never blocks.
        In:  folder  Path  existing folder of one session
        Out:         None
        """
        self._queue.put(folder)

    def put(self, buffers: dict[str, Rows]) -> None:
        """Queue buffered rows for the open folder; never blocks.
        In:  buffers  dict[str, Rows]  table name -> rows; empty = nothing is queued
        Out:          None
        """
        if buffers:
            self._queue.put(buffers)

    def end(self) -> None:
        """Close the open part files; never blocks.
        In:  -
        Out: None
        """
        self._queue.put(END)

    def close(self) -> None:
        """Have everything queued written and wait for the writer process to end; Ctrl+C while waiting is answered
        with a notice and the waiting goes on.
        In:  -
        Out: None
        """
        self._queue.put(STOP)
        self._queue.close()
        while True:
            try:
                self._queue.join_thread()
                self._process.join()
                return
            except KeyboardInterrupt:
                print("still writing to disk, please wait ...")


def _write_parts(queue: mp.Queue, to_columns: ToColumns, part_s: float) -> None:
    """Writer process: write what the queue brings until the stop mark; Ctrl+C is left to the polling process.
    In:  queue       mp.Queue   buffers (dict), a folder (Path), END or STOP
         to_columns  ToColumns  (table name, its buffered rows) -> column name -> values
         part_s      float      s per part file
    Out:             None
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    PartFiles(queue, to_columns, part_s).loop()


class PartFiles:
    """The open part files of one session folder, inside the writer process.
    _queue       mp.Queue   buffers (dict), a folder (Path), END or STOP
    _to_columns  ToColumns  (table name, its buffered rows) -> column name -> values
    _part_s      float      s per part file
    _folder      Path | None  the open session folder
    _writers     dict[str, tuple[pq.ParquetWriter, float, int]]  table -> (open file, s monotonic clock at which
                                                                 it was started, part number)
    """

    def __init__(self, queue: mp.Queue, to_columns: ToColumns, part_s: float) -> None:
        """Store the queue and how rows become columns.
        In:  queue       mp.Queue
             to_columns  ToColumns
             part_s      float      s per part file
        Out:             None
        """
        self._queue: mp.Queue = queue
        self._to_columns: ToColumns = to_columns
        self._part_s: float = part_s
        self._folder: Path | None = None
        self._writers: dict[str, tuple[pq.ParquetWriter, float, int]] = {}

    def loop(self) -> None:
        """Write until the stop mark. A table that cannot be converted or written is reported on the console and
        the writing goes on.
        In:  -
        Out: None
        """
        while (item := self._queue.get()) is not STOP:
            if isinstance(item, Path | str):
                self._close_all()
                self._folder = item if isinstance(item, Path) else None
                continue
            for table, rows in item.items():
                if rows:
                    try:
                        self._write(table, self._to_columns(table, rows))
                    except Exception as error:
                        print(f"write failed for {table}: {error!r}")
        self._close_all()

    def _close_all(self) -> None:
        """Close every open part file.
        In:  -
        Out: None
        """
        for writer, _, _ in self._writers.values():
            writer.close()
        self._writers = {}

    def _write(self, table: str, columns: dict[str, object]) -> None:
        """Append columns to the table's open part file; the first rows of a part set its schema, and a part older
        than part_s is closed and the next one started.
        In:  table    str                table name
             columns  dict[str, object]  column name -> values
        Out:          None
        """
        now = time.monotonic()
        entry = self._writers.get(table)
        rolled = entry is not None and now - entry[1] >= self._part_s
        if rolled:
            entry[0].close()
        if entry is None or rolled:
            part = 0 if entry is None else entry[2] + 1
            data = pa.Table.from_pydict(columns)
            (self._folder / table).mkdir(exist_ok=True)
            file = self._folder / table / f"part-{part:03d}.parquet"
            self._writers[table] = (pq.ParquetWriter(file, data.schema, compression="zstd"), now, part)
            self._writers[table][0].write_table(data)
            return
        entry[0].write_table(pa.Table.from_pydict(columns, schema=entry[0].schema))
