import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ai_race_engineer.domains.common.recording.layout import SESSION_FILE, TABLES
from ai_race_engineer.domains.common.recording.schema import schema_of
from ai_race_engineer.domains.common.data.frame import SessionInfo


class ParquetSessionWriter:
    """Writes one session folder: session.yaml plus one folder of rolling Parquet part files per table.
    Raises OSError on disk failures; the caller handles them.
    _part_s    float                       s a part file stays open before rolling to the next
    _folder    Path | None                 session folder (None before open)
    _writers   dict[str, pq.ParquetWriter] table name -> writer of its open part
    _parts     dict[str, int]              table name -> index of its latest part
    _opened_s  dict[str, float]            table name -> s monotonic clock when its open part was opened
    """

    def __init__(self, part_s: float) -> None:
        """Set the part length.
        In:  part_s  float  s per part file
        Out:         None
        """
        self._part_s: float = part_s
        self._folder: Path | None = None
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._parts: dict[str, int] = {}
        self._opened_s: dict[str, float] = {}

    def open(self, folder: Path, info: SessionInfo, started: datetime) -> None:
        """Create the session folder and write session.yaml.
        In:  folder   Path         session folder
             info     SessionInfo
             started  datetime     local wall-clock start of the recording
        Out:          None
        """
        folder.mkdir(parents=True, exist_ok=True)
        self._folder = folder
        self._parts = {}
        record = {**asdict(info), "session_type": info.session_type.name, "started": started.isoformat()}
        (folder / SESSION_FILE).write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True), "utf-8")

    def append(self, table: str, rows: pd.DataFrame) -> None:
        """Append rows as one row group to the table's open part, rolling to a new part after part_s.
        In:  table  str           key of TABLES
             rows   pd.DataFrame  columns named like the table's row fields (missing ones become null)
        Out:        None
        """
        if table in self._writers and time.monotonic() - self._opened_s[table] >= self._part_s:
            self._writers.pop(table).close()
        if table not in self._writers:
            self._parts[table] = self._parts.get(table, -1) + 1
            self._writers[table] = pq.ParquetWriter(
                self._part_path(table, self._parts[table]), schema_of(TABLES[table])
            )
            self._opened_s[table] = time.monotonic()
        writer = self._writers[table]
        writer.write_table(_arrow_table(rows, writer.schema))

    def close(self) -> None:
        """Close all open part files; each is dropped before closing, so a failed close can't leak into a later session.
        In:  -
        Out: None
        """
        while self._writers:
            self._writers.popitem()[1].close()

    def write(self, table: str, rows: pd.DataFrame) -> None:
        """Write a complete table as its first part.
        In:  table  str           key of TABLES
             rows   pd.DataFrame  columns named like the table's row fields (missing ones become null)
        Out:        None
        """
        pq.write_table(_arrow_table(rows, schema_of(TABLES[table])), self._part_path(table, 0))

    def _part_path(self, table: str, index: int) -> Path:
        """Path of one part file of a table; creates the table folder.
        In:  table  str   key of TABLES
             index  int   part number, from 0
        Out:        Path  <folder>/<table>/part-NNN.parquet
        """
        (self._folder / table).mkdir(exist_ok=True)
        return self._folder / table / f"part-{index:03d}.parquet"


def _arrow_table(rows: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Arrow table with exactly the schema's columns, in order.
    In:  rows    pd.DataFrame
         schema  pa.Schema
    Out:         pa.Table
    """
    return pa.Table.from_pandas(rows.reindex(columns=schema.names), schema=schema, preserve_index=False)
