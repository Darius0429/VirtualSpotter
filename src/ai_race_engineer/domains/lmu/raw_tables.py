from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

STREAM_TABLE: str = "raw_scoring_stream"
STAMP_KEYS: tuple[str, ...] = ("poll_s", "wall_s")
UNPREFIXED_KEYS: tuple[str, ...] = ("telem", "info", "veh")
MAX_SPLIT_BYTES: int = 16


def to_columns(table: str, rows: list[dict[str, object]]) -> dict[str, object]:
    """Buffered rows of one raw table as columns. A row holds its stamps and one or more arrays of game structs of
    the same length; every struct field becomes a column named like mWheels.mTemperature.1.2 (arrays of structs:
    field, then index), every struct gets the stamps of its row.
    In:  table  str                      table name; the scoring stream holds text instead of structs
         rows   list[dict[str, object]]  poll_s (s since the recorder started), wall_s (s unix time) and struct
                                         arrays by key; the fields of the keys telem, info and veh keep their names,
                                         those of any other key get "<key>." in front
    Out:        dict[str, object]        column name -> numpy or Arrow array, one value per struct
    """
    if table == STREAM_TABLE:
        return {key: [row[key] for row in rows] for key in (*STAMP_KEYS, "text")}
    struct_keys = [key for key in rows[0] if key not in STAMP_KEYS]
    counts = [len(row[struct_keys[0]]) for row in rows]
    columns: dict[str, object] = {key: np.repeat([row[key] for row in rows], counts) for key in STAMP_KEYS}
    for key in struct_keys:
        flatten(np.concatenate([row[key] for row in rows]), "" if key in UNPREFIXED_KEYS else f"{key}.", columns)
    return columns


def flatten_one(struct: np.void) -> dict[str, object]:
    """One game struct as plain values.
    In:  struct  np.void            one struct
    Out:         dict[str, object]  flattened field name -> number, or text cut at the first null
    """
    columns: dict[str, object] = {}
    flatten(np.array([struct]), "", columns)
    return {
        key: column[0].item() if isinstance(column, np.ndarray) else text(column[0].as_py())
        for key, column in columns.items()
    }


def flatten(values: NDArray[Any], key: str, columns: dict[str, object]) -> None:
    """Add the columns of a struct array: one per number field and array element; char arrays as bytes
    (null-padded), byte arrays of more than 16 bytes (padding) as one fixed-size binary column.
    In:  values   NDArray[Any]       structs or values, first axis = rows
         key      str                column name so far; "" or ending in "." = no dot is put before the next name
         columns  dict[str, object]  column name -> numpy or Arrow array; filled in place
    Out:          None
    """
    if values.dtype.names:
        for name in values.dtype.names:
            flatten(values[name], f"{key}{name}" if not key or key.endswith(".") else f"{key}.{name}", columns)
        return
    if values.ndim == 1:
        columns[key] = pa.array(values) if values.dtype.kind == "S" else values
        return
    if values.dtype.kind == "S" and values.ndim == 2:
        width = values.shape[1] * values.dtype.itemsize
        columns[key] = pa.array(np.ascontiguousarray(values).view(f"S{width}").ravel())
        return
    width = int(np.prod(values.shape[1:]))
    if values.dtype.itemsize == 1 and values.dtype.kind in "iub" and width > MAX_SPLIT_BYTES:
        raw = pa.py_buffer(np.ascontiguousarray(values.reshape(len(values), width)).tobytes())
        columns[key] = pa.FixedSizeBinaryArray.from_buffers(pa.binary(width), len(values), [None, raw])
        return
    for index in range(values.shape[1]):
        flatten(values[:, index], f"{key}.{index}", columns)


def text(raw: bytes) -> str:
    """Text of a null-padded char array.
    In:  raw  bytes  UTF-8, padded with nulls
    Out:      str    the text before the first null
    """
    return bytes(raw).split(bytes(1), 1)[0].decode("utf-8", "replace")


def safe(name: str) -> str:
    """A name that can be used as a folder name.
    In:  name  str
    Out:       str  letters, digits, "-", "_" and "." kept, everything else "_"
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
