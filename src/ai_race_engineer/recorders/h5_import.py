import logging
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from numpy.typing import NDArray

from ai_race_engineer.recorders.h5_convert import (
    cars_tables,
    channels_frame,
    clock_offset_s,
    group_split_sessions,
    player_state_table,
    player_telemetry_table,
    session_info,
)
from ai_race_engineer.recorders.layout import session_dir
from ai_race_engineer.recorders.parquet_session import ParquetSessionWriter
from ai_race_engineer.strategy.lap_table import lap_table

logger = logging.getLogger(__name__)

_H5_FILE: str = "telemetry.h5"
_TELEMETRY_GROUPS: tuple[str, ...] = ("telemetry_data", "vehicle_data")
_UNUSED_CHANNELS: frozenset[str] = frozenset({"relative_x", "relative_y", "relative_z"})
_META_KEYS: tuple[str, ...] = ("timestamp", "combo_id", "session_id", "session_type", "track_name", "vehicle_name")


def import_all(root: Path, out_root: Path) -> list[Path]:
    """Import every telemetry.h5 folder under root, merging consecutive folders split from one session.
    Files with unreadable or incomplete metadata are logged and dropped; failing sessions are logged and skipped.
    In:  root      Path        folder of telemetry.h5 folders
         out_root  Path        root of the session folders
    Out:           list[Path]  session folders written or already present
    """
    metas: list[tuple[Path, dict[str, str]]] = []
    for h5 in sorted(root.glob(f"*/{_H5_FILE}")):
        try:
            metas.append((h5.parent, _metadata(h5)))
        except (OSError, KeyError) as err:
            logger.error("dropped %s: %r", h5.parent.name, err)
    done: list[Path] = []
    for folders in group_split_sessions(metas):
        try:
            done.append(import_folders(folders, out_root))
        except (OSError, KeyError, ValueError) as err:
            logger.error("skipped %s: %r", folders[0].name, err)
    return done


def import_folders(folders: Sequence[Path], out_root: Path) -> Path:
    """Import the telemetry.h5 folders of one session into one session folder; skipped if it already exists.
    Written into '<folder>.partial' and renamed when complete, so a crashed import is redone on the next run.
    In:  folders   Sequence[Path]  folders of one session, in time order
         out_root  Path            root of the session folders
    Out:           Path            session folder
    """
    files = [folder / _H5_FILE for folder in folders]
    statics = [_read_group(path, ("static_data",)) for path in files]
    meta = _metadata(files[0])
    info = session_info(meta, statics[0])
    started = datetime.strptime(meta["timestamp"], "%Y%m%d_%H%M%S")
    out = session_dir(out_root, started, info)
    if out.exists():
        logger.info("already imported: %s", out)
        return out
    parts = [_convert(path, static) for path, static in zip(files, statics, strict=True)]
    tables = {name: pd.concat([part[name] for part in parts], ignore_index=True) for name in parts[0]}
    tables["laps"] = lap_table(tables["cars_state_5hz"], tables["player_state_5hz"])
    partial = out.with_name(f"{out.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    writer = ParquetSessionWriter(part_s=0.0)
    writer.open(partial, info, started)
    for name, rows in tables.items():
        writer.write(name, rows)
    partial.rename(out)
    logger.info("imported %s (%d laps)", out, len(tables["laps"]))
    return out


def _convert(path: Path, static: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Session tables (without laps) of one telemetry.h5, on its own telemetry clock.
    In:  path    Path                     telemetry.h5
         static  pd.DataFrame             its static_data channels (may be empty)
    Out:         dict[str, pd.DataFrame]  table name -> rows
    """
    telemetry = _read_group(path, _TELEMETRY_GROUPS)
    offset_s = clock_offset_s(telemetry["timestamp"], telemetry["session_time"])
    cars_positions, cars_state = cars_tables(_read_group(path, ("vehicles_data",)), offset_s)
    return {
        "player_telemetry_100hz": player_telemetry_table(telemetry, offset_s),
        "cars_positions_50hz": cars_positions,
        "player_state_5hz": player_state_table(telemetry, static, offset_s),
        "cars_state_5hz": cars_state,
    }


def _metadata(path: Path) -> dict[str, str]:
    """Metadata attributes the import needs; KeyError when one is missing.
    In:  path  Path            telemetry.h5
    Out:       dict[str, str]  attribute name -> value as text (keys of _META_KEYS)
    """
    with h5py.File(path, "r") as f:
        attrs = f["metadata"].attrs
        return {key: str(attrs[key]) for key in _META_KEYS}


def _read_group(path: Path, names: Sequence[str]) -> pd.DataFrame:
    """Channels of the first present group of names as a frame (strings decoded); empty when none is present.
    In:  path   Path           telemetry.h5
         names  Sequence[str]  group names to try, in order
    Out:        pd.DataFrame   one column per channel
    """
    with h5py.File(path, "r") as f:
        name = next((n for n in names if n in f), None)
        if name is None:
            return pd.DataFrame()
        return channels_frame({key: _values(data) for key, data in f[name].items() if key not in _UNUSED_CHANNELS})


def _values(data: h5py.Dataset) -> NDArray[Any]:
    """All values of a dataset, with byte strings decoded to str.
    In:  data  h5py.Dataset
    Out:       NDArray[Any]
    """
    return data.asstr()[()] if h5py.check_string_dtype(data.dtype) else data[()]
