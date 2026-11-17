"""Raw LMU dump: every field the game publishes, losslessly, until Ctrl+C.

One folder per session: <out_root>/<YYYYmmdd_HHMMSS>_<track>_<session>/ (e.g. ..._qualify1). The current folder is
closed when scoring mSession or the track changes, or the scoring clock mCurrentET jumps back (restart). The next
folder is only opened once the new session's clock actually runs, so frozen leftovers (results screen, loading) are
never written. The folder is also closed FINISH_GRACE_S after the player's mFinishStatus leaves 0 (finished, DNF,
DQ); nothing more is recorded until the next session or a restart. Telemetry rows whose clock is more than STALE_S from the session clock (stale data from the previous
session) are dropped. Formation lap and pre-race garage time belong to the race session and are recorded.
  raw_player_telemetry_100hz/  full LMUVehicleTelemetry of the player + generic block (FFB, events, app) + header,
                               one row per player telemetry tick (mElapsedTime change)
  raw_cars_telemetry_50hz/     full LMUVehicleTelemetry of every other car, one row per that car's tick
  raw_scoring_info_5hz/        full LMUScoringInfo, one row per scoring tick (mCurrentET change)
  raw_cars_scoring_5hz/        full LMUVehicleScoring of every car, one row per car per scoring tick
  raw_scoring_stream/          scoring stream text, one row when it changes
  paths.json                   game paths (once per session)
  rest/<name>.jsonl            (garage setup is only re-read while the player is in the pits)
                               the game's local REST API (localhost:6397): garage setup, repair/refuel screen
                               (brake wear, damage, virtual energy), pit stop estimate, weather forecast, session
                               settings. One line each time the answer changes: poll_s, wall_s, session_et, data
Every row has poll_s (s, monotonic since start) and wall_s (unix time). Struct fields become columns named like
mWheels.mTemperature.1.2 (arrays of structs: field, then index); char arrays are stored as bytes (decode with .str.decode("utf-8"), they are null-padded);
byte padding arrays (mExpansion etc.) are kept as fixed-size binary.
Parquet part files roll every PART_S seconds, so a hard kill loses at most the open part.

Usage: python record_raw.py [out_root=data/raw/raw_dumps] [stop_after_s]
"""

import json
import multiprocessing as mp
import signal
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pyLMUSharedMemory.lmu_data import LMUConstants, LMUObjectOut
from pyLMUSharedMemory.lmu_mmap import MMapControl

POLL_HZ = 250.0
FLUSH_S = 1.0
PART_S = 300.0
STALE_S = 2.0
FINISH_GRACE_S = 5.0
REST_URL = "http://localhost:6397"
REST_ENDPOINTS = {
    "garage_setup": ("/rest/garage/getPlayerGarageData", 5.0),
    "repair_and_refuel": ("/rest/garage/UIScreen/RepairAndRefuel", 5.0),
    "pitstop_estimate": ("/rest/strategy/pitstop-estimate", 5.0),
    "weather_forecast": ("/rest/sessions/weather", 120.0),
    "session_settings": ("/rest/sessions", 120.0),
}
OBJ = np.dtype(LMUObjectOut)
SESSION_NAMES = ["testday"] + [f"practice{i}" for i in range(1, 5)] + [f"qualify{i}" for i in range(1, 5)]
SESSION_NAMES += ["warmup"] + [f"race{i}" for i in range(1, 5)]


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/raw_dumps")
    stop_after_s = float(sys.argv[2]) if len(sys.argv) > 2 else float("inf")
    writer = Writer()
    rest = RestCapture()
    folder: Path | None = None
    session_key: tuple[int, str] | None = None
    start_et = 0.0
    finished_at: float | None = None
    session_done = False
    mmap = MMapControl(LMUConstants.LMU_SHARED_MEMORY_FILE, LMUObjectOut)
    mmap.create(access_mode=0)
    buffers: dict[str, list[dict[str, object]]] = {}
    last_tel = np.full(65536, np.nan)
    last_scoring_et: float | None = None
    last_stream = b""
    t0 = last_flush = last_print = time.perf_counter()
    rest.start(t0)
    print("waiting for a session (Ctrl+C to stop)")
    try:
        while True:
            mmap.update()
            now = time.perf_counter()
            if now - t0 > stop_after_s:
                break
            stamp = {"poll_s": now - t0, "wall_s": time.time()}
            obj = np.frombuffer(memoryview(mmap.data).cast("B"), dtype=OBJ, count=1)[0]
            tel, sco = obj["telemetry"], obj["scoring"]
            info = sco["scoringInfo"]
            if info["mNumVehicles"] == 0:
                time.sleep(1 / POLL_HZ)
                continue
            key = (int(info["mSession"]), text(info["mTrackName"]))
            session_et = float(info["mCurrentET"])
            restarted = last_scoring_et is not None and session_et < last_scoring_et - 1.0
            if key != session_key or restarted:
                writer.put(buffers)
                writer.end()
                if folder is not None:
                    print(f"{now - t0:7.0f}s  closed {folder}")
                buffers, last_stream, folder = {}, b"", None
                rest.folder = None
                last_tel[:] = np.nan
                session_key, start_et, last_scoring_et = key, session_et, session_et
                finished_at, session_done = None, False
            rest.session_et = session_et
            if finished_at is not None and now - finished_at >= FINISH_GRACE_S:
                writer.put(buffers)
                writer.end()
                print(f"{now - t0:7.0f}s  finished, closed {folder}")
                buffers, last_stream, folder = {}, b"", None
                rest.folder = None
                finished_at, session_done = None, True
            if folder is None:
                if session_done or session_et == start_et:
                    last_scoring_et = session_et
                    time.sleep(1 / POLL_HZ)
                    continue
                name = SESSION_NAMES[key[0]] if 0 <= key[0] < len(SESSION_NAMES) else f"session{key[0]}"
                folder = root / safe(f"{datetime.now():%Y%m%d_%H%M%S}_{key[1]}_{name}")
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "paths.json").write_text(json.dumps(flatten_one(obj["paths"]), indent=1))
                writer.open(folder)
                rest.folder = folder
                last_scoring_et = None
                print(f"{now - t0:7.0f}s  new session -> {folder}")
            n_tel = int(tel["activeVehicles"])
            cars = tel["telemInfo"][:n_tel]
            ids, clocks = cars["mID"].astype(np.int64), cars["mElapsedTime"]
            changed = np.flatnonzero((last_tel[ids] != clocks) & (np.abs(clocks - session_et) <= STALE_S)).tolist()
            last_tel[ids] = clocks
            player_idx = int(tel["playerVehicleIdx"]) if tel["playerHasVehicle"] else -1
            if player_idx in changed:
                buffers.setdefault("raw_player_telemetry_100hz", []).append(
                    {
                        **stamp,
                        "header": np.array(
                            [(n_tel, player_idx, bool(tel["playerHasVehicle"]))],
                            dtype=[("activeVehicles", "u1"), ("playerVehicleIdx", "i2"), ("playerHasVehicle", "?")],
                        ),
                        "telem": cars[player_idx : player_idx + 1].copy(),
                        "generic": obj["generic"].reshape(1).copy(),
                    }
                )
            others = [i for i in changed if i != player_idx]
            if others:
                buffers.setdefault("raw_cars_telemetry_50hz", []).append({**stamp, "telem": cars[others].copy()})
            if float(info["mCurrentET"]) != last_scoring_et:
                last_scoring_et = float(info["mCurrentET"])
                buffers.setdefault("raw_scoring_info_5hz", []).append({**stamp, "info": info.reshape(1).copy()})
                n_sco = int(info["mNumVehicles"])
                buffers.setdefault("raw_cars_scoring_5hz", []).append(
                    {**stamp, "veh": sco["vehScoringInfo"][:n_sco].copy()}
                )
                me = sco["vehScoringInfo"][:n_sco]
                me = me[me["mIsPlayer"] != 0]
                if len(me):
                    rest.in_pits = bool(me["mInPits"][0])
                    if finished_at is None and int(me["mFinishStatus"][0]) != 0:
                        finished_at = now
                size = int.from_bytes(bytes(sco["scoringStreamSize"][:4]), "little")
                stream = bytes(sco["scoringStream"])[: max(0, min(size, 65536))]
                if stream != last_stream:
                    last_stream = stream
                    buffers.setdefault("raw_scoring_stream", []).append(
                        {**stamp, "text": stream.decode("utf-8", "replace")}
                    )
            if now - last_flush >= FLUSH_S:
                writer.put(buffers)
                buffers, last_flush = {}, now
                if now - last_print >= 30:
                    last_print = now
                    print(f"{now - t0:7.0f}s  {SESSION_NAMES[key[0]] if 0 <= key[0] < 14 else key[0]}  "
                          f"cars {n_tel:2d}  session time {last_scoring_et}")
            time.sleep(max(0.0, 1 / POLL_HZ - (time.perf_counter() - now)))
    except KeyboardInterrupt:
        print("stopping, writing the last data (do not press Ctrl+C again) ...")
    finally:
        rest.folder = None
        writer.put(buffers)
        writer.close()
        mmap.close()
        print(f"done, last session: {folder}")


class RestCapture:
    """Background thread: polls the game's REST API and appends each changed answer to <folder>/rest/<name>.jsonl."""

    def __init__(self) -> None:
        self.folder: Path | None = None
        self.session_et = 0.0
        self.in_pits = True
        self._t0 = 0.0

    def start(self, t0: float) -> None:
        self._t0 = t0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        last: dict[str, bytes] = {}
        due = dict.fromkeys(REST_ENDPOINTS, 0.0)
        current: Path | None = None
        while True:
            folder = self.folder
            if folder != current:
                current, last = folder, {}
                due = dict.fromkeys(REST_ENDPOINTS, 0.0)
            now = time.perf_counter()
            for name, (path, every_s) in REST_ENDPOINTS.items():
                if folder is None or now < due[name]:
                    continue
                if name == "garage_setup" and not self.in_pits and name in last:
                    continue
                due[name] = now + every_s
                try:
                    with urllib.request.urlopen(REST_URL + path, timeout=1.0) as r:
                        body = r.read()
                    data = json.loads(body)
                except (OSError, ValueError):
                    continue
                if body == last.get(name):
                    continue
                last[name] = body
                (folder / "rest").mkdir(exist_ok=True)
                row = {"poll_s": now - self._t0, "wall_s": time.time(), "session_et": self.session_et, "data": data}
                line = json.dumps(row, ensure_ascii=False) + chr(10)
                with open(folder / "rest" / f"{name}.jsonl", "a", encoding="utf-8") as f:
                    f.write(line)
            time.sleep(0.2)


class Writer:
    """Separate writer process (no GIL contention with the poll loop) fed through a multiprocessing queue."""

    def __init__(self) -> None:
        self._queue: mp.Queue = mp.Queue()
        self._process = mp.Process(target=_writer_main, args=(self._queue,), daemon=False)
        self._process.start()

    def put(self, buffers: dict[str, list[dict[str, object]]]) -> None:
        if buffers:
            self._queue.put(buffers)

    def open(self, folder: Path) -> None:
        self._queue.put(folder)

    def end(self) -> None:
        self._queue.put("end")

    def close(self) -> None:
        self._queue.put(None)
        self._queue.close()
        while True:
            try:
                self._queue.join_thread()
                self._process.join()
                return
            except KeyboardInterrupt:
                print("still writing to disk, please wait ...")


def _writer_main(q: mp.Queue) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    PartWriter(q).loop()


class PartWriter:
    """Turns buffered struct rows into Parquet columns and rolling part files, one folder per session."""

    def __init__(self, q: mp.Queue) -> None:
        self._queue = q
        self._folder: Path | None = None
        self._writers: dict[str, tuple[pq.ParquetWriter, float, int]] = {}

    def loop(self) -> None:
        while (buffers := self._queue.get()) is not None:
            if isinstance(buffers, (Path, str)):
                self._close_all()
                self._folder = buffers if isinstance(buffers, Path) else None
                continue
            for table, rows in buffers.items():
                if rows:
                    try:
                        self._write(table, to_columns(table, rows))
                    except Exception as e:  # noqa: BLE001
                        print(f"write failed for {table}: {e!r}")
        self._close_all()

    def _close_all(self) -> None:
        for w, _, _ in self._writers.values():
            if w is not None:
                w.close()
        self._writers = {}

    def _write(self, table: str, cols: dict[str, object]) -> None:
        now = time.monotonic()
        entry = self._writers.get(table)
        if entry is not None and now - entry[1] >= PART_S:
            entry[0].close()
            entry = (None, now, entry[2] + 1)
        if entry is None or entry[0] is None:
            part = 0 if entry is None else entry[2]
            data = pa.Table.from_pydict(cols)
            (self._folder / table).mkdir(exist_ok=True)
            w = pq.ParquetWriter(self._folder / table / f"part-{part:03d}.parquet", data.schema, compression="zstd")
            self._writers[table] = (w, now, part)
            w.write_table(data)
            return
        w = entry[0]
        w.write_table(pa.Table.from_pydict(cols, schema=w.schema))


def to_columns(table: str, rows: list[dict[str, object]]) -> dict[str, object]:
    if table == "raw_scoring_stream":
        return {k: [r[k] for r in rows] for k in ("poll_s", "wall_s", "text")}
    struct_keys = [k for k in rows[0] if k not in ("poll_s", "wall_s")]
    counts = [len(r[struct_keys[0]]) for r in rows]
    cols: dict[str, object] = {
        "poll_s": np.repeat([r["poll_s"] for r in rows], counts),
        "wall_s": np.repeat([r["wall_s"] for r in rows], counts),
    }
    for key in struct_keys:
        arr = np.concatenate([r[key] for r in rows])
        prefix = "" if key in ("telem", "info", "veh") else f"{key}."
        flatten(arr, prefix, cols)
    return cols


def flatten(v: NDArray, key: str, cols: dict[str, object]) -> None:
    if v.dtype.names:
        for name in v.dtype.names:
            flatten(v[name], f"{key}{name}" if not key or key.endswith(".") else f"{key}.{name}", cols)
        return
    if v.ndim == 1:
        cols[key] = pa.array(v) if v.dtype.kind == "S" else v
        return
    if v.dtype.kind == "S" and v.ndim == 2:
        width = v.shape[1] * v.dtype.itemsize
        cols[key] = pa.array(np.ascontiguousarray(v).view(f"S{width}").ravel())
        return
    if v.dtype.itemsize == 1 and v.dtype.kind in "iub" and int(np.prod(v.shape[1:])) > 16:
        width = int(np.prod(v.shape[1:]))
        raw = pa.py_buffer(np.ascontiguousarray(v.reshape(len(v), width)).tobytes())
        cols[key] = pa.FixedSizeBinaryArray.from_buffers(pa.binary(width), len(v), [None, raw])
        return
    for i in range(v.shape[1]):
        flatten(v[:, i], f"{key}.{i}", cols)


def text(raw: bytes) -> str:
    return bytes(raw).split(bytes(1), 1)[0].decode("utf-8", "replace")


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def flatten_one(v: np.void) -> dict[str, object]:
    cols: dict[str, object] = {}
    flatten(np.array([v]), "", cols)
    return {k: (c[0].item() if isinstance(c, np.ndarray) else text(c[0].as_py())) for k, c in cols.items()}


if __name__ == "__main__":
    main()
