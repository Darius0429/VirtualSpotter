import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyLMUSharedMemory.lmu_data import LMUConstants, LMUObjectOut
from pyLMUSharedMemory.lmu_mmap import MMapControl

from ai_race_engineer.domains.common.recording.part_writer import PartWriter, Rows
from ai_race_engineer.domains.lmu.raw_tables import flatten_one, safe, text
from ai_race_engineer.domains.lmu.rest_capture import RestCapture

OBJECT: np.dtype = np.dtype(LMUObjectOut)
HEADER: np.dtype = np.dtype([("activeVehicles", "u1"), ("playerVehicleIdx", "i2"), ("playerHasVehicle", "?")])
SESSION_NAMES: tuple[str, ...] = (
    "testday",
    *(f"practice{number}" for number in range(1, 5)),
    *(f"qualify{number}" for number in range(1, 5)),
    "warmup",
    *(f"race{number}" for number in range(1, 5)),
)
MAX_SLOT_IDS: int = 65536
MAX_STREAM_BYTES: int = 65536


class RawRecorder:
    """Records every field the game publishes in shared memory, losslessly, until Ctrl+C: one folder per session
    under the root, named <track>_<session>_<HHMMSS>, with
      raw_player_telemetry_100hz/  the player's telemetry struct, the generic block and a header, one row per tick
                                   of the player's telemetry clock
      raw_cars_telemetry_50hz/     the telemetry struct of every other car, one row per tick of that car's clock
      raw_scoring_info_5hz/        the scoring info struct, one row per scoring tick
      raw_cars_scoring_5hz/        the scoring struct of every car, one row per car per scoring tick
      raw_scoring_stream/          the scoring stream text, one row when it changes
      paths.json                   the game's paths, once per session
      rest/<name>.jsonl            the answers of the game's REST API
    Every row has poll_s (s since the recorder started) and wall_s (s unix time).
    _root          Path             folder that holds the session folders
    _writer        PartWriter       writes the buffered rows in its own process
    _rest          RestCapture      polls the REST API in its own thread
    _poll_hz       float            shared-memory polls per second
    _flush_s       float            s between two hand-overs of the buffered rows to the writer
    _stale_s       float            s a car's telemetry clock may differ from the session clock
    _progress_s    float            s between two progress lines on the console
    _stop_after_s  float            s after which the recording stops by itself; inf = never
    _gate          SessionGate      decides when a session folder is opened and closed
    _folder        Path | None      the open session folder
    _buffers       dict[str, Rows]  table name -> rows not yet handed to the writer
    _last_tel      NDArray[np.float64]  s, telemetry clock of the last row of every slot id; NaN = none yet
    _last_stream   bytes            the scoring stream last recorded
    _t0            float            s monotonic clock at the start, the zero of poll_s
    _last_flush    float            s monotonic clock of the last hand-over
    _last_print    float            s monotonic clock of the last progress line
    """

    def __init__(
        self,
        root: Path,
        writer: PartWriter,
        rest: RestCapture,
        poll_hz: float,
        flush_s: float,
        stale_s: float,
        finish_grace_s: float,
        restart_back_s: float,
        progress_s: float,
        stop_after_s: float | None,
    ) -> None:
        """Store the blocks and the settings.
        In:  root            Path          folder that holds the session folders; created with the first session
             writer          PartWriter    not yet started
             rest            RestCapture   not yet started
             poll_hz         float         shared-memory polls per second
             flush_s         float         s between two hand-overs of the buffered rows to the writer
             stale_s         float         s a car's telemetry clock may differ from the session clock; rows further
                                           off (left over from the session before) are dropped
             finish_grace_s  float         s the recording goes on after the player's finish status left 0
             restart_back_s  float         s the session clock must jump back to count as a restart
             progress_s      float         s between two progress lines on the console
             stop_after_s    float | None  s after which the recording stops by itself; None = until Ctrl+C
        Out:                 None
        """
        self._root: Path = root
        self._writer: PartWriter = writer
        self._rest: RestCapture = rest
        self._poll_hz: float = poll_hz
        self._flush_s: float = flush_s
        self._stale_s: float = stale_s
        self._stop_after_s: float = float("inf") if stop_after_s is None else stop_after_s
        self._progress_s: float = progress_s
        self._gate: SessionGate = SessionGate(finish_grace_s, restart_back_s)
        self._folder: Path | None = None
        self._buffers: dict[str, Rows] = {}
        self._last_tel: NDArray[np.float64] = np.full(MAX_SLOT_IDS, np.nan)
        self._last_stream: bytes = b""
        self._t0: float = 0.0
        self._last_flush: float = 0.0
        self._last_print: float = 0.0

    def run(self) -> None:
        """Poll the shared memory and record until Ctrl+C or the stop time (blocking); what is buffered is written
        and the writer process ended before it returns.
        In:  -
        Out: None
        """
        mmap = MMapControl(LMUConstants.LMU_SHARED_MEMORY_FILE, LMUObjectOut)
        mmap.create(access_mode=0)
        self._writer.start()
        try:
            self._t0 = self._last_flush = self._last_print = time.perf_counter()
            self._rest.start(self._t0)
            print("waiting for a session (Ctrl+C to stop)")
            while True:
                mmap.update()
                now = time.perf_counter()
                if now - self._t0 > self._stop_after_s:
                    break
                self._poll(np.frombuffer(memoryview(mmap.data).cast("B"), dtype=OBJECT, count=1)[0], now)
                time.sleep(max(0.0, 1 / self._poll_hz - (time.perf_counter() - now)))
        except KeyboardInterrupt:
            print("stopping, writing the last data (do not press Ctrl+C again) ...")
        finally:
            self._rest.folder = None
            self._writer.put(self._buffers)
            self._writer.close()
            mmap.close()
            print(f"done, last session: {self._folder}")

    def _poll(self, obj: np.void, now: float) -> None:
        """Handle one copy of the shared memory: open or close the session folder as the gate decides and buffer
        the rows that are new.
        In:  obj  np.void  the whole shared-memory struct
             now  float    s monotonic clock of the poll
        Out:      None
        """
        stamp = {"poll_s": now - self._t0, "wall_s": time.time()}
        info = obj["scoring"]["scoringInfo"]
        if info["mNumVehicles"] == 0:
            return
        key = (int(info["mSession"]), text(info["mTrackName"]))
        session_et = float(info["mCurrentET"])
        decision = self._gate.update(key, session_et, now)
        self._rest.session_et = session_et
        if decision in (Decision.CLOSE, Decision.FINISH):
            self._close(decision, now)
        if decision is Decision.OPEN:
            self._open(obj["paths"], key, now)
        if decision not in (Decision.OPEN, Decision.RECORD):
            return
        self._telemetry(obj, session_et, stamp)
        if self._gate.tick(session_et):
            self._scoring(obj["scoring"], stamp, now)
        self._flush(key, int(obj["telemetry"]["activeVehicles"]), session_et, now)

    def _open(self, paths: np.void, key: tuple[int, str], now: float) -> None:
        """Open the folder of a new session, write its paths.json and point the writer and the REST capture to it;
        a folder or file that cannot be made raises OSError.
        In:  paths  np.void          the game's paths struct
             key    tuple[int, str]  (the game's session number, track name)
             now    float            s monotonic clock of the poll
        Out:        None
        """
        folder = self._root / safe(f"{key[1]}_{session_name(key[0])}_{datetime.now():%H%M%S}")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "paths.json").write_text(json.dumps(flatten_one(paths), indent=1))
        self._writer.open(folder)
        self._rest.folder = self._folder = folder
        self._last_tel[:] = np.nan
        self._last_stream = b""
        print(f"{now - self._t0:7.0f}s  new session -> {folder}")

    def _close(self, decision: "Decision", now: float) -> None:
        """Hand the buffered rows to the writer and close the session folder.
        In:  decision  Decision  CLOSE or FINISH
             now       float     s monotonic clock of the poll
        Out:           None
        """
        self._writer.put(self._buffers)
        self._writer.end()
        print(f"{now - self._t0:7.0f}s  {'finished, ' if decision is Decision.FINISH else ''}closed {self._folder}")
        self._buffers = {}
        self._rest.folder = self._folder = None

    def _telemetry(self, obj: np.void, session_et: float, stamp: dict[str, float]) -> None:
        """Buffer the telemetry of every car whose telemetry clock moved since its last row and is within stale_s
        of the session clock: the player's with the generic block and a header, the others' in one row.
        In:  obj         np.void           the whole shared-memory struct
             session_et  float             s session time
             stamp       dict[str, float]  poll_s, wall_s of the poll
        Out:             None
        """
        tel = obj["telemetry"]
        count = int(tel["activeVehicles"])
        cars = tel["telemInfo"][:count]
        ids, clocks = cars["mID"].astype(np.int64), cars["mElapsedTime"]
        fresh = (self._last_tel[ids] != clocks) & (np.abs(clocks - session_et) <= self._stale_s)
        changed = np.flatnonzero(fresh).tolist()
        self._last_tel[ids] = clocks
        player = int(tel["playerVehicleIdx"]) if tel["playerHasVehicle"] else -1
        if player in changed:
            header = np.array([(count, player, bool(tel["playerHasVehicle"]))], dtype=HEADER)
            telem, generic = cars[player : player + 1].copy(), obj["generic"].reshape(1).copy()
            row = {**stamp, "header": header, "telem": telem, "generic": generic}
            self._buffers.setdefault("raw_player_telemetry_100hz", []).append(row)
        others = [index for index in changed if index != player]
        if others:
            self._buffers.setdefault("raw_cars_telemetry_50hz", []).append({**stamp, "telem": cars[others].copy()})

    def _scoring(self, sco: np.void, stamp: dict[str, float], now: float) -> None:
        """Buffer one scoring tick: the scoring info, the scoring of every car and the scoring stream when it
        changed; tell the REST capture whether the player is in the pits and the gate when the player has finished.
        In:  sco    np.void           the scoring part of the shared memory
             stamp  dict[str, float]  poll_s, wall_s of the poll
             now    float             s monotonic clock of the poll
        Out:        None
        """
        info = sco["scoringInfo"]
        cars = sco["vehScoringInfo"][: int(info["mNumVehicles"])]
        self._buffers.setdefault("raw_scoring_info_5hz", []).append({**stamp, "info": info.reshape(1).copy()})
        self._buffers.setdefault("raw_cars_scoring_5hz", []).append({**stamp, "veh": cars.copy()})
        me = cars[cars["mIsPlayer"] != 0]
        if len(me):
            self._rest.in_pits = bool(me["mInPits"][0])
            if int(me["mFinishStatus"][0]) != 0:
                self._gate.finish(now)
        size = int.from_bytes(bytes(sco["scoringStreamSize"][:4]), "little")
        stream = bytes(sco["scoringStream"])[: max(0, min(size, MAX_STREAM_BYTES))]
        if stream != self._last_stream:
            self._last_stream = stream
            row = {**stamp, "text": stream.decode("utf-8", "replace")}
            self._buffers.setdefault("raw_scoring_stream", []).append(row)

    def _flush(self, key: tuple[int, str], cars: int, session_et: float, now: float) -> None:
        """Hand the buffered rows to the writer every flush_s and print a progress line every 30 s.
        In:  key         tuple[int, str]  (the game's session number, track name)
             cars        int              cars with telemetry
             session_et  float            s session time
             now         float            s monotonic clock of the poll
        Out:             None
        """
        if now - self._last_flush < self._flush_s:
            return
        self._writer.put(self._buffers)
        self._buffers, self._last_flush = {}, now
        if now - self._last_print >= self._progress_s:
            self._last_print = now
            print(f"{now - self._t0:7.0f}s  {session_name(key[0])}  cars {cars:2d}  session time {session_et}")


class Decision(Enum):
    """What the recorder does with one poll.
    SKIP    nothing is recorded
    OPEN    a new session folder is opened and the poll recorded into it
    RECORD  the poll is recorded into the open folder
    CLOSE   the folder is closed: another session or track, or a restart
    FINISH  the folder is closed: the player has finished
    """

    SKIP = 0
    OPEN = 1
    RECORD = 2
    CLOSE = 3
    FINISH = 4


class SessionGate:
    """When a session folder is open. It is closed when the game's session number or the track changes or the
    session clock jumps back (restart). The next one is only opened once the new session's clock runs, so frozen
    leftovers (results screen, loading) are never recorded. It is also closed finish_grace_s after the player's
    finish status left 0 (finished, DNF, DQ); nothing more is recorded until the next session or a restart.
    Formation lap and garage time before a race belong to the race session and are recorded.
    _finish_grace_s  float                   s recorded after the player has finished
    _restart_back_s  float                   s the session clock must jump back to count as a restart
    _key             tuple[int, str] | None  (session number, track name) of the session seen last
    _start_et        float                   s session time at which that session was first seen
    _last_et         float | None            s session time of the last poll while closed, of the last scoring tick
                                             while open; None = no tick recorded yet
    _open            bool                    a folder is open
    _finished_at     float | None            s monotonic clock at which the player was seen finished
    _done            bool                    closed after the finish; stays closed until another session or a restart
    """

    def __init__(self, finish_grace_s: float, restart_back_s: float) -> None:
        """Start closed, before any session.
        In:  finish_grace_s  float  s recorded after the player's finish status left 0
             restart_back_s  float  s the session clock must jump back to count as a restart
        Out:                 None
        """
        self._finish_grace_s: float = finish_grace_s
        self._restart_back_s: float = restart_back_s
        self._key: tuple[int, str] | None = None
        self._start_et: float = 0.0
        self._last_et: float | None = None
        self._open: bool = False
        self._finished_at: float | None = None
        self._done: bool = False

    def update(self, key: tuple[int, str], session_et: float, now: float) -> Decision:
        """Decide what is done with a poll.
        In:  key         tuple[int, str]  (the game's session number, track name)
             session_et  float            s session time
             now         float            s monotonic clock of the poll
        Out:             Decision
        """
        restarted = self._last_et is not None and session_et < self._last_et - self._restart_back_s
        if key != self._key or restarted:
            was_open = self._open
            self._key, self._start_et, self._last_et = key, session_et, session_et
            self._open, self._finished_at, self._done = False, None, False
            return Decision.CLOSE if was_open else Decision.SKIP
        if self._finished_at is not None and now - self._finished_at >= self._finish_grace_s:
            self._open, self._finished_at, self._done = False, None, True
            self._last_et = session_et
            return Decision.FINISH
        if self._open:
            return Decision.RECORD
        if self._done or session_et == self._start_et:
            self._last_et = session_et
            return Decision.SKIP
        self._open, self._last_et = True, None
        return Decision.OPEN

    def tick(self, session_et: float) -> bool:
        """Whether the scoring was updated since the last tick; call once per recorded poll.
        In:  session_et  float  s session time
        Out:             bool   True = the scoring of this poll is new
        """
        new = session_et != self._last_et
        self._last_et = session_et
        return new

    def finish(self, now: float) -> None:
        """Note that the player's finish status is no longer 0; the first time counts.
        In:  now  float  s monotonic clock of the poll
        Out:      None
        """
        if self._finished_at is None:
            self._finished_at = now


def session_name(session: int) -> str:
    """Name of a session of the game.
    In:  session  int  the game's session number (mSession)
    Out:          str  testday, practice1-4, qualify1-4, warmup, race1-4; session<number> for any other
    """
    return SESSION_NAMES[session] if 0 <= session < len(SESSION_NAMES) else f"session{session}"
