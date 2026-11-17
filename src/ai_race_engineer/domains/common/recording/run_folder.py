import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import psutil

LOCK_SUFFIX: str = ".lock"
RUN_NAME: re.Pattern[str] = re.compile(r"\d{8}_\d{6}(_\d+)?")
SAME_START_S: float = 1.0


class RunFolder:
    """The folder of one run, shared by the programs that run at the same time (recorder, engineer): the first one
    creates it, one of another kind started while the first is alive joins it. While it is entered, each program
    marks the folder with a lock file <kind>.lock that holds its process id and start time.
    path   Path  the run folder, <root>/<YYYYmmdd_HHMMSS> of the program that created it (_2, _3, ... added when
                 that name is taken)
    _lock  Path  this program's lock file
    """

    def __init__(self, root: Path, kind: str, other: str | None) -> None:
        """Choose the run folder: one to join, or the name of a new one. Nothing is written yet.
        In:  root   Path        folder that holds one folder per run
             kind   str         what this program is, e.g. "recorder"
             other  str | None  the kind of program whose run folder is joined; None = never join
        Out:        None
        """
        folders = [f for f in root.iterdir() if f.is_dir() and RUN_NAME.fullmatch(f.name)] if root.is_dir() else []
        joined = run_to_join({folder.name: _locks(folder) for folder in folders}, kind, other, psutil.pid_exists)
        self.path: Path = root / (joined or _new_name({folder.name for folder in folders}, time.localtime()))
        self._lock: Path = self.path / f"{kind}{LOCK_SUFFIX}"

    def close(self) -> None:
        """Remove the lock file.
        In:  -
        Out: None
        """
        self._lock.unlink(missing_ok=True)

    def __enter__(self) -> "RunFolder":
        """Make the folder and write the lock file; a folder or lock file that cannot be made raises OSError.
        In:  -
        Out: RunFolder  this run folder
        """
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock.write_text(f"{os.getpid()} {psutil.Process().create_time()}")
        return self

    def __exit__(self, *exc: object) -> None:
        """Remove the lock file.
        In:  exc  object  exception info (ignored)
        Out:      None
        """
        self.close()


def run_to_join(
    runs: Mapping[str, Mapping[str, int]], kind: str, other: str | None, alive: Callable[[int], bool]
) -> str | None:
    """The run folder a starting program joins: the newest one in which a program of the other kind is alive and
    none of its own kind.
    In:  runs   Mapping[str, Mapping[str, int]]  run folder name (YYYYmmdd_HHMMSS) -> lock kind -> process id
         kind   str                              what the starting program is
         other  str | None                       the kind of program whose run folder is joined; None = never join
         alive  Callable[[int], bool]            whether a process id is a running process
    Out:        str | None                       folder name; None = create a new run folder
    """
    joinable = [
        name
        for name, locks in runs.items()
        if other in locks and alive(locks[other]) and not (kind in locks and alive(locks[kind]))
    ]
    return max(joinable, default=None)


def _new_name(taken: set[str], now: time.struct_time) -> str:
    """Name of a new run folder.
    In:  taken  set[str]          names of the run folders there are
         now    time.struct_time  local time
    Out:        str               YYYYmmdd_HHMMSS, with _2, _3, ... when that name is taken
    """
    name = time.strftime("%Y%m%d_%H%M%S", now)
    names = (name, *(f"{name}_{number}" for number in range(2, len(taken) + 3)))
    return next(candidate for candidate in names if candidate not in taken)


def _locks(folder: Path) -> dict[str, int]:
    """The lock files of a run folder. Left out: one that cannot be read or is not "<process id> <start time>", and
    one whose process id now belongs to another process than the one that wrote it.
    In:  folder  Path            run folder
    Out:         dict[str, int]  lock kind -> process id
    """
    locks: dict[str, int] = {}
    for file in folder.glob(f"*{LOCK_SUFFIX}"):
        try:
            pid, started_s = file.read_text().split()
            reused = abs(psutil.Process(int(pid)).create_time() - float(started_s)) > SAME_START_S
        except psutil.NoSuchProcess:
            reused = False
        except (OSError, ValueError, psutil.Error):
            continue
        if not reused:
            locks[file.stem] = int(pid)
    return locks
