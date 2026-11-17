from collections.abc import Callable
from pathlib import Path

import pandas as pd


class ReplayWallClock:
    """Wall-clock time of a raw dump at the replay time, so ages (a qualifying results file) are measured against
    the day of the recording and not the day of the replay.
    _time_s    Callable[[], float]  current replay time, s on the poll_s axis
    _offset_s  float                s, unix time of poll_s = 0
    """

    def __init__(self, folder: Path, time_s: Callable[[], float]) -> None:
        """Read the dump's first poll and wall time; a missing folder or table raises OSError.
        In:  folder  Path                 raw-dump folder
             time_s  Callable[[], float]  current replay time, s on the poll_s axis
        Out:         None
        """
        first = pd.read_parquet(folder / "raw_scoring_info_5hz", columns=["poll_s", "wall_s"]).sort_values("poll_s")
        self._time_s: Callable[[], float] = time_s
        self._offset_s: float = float(first["wall_s"].iloc[0] - first["poll_s"].iloc[0])

    def unix_s(self) -> float:
        """Wall-clock time of the recording at the replay time.
        In:  -
        Out: float  s unix time
        """
        return self._offset_s + self._time_s()
