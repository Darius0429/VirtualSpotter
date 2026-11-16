import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.telemetry.garage import GarageInfo, GarageSource, garage_info


class ReplayGarage(GarageSource):
    """Garage figures recorded in a raw dump (rest/garage_setup.jsonl), looked up at the replay time.
    _time_s  Callable[[], float]      current replay time, s on the poll_s axis
    _poll_s  NDArray[np.float64]      s, poll time of each recorded line
    _infos   list[GarageInfo | None]  the figures of each line
    """

    def __init__(self, folder: Path, time_s: Callable[[], float]) -> None:
        """Load the recorded garage data; a dump without the file gives no figures.
        In:  folder  Path                 raw-dump folder
             time_s  Callable[[], float]  current replay time, s on the poll_s axis
        Out:         None
        """
        self._time_s: Callable[[], float] = time_s
        try:
            lines = [json.loads(line) for line in (folder / "rest" / "garage_setup.jsonl").read_text().splitlines()]
        except (OSError, ValueError):
            lines = []
        self._poll_s: NDArray[np.float64] = np.array([line["poll_s"] for line in lines], dtype=np.float64)
        self._infos: list[GarageInfo | None] = [garage_info(line["data"]) for line in lines]

    def read(self) -> GarageInfo | None:
        """Figures of the latest line recorded at or before the replay time.
        In:  -
        Out: GarageInfo | None  None before the first line, for a null line or without the file
        """
        index = int(np.searchsorted(self._poll_s, self._time_s(), side="right")) - 1
        return self._infos[index] if index >= 0 else None
