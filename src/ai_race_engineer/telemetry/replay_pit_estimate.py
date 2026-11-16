import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.telemetry.pit_estimate import PitEstimate, PitEstimateSource, pit_estimate


class ReplayPitEstimate(PitEstimateSource):
    """Pit estimate recorded in a raw dump (rest/pitstop_estimate.jsonl), looked up at the replay time.
    _time_s  Callable[[], float]               current replay time, s on the poll_s axis
    _poll_s  NDArray[np.float64]               s, poll time of each recorded line
    _data    list[Mapping[str, float] | None]  the game's JSON object of each line
    """

    def __init__(self, folder: Path, time_s: Callable[[], float]) -> None:
        """Load the recorded estimates; a dump without the file gives no estimates.
        In:  folder  Path                 raw-dump folder
             time_s  Callable[[], float]  current replay time, s on the poll_s axis
        Out:         None
        """
        self._time_s: Callable[[], float] = time_s
        try:
            lines = [json.loads(line) for line in (folder / "rest" / "pitstop_estimate.jsonl").read_text().splitlines()]
        except OSError:
            lines = []
        self._poll_s: NDArray[np.float64] = np.array([line["poll_s"] for line in lines], dtype=np.float64)
        self._data: list[Mapping[str, float] | None] = [line["data"] for line in lines]

    def read(self) -> PitEstimate | None:
        """Estimate of the latest line recorded at or before the replay time.
        In:  -
        Out: PitEstimate | None  None before the first line, for a null line or without the file
        """
        index = int(np.searchsorted(self._poll_s, self._time_s(), side="right")) - 1
        return pit_estimate(self._data[index]) if index >= 0 else None
