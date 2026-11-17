from collections.abc import Callable
from typing import Any

from ai_race_engineer.domains.common.data.pit_estimate import PitEstimate, PitEstimateSource


class TimedPitEstimate(PitEstimateSource):
    """Wraps a pit-estimate source so every read is timed and counted in the health log.
    _read  Callable[[], PitEstimate | None]  the wrapped source's read under the timer
    """

    def __init__(self, source: PitEstimateSource, timed: Callable[..., Any]) -> None:
        """Wrap the source's read.
        In:  source  PitEstimateSource
             timed   Callable[..., Any]  (name, call) -> the same call, timed (HealthLog.timed)
        Out:         None
        """
        self._read: Callable[[], PitEstimate | None] = timed("pit_estimate_read", source.read)

    def read(self) -> PitEstimate | None:
        """Latest pit estimate of the wrapped source.
        In:  -
        Out: PitEstimate | None  None when the game gives none
        """
        return self._read()
