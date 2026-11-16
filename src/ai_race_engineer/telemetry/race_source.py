from abc import abstractmethod
from pathlib import Path

from ai_race_engineer.telemetry.base import TelemetrySource
from ai_race_engineer.telemetry.race_snapshot import RaceSnapshot


class RaceSource(TelemetrySource):
    """Common interface for telemetry sources that also give every car's resources and the flags."""

    @abstractmethod
    def read(self) -> RaceSnapshot | None:
        """Latest race snapshot of the game state.
        In:  -
        Out: RaceSnapshot | None  None when there is no live session
        """

    def results_path(self) -> Path | None:
        """Results file the game points to (the previous session of the event).
        In:  -
        Out: Path | None  None when the source has none
        """
        return None
