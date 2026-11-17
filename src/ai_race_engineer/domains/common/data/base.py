from abc import ABC, abstractmethod

from ai_race_engineer.domains.common.data.snapshot import Snapshot


class TelemetrySource(ABC):
    """Common interface for all game telemetry sources; usable as a context manager (open on enter, close on exit)."""

    @abstractmethod
    def open(self) -> None:
        """Map the game's shared memory.
        In:  -
        Out: None
        """

    @abstractmethod
    def read(self) -> Snapshot | None:
        """Latest snapshot of the game state.
        In:  -
        Out: Snapshot | None  None when there is no live session (game not running, in menus, plugin missing)
        """

    @abstractmethod
    def close(self) -> None:
        """Release the shared memory.
        In:  -
        Out: None
        """

    def __enter__(self) -> "TelemetrySource":
        """Open the source.
        In:  -
        Out: TelemetrySource  this source
        """
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the source.
        In:  exc  object  exception info (ignored)
        Out:      None
        """
        self.close()
