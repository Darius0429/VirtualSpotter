from abc import ABC, abstractmethod

from ai_race_engineer.telemetry.frame import Frame


class FrameSource(ABC):
    """Common interface for sources of recordable frames; usable as a context manager (open on enter, close on exit)."""

    @abstractmethod
    def open(self) -> None:
        """Map the game's shared memory.
        In:  -
        Out: None
        """

    @abstractmethod
    def read_new(self) -> Frame | None:
        """Frame of the latest telemetry tick, only if the player's telemetry clock advanced since the last call.
        In:  -
        Out: Frame | None  None when the clock did not advance or there is no live session
        """

    @abstractmethod
    def close(self) -> None:
        """Release the shared memory.
        In:  -
        Out: None
        """

    def __enter__(self) -> "FrameSource":
        """Open the source.
        In:  -
        Out: FrameSource  this source
        """
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the source.
        In:  exc  object  exception info (ignored)
        Out:      None
        """
        self.close()
