from collections.abc import Callable, Sequence

from ai_race_engineer.domains.common.data.base import TelemetrySource
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot
from ai_race_engineer.domains.common.data.race_source import RaceSource


class ObservedSource(TelemetrySource):
    """Wraps a race source and hands every snapshot to observers before returning it to the reader.
    _source     RaceSource
    _observers  Sequence[Callable[[RaceSnapshot], None]]  called in order, in the reading thread
    """

    def __init__(self, source: RaceSource, observers: Sequence[Callable[[RaceSnapshot], None]]) -> None:
        """Store the wrapped source and its observers.
        In:  source     RaceSource
             observers  Sequence[Callable[[RaceSnapshot], None]]  called in order with every snapshot
        Out:            None
        """
        self._source: RaceSource = source
        self._observers: Sequence[Callable[[RaceSnapshot], None]] = observers

    def open(self) -> None:
        """Open the wrapped source.
        In:  -
        Out: None
        """
        self._source.open()

    def read(self) -> RaceSnapshot | None:
        """Latest race snapshot, shown to every observer first.
        In:  -
        Out: RaceSnapshot | None  None when the wrapped source has no live session (observers are not called)
        """
        snapshot = self._source.read()
        if snapshot is not None:
            for observer in self._observers:
                observer(snapshot)
        return snapshot

    def close(self) -> None:
        """Close the wrapped source.
        In:  -
        Out: None
        """
        self._source.close()
