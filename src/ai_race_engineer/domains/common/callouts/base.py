from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

from ai_race_engineer.domains.common.data.snapshot import Snapshot


class Priority(IntEnum):
    """Urgency of a callout; higher is more urgent."""

    INFO = 0
    STRATEGY = 1
    SPOTTER = 2


@dataclass(frozen=True)
class Callout:
    """One message for the driver.
    kind      str                          e.g. "spotter", "fastest_lap", "personal_best", "sector_comparison",
                                           "low_fuel", "low_energy"
    text      str                          final short speech text
    priority  Priority
    facts     dict[str, str | float | int]  structured values behind the text (units in the key suffix)
    """

    kind: str
    text: str
    priority: Priority
    facts: dict[str, str | float | int]


class Event(ABC):
    """Common interface for all race-situation detectors.
    name  ClassVar[str]  key in configs/events.yaml
    """

    name: ClassVar[str]

    @abstractmethod
    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Inspect one snapshot and return at most one callout.
        In:  snapshot  Snapshot
             now_s     float     s monotonic clock
        Out:           Callout | None
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget all per-session state.
        In:  -
        Out: None
        """
