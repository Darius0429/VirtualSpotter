from abc import ABC, abstractmethod

from ai_race_engineer.domains.common.data.snapshot import Snapshot


class StraightDetector(ABC):
    """Common interface for all straight detectors."""

    @abstractmethod
    def update(self, snapshot: Snapshot) -> None:
        """Feed one snapshot.
        In:  snapshot  Snapshot
        Out:           None
        """

    @abstractmethod
    def is_straight(self) -> bool:
        """Whether the player is on a straight where a message can be spoken.
        In:  -
        Out: bool
        """


def on_straight(
    steering: float,
    accel_lateral_mps2: float,
    speed_mps: float,
    max_steering: float,
    max_lateral_mps2: float,
    min_speed_mps: float,
) -> bool:
    """Whether the car is driving straight at this instant.
    In:  steering            float  -1 (left) to 1 (right), fraction of lock
         accel_lateral_mps2  float  m/s^2
         speed_mps           float  m/s
         max_steering        float  |steering| limit, fraction of lock
         max_lateral_mps2    float  |lateral acceleration| limit, m/s^2
         min_speed_mps       float  m/s
    Out:                     bool
    """
    return abs(steering) <= max_steering and abs(accel_lateral_mps2) <= max_lateral_mps2 and speed_mps >= min_speed_mps
