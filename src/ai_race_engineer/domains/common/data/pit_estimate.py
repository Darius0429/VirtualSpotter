from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PitEstimate:
    """The game's pit-stop time estimate for the player.
    fuel_s    float  s to add the fuel
    energy_s  float  s to add the virtual energy
    total_s   float  s standing still in total
    """

    fuel_s: float
    energy_s: float
    total_s: float


def pit_estimate(data: Mapping[str, float] | None) -> PitEstimate | None:
    """Pit estimate from the game's JSON object.
    In:  data  Mapping[str, float] | None  s per work item; keys used: fuel, ve, total (None = the body was null)
    Out:       PitEstimate | None          None when data is None
    """
    if data is None:
        return None
    return PitEstimate(fuel_s=data["fuel"], energy_s=data["ve"], total_s=data["total"])


class PitEstimateSource(ABC):
    """Common interface for all sources of the game's pit-stop estimate."""

    @abstractmethod
    def read(self) -> PitEstimate | None:
        """Latest pit estimate.
        In:  -
        Out: PitEstimate | None  None when the game gives none
        """
