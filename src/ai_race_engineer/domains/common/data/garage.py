import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ENERGY_PATTERN: re.Pattern[str] = re.compile(r"([\d.]+)%\s*\(([\d.]+) laps\)")
RATIO_PATTERN: re.Pattern[str] = re.compile(r"[\d.]+")


@dataclass(frozen=True)
class GarageInfo:
    """What the game's garage screen says about the player's fuel and energy.
    fuel_ratio      float | None  litres per 1 % energy (VM_FUEL_LEVEL "0.86"); None when the game shows no number
    energy_per_lap  float | None  fraction/lap (VM_VIRTUAL_ENERGY "100% (36.2 laps)" -> 1.00 / 36.2); None when the
                                  game shows no such text
    """

    fuel_ratio: float | None
    energy_per_lap: float | None


def garage_info(data: Mapping[str, Any] | None) -> GarageInfo | None:
    """Garage figures from the game's garage JSON object.
    In:  data  Mapping[str, Any] | None  setting id -> setting; used: VM_FUEL_LEVEL and VM_VIRTUAL_ENERGY, each with
                                         "stringValue" (None = the body was null)
    Out:       GarageInfo | None         None when data is None
    """
    if data is None:
        return None
    ratio = RATIO_PATTERN.fullmatch(_text(data, "VM_FUEL_LEVEL"))
    energy = ENERGY_PATTERN.search(_text(data, "VM_VIRTUAL_ENERGY"))
    return GarageInfo(
        fuel_ratio=float(ratio.group()) if ratio else None,
        energy_per_lap=float(energy.group(1)) / 100 / float(energy.group(2))
        if energy and float(energy.group(2)) > 0
        else None,
    )


def merge_garage(last: GarageInfo | None, new: GarageInfo | None) -> GarageInfo | None:
    """Garage figures after a new read: each figure the read does not give keeps its last good value.
    In:  last  GarageInfo | None  the figures kept so far
         new   GarageInfo | None  the read (None = the game gave nothing)
    Out:       GarageInfo | None  None while nothing was ever read
    """
    if last is None or new is None:
        return new or last
    return GarageInfo(
        fuel_ratio=last.fuel_ratio if new.fuel_ratio is None else new.fuel_ratio,
        energy_per_lap=last.energy_per_lap if new.energy_per_lap is None else new.energy_per_lap,
    )


class GarageSource(ABC):
    """Common interface for all sources of the game's garage figures."""

    @abstractmethod
    def read(self) -> GarageInfo | None:
        """Latest garage figures.
        In:  -
        Out: GarageInfo | None  None when the game gives none
        """


def _text(data: Mapping[str, Any], key: str) -> str:
    """Displayed text of one garage setting.
    In:  data  Mapping[str, Any]  setting id -> setting
         key   str                setting id
    Out:       str                "" when the setting or its text is missing
    """
    return str(data.get(key, {}).get("stringValue", ""))
