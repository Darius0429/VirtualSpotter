import logging
import re
from pathlib import Path

import numpy as np

from ai_race_engineer.strategy.lap_segments import LapSamples

logger = logging.getLogger(__name__)

LAP_TIME: str = "lap_time_s"


class BestLapStore:
    """The driver's best lap of each car on each track, kept across sessions: one file with the lap time and the
    telemetry of the lap, so a new session has a lap to compare with from its first corner.
    _folder  Path  folder of the files
    """

    def __init__(self, folder: Path) -> None:
        """Set the folder.
        In:  folder  Path  created on the first save
        Out:         None
        """
        self._folder: Path = folder

    def load(self, track_name: str, car: str) -> tuple[float, LapSamples] | None:
        """Stored best lap of a car on a track.
        In:  track_name  str
             car         str  class and car as the tracker names it
        Out:             tuple[float, LapSamples] | None  lap time (s) and telemetry; None when there is none or the
                                                          file cannot be read
        """
        return _read(self._path(track_name, car))

    def load_class(self, track_name: str, class_name: str) -> tuple[float, LapSamples] | None:
        """Fastest stored lap on a track in any car of a class.
        In:  track_name  str
             class_name  str
        Out:             tuple[float, LapSamples] | None  lap time (s) and telemetry; None when there is none
        """
        start = self._path(track_name, f"{class_name} ").stem
        laps = [_read(path) for path in self._folder.glob("*.npz") if path.name.startswith(start)]
        return min((lap for lap in laps if lap is not None), key=lambda lap: lap[0], default=None)

    def save(self, track_name: str, car: str, lap_time_s: float, samples: LapSamples) -> None:
        """Store a best lap; a disk error is logged and the lap stays in memory only.
        In:  track_name  str
             car         str
             lap_time_s  float       s
             samples     LapSamples
        Out:             None
        """
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self._path(track_name, car), **{LAP_TIME: lap_time_s}, **samples.__dict__)
        except OSError as error:
            logger.warning("best lap of %s at %s not saved: %s", car, track_name, error)

    def _path(self, track_name: str, car: str) -> Path:
        """File of a car on a track.
        In:  track_name  str
             car         str
        Out:             Path
        """
        return self._folder / re.sub(r'[<>:"/\\|?*]', "_", f"{track_name} - {car}.npz")


def _read(path: Path) -> tuple[float, LapSamples] | None:
    """One stored lap.
    In:  path  Path
    Out:       tuple[float, LapSamples] | None  lap time (s) and telemetry; None when the file is missing or cannot
                                                be read
    """
    try:
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        return float(arrays.pop(LAP_TIME)), LapSamples(**arrays)
    except (OSError, ValueError, TypeError, KeyError):
        return None
