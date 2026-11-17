import dataclasses
import json
import logging
from pathlib import Path

from ai_race_engineer.domains.common.analysis.corners.track_map import Segment

logger = logging.getLogger(__name__)


class TrackMapStore:
    """The track maps on disk: one JSON file per track with its corners and straights; names can be edited by hand.
    _folder  Path  folder of the files
    """

    def __init__(self, folder: Path) -> None:
        """Set the folder.
        In:  folder  Path  created on the first save
        Out:         None
        """
        self._folder: Path = folder

    def load(self, track_name: str) -> tuple[Segment, ...]:
        """Stored map of a track.
        In:  track_name  str
        Out:             tuple[Segment, ...]  empty when the file is missing or unreadable
        """
        try:
            return tuple(Segment(**entry) for entry in json.loads(self._path(track_name).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return ()

    def save(self, track_name: str, segments: tuple[Segment, ...]) -> None:
        """Store the map of a track; a disk error is logged and the map stays in memory only.
        In:  track_name  str
             segments    tuple[Segment, ...]
        Out:             None
        """
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            self._path(track_name).write_text(
                json.dumps([dataclasses.asdict(segment) for segment in segments], indent=1), encoding="utf-8"
            )
        except OSError as error:
            logger.warning("track map of %s not saved: %s", track_name, error)

    def _path(self, track_name: str) -> Path:
        """File of a track.
        In:  track_name  str
        Out:             Path
        """
        return self._folder / f"{track_name}.json"
