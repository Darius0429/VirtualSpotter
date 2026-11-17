import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import yaml

from ai_race_engineer.domains.common.analysis.session.session_summary import SessionSummary


class SessionStore:
    """The session files: one YAML summary per session, named <session id>_<track>.yaml.
    _root  Path  folder of the files
    """

    def __init__(self, root: Path) -> None:
        """Set the folder.
        In:  root  Path  created on the first write
        Out:       None
        """
        self._root: Path = root

    def write(self, summary: SessionSummary) -> Path:
        """Store a summary, replacing the file of the same session (written to a temporary file, then renamed, so a
        crash never leaves half a file); a disk error raises OSError.
        In:  summary  SessionSummary
        Out:          Path  the file written
        """
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{summary.session_id}_{_file_part(summary.track)}.yaml"
        temporary = path.with_suffix(".tmp")
        data = json.loads(json.dumps(dataclasses.asdict(summary)))
        temporary.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        temporary.replace(path)
        return path

    def latest(self, track: str, kind: str, before_unix: float) -> dict[str, Any] | None:
        """The newest stored summary of a track and session kind that started before a time.
        In:  track        str
             kind         str    "practice" | "qualify" | "race" | ...
             before_unix  float  s unix time; a session that started at or after it is not wanted (the running one,
                                 or a later one in a replay)
        Out:              dict[str, Any] | None  the stored summary; None without such a file or when a file cannot
                                                 be read
        """
        try:
            for path in sorted(self._root.glob(f"*_{kind}_{_file_part(track)}.yaml"), reverse=True):
                summary = yaml.safe_load(path.read_text(encoding="utf-8"))
                if summary["started_unix"] < before_unix:
                    return summary
        except (OSError, yaml.YAMLError, KeyError, TypeError):
            return None
        return None


def _file_part(text: str) -> str:
    """Text made safe for a file name.
    In:  text  str
    Out:       str  letters, digits and underscores
    """
    return re.sub(r"\W+", "_", text).strip("_")
