import dataclasses
import json
from pathlib import Path

from ai_race_engineer.domains.common.analysis.session.history import SessionFigures, figures_from_dict


class HistoryStore:
    """The history file: one JSON line of figures per session, append-only, so the history can always be recomputed.
    _path  Path  the .jsonl file
    """

    def __init__(self, path: Path) -> None:
        """Set the file.
        In:  path  Path  .jsonl file (created with its folder on the first append)
        Out:       None
        """
        self._path: Path = path

    def load(self) -> tuple[SessionFigures, ...]:
        """All stored sessions; a line that cannot be read is skipped.
        In:  -
        Out: tuple[SessionFigures, ...]  in file order; empty when the file is missing or unreadable
        """
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        entries: list[SessionFigures] = []
        for line in lines:
            try:
                entries.append(figures_from_dict(json.loads(line)))
            except (ValueError, TypeError, KeyError):
                continue
        return tuple(entries)

    def append(self, figures: SessionFigures) -> None:
        """Add one session; a disk error raises OSError.
        In:  figures  SessionFigures
        Out:          None
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(dataclasses.asdict(figures)) + "\n")

    def session_ids(self) -> set[str]:
        """Ids of the stored sessions.
        In:  -
        Out: set[str]
        """
        return {entry.session_id for entry in self.load()}
