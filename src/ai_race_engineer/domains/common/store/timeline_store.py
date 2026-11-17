import dataclasses
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

from ai_race_engineer.domains.common.analysis.session.timeline import (
    TIMELINE_SUFFIX,
    TimelineEntry,
    file_name,
    files_in_window,
    matching_file,
)


class TimelineStore:
    """The timeline files: one JSON line per entry in <root>/<YYYYmmdd_HHMMSS>_<kind>.timeline.jsonl, append-only.
    _root             Path   folder of the session files
    _resume_margin_s  float  s two session starts may differ and still be the same session
    """

    def __init__(self, root: Path, resume_margin_s: float) -> None:
        """Set the folder and the margin.
        In:  root             Path   folder of the session files (created on the first open)
             resume_margin_s  float  s two session starts may differ and still be the same session
        Out:                  None
        """
        self._root: Path = root
        self._resume_margin_s: float = resume_margin_s

    def open(self, kind: str, started_unix: float, time_s: float) -> tuple[Path, tuple[TimelineEntry, ...]]:
        """File of a session and the entries already in it. A matching file whose last entry is not later than
        time_s is continued; one that is later (a recording replayed again) is renamed to
        <stem>.timeline.<n>.jsonl and a new file is started under its name, or under the free numbered name itself
        when the rename fails (a file held open by a sync client). Without a matching file the newest file of the
        kind stamped between the start and now whose last entry is not later than time_s is continued (the start
        computed from the session clock moves when the game is paused). Unreadable lines are skipped. A disk error
        raises OSError.
        In:  kind          str    "practice" | "qualify" | "race" | ...
             started_unix  float  s unix time the session started
             time_s        float  s session time now
        Out:               tuple[Path, tuple[TimelineEntry, ...]]  (file to append to, entries to continue from)
        """
        self._root.mkdir(parents=True, exist_ok=True)
        names = [p.name for p in self._root.iterdir()]
        name = matching_file(names, kind, started_unix, self._resume_margin_s)
        if name is None:
            window = files_in_window(names, kind, started_unix - self._resume_margin_s, started_unix + time_s)
            for candidate in window:
                entries = _read(self._root / candidate)
                if entries and entries[-1].time_s <= time_s:
                    return self._root / candidate, entries
            return self._root / file_name(kind, started_unix), ()
        path = self._root / name
        entries = _read(path)
        if not entries or entries[-1].time_s <= time_s:
            return path, entries
        stem = name.removesuffix(TIMELINE_SUFFIX)
        spare = next(
            spare
            for spare in (self._root / f"{stem}.timeline.{n}.jsonl" for n in itertools.count(1))
            if not spare.exists()
        )
        try:
            path.rename(spare)
        except OSError:
            return spare, ()
        return path, ()

    def append(self, path: Path, entries: Sequence[TimelineEntry]) -> None:
        """Append one JSON line per entry; a disk error raises OSError.
        In:  path     Path                     file returned by open
             entries  Sequence[TimelineEntry]
        Out:          None
        """
        with path.open("a", encoding="utf-8") as file:
            for entry in entries:
                file.write(json.dumps(dataclasses.asdict(entry), ensure_ascii=False) + "\n")


def _read(path: Path) -> tuple[TimelineEntry, ...]:
    """Entries stored in a timeline file; a line that cannot be read is skipped, also one cut inside a character.
    A disk error raises OSError.
    In:  path  Path
    Out:       tuple[TimelineEntry, ...]  in file order
    """
    entries: list[TimelineEntry] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entries.append(TimelineEntry(**json.loads(line)))
        except (ValueError, TypeError):
            continue
    return tuple(entries)
