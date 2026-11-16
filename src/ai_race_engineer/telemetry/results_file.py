import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ai_race_engineer.telemetry.results import SessionResults, parse_results

KIND_LETTERS: dict[str, str] = {"practice": "P", "qualify": "Q", "race": "R"}


def read_results(path: Path) -> SessionResults | None:
    """Session results of one results file.
    In:  path  Path                   results file (.xml)
    Out:       SessionResults | None  None for a missing, unreadable or half-written file, also one with a number
                                      field that is empty or not a number
    """
    try:
        return parse_results(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def latest_results(folder: Path, kind: str) -> SessionResults | None:
    """Session results of the newest file of a kind in the game's results folder (file names start with the date).
    In:  folder  Path                   results folder
         kind    str                    "practice" | "qualify" | "race"
    Out:         SessionResults | None  None without such a file or when it cannot be read
    """
    pattern = re.compile(rf"{KIND_LETTERS[kind]}\d+$")
    try:
        paths = sorted(p for p in folder.glob("*.xml") if pattern.search(p.stem))
    except OSError:
        return None
    return read_results(paths[-1]) if paths else None


def most_flagged_player(paths: Sequence[Path]) -> tuple[str, int]:
    """The driver name flagged as the player in the largest number of results files. Offline files flag only the
    local player, online files every human driver, so over a folder the local player is flagged most often.
    In:  paths  Sequence[Path]   results files; unreadable ones are skipped
    Out:        tuple[str, int]  (driver name, files that flag it); ("", 0) when no file flags anyone
    """
    counts: Counter[str] = Counter()
    for path in paths:
        results = read_results(path)
        if results is not None:
            counts.update({d.name for d in results.drivers if d.is_player})
    return counts.most_common(1)[0] if counts else ("", 0)
