from collections.abc import Mapping, Sequence
from pathlib import Path

from ai_race_engineer.recorders.history_store import HistoryStore
from ai_race_engineer.strategy.history import figures_from_results
from ai_race_engineer.telemetry.results_file import read_results


def import_results(paths: Sequence[Path], store: HistoryStore, track_field: str, player_name: str) -> dict[str, int]:
    """Add the sessions of the game's results files to the history; files already in it are skipped, so it can be
    run again at any time. A disk error while appending raises OSError.
    In:  paths        Sequence[Path]  results files (.xml)
         store        HistoryStore
         track_field  str             results field equal to the game's track name
         player_name  str             the player's driver name
    Out:              dict[str, int]  files per outcome: "already imported", "unreadable", "without the driver or a
                                      lap", "added"
    """
    known = store.session_ids()
    counts = {"already imported": 0, "unreadable": 0, "without the driver or a lap": 0, "added": 0}
    for path in paths:
        if path.stem in known:
            counts["already imported"] += 1
            continue
        results = read_results(path)
        figures = None if results is None else figures_from_results(results, path.stem, track_field, player_name)
        if results is None:
            counts["unreadable"] += 1
        elif figures is None:
            counts["without the driver or a lap"] += 1
        else:
            store.append(figures)
            counts["added"] += 1
    return counts


def import_report(counts: Mapping[str, int]) -> str:
    """One line telling what an import did.
    In:  counts  Mapping[str, int]  files per outcome, from import_results
    Out:         str                e.g. "files: 12, already imported: 10, unreadable: 0, ..., added: 2"
    """
    return f"files: {sum(counts.values())}, " + ", ".join(f"{label}: {count}" for label, count in counts.items())
