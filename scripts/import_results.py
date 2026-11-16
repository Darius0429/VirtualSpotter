from pathlib import Path

import typer
import yaml

from ai_race_engineer.recorders.history_store import HistoryStore
from ai_race_engineer.recorders.results_import import import_report, import_results
from ai_race_engineer.telemetry.results_file import most_flagged_player


def main(folder: Path, driver: str = "", history: Path = Path("configs/history.yaml")) -> None:
    """Fill the history from the game's results files; it can be run again at any time, files already imported are
    skipped. Nothing is written into the results folder.
    In:  folder   Path  the game's results folder (<game>/UserData/Log/Results)
         driver   str   the player's driver name; "" = the name flagged as the player in the most files
         history  Path  history config YAML (history_file, track_field)
    Out:          None
    """
    hist_cfg = yaml.safe_load(history.read_text())
    store = HistoryStore(Path(hist_cfg["history_file"]))
    paths = sorted(folder.glob("*.xml"))
    name, flagged = (driver, 0) if driver else most_flagged_player(paths)
    print(f"driver: {name}" + ("" if driver else f" (flagged as the player in {flagged} of {len(paths)} files)"))
    print(import_report(import_results(paths, store, hist_cfg["track_field"], name)))


if __name__ == "__main__":
    typer.run(main)
