import logging
from pathlib import Path

import typer
import yaml

from ai_race_engineer.recorders.h5_import import import_all, import_folders

app = typer.Typer()


@app.command()
def one(folder: Path, recorder: Path = Path("configs/recorder.yaml")) -> None:
    """Convert one old telemetry.h5 folder into the session layout.
    In:  folder    Path  folder containing telemetry.h5
         recorder  Path  recorder config YAML (root: where session folders go)
    Out:           None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rec_cfg = yaml.safe_load(recorder.read_text())
    import_folders([folder], Path(rec_cfg["root"]))


@app.command(name="all")
def all_(recorder: Path = Path("configs/recorder.yaml")) -> None:
    """Convert every old telemetry.h5 folder, merging folders split from one session.
    In:  recorder  Path  recorder config YAML (h5_root: old folders, root: where session folders go)
    Out:           None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rec_cfg = yaml.safe_load(recorder.read_text())
    import_all(Path(rec_cfg["h5_root"]), Path(rec_cfg["root"]))


if __name__ == "__main__":
    app()
