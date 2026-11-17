from collections.abc import Callable, Mapping
from pathlib import Path

import typer
import yaml

from ai_race_engineer.domains.common.recording.part_writer import PartWriter
from ai_race_engineer.domains.common.recording.run_folder import RunFolder
from ai_race_engineer.domains.lmu.raw_recorder import RawRecorder
from ai_race_engineer.domains.lmu.raw_tables import to_columns
from ai_race_engineer.domains.lmu.rest_capture import RestCapture

RECORDERS: dict[str, Callable[[Mapping[str, object], Path, float | None], RawRecorder]] = {
    "lmu": lambda cfg, folder, stop_after_s: RawRecorder(
        folder,
        PartWriter(to_columns, cfg["part_s"]),
        RestCapture(**cfg["rest"]),
        cfg["poll_hz"],
        cfg["flush_s"],
        cfg["stale_s"],
        cfg["finish_grace_s"],
        cfg["restart_back_s"],
        cfg["progress_s"],
        stop_after_s,
    ),
}


def main(domain: str = "lmu", config: Path | None = None, stop_after_s: float | None = None) -> None:
    """Record everything the game publishes until Ctrl+C: one folder per session under <run folder>/game, the run
    folder shared with an engineer program that is running.
    In:  domain        str           key of RECORDERS, e.g. "lmu"
         config        Path | None   the domain's YAML (runs, recorder); None = configs/<domain>.yaml
         stop_after_s  float | None  s after which the recording stops by itself; None = until Ctrl+C
    Out:               None
    """
    cfg = yaml.safe_load((config or Path(f"configs/{domain}.yaml")).read_text())
    with RunFolder(Path(cfg["runs"]), "recorder", "engineer") as run:
        RECORDERS[domain](cfg["recorder"], run.path / "game", stop_after_s).run()


if __name__ == "__main__":
    typer.run(main)
