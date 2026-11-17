import logging
from pathlib import Path

import typer
import yaml

from ai_race_engineer.domains.common.recording.parquet_session import ParquetSessionWriter
from ai_race_engineer.domains.common.recording.session_recorder import SessionRecorder
from ai_race_engineer.domains.common.data.frame_source import FrameSource
from ai_race_engineer.domains.lmu.lmu_frames import LMUFrameSource
from ai_race_engineer.domains.rf2.rf2_frames import RF2FrameSource

FRAME_SOURCES: dict[str, type[FrameSource]] = {"lmu": LMUFrameSource, "rf2": RF2FrameSource}


def main(domain: str = "lmu", config: Path | None = None) -> None:
    """Record every live session to Parquet until Ctrl+C.
    In:  domain  str          key of FRAME_SOURCES, e.g. "lmu"
         config  Path | None  the domain's YAML (recorder: root, poll_hz, flush_s, part_s); None = configs/<domain>.yaml
    Out:         None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rec_cfg = yaml.safe_load((config or Path(f"configs/{domain}.yaml")).read_text())["recorder"]
    source: FrameSource = FRAME_SOURCES[domain]()
    rec = SessionRecorder(
        source, ParquetSessionWriter(rec_cfg["part_s"]), Path(rec_cfg["root"]), rec_cfg["poll_hz"], rec_cfg["flush_s"]
    )
    with source:
        rec.run()


if __name__ == "__main__":
    typer.run(main)
