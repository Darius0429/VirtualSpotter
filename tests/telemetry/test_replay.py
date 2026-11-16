from pathlib import Path

import pandas as pd
import pytest

from ai_race_engineer.telemetry import replay
from ai_race_engineer.telemetry.replay import CAR_COLUMNS, INFO_COLUMNS, PLAYER_COLUMNS, SCORING_COLUMNS, ReplaySource

TEXT_COLUMNS: tuple[str, ...] = (
    "mTrackName",
    "mYellowFlagState",
    "mDriverName",
    "mVehicleName",
    "mVehicleClass",
    "mFrontTireCompoundName",
    "mRearTireCompoundName",
)


def _write(folder: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Write a raw-dump table whose unnamed columns are zero (text columns empty).
    In:  folder   Path                      table folder
         columns  tuple[str, ...]           all column names
         rows     list[dict[str, object]]   the values that matter, per row
    Out:          None
    """
    blank = {name: b"" if name in TEXT_COLUMNS else 0.0 for name in columns}
    folder.mkdir(parents=True)
    pd.DataFrame([{**blank, **row} for row in rows]).to_parquet(folder / "part-000.parquet")


@pytest.fixture
def dump(tmp_path: Path) -> Path:
    """Raw dump of 10 s at 1 Hz with the player (slot 0) and one car (slot 5) whose telemetry stops after 3 s.
    In:  tmp_path  Path
    Out:           Path  dump folder
    """
    times = [float(t) for t in range(11)]
    _write(tmp_path / "raw_scoring_info_5hz", INFO_COLUMNS, [{"poll_s": t, "mSession": 10} for t in times])
    _write(
        tmp_path / "raw_cars_scoring_5hz",
        SCORING_COLUMNS,
        [{"poll_s": t, "mID": slot_id, "mIsPlayer": slot_id == 0} for t in times for slot_id in (0, 5)],
    )
    _write(
        tmp_path / "raw_player_telemetry_100hz", PLAYER_COLUMNS, [{"poll_s": t, "mID": 0, "mFuel": 50.0} for t in times]
    )
    _write(
        tmp_path / "raw_cars_telemetry_50hz", CAR_COLUMNS, [{"poll_s": t, "mID": 5, "mFuel": 40.0} for t in times[:4]]
    )
    return tmp_path


def _read_at(source: ReplaySource, monkeypatch: pytest.MonkeyPatch, time_s: float) -> list[int]:
    """Slot ids with resources at a replay time.
    In:  source       ReplaySource  opened at wall time 0
         monkeypatch  pytest.MonkeyPatch
         time_s       float         s replay time
    Out:              list[int]
    """
    monkeypatch.setattr(replay.time, "monotonic", lambda: time_s)
    return [r.slot_id for r in source.read().resources]


def test_stale_car_telemetry_is_not_used_for_resources(dump: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A car's last telemetry row counts for 2 s of replay time; after that the car has no resources."""
    monkeypatch.setattr(replay.time, "monotonic", lambda: 0.0)
    source = ReplaySource(dump)
    source.open()
    assert _read_at(source, monkeypatch, 3.5) == [0, 5]
    assert _read_at(source, monkeypatch, 5.0) == [0, 5]
    assert _read_at(source, monkeypatch, 5.5) == [0]
    assert len(source.read().vehicles) == 2


def test_no_snapshot_outside_the_recording(dump: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the last recorded row there is no snapshot."""
    monkeypatch.setattr(replay.time, "monotonic", lambda: 0.0)
    source = ReplaySource(dump)
    source.open()
    monkeypatch.setattr(replay.time, "monotonic", lambda: 10.5)
    assert source.read() is None
