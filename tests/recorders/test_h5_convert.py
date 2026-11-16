import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_race_engineer.recorders.h5_convert import (
    cars_tables,
    channels_frame,
    clock_offset_s,
    group_split_sessions,
    player_telemetry_table,
)


def _telemetry() -> pd.DataFrame:
    """Old telemetry_data with 4 polls: poll 2 repeats poll 1, poll 3 only changes the scoring lap distance.
    In:  -
    Out: pd.DataFrame  old telemetry_data channels; wall clock = session clock + 1000 s
    """
    rows = pd.DataFrame(
        {
            "timestamp": [1000.0, 1000.01, 1000.02, 1000.03],
            "session_time": [0.0, 0.0, 0.0, 0.0],
            "lap_distance": [5.0, 5.0, 6.0, 6.0],
            "lap_progress": 0.0,
            "sector": 1,
            "throttle": [0.5, 0.5, 0.5, 0.7],
            "orientation_yaw": 0.0,
            "fuel": 50.0,
        }
    )
    for name in ("brake", "clutch", "steering", "gear", "rpm", "speed", "lap_number"):
        rows[name] = 1
    for axis in ("x", "y", "z"):
        rows[f"position_{axis}"] = 0.0
    for name in ("accel_lat", "accel_long", "accel_vert"):
        rows[name] = 0.0
    for corner in ("fl", "fr", "rl", "rr"):
        rows[f"temp_tire_{corner}"] = 80.0
    return rows


def test_channels_frame_truncates() -> None:
    """Channels are cut to the shortest one."""
    frame = channels_frame({"a": np.arange(5), "b": np.arange(3)})
    assert len(frame) == 3


def test_duplicates_dropped_first_kept() -> None:
    """Repeated polls are dropped (scoring-only changes too); the first poll of each run is kept."""
    rows = player_telemetry_table(_telemetry(), 1000.0)
    assert list(rows["throttle"]) == [0.5, 0.7]
    assert list(rows["time_s"]) == pytest.approx([0.0, 0.03])


def test_clock_offset() -> None:
    """Offset is the median of wall minus session clock where the session clock ticks."""
    timestamp = pd.Series([1000.1, 1000.15, 1000.3, 1000.5, 1000.55])
    session_time = pd.Series([0.0, 0.0, 0.2, 0.4, 0.4])
    assert clock_offset_s(timestamp, session_time) == pytest.approx(1000.1)


def test_columns_renamed() -> None:
    """Old channel names become PlayerSample fields; heading uses forward = (sin h, cos h)."""
    rows = player_telemetry_table(_telemetry(), 1000.0)
    assert {"speed_mps", "pos_x_m", "accel_x_mps2", "accel_z_mps2", "lap_distance_m"} <= set(rows.columns)
    assert rows["heading_rad"].iloc[0] == pytest.approx(-math.pi)


def test_vehicle_strings_parsed_and_missing_null() -> None:
    """Text numbers of vehicles_data are parsed, old pit state 4 becomes stopped (3), no time and missing give null."""
    vehicles = pd.DataFrame(
        {
            "timestamp": ["1000.500"],
            "index": [3],
            "world_x": ["1.5"],
            "speed": ["60.000"],
            "last_laptime": ["-1.000"],
            "lap_number": [4],
            "pit_state": [4],
        }
    )
    positions, state = cars_tables(vehicles, 1000.0)
    assert (positions["time_s"].iloc[0], positions["pos_x_m"].iloc[0], positions["speed_mps"].iloc[0]) == (
        0.5,
        1.5,
        60.0,
    )
    assert pd.isna(positions["pos_z_m"].iloc[0])
    assert pd.isna(state["last_lap_s"].iloc[0])
    assert state["completed_laps"].iloc[0] == 4
    assert state["pit_state"].iloc[0] == 3


def test_ambiguous_pit_state_null() -> None:
    """Old pit state 4 becomes stopped (3); the ambiguous old 5 becomes null."""
    _, state = cars_tables(pd.DataFrame({"timestamp": ["1.0", "2.0"], "pit_state": [4, 5]}), 0.0)
    assert state["pit_state"].iloc[0] == 3
    assert pd.isna(state["pit_state"].iloc[1])


def _meta(timestamp: str, session_id: str, combo_id: str = "Sebring - GT3") -> dict[str, str]:
    """Old metadata attributes.
    In:  timestamp   str  YYYYmmdd_HHMMSS
         session_id  str  '(stamp, session clock at start, laps)'
         combo_id    str
    Out:             dict[str, str]
    """
    return {"timestamp": timestamp, "session_id": session_id, "combo_id": combo_id, "vehicle_name": "Car #1"}


def test_split_sessions_grouped() -> None:
    """Consecutive folders of the same session (same stamp, clock increasing) form one group."""
    metas = [
        (Path("b"), _meta("20250101_100500", "(93001, 300, 2)")),
        (Path("a"), _meta("20250101_100000", "(93001, 10, 0)")),
    ]
    assert group_split_sessions(metas) == [[Path("a"), Path("b")]]


def test_split_sessions_separated() -> None:
    """A different stamp or combo, or the session clock restarting, splits groups."""
    metas = [
        (Path("a"), _meta("20250101_100000", "(93001, 10, 0)")),
        (Path("b"), _meta("20250101_100100", "(93001, 70, 0)", combo_id="Spa - GT3")),
        (Path("c"), _meta("20250101_100200", "(93001, 130, 0)")),
        (Path("d"), _meta("20250101_100300", "(123005, 190, 0)")),
        (Path("e"), _meta("20250101_100400", "(123005, 5, 0)")),
    ]
    assert group_split_sessions(metas) == [[Path("a")], [Path("b")], [Path("c")], [Path("d")], [Path("e")]]
