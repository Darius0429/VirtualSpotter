import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from numpy.typing import NDArray

from ai_race_engineer.telemetry.frame import SessionInfo
from ai_race_engineer.telemetry.snapshot import SessionType

_PLAYER_COLUMNS: dict[str, str] = {
    "throttle": "throttle",
    "brake": "brake",
    "clutch": "clutch",
    "steering": "steering",
    "gear": "gear",
    "rpm": "rpm",
    "speed": "speed_mps",
    "position_x": "pos_x_m",
    "position_y": "pos_y_m",
    "position_z": "pos_z_m",
    "accel_lat": "accel_x_mps2",
    "accel_vert": "accel_y_mps2",
    "accel_long": "accel_z_mps2",
    "lap_number": "lap_number",
}
_SCORING_CHANNELS: tuple[str, ...] = ("timestamp", "session_time", "lap_distance", "lap_progress", "sector")
_CORNERS: tuple[str, ...] = ("fl", "fr", "rl", "rr")
_PIT_STATES: dict[int, int] = {0: 0, 1: 1, 2: 2, 4: 3}


def channels_frame(channels: Mapping[str, NDArray[Any]]) -> pd.DataFrame:
    """Frame of equally sampled channels, truncated to the shortest one (empty without channels).
    In:  channels  Mapping[str, NDArray[Any]]  channel name -> values
    Out:           pd.DataFrame                one column per channel
    """
    length = min((len(values) for values in channels.values()), default=0)
    return pd.DataFrame({name: values[:length] for name, values in channels.items()})


def clock_offset_s(timestamp: pd.Series, session_time: pd.Series) -> float:
    """Wall clock minus session clock, measured where the session clock ticks.
    In:  timestamp     pd.Series  s wall clock (unix)
         session_time  pd.Series  s session clock (steps at the scoring rate)
    Out:               float      s
    """
    step = session_time.diff()
    ticks = step.notna() & step.ne(0)
    return float((timestamp - session_time)[ticks].median())


def player_telemetry_table(telemetry: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    """player_telemetry_100hz rows: one per telemetry tick (repeated polls dropped), on the session clock.
    In:  telemetry  pd.DataFrame  old telemetry_data channels
         offset_s   float         s wall clock minus session clock
    Out:            pd.DataFrame  PlayerSample columns (heading from yaw: forward = (sin h, cos h) in world (x, z))
    """
    physics = telemetry.drop(columns=list(_SCORING_CHANNELS))
    ticks = telemetry[physics.ne(physics.shift()).any(axis=1)]
    rows = ticks[list(_PLAYER_COLUMNS)].rename(columns=_PLAYER_COLUMNS)
    rows.insert(0, "time_s", ticks["timestamp"] - offset_s)
    rows["heading_rad"] = (math.pi - ticks["orientation_yaw"] + math.pi) % (2 * math.pi) - math.pi
    rows["lap_distance_m"] = ticks["lap_distance"]
    return rows.reset_index(drop=True)


def player_state_table(telemetry: pd.DataFrame, static: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    """player_state_5hz rows: one per session clock tick, with remaining time and temperatures from static data.
    In:  telemetry  pd.DataFrame  old telemetry_data channels
         static     pd.DataFrame  old static_data channels (may be empty)
         offset_s   float         s wall clock minus session clock
    Out:            pd.DataFrame  StateSample columns that the old data has
    """
    ticks = telemetry[telemetry["session_time"].ne(telemetry["session_time"].shift())]
    rows = pd.DataFrame(
        {
            "time_s": ticks["timestamp"] - offset_s,
            "session_time_s": ticks["session_time"],
            "fuel_l": ticks["fuel"],
            **{f"tyre_temp_c_{c}": ticks[f"temp_tire_{c}"] for c in _CORNERS},
        }
    ).reset_index(drop=True)
    columns = {"session_remaining": "remaining_s", "track_temp": "track_temp_c", "ambient_temp": "ambient_temp_c"}
    weather = static.reindex(columns=["timestamp", *columns]).astype(float).rename(columns=columns)
    weather.insert(0, "time_s", weather.pop("timestamp") - offset_s)
    return pd.merge_asof(rows, weather.sort_values("time_s"), on="time_s")


def cars_tables(vehicles: pd.DataFrame, offset_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """cars_positions_50hz and cars_state_5hz rows from the old vehicles_data (text numbers parsed), at its own rate.
    In:  vehicles  pd.DataFrame  old vehicles_data channels (may be empty)
         offset_s  float         s wall clock minus session clock
    Out:           tuple[pd.DataFrame, pd.DataFrame]  (CarSample columns, CarStatus columns); heading is null,
                   slot_id is the old logger's scoring array index (not mID), the old pit_state 4 (stopped)
                   becomes the game's 3 and the ambiguous 5 becomes null
    """
    v = vehicles.reindex(
        columns=[
            "timestamp",
            "index",
            "is_player",
            "position",
            "driver_name",
            "vehicle_name",
            "vehicle_class",
            "lap_number",
            "lap_distance",
            "last_laptime",
            "best_laptime",
            "world_x",
            "world_y",
            "world_z",
            "speed",
            "in_pit",
            "pit_state",
            "num_pitstops",
        ]
    )
    time_s = pd.to_numeric(v["timestamp"]) - offset_s
    lap_distance_m = pd.to_numeric(v["lap_distance"])
    positions = pd.DataFrame(
        {
            "time_s": time_s,
            "slot_id": v["index"],
            "pos_x_m": pd.to_numeric(v["world_x"]),
            "pos_y_m": pd.to_numeric(v["world_y"]),
            "pos_z_m": pd.to_numeric(v["world_z"]),
            "speed_mps": pd.to_numeric(v["speed"]),
            "lap_distance_m": lap_distance_m,
        }
    )
    state = pd.DataFrame(
        {
            "time_s": time_s,
            "slot_id": v["index"],
            "is_player": v["is_player"],
            "driver_name": v["driver_name"],
            "vehicle_name": v["vehicle_name"],
            "class_name": v["vehicle_class"],
            "place": v["position"],
            "completed_laps": v["lap_number"],
            "lap_distance_m": lap_distance_m,
            "best_lap_s": _lap_time(v["best_laptime"]),
            "last_lap_s": _lap_time(v["last_laptime"]),
            "in_pits": v["in_pit"],
            "pit_state": v["pit_state"].map(_PIT_STATES),
            "pitstops": v["num_pitstops"],
        }
    )
    return positions.reset_index(drop=True), state.reset_index(drop=True)


def session_info(meta: Mapping[str, str], static: pd.DataFrame) -> SessionInfo:
    """Session info from the old metadata attributes and static data; values the old data lacks are None.
    In:  meta    Mapping[str, str]  metadata attributes (track_name, vehicle_name, session_type 0-4)
         static  pd.DataFrame       old static_data channels (may be empty)
    Out:         SessionInfo
    """
    last = static.reindex(columns=["track_length", "vehicle_class"]).dropna().tail(1)
    return SessionInfo(
        game="lmu",
        track_name=meta["track_name"],
        track_length_m=float(last["track_length"].iloc[0]) if len(last) else None,
        vehicle_name=meta["vehicle_name"],
        class_name=str(last["vehicle_class"].iloc[0]) if len(last) else None,
        session_type=SessionType(int(meta["session_type"])),
        session_number=None,
        max_laps=None,
    )


def group_split_sessions(metas: Sequence[tuple[Path, Mapping[str, str]]]) -> list[list[Path]]:
    """Group consecutive folders that one session was split into.
    Folders belong together when combo_id, vehicle_name and the session stamp (first number of session_id) match
    and the session clock at the folder start (second number of session_id) keeps increasing.
    In:  metas  Sequence[tuple[Path, Mapping[str, str]]]  (folder, metadata attributes) per folder
    Out:        list[list[Path]]                          folders per session, in time order
    """
    ordered = sorted(metas, key=lambda item: (item[1]["timestamp"], item[0].name))
    groups: list[list[Path]] = []
    prev: tuple[str, str, int, int] | None = None
    for folder, meta in ordered:
        stamp, start_s, *_ = (int(n) for n in meta["session_id"].strip("()").split(","))
        key = (meta["combo_id"], meta["vehicle_name"], stamp, start_s)
        if prev is not None and key[:3] == prev[:3] and start_s > prev[3]:
            groups[-1].append(folder)
        else:
            groups.append([folder])
        prev = key
    return groups


def _lap_time(text: pd.Series) -> pd.Series:
    """Lap times parsed from text; the old logger's values <= 0 (no time) become null.
    In:  text  pd.Series  s as text
    Out:       pd.Series  s
    """
    seconds = pd.to_numeric(text)
    return seconds.where(seconds > 0)
